import json
import logging
import os
import uuid
from datetime import datetime, timezone

import functions_framework
import pandas as pd
from cloudevents.http import CloudEvent
import google.auth

from google.cloud import bigquery
from google.cloud import storage

# google genai (Vertex AI)
from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("review-pipeline")

# -----------------------------
# Config (Project ID 안전하게 가져오기)
# -----------------------------
def _get_project_id() -> str:
    env_pid = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("PROJECT_ID")
        or os.getenv("BQ_PROJECT_ID")
    )
    if env_pid:
        return env_pid

    _, pid = google.auth.default()
    if not pid:
        raise RuntimeError(
            "Project ID를 찾지 못했습니다. Cloud Run에 PROJECT_ID(또는 GOOGLE_CLOUD_PROJECT) env를 설정하세요."
        )
    return pid

PROJECT_ID = _get_project_id()
DATASET = os.getenv("BQ_DATASET", "ths_review_analytics")

TABLE_INGEST = f"{PROJECT_ID}.{DATASET}.ingestion_files"
TABLE_RAW = f"{PROJECT_ID}.{DATASET}.reviews_raw"
TABLE_CLEAN = f"{PROJECT_ID}.{DATASET}.reviews_clean"

# Batch 관련
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "asia-northeast3")
VERTEX_GEMINI_MODEL = os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.5-flash-lite")

# JSONL 저장 버킷/프리픽스
BATCH_BUCKET = os.getenv("BATCH_BUCKET", "ths-review-upload-bkt")
BATCH_INPUT_PREFIX = os.getenv("BATCH_INPUT_PREFIX", "batch_inputs")
BATCH_OUTPUT_PREFIX = os.getenv("BATCH_OUTPUT_PREFIX", f"gs://{BATCH_BUCKET}/batch_outputs/")

# 엑셀 헤더(한글) -> 표준 컬럼명(BQ)
EXCEL_TO_STD = {
    "상품평작성일자": "write_date",
    "상품평리뷰번호": "review_no",
    "리뷰SEQ": "review_seq",
    "채널구분": "channel",
    "브랜드코드": "brand_no",
    "스타일코드": "product_no",          # 내부에서 product_no로 쓰기로 했으면 OK
    "대카테고리": "review_1depth",
    "소카테고리": "review_2depth",
    "리뷰별점": "review_score",
    "상품평리뷰내용(원글)": "review_contents",
    "리뷰분석점수": "review_ai_score",
    "리뷰분석내용": "review_ai_contents",
}

EXPECTED_STD_COLS = [
    "write_date",
    "review_no",
    "review_seq",
    "channel",
    "brand_no",
    "product_no",
    "review_1depth",
    "review_2depth",
    "review_score",
    "review_contents",
    "review_ai_score",
    "review_ai_contents",
]


# -----------------------------
# Clients
# -----------------------------
def _bq() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)

def _gcs() -> storage.Client:
    return storage.Client(project=PROJECT_ID)

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# -----------------------------
# Ingestion idempotency / safety
# -----------------------------
def _already_done(bucket: str, object_name: str, generation: str) -> bool:
    client = _bq()
    sql = f"""
    SELECT status
    FROM `{TABLE_INGEST}`
    WHERE bucket=@bucket AND object_name=@object_name AND generation=@generation
    ORDER BY received_at DESC
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("bucket", "STRING", bucket),
            bigquery.ScalarQueryParameter("object_name", "STRING", object_name),
            bigquery.ScalarQueryParameter("generation", "STRING", generation),
        ]
    )
    rows = list(client.query(sql, job_config=job_config).result())
    return bool(rows) and rows[0]["status"] == "DONE"

def _mark_ingestion(status: str, bucket: str, object_name: str, generation: str, error_message: str = None):
    client = _bq()
    sql = f"""
    INSERT INTO `{TABLE_INGEST}` (bucket, object_name, generation, received_at, status, error_message)
    VALUES (@bucket, @object_name, @generation, @received_at, @status, @error_message)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("bucket", "STRING", bucket),
            bigquery.ScalarQueryParameter("object_name", "STRING", object_name),
            bigquery.ScalarQueryParameter("generation", "STRING", generation),
            bigquery.ScalarQueryParameter("received_at", "TIMESTAMP", _now_utc()),
            bigquery.ScalarQueryParameter("status", "STRING", status),
            bigquery.ScalarQueryParameter("error_message", "STRING", error_message),
        ]
    )
    client.query(sql, job_config=job_config).result()

def _raw_already_loaded(bucket: str, object_name: str, generation: str) -> bool:
    """재시도 시 reviews_raw 중복 append 방지용"""
    client = _bq()
    sql = f"""
    SELECT 1
    FROM `{TABLE_RAW}`
    WHERE source_bucket=@bucket AND source_object=@object_name AND source_generation=@generation
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("bucket", "STRING", bucket),
            bigquery.ScalarQueryParameter("object_name", "STRING", object_name),
            bigquery.ScalarQueryParameter("generation", "STRING", generation),
        ]
    )
    rows = list(client.query(sql, job_config=job_config).result())
    return bool(rows)


# -----------------------------
# GCS + Excel load
# -----------------------------
def _download_xlsx(bucket: str, name: str) -> str:
    local_path = f"/tmp/{uuid.uuid4().hex}.xlsx"
    client = _gcs()
    blob = client.bucket(bucket).blob(name)
    blob.download_to_filename(local_path)
    return local_path

def _load_excel_mapped(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    df = df.rename(columns=EXCEL_TO_STD)

    missing = [c for c in EXPECTED_STD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"컬럼 매핑 후 누락: {missing}. 현재 컬럼={list(df.columns)}")

    return df[EXPECTED_STD_COLS].copy()


# -----------------------------
# BigQuery writes
# -----------------------------
def _append_reviews_raw(df_std: pd.DataFrame, bucket: str, name: str, generation: str):
    """
    reviews_raw는 STRING 위주(loaded_at만 TIMESTAMP)
    """
    client = _bq()
    loaded_at = _now_utc()
    ingest_id = uuid.uuid4().hex

    df = df_std.copy()

    df_raw = pd.DataFrame({
        "ingest_id": ingest_id,
        "source_bucket": bucket,
        "source_object": name,
        "source_generation": generation,
        "loaded_at": loaded_at,

        "write_date": df["write_date"].astype(str),
        "review_no": df["review_no"].astype(str),
        "review_seq": df["review_seq"].astype(str),
        "channel": df["channel"].astype(str),
        "brand_no": df["brand_no"].astype(str),
        "product_no": df["product_no"].astype(str),
        "review_1depth": df["review_1depth"].astype(str),
        "review_2depth": df["review_2depth"].astype(str),
        "review_score": df["review_score"].astype(str),
        "review_contents": df["review_contents"].astype(str),
        "review_ai_score": df["review_ai_score"].astype(str),
        "review_ai_contents": df["review_ai_contents"].astype(str),
    })

    job = client.load_table_from_dataframe(
        df_raw,
        TABLE_RAW,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    )
    job.result()
    logger.info("BQ append reviews_raw rows=%d ingest_id=%s", len(df_raw), ingest_id)

def _merge_reviews_clean(df_std: pd.DataFrame):
    """
    reviews_clean은 타입 파싱(날짜/숫자) + review_key 생성 + MERGE upsert
    """
    client = _bq()
    loaded_at = _now_utc()

    df = df_std.copy()
    df["review_seq"] = pd.to_numeric(df["review_seq"], errors="coerce").fillna(0).astype(int)
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce").fillna(0).astype(int)
    df["write_date"] = pd.to_datetime(df["write_date"], errors="coerce").dt.date

    df["review_no"] = df["review_no"].astype(str)
    df["review_key"] = df["review_no"] + "-" + df["review_seq"].astype(str)

    # MVP: 아직 DLP 마스킹 전이므로 원문을 그대로 masked로 채움
    df["review_text_masked"] = df["review_contents"].astype(str)
    df["normalized_text"] = df["review_contents"].astype(str)

    df_stage = pd.DataFrame({
        "review_key": df["review_key"].astype(str),
        "review_no": df["review_no"].astype(str),
        "review_seq": df["review_seq"].astype(int),
        "channel": df["channel"].astype(str),
        "brand_no": df["brand_no"].astype(str),
        "product_no": df["product_no"].astype(str),

        "write_date": df["write_date"],
        "review_score": df["review_score"].astype(int),

        "review_1depth": df["review_1depth"].astype(str),
        "review_2depth": df["review_2depth"].astype(str),

        "review_contents": df["review_contents"].astype(str),
        "review_text_masked": df["review_text_masked"].astype(str),
        "normalized_text": df["normalized_text"].astype(str),

        "legacy_review_ai_score": pd.to_numeric(df["review_ai_score"], errors="coerce"),
        "legacy_review_ai_contents": df["review_ai_contents"].astype(str),

        "loaded_at": loaded_at,
    })

    staging_table = f"{PROJECT_ID}.{DATASET}.staging_reviews_clean_{uuid.uuid4().hex[:10]}"
    client.load_table_from_dataframe(
        df_stage,
        staging_table,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    ).result()

    merge_sql = f"""
    MERGE `{TABLE_CLEAN}` T
    USING `{staging_table}` S
    ON T.review_key = S.review_key
    WHEN MATCHED THEN UPDATE SET
      review_no = S.review_no,
      review_seq = S.review_seq,
      channel = S.channel,
      brand_no = S.brand_no,
      product_no = S.product_no,
      write_date = S.write_date,
      review_score = S.review_score,
      review_1depth = S.review_1depth,
      review_2depth = S.review_2depth,
      review_contents = S.review_contents,
      review_text_masked = S.review_text_masked,
      normalized_text = S.normalized_text,
      legacy_review_ai_score = S.legacy_review_ai_score,
      legacy_review_ai_contents = S.legacy_review_ai_contents,
      loaded_at = S.loaded_at
    WHEN NOT MATCHED THEN
      INSERT ROW
    """
    client.query(merge_sql).result()
    client.delete_table(staging_table, not_found_ok=True)

    logger.info("BQ MERGE reviews_clean done rows=%d", len(df_stage))


# -----------------------------
# Batch input(JSONL) builder
# -----------------------------
SQL_NEW_REVIEWS_FOR_FILE = f"""
WITH file_rows AS (
  SELECT DISTINCT
    review_no,
    SAFE_CAST(review_seq AS INT64) AS review_seq
  FROM `{PROJECT_ID}.{DATASET}.reviews_raw`
  WHERE source_bucket = @bucket
    AND source_object = @object_name
    AND source_generation = @generation
),
targets AS (
  SELECT
    c.review_key,
    c.brand_no,
    c.product_no,
    c.channel,
    c.review_text_masked,
    c.write_date,
    c.review_score
  FROM `{PROJECT_ID}.{DATASET}.reviews_clean` c
  JOIN file_rows r
    ON c.review_no = r.review_no
   AND c.review_seq = r.review_seq
)
SELECT t.*
FROM targets t
LEFT JOIN `{PROJECT_ID}.{DATASET}.review_llm_extract` e
  ON e.review_key = t.review_key
WHERE e.review_key IS NULL
"""

def _build_prompt(row: dict) -> str:
    review_text = (row.get("review_text_masked") or "").strip()
    if len(review_text) > 1200:
        review_text = review_text[:1200] + "…"

    return f"""
너는 패션/의류 상품평을 생산/디자인/QC 관점에서 구조화하는 분석기다.
반드시 JSON 한 개만 출력한다. (설명/문장/코드블록 금지)

허용값:
- issue_category: ["봉제","원단","색상","사이즈핏","내구성","냄새","배송포장","기타"]
- severity: ["경미","보통","중대"]
- sentiment: ["불만","중립","칭찬"]
- size_feedback: ["작다","크다","정사이즈","불명"]
- repurchase_intent: ["있음","없음","불명"]

출력 스키마(키 이름 고정):
{{
  "review_key": "{row["review_key"]}",
  "brand_no": "{row.get("brand_no","")}",
  "product_no": "{row.get("product_no","")}",
  "channel": "{row.get("channel","")}",
  "issue_category": "",
  "severity": "",
  "sentiment": "",
  "signals": {{
    "size_feedback": "",
    "defect_part": "",
    "color_mentioned": "",
    "repurchase_intent": ""
  }},
  "evidence": "",
  "action_suggestion": ""
}}

리뷰 텍스트:
{review_text}
""".strip()

def make_batch_input_jsonl_and_upload(bucket: str, object_name: str, generation: str) -> str:
    bq = _bq()
    gcs = _gcs()

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("bucket", "STRING", bucket),
            bigquery.ScalarQueryParameter("object_name", "STRING", object_name),
            bigquery.ScalarQueryParameter("generation", "STRING", generation),
        ]
    )

    tmp_path = f"/tmp/batch_input_{uuid.uuid4().hex}.jsonl"
    rows_written = 0

    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in bq.query(SQL_NEW_REVIEWS_FOR_FILE, job_config=job_config).result(page_size=1000):
            row = dict(r)
            prompt = _build_prompt(row)

            line = {
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 256,
                        "responseMimeType": "application/json",
                    },
                }
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            rows_written += 1

    if rows_written == 0:
        logger.info("BATCH INPUT: no targets (already analyzed or empty)")
        return ""

    dest_blob = f"{BATCH_INPUT_PREFIX}/{object_name}/{generation}/batch_input.jsonl"
    gcs.bucket(BATCH_BUCKET).blob(dest_blob).upload_from_filename(
        tmp_path, content_type="application/jsonl"
    )

    input_uri = f"gs://{BATCH_BUCKET}/{dest_blob}"
    logger.info("BATCH INPUT uploaded: %s (rows=%d)", input_uri, rows_written)
    return input_uri


# -----------------------------
# Vertex AI Gemini Batch submit
# -----------------------------
def submit_vertex_batch_job(input_jsonl_gcs_uri: str, object_name: str, generation: str) -> str:
    """
    input_jsonl_gcs_uri: gs://.../batch_input.jsonl
    return: job.name (string)
    """
    if not input_jsonl_gcs_uri:
        return ""

    # output prefix를 파일별로 분리(결과 혼합 방지)
    # 예: gs://.../batch_outputs/review_test.xlsx/<gen>/
    output_prefix = BATCH_OUTPUT_PREFIX.rstrip("/") + f"/{object_name}/{generation}/"

    # Vertex AI 사용 설정 (SDK가 env를 참조)
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = VERTEX_LOCATION

    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    job = client.batches.create(
        model=VERTEX_GEMINI_MODEL,
        src=input_jsonl_gcs_uri,
        config=CreateBatchJobConfig(dest=output_prefix),
    )

    job_name = getattr(job, "name", "") or str(job)
    logger.info("BATCH SUBMITTED model=%s location=%s input=%s output=%s job=%s",
                VERTEX_GEMINI_MODEL, VERTEX_LOCATION, input_jsonl_gcs_uri, output_prefix, job_name)
    return job_name


# -----------------------------
# CloudEvent handler
# -----------------------------
@functions_framework.cloud_event
def ingest_from_gcs(cloud_event: CloudEvent):
    data = cloud_event.data or {}

    bucket = data.get("bucket")
    name = data.get("name")
    generation = str(data.get("generation", ""))

    logger.info("EVENT bucket=%s name=%s generation=%s", bucket, name, generation)
    logger.info("EVENT type=%s source=%s id=%s",
                cloud_event.get("type"), cloud_event.get("source"), cloud_event.get("id"))

    if not bucket or not name:
        logger.warning("Missing bucket/name in event payload. data=%s", data)
        return ("OK", 200)

    # 중복 처리 방지(완료된 파일/세대면 바로 종료)
    if _already_done(bucket, name, generation):
        logger.info("SKIP already DONE: %s/%s gen=%s", bucket, name, generation)
        return ("OK", 200)

    _mark_ingestion("STARTED", bucket, name, generation)

    try:
        # 1) 다운로드 & 로드
        local_path = _download_xlsx(bucket, name)
        df_std = _load_excel_mapped(local_path)

        # 2) raw 적재(재시도 시 중복 append 방지)
        if not _raw_already_loaded(bucket, name, generation):
            _append_reviews_raw(df_std, bucket, name, generation)
        else:
            logger.info("SKIP reviews_raw already loaded for %s/%s gen=%s", bucket, name, generation)

        # 3) clean merge (idempotent)
        _merge_reviews_clean(df_std)

        # 4) batch input 생성 + 업로드
        input_uri = make_batch_input_jsonl_and_upload(bucket, name, generation)

        # 5) batch job 제출 + job_name 로그
        job_name = submit_vertex_batch_job(input_uri, name, generation)
        if input_uri and not job_name:
            raise RuntimeError("Batch job submission failed (job_name empty)")

        # 6) 완료 마킹 (job_name은 로그로 남김)
        _mark_ingestion("DONE", bucket, name, generation, error_message=(f"BATCH_JOB={job_name}" if job_name else "NO_TARGETS"))
        return ("OK", 200)

    except Exception as e:
        logger.exception("FAILED processing %s/%s gen=%s", bucket, name, generation)
        _mark_ingestion("FAILED", bucket, name, generation, error_message=str(e)[:5000])
        return ("ERROR", 500)
