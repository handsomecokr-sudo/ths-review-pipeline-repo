import json
import logging
import os
import re
import uuid
import zipfile
from datetime import datetime, timezone

import functions_framework
import pandas as pd
from cloudevents.http import CloudEvent
import google.auth

from google.cloud import bigquery
from google.cloud import storage

# google genai (Vertex AI Batch)
from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions
from google.genai import errors as genai_errors

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

# -----------------------------
# Bucket split (중요)
# -----------------------------
# 업로드(트리거 대상) 버킷: 엑셀만 들어오는 곳
UPLOAD_BUCKET = os.getenv("UPLOAD_BUCKET", "ths-review-upload-bkt")

# 아카이브(결과 저장) 버킷: batch_inputs / batch_outputs 저장 (트리거 X 권장)
ARCHIVE_BUCKET = os.getenv("ARCHIVE_BUCKET", "ths-review-archive-bkt")

# 결과물 prefix
BATCH_INPUT_PREFIX = os.getenv("BATCH_INPUT_PREFIX", "batch_inputs")
BATCH_OUTPUT_PREFIX = os.getenv("BATCH_OUTPUT_PREFIX", f"gs://{ARCHIVE_BUCKET}/batch_outputs")

# 업로드 파일명 패턴(원하면 더 엄격하게): review_yyyymmdd.xlsx
UPLOAD_XLSX_PATTERN = os.getenv("UPLOAD_XLSX_PATTERN", r"^review_\d{8}\.xlsx$")

# -----------------------------
# Model / Vertex Batch (global 강제)
# -----------------------------
# 기본 모델(원하면 env로 교체)
VERTEX_GEMINI_MODEL = os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.5-flash-lite")  # :contentReference[oaicite:2]{index=2}
# 404 대비 fallback (가급적 “동작하는 모델” 우선)
MODEL_FALLBACKS = [
    os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.5-flash-lite"),
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# Batch는 global endpoint 사용
GENAI_LOCATION = "global"  # :contentReference[oaicite:3]{index=3}

# -----------------------------
# 엑셀 헤더(한글) -> 표준 컬럼명(BQ)
# -----------------------------
EXCEL_TO_STD = {
    "상품평작성일자": "write_date",
    "상품평리뷰번호": "review_no",
    "리뷰SEQ": "review_seq",
    "채널구분": "channel",
    "브랜드코드": "brand_no",
    "스타일코드": "product_no",
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
# (2) XLSX 검증
# -----------------------------
def _assert_xlsx(local_path: str, object_name: str):
    if not object_name.lower().endswith(".xlsx"):
        raise ValueError(f"Not an .xlsx file: {object_name}")

    with open(local_path, "rb") as f:
        sig = f.read(2)
    if sig != b"PK":
        raise ValueError("Uploaded file is not a valid .xlsx (zip header 'PK' not found)")

    try:
        with zipfile.ZipFile(local_path, "r") as zf:
            _ = zf.namelist()[:1]
    except zipfile.BadZipFile:
        raise ValueError("Uploaded file is not a valid .xlsx (bad zip archive)")


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
    client = _bq()
    loaded_at = _now_utc()
    ingest_id = uuid.uuid4().hex

    df = df_std.copy()

    df_raw = pd.DataFrame(
        {
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
        }
    )

    job = client.load_table_from_dataframe(
        df_raw,
        TABLE_RAW,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    )
    job.result()
    logger.info("BQ append reviews_raw rows=%d ingest_id=%s", len(df_raw), ingest_id)


def _merge_reviews_clean(df_std: pd.DataFrame):
    """
    안정화:
    - review_seq 파싱 실패는 0으로 몰지 말고 drop
    - staging 중복 review_key는 MERGE 소스에서 ROW_NUMBER로 1행만 남김
    """
    client = _bq()
    loaded_at = _now_utc()

    df = df_std.copy()
    df["review_no"] = df["review_no"].astype(str).str.strip()

    df["review_seq_num"] = pd.to_numeric(df["review_seq"], errors="coerce")
    df["review_score_num"] = pd.to_numeric(df["review_score"], errors="coerce")
    df["write_date_dt"] = pd.to_datetime(df["write_date"], errors="coerce").dt.date

    before = len(df)
    df = df[df["review_no"].notna() & (df["review_no"] != "")]
    df = df[df["review_seq_num"].notna()]
    df = df[df["write_date_dt"].notna()]
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("DROP rows due to invalid keys/parse (count=%d)", dropped)

    if len(df) == 0:
        logger.info("No valid rows to merge into reviews_clean")
        return

    df["review_seq"] = df["review_seq_num"].astype(int)
    df["review_score"] = df["review_score_num"].fillna(0).astype(int)
    df["write_date"] = df["write_date_dt"]

    df["review_key"] = df["review_no"] + "-" + df["review_seq"].astype(str)

    df["review_text_masked"] = df["review_contents"].astype(str)
    df["normalized_text"] = df["review_contents"].astype(str)

    df_stage = pd.DataFrame(
        {
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
        }
    )

    staging_table = f"{PROJECT_ID}.{DATASET}.staging_reviews_clean_{uuid.uuid4().hex[:10]}"
    client.load_table_from_dataframe(
        df_stage,
        staging_table,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    ).result()

    merge_sql = f"""
    MERGE `{TABLE_CLEAN}` T
    USING (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT
          S.*,
          ROW_NUMBER() OVER (PARTITION BY review_key ORDER BY loaded_at DESC) AS rn
        FROM `{staging_table}` S
      )
      WHERE rn = 1
    ) S
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
    c.review_text_masked
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


def make_batch_input_jsonl_and_upload(source_bucket: str, object_name: str, generation: str) -> str:
    """
    ⚠️ JSONL은 반드시 ARCHIVE_BUCKET에 저장 (업로드 버킷에 저장하면 재귀 트리거 위험)
    """
    bq = _bq()
    gcs = _gcs()

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("bucket", "STRING", source_bucket),
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

    safe_name = object_name.replace("/", "_")
    dest_blob = f"{BATCH_INPUT_PREFIX}/{safe_name}/{generation}/batch_input.jsonl"
    gcs.bucket(ARCHIVE_BUCKET).blob(dest_blob).upload_from_filename(
        tmp_path, content_type="application/jsonl"
    )

    input_uri = f"gs://{ARCHIVE_BUCKET}/{dest_blob}"
    logger.info("BATCH INPUT uploaded: %s (rows=%d)", input_uri, rows_written)
    return input_uri


# -----------------------------
# Vertex AI Batch submit (global 강제 + model fallback)
# -----------------------------
def _create_genai_client_global() -> genai.Client:
    # 문서 예시처럼 env로 VertexAI 사용 + global 설정 :contentReference[oaicite:4]{index=4}
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = GENAI_LOCATION
    return genai.Client(http_options=HttpOptions(api_version="v1"))


def submit_vertex_batch_job_global(input_jsonl_gcs_uri: str, object_name: str, generation: str) -> str:
    if not input_jsonl_gcs_uri:
        return ""

    safe_name = object_name.replace("/", "_")
    output_prefix = BATCH_OUTPUT_PREFIX.rstrip("/") + f"/{safe_name}/{generation}/"

    client = _create_genai_client_global()

    last_err = None
    tried = []

    for model_id in MODEL_FALLBACKS:
        if not model_id or model_id in tried:
            continue
        tried.append(model_id)

        try:
            job = client.batches.create(
                model=model_id,
                src=input_jsonl_gcs_uri,
                config=CreateBatchJobConfig(dest=output_prefix),
            )
            job_name = getattr(job, "name", "") or str(job)
            logger.info(
                "BATCH SUBMITTED location=%s model=%s input=%s output=%s job=%s",
                GENAI_LOCATION,
                model_id,
                input_jsonl_gcs_uri,
                output_prefix,
                job_name,
            )
            return job_name

        except genai_errors.ClientError as e:
            last_err = e
            # 404면 다음 모델로 fallback
            msg = str(e)
            logger.warning("Batch create failed model=%s err=%s", model_id, msg)
            if "404" in msg or "NOT_FOUND" in msg:
                continue
            # 404가 아닌 다른 에러면 즉시 raise(권한/쿼터/입력 포맷 등)
            raise

    # 모든 모델이 404면 여기로
    raise RuntimeError(f"All batch models failed. tried={tried} last_err={last_err}")


# -----------------------------
# CloudEvent handler
# -----------------------------
@functions_framework.cloud_event
def ingest_from_gcs(cloud_event: CloudEvent):
    data = cloud_event.data or {}

    bucket = data.get("bucket") or ""
    name = data.get("name") or ""
    generation = str(data.get("generation", ""))

    logger.info("EVENT bucket=%s name=%s generation=%s", bucket, name, generation)

    # -----------------------------
    # ✅ 0) 필터링 (재귀/잡파일 차단)
    # -----------------------------
    # 업로드 버킷만 처리
    if bucket != UPLOAD_BUCKET:
        logger.info("SKIP: bucket not upload bucket. bucket=%s expected=%s", bucket, UPLOAD_BUCKET)
        return ("OK", 200)

    # xlsx만 처리
    if not name.lower().endswith(".xlsx"):
        logger.info("SKIP: non-xlsx object: %s", name)
        return ("OK", 200)

    # 업로드 파일명 패턴(원하면 완화/비활성화 가능)
    if UPLOAD_XLSX_PATTERN and not re.match(UPLOAD_XLSX_PATTERN, name):
        logger.info("SKIP: filename not match pattern. name=%s pattern=%s", name, UPLOAD_XLSX_PATTERN)
        return ("OK", 200)

    # 혹시 업로드 버킷에 내부 prefix가 들어오면 차단
    if name.startswith("batch_inputs/") or name.startswith("batch_outputs/"):
        logger.info("SKIP: internal artifact in upload bucket: %s", name)
        return ("OK", 200)

    # -----------------------------
    # ✅ 1) 멱등성 체크
    # -----------------------------
    if _already_done(bucket, name, generation):
        logger.info("SKIP already DONE: %s/%s gen=%s", bucket, name, generation)
        return ("OK", 200)

    _mark_ingestion("STARTED", bucket, name, generation)

    try:
        # 2) 다운로드
        local_path = _download_xlsx(bucket, name)

        # 3) xlsx 검증
        _assert_xlsx(local_path, name)

        # 4) 로드 & 매핑
        df_std = _load_excel_mapped(local_path)

        # 5) raw 적재 (재시도 중복 방지)
        if not _raw_already_loaded(bucket, name, generation):
            _append_reviews_raw(df_std, bucket, name, generation)
        else:
            logger.info("SKIP reviews_raw already loaded for %s/%s gen=%s", bucket, name, generation)

        # 6) clean merge
        _merge_reviews_clean(df_std)

        # 7) batch input 생성 + 업로드(ARCHIVE_BUCKET!)
        input_uri = make_batch_input_jsonl_and_upload(bucket, name, generation)

        # 8) batch job 제출(global 강제 + fallback)
        job_name = submit_vertex_batch_job_global(input_uri, name, generation)

        _mark_ingestion(
            "DONE",
            bucket,
            name,
            generation,
            error_message=(f"BATCH_JOB={job_name}" if job_name else "NO_TARGETS"),
        )
        return ("OK", 200)

    except Exception as e:
        logger.exception("FAILED processing %s/%s gen=%s", bucket, name, generation)
        _mark_ingestion("FAILED", bucket, name, generation, error_message=str(e)[:5000])
        return ("ERROR", 500)
