import json
import logging
import os
import uuid
import zipfile
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional

import functions_framework
import google.auth
import pandas as pd
from cloudevents.http import CloudEvent

from google.cloud import bigquery
from google.cloud import storage

# google genai (Vertex AI Batch)
from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("review-pipeline")

# -----------------------------
# Project / Config
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
            "Project ID를 찾지 못했습니다. Cloud Run에 GOOGLE_CLOUD_PROJECT env를 설정하세요."
        )
    return pid


PROJECT_ID = _get_project_id()
DATASET = os.getenv("BQ_DATASET", "ths_review_analytics")

UPLOAD_BUCKET = (os.getenv("UPLOAD_BUCKET") or "ths-review-upload-bkt").strip()
ARCHIVE_BUCKET = (os.getenv("ARCHIVE_BUCKET") or "ths-review-archive-bkt").strip()

# Batch
VERTEX_GEMINI_MODEL = os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.0-flash").strip()
BATCH_INPUT_PREFIX = (os.getenv("BATCH_INPUT_PREFIX") or "batch_inputs").strip().strip("/")
BATCH_OUTPUT_PREFIX = (os.getenv("BATCH_OUTPUT_PREFIX") or "batch_outputs").strip().strip("/")

# Tables
TABLE_INGEST = f"{PROJECT_ID}.{DATASET}.ingestion_files"
TABLE_RAW = f"{PROJECT_ID}.{DATASET}.reviews_raw"
TABLE_CLEAN = f"{PROJECT_ID}.{DATASET}.reviews_clean"
TABLE_LLM = f"{PROJECT_ID}.{DATASET}.review_llm_extract"

# Fixed staging table (do not create tons of tables)
STG_CLEAN = f"{PROJECT_ID}.{DATASET}.staging_reviews_clean"

# Excel header mapping (Korean -> Std)
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
# Utils
# -----------------------------
def _normalize_write_date_to_ymd(v: Any) -> str:
    """
    입력: 20251215 / "20251215" / "2025-12-15" / "2025.12.15" / datetime/date
    출력: "YYYY-MM-DD" or ""
    """
    if v is None:
        return ""

    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()

    s = str(v).strip()
    if not s:
        return ""

    # Excel numeric like 20251215
    if s.isdigit() and len(s) == 8:
        y, m, d = s[0:4], s[4:6], s[6:8]
        return f"{y}-{m}-{d}"

    # remove common separators
    s2 = s.replace(".", "").replace("-", "").replace("/", "")
    if s2.isdigit() and len(s2) == 8:
        y, m, d = s2[0:4], s2[4:6], s2[6:8]
        return f"{y}-{m}-{d}"

    # last resort: pandas parsing
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return ""
        return dt.date().isoformat()
    except Exception:
        return ""


def _is_internal_object(name: str) -> bool:
    n = (name or "").lstrip("/")
    return (
        n.startswith(BATCH_INPUT_PREFIX + "/")
        or n.startswith(BATCH_OUTPUT_PREFIX + "/")
        or n.lower().endswith(".jsonl")
        or n.lower().endswith(".ndjson")
    )


# -----------------------------
# XLSX Validation / Download
# -----------------------------
def _download_from_gcs(bucket: str, name: str, suffix: str = "") -> str:
    local_path = f"/tmp/{uuid.uuid4().hex}{suffix}"
    client = _gcs()
    client.bucket(bucket).blob(name).download_to_filename(local_path)
    return local_path


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


def _load_excel_mapped(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=EXCEL_TO_STD)

    missing = [c for c in EXPECTED_STD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"컬럼 매핑 후 누락: {missing}. 현재 컬럼={list(df.columns)}")

    return df[EXPECTED_STD_COLS].copy()


# -----------------------------
# BigQuery helpers
# -----------------------------
def _load_ndjson_to_table(table_id: str, local_ndjson_path: str, write_disposition: str):
    client = _bq()
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=write_disposition,
        ignore_unknown_values=True,
        max_bad_records=0,
    )
    with open(local_ndjson_path, "rb") as f:
        job = client.load_table_from_file(f, table_id, job_config=job_config)
    job.result()


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


def _mark_ingestion(status: str, bucket: str, object_name: str, generation: str, error_message: Optional[str] = None):
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
# reviews_raw append (NDJSON load)
# -----------------------------
def _append_reviews_raw_ndjson(df_std: pd.DataFrame, bucket: str, name: str, generation: str):
    loaded_at = _now_utc().isoformat()
    base_ingest_id = uuid.uuid4().hex

    rows: List[Dict[str, Any]] = []
    for idx, r in df_std.iterrows():
        rows.append({
            "ingest_id": f"{base_ingest_id}-{idx}",
            "source_bucket": bucket,
            "source_object": name,
            "source_generation": generation,
            "loaded_at": loaded_at,
            "write_date": str(r.get("write_date", "")),
            "review_no": str(r.get("review_no", "")),
            "review_seq": str(r.get("review_seq", "")),
            "channel": str(r.get("channel", "")),
            "brand_no": str(r.get("brand_no", "")),
            "product_no": str(r.get("product_no", "")),
            "review_1depth": str(r.get("review_1depth", "")),
            "review_2depth": str(r.get("review_2depth", "")),
            "review_score": str(r.get("review_score", "")),
            "review_contents": str(r.get("review_contents", "")),
            "review_ai_score": str(r.get("review_ai_score", "")),
            "review_ai_contents": str(r.get("review_ai_contents", "")),
        })

    tmp = f"/tmp/raw_{uuid.uuid4().hex}.ndjson"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _load_ndjson_to_table(TABLE_RAW, tmp, write_disposition="WRITE_APPEND")
    logger.info("BQ append reviews_raw rows=%d ingest_id=%s", len(rows), base_ingest_id)


# -----------------------------
# (1) reviews_clean MERGE (review_seq STRING 버전) - 통째 교체
# -----------------------------
def _merge_reviews_clean_fixed_staging(df_std: pd.DataFrame):
    """
    reviews_clean.review_seq 가 STRING인 버전.
    - staging은 STRING 적재
    - write_date=YYYYMMDD도 YYYY-MM-DD로 정규화
    - MERGE source 중복 review_key는 ROW_NUMBER로 1개만 남김
    """
    client = _bq()
    loaded_at = _now_utc().isoformat()

    # staging 고정 테이블 (전부 STRING + loaded_at STRING)  ← 파싱 오류 방지
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS `{STG_CLEAN}` (
      review_key STRING,
      review_no STRING,
      review_seq STRING,
      channel STRING,
      brand_no STRING,
      product_no STRING,
      write_date STRING,
      review_score STRING,
      review_1depth STRING,
      review_2depth STRING,
      review_contents STRING,
      review_text_masked STRING,
      normalized_text STRING,
      legacy_review_ai_score STRING,
      legacy_review_ai_contents STRING,
      loaded_at STRING
    )
    """
    client.query(create_sql).result()

    df = df_std.copy()

    df["review_no"] = df["review_no"].astype(str).str.strip()
    df["review_seq"] = df["review_seq"].astype(str).str.strip()

    df["write_date_str"] = df["write_date"].apply(_normalize_write_date_to_ymd)

    # 필수키 검증
    before = len(df)
    df = df[df["review_no"].notna() & (df["review_no"] != "")]
    df = df[df["review_seq"].notna() & (df["review_seq"] != "")]
    df = df[df["write_date_str"].notna() & (df["write_date_str"] != "")]
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("DROP rows due to invalid keys/parse (count=%d)", dropped)

    if len(df) == 0:
        logger.info("No valid rows to merge into reviews_clean")
        return

    # review_key = review_no-review_seq (STRING)
    df["review_key"] = df["review_no"] + "-" + df["review_seq"]

    # MVP: DLP 전이면 원문 그대로
    df["review_text_masked"] = df["review_contents"].astype(str)
    df["normalized_text"] = df["review_contents"].astype(str)

    # staging rows(전부 STRING)
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append({
            "review_key": str(r.get("review_key", "")),
            "review_no": str(r.get("review_no", "")),
            "review_seq": str(r.get("review_seq", "")),
            "channel": str(r.get("channel", "")),
            "brand_no": str(r.get("brand_no", "")),
            "product_no": str(r.get("product_no", "")),
            "write_date": str(r.get("write_date_str", "")),  # YYYY-MM-DD
            "review_score": str(r.get("review_score", "")),
            "review_1depth": str(r.get("review_1depth", "")),
            "review_2depth": str(r.get("review_2depth", "")),
            "review_contents": str(r.get("review_contents", "")),
            "review_text_masked": str(r.get("review_text_masked", "")),
            "normalized_text": str(r.get("normalized_text", "")),
            "legacy_review_ai_score": str(r.get("review_ai_score", "")),
            "legacy_review_ai_contents": str(r.get("review_ai_contents", "")),
            "loaded_at": loaded_at,
        })

    tmp = f"/tmp/stg_clean_{uuid.uuid4().hex}.ndjson"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _load_ndjson_to_table(STG_CLEAN, tmp, write_disposition="WRITE_TRUNCATE")

    # ✅ MERGE: review_seq는 STRING 그대로
    merge_sql = f"""
    MERGE `{TABLE_CLEAN}` T
    USING (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT
          S.*,
          ROW_NUMBER() OVER (PARTITION BY review_key ORDER BY loaded_at DESC) AS rn
        FROM `{STG_CLEAN}` S
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
      write_date = SAFE_CAST(NULLIF(TRIM(S.write_date), '') AS DATE),
      review_score = SAFE_CAST(NULLIF(TRIM(S.review_score), '') AS INT64),
      review_1depth = S.review_1depth,
      review_2depth = S.review_2depth,
      review_contents = S.review_contents,
      review_text_masked = S.review_text_masked,
      normalized_text = S.normalized_text,
      legacy_review_ai_score = SAFE_CAST(NULLIF(TRIM(S.legacy_review_ai_score), '') AS FLOAT64),
      legacy_review_ai_contents = S.legacy_review_ai_contents,
      loaded_at = SAFE_CAST(NULLIF(TRIM(S.loaded_at), '') AS TIMESTAMP)
    WHEN NOT MATCHED THEN
      INSERT (
        review_key, review_no, review_seq, channel, brand_no, product_no,
        write_date, review_score, review_1depth, review_2depth,
        review_contents, review_text_masked, normalized_text,
        legacy_review_ai_score, legacy_review_ai_contents, loaded_at
      )
      VALUES (
        S.review_key, S.review_no, S.review_seq,
        S.channel, S.brand_no, S.product_no,
        SAFE_CAST(NULLIF(TRIM(S.write_date), '') AS DATE),
        SAFE_CAST(NULLIF(TRIM(S.review_score), '') AS INT64),
        S.review_1depth, S.review_2depth,
        S.review_contents, S.review_text_masked, S.normalized_text,
        SAFE_CAST(NULLIF(TRIM(S.legacy_review_ai_score), '') AS FLOAT64),
        S.legacy_review_ai_contents,
        SAFE_CAST(NULLIF(TRIM(S.loaded_at), '') AS TIMESTAMP)
      )
    """
    client.query(merge_sql).result()
    logger.info("BQ MERGE reviews_clean done rows=%d (seq=STRING)", len(rows))


# -----------------------------
# (2) “이번 파일 신규 대상” SQL (STRING 조인 버전) - 통째 교체
# -----------------------------
SQL_NEW_REVIEWS_FOR_FILE = f"""
WITH file_rows AS (
  SELECT DISTINCT
    review_no,
    review_seq
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


def make_batch_input_jsonl_and_upload(bucket: str, object_name: str, generation: str) -> str:
    """
    batch input은 ARCHIVE_BUCKET에 업로드 (UPLOAD_BUCKET 재트리거 방지)
    """
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
    gcs.bucket(ARCHIVE_BUCKET).blob(dest_blob).upload_from_filename(
        tmp_path, content_type="application/jsonl"
    )

    input_uri = f"gs://{ARCHIVE_BUCKET}/{dest_blob}"
    logger.info("BATCH INPUT uploaded: %s (rows=%d)", input_uri, rows_written)
    return input_uri


def _normalize_model_name(model: str) -> str:
    """
    google-genai batches.create에서 모델명을 publisher 경로로 요구하는 경우가 있어 안전하게 보정
    """
    m = (model or "").strip()
    if not m:
        return "publishers/google/models/gemini-2.0-flash"
    if "/" in m:
        return m
    return f"publishers/google/models/{m}"


def submit_vertex_batch_job_global(input_jsonl_gcs_uri: str, object_name: str, generation: str) -> str:
    if not input_jsonl_gcs_uri:
        return ""

    output_prefix = f"gs://{ARCHIVE_BUCKET}/{BATCH_OUTPUT_PREFIX}/{object_name}/{generation}/"

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="global",
        http_options=HttpOptions(api_version="v1"),
    )

    model_name = _normalize_model_name(VERTEX_GEMINI_MODEL)
    job = client.batches.create(
        model=model_name,
        src=input_jsonl_gcs_uri,
        config=CreateBatchJobConfig(dest=output_prefix),
    )

    job_name = getattr(job, "name", "") or str(job)
    logger.info(
        "BATCH SUBMITTED location=global model=%s input=%s output=%s job=%s",
        model_name,
        input_jsonl_gcs_uri,
        output_prefix,
        job_name,
    )
    return job_name


# -----------------------------
# Main handler
# -----------------------------
def handle_xlsx_upload_event(bucket: str, name: str, generation: str):
    logger.info("CONFIG UPLOAD_BUCKET=%s ARCHIVE_BUCKET=%s DATASET=%s", UPLOAD_BUCKET, ARCHIVE_BUCKET, DATASET)

    # 0) only target upload bucket + xlsx + not internal objects
    if bucket != UPLOAD_BUCKET:
        logger.info("SKIP not a target bucket: bucket=%s name=%s", bucket, name)
        return

    if _is_internal_object(name):
        logger.info("SKIP internal object event: bucket=%s name=%s", bucket, name)
        return

    if not name.lower().endswith(".xlsx"):
        logger.info("SKIP not an xlsx: bucket=%s name=%s", bucket, name)
        return

    # 1) download + validate
    local_path = _download_from_gcs(bucket, name, suffix=".xlsx")
    _assert_xlsx(local_path, name)

    # 2) load excel
    df_std = _load_excel_mapped(local_path)

    # 3) raw append (idempotent by source)
    if not _raw_already_loaded(bucket, name, generation):
        _append_reviews_raw_ndjson(df_std, bucket, name, generation)
    else:
        logger.info("SKIP reviews_raw already loaded for %s/%s gen=%s", bucket, name, generation)

    # 4) clean merge (STRING seq)
    _merge_reviews_clean_fixed_staging(df_std)

    # 5) build batch input jsonl (ARCHIVE_BUCKET)
    input_uri = make_batch_input_jsonl_and_upload(bucket, name, generation)

    # 6) submit batch (global)
    job_name = submit_vertex_batch_job_global(input_uri, name, generation)

    # 7) done mark
    _mark_ingestion(
        "DONE",
        bucket,
        name,
        generation,
        error_message=(f"BATCH_JOB={job_name}" if job_name else "NO_TARGETS"),
    )


@functions_framework.cloud_event
def ingest_from_gcs(cloud_event: CloudEvent):
    data = cloud_event.data or {}

    bucket = (data.get("bucket") or "").strip()
    name = (data.get("name") or "").strip()
    generation = str(data.get("generation", "")).strip()

    logger.info("EVENT bucket=%s name=%s generation=%s", bucket, name, generation)
    logger.info("EVENT type=%s source=%s id=%s", cloud_event.get("type"), cloud_event.get("source"), cloud_event.get("id"))

    if not bucket or not name:
        logger.warning("Missing bucket/name in event payload. data=%s", data)
        return ("OK", 200)

    # idempotency on ingestion table
    if _already_done(bucket, name, generation):
        logger.info("SKIP already DONE: %s/%s gen=%s", bucket, name, generation)
        return ("OK", 200)

    _mark_ingestion("STARTED", bucket, name, generation)

    try:
        handle_xlsx_upload_event(bucket, name, generation)
        return ("OK", 200)

    except Exception as e:
        logger.exception("FAILED upload %s/%s gen=%s", bucket, name, generation)
        _mark_ingestion("FAILED", bucket, name, generation, error_message=str(e)[:5000])
        return ("ERROR", 500)
