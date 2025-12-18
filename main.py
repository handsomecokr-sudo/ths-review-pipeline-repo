import json
import logging
import os
import re
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import functions_framework
import pandas as pd
from cloudevents.http import CloudEvent

import google.auth
from google.cloud import bigquery
from google.cloud import storage

# Vertex AI Batch (Google Gen AI SDK)
from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("review-pipeline")

# -----------------------------
# Config
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
        raise RuntimeError("Project ID를 찾지 못했습니다. Cloud Run env에 GOOGLE_CLOUD_PROJECT 설정 권장")
    return pid

PROJECT_ID = _get_project_id()
DATASET = os.getenv("BQ_DATASET", "ths_review_analytics")

UPLOAD_BUCKET = os.getenv("UPLOAD_BUCKET", "ths-review-upload-bkt")
ARCHIVE_BUCKET = os.getenv("ARCHIVE_BUCKET", "ths-review-archive-bkt")

BATCH_INPUT_PREFIX = os.getenv("BATCH_INPUT_PREFIX", "batch_inputs").strip("/")
BATCH_OUTPUT_PREFIX = os.getenv("BATCH_OUTPUT_PREFIX", "batch_outputs").strip("/")

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")

VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "global")
VERTEX_GEMINI_MODEL = os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.0-flash-lite-001")

# Tables
TABLE_INGEST = f"{PROJECT_ID}.{DATASET}.ingestion_files"
TABLE_RAW = f"{PROJECT_ID}.{DATASET}.reviews_raw"
TABLE_CLEAN = f"{PROJECT_ID}.{DATASET}.reviews_clean"
TABLE_EXTRACT = f"{PROJECT_ID}.{DATASET}.review_llm_extract"
TABLE_METRICS = f"{PROJECT_ID}.{DATASET}.style_daily_metrics"

# 고정 staging 테이블(폭증 방지)
STG_CLEAN = f"{PROJECT_ID}.{DATASET}.staging_reviews_clean"
STG_EXTRACT = f"{PROJECT_ID}.{DATASET}.staging_review_llm_extract"
STG_KEYS = f"{PROJECT_ID}.{DATASET}.staging_new_extract_keys"

# Excel header mapping
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
# Clients / utils
# -----------------------------
def _bq() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)

def _gcs() -> storage.Client:
    return storage.Client(project=PROJECT_ID)

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _iso_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()

def _ensure_table(table_id: str, schema: List[bigquery.SchemaField], partition_field: str = None):
    client = _bq()
    try:
        client.get_table(table_id)
        return
    except Exception:
        pass

    t = bigquery.Table(table_id, schema=schema)
    if partition_field:
        t.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_field,
        )
    client.create_table(t)
    logger.info("Created table: %s", table_id)

def _init_staging_tables():
    # staging_reviews_clean: reviews_clean과 동일한 스키마
    clean_schema = [
        bigquery.SchemaField("review_key", "STRING"),
        bigquery.SchemaField("review_no", "STRING"),
        bigquery.SchemaField("review_seq", "INT64"),
        bigquery.SchemaField("channel", "STRING"),
        bigquery.SchemaField("brand_no", "STRING"),
        bigquery.SchemaField("product_no", "STRING"),
        bigquery.SchemaField("write_date", "DATE"),
        bigquery.SchemaField("review_score", "INT64"),
        bigquery.SchemaField("review_1depth", "STRING"),
        bigquery.SchemaField("review_2depth", "STRING"),
        bigquery.SchemaField("review_contents", "STRING"),
        bigquery.SchemaField("review_text_masked", "STRING"),
        bigquery.SchemaField("normalized_text", "STRING"),
        bigquery.SchemaField("legacy_review_ai_score", "FLOAT64"),
        bigquery.SchemaField("legacy_review_ai_contents", "STRING"),
        bigquery.SchemaField("loaded_at", "TIMESTAMP"),
    ]
    extract_schema = [
        bigquery.SchemaField("review_key", "STRING"),
        bigquery.SchemaField("extracted_at", "TIMESTAMP"),
        bigquery.SchemaField("model_name", "STRING"),
        bigquery.SchemaField("model_version", "STRING"),
        bigquery.SchemaField("brand_no", "STRING"),
        bigquery.SchemaField("product_no", "STRING"),
        bigquery.SchemaField("issue_category", "STRING"),
        bigquery.SchemaField("severity", "STRING"),
        bigquery.SchemaField("sentiment", "STRING"),
        bigquery.SchemaField("size_feedback", "STRING"),
        bigquery.SchemaField("defect_part", "STRING"),
        bigquery.SchemaField("color_mentioned", "STRING"),
        bigquery.SchemaField("repurchase_intent", "STRING"),
        bigquery.SchemaField("evidence", "STRING"),
        bigquery.SchemaField("raw_json", "JSON"),
        bigquery.SchemaField("prompt_version", "STRING"),
    ]
    keys_schema = [bigquery.SchemaField("review_key", "STRING")]

    _ensure_table(STG_CLEAN, clean_schema)
    _ensure_table(STG_EXTRACT, extract_schema)
    _ensure_table(STG_KEYS, keys_schema)

def _ndjson_write(path: str, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _load_ndjson_to_table(table_id: str, ndjson_path: str, write_disposition: str):
    client = _bq()
    with open(ndjson_path, "rb") as fp:
        job = client.load_table_from_file(
            fp,
            table_id,
            job_config=bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                write_disposition=write_disposition,
            ),
        )
    job.result()

# -----------------------------
# XLSX validation
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
# GCS IO
# -----------------------------
def _download_to_tmp(bucket: str, name: str) -> str:
    local_path = f"/tmp/{uuid.uuid4().hex}_{os.path.basename(name)}"
    gcs = _gcs()
    gcs.bucket(bucket).blob(name).download_to_filename(local_path)
    return local_path

def _upload_file(bucket: str, blob_name: str, local_path: str, content_type: str = None) -> str:
    gcs = _gcs()
    b = gcs.bucket(bucket).blob(blob_name)
    b.upload_from_filename(local_path, content_type=content_type)
    return f"gs://{bucket}/{blob_name}"

# -----------------------------
# Excel load
# -----------------------------
def _load_excel_mapped(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=EXCEL_TO_STD)

    missing = [c for c in EXPECTED_STD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"컬럼 매핑 후 누락: {missing}. 현재 컬럼={list(df.columns)}")

    df = df[EXPECTED_STD_COLS].copy()
    return df

# -----------------------------
# BigQuery writes (NDJSON 방식: pyarrow 불필요)
# -----------------------------
def _append_reviews_raw(df_std: pd.DataFrame, bucket: str, name: str, generation: str):
    loaded_at = _now_utc()
    ingest_id = uuid.uuid4().hex

    rows = []
    for _, r in df_std.iterrows():
        rows.append(
            {
                "ingest_id": ingest_id,
                "source_bucket": bucket,
                "source_object": name,
                "source_generation": generation,
                "loaded_at": _iso_ts(loaded_at),

                "write_date": "" if pd.isna(r["write_date"]) else str(r["write_date"]),
                "review_no": "" if pd.isna(r["review_no"]) else str(r["review_no"]),
                "review_seq": "" if pd.isna(r["review_seq"]) else str(r["review_seq"]),
                "channel": "" if pd.isna(r["channel"]) else str(r["channel"]),
                "brand_no": "" if pd.isna(r["brand_no"]) else str(r["brand_no"]),
                "product_no": "" if pd.isna(r["product_no"]) else str(r["product_no"]),
                "review_1depth": "" if pd.isna(r["review_1depth"]) else str(r["review_1depth"]),
                "review_2depth": "" if pd.isna(r["review_2depth"]) else str(r["review_2depth"]),
                "review_score": "" if pd.isna(r["review_score"]) else str(r["review_score"]),
                "review_contents": "" if pd.isna(r["review_contents"]) else str(r["review_contents"]),
                "review_ai_score": "" if pd.isna(r["review_ai_score"]) else str(r["review_ai_score"]),
                "review_ai_contents": "" if pd.isna(r["review_ai_contents"]) else str(r["review_ai_contents"]),
            }
        )

    tmp = f"/tmp/raw_{uuid.uuid4().hex}.ndjson"
    _ndjson_write(tmp, rows)
    _load_ndjson_to_table(TABLE_RAW, tmp, write_disposition="WRITE_APPEND")
    logger.info("BQ append reviews_raw rows=%d ingest_id=%s", len(rows), ingest_id)

def _merge_reviews_clean_fixed_staging(df_std: pd.DataFrame):
    """
    고정 staging 테이블 1개(STG_CLEAN)만 사용:
    - 파싱 실패 행 drop
    - staging 내 review_key 중복은 ROW_NUMBER로 1개만 남겨 MERGE 안정화
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
    if dropped:
        logger.warning("DROP invalid rows (count=%d)", dropped)

    if len(df) == 0:
        logger.info("No valid rows to merge into reviews_clean")
        return

    df["review_seq"] = df["review_seq_num"].astype(int)
    df["review_score"] = df["review_score_num"].fillna(0).astype(int)
    df["write_date"] = df["write_date_dt"]
    df["review_key"] = df["review_no"] + "-" + df["review_seq"].astype(str)

    # MVP: DLP 전이라 원문을 그대로 masked로
    df["review_text_masked"] = df["review_contents"].astype(str)
    df["normalized_text"] = df["review_contents"].astype(str)

    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "review_key": str(r["review_key"]),
                "review_no": str(r["review_no"]),
                "review_seq": int(r["review_seq"]),
                "channel": "" if pd.isna(r["channel"]) else str(r["channel"]),
                "brand_no": "" if pd.isna(r["brand_no"]) else str(r["brand_no"]),
                "product_no": "" if pd.isna(r["product_no"]) else str(r["product_no"]),
                "write_date": str(r["write_date"]),
                "review_score": int(r["review_score"]),
                "review_1depth": "" if pd.isna(r["review_1depth"]) else str(r["review_1depth"]),
                "review_2depth": "" if pd.isna(r["review_2depth"]) else str(r["review_2depth"]),
                "review_contents": "" if pd.isna(r["review_contents"]) else str(r["review_contents"]),
                "review_text_masked": "" if pd.isna(r["review_text_masked"]) else str(r["review_text_masked"]),
                "normalized_text": "" if pd.isna(r["normalized_text"]) else str(r["normalized_text"]),
                "legacy_review_ai_score": None if pd.isna(r["review_ai_score"]) else float(r["review_ai_score"]) if str(r["review_ai_score"]).strip() != "" else None,
                "legacy_review_ai_contents": "" if pd.isna(r["review_ai_contents"]) else str(r["review_ai_contents"]),
                "loaded_at": _iso_ts(loaded_at),
            }
        )

    tmp = f"/tmp/clean_stage_{uuid.uuid4().hex}.ndjson"
    _ndjson_write(tmp, rows)
    _load_ndjson_to_table(STG_CLEAN, tmp, write_disposition="WRITE_TRUNCATE")

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
    logger.info("BQ MERGE reviews_clean done rows=%d", len(rows))

# -----------------------------
# Build batch input JSONL
# -----------------------------
SQL_NEW_REVIEWS_FOR_FILE = f"""
WITH file_rows AS (
  SELECT DISTINCT
    review_no,
    SAFE_CAST(review_seq AS INT64) AS review_seq
  FROM `{TABLE_RAW}`
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
  FROM `{TABLE_CLEAN}` c
  JOIN file_rows r
    ON c.review_no = r.review_no
   AND c.review_seq = r.review_seq
)
SELECT t.*
FROM targets t
LEFT JOIN `{TABLE_EXTRACT}` e
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
    input_uri = _upload_file(
        ARCHIVE_BUCKET,
        dest_blob,
        tmp_path,
        content_type="application/jsonl",
    )
    logger.info("BATCH INPUT uploaded: %s (rows=%d)", input_uri, rows_written)
    return input_uri

# -----------------------------
# Vertex AI Batch submit (global)
# -----------------------------
def _normalize_model_name(model: str) -> str:
    """
    모델명이 짧게 들어와도 batches.create에서 안정적으로 찾도록 publisher 경로로 정규화.
    """
    m = (model or "").strip()
    if not m:
        return "publishers/google/models/gemini-2.0-flash-lite-001"
    if m.startswith("publishers/"):
        return m
    # gemini-xxx 형태면 publisher 경로로
    if re.match(r"^gemini-[\w\.\-]+$", m):
        return f"publishers/google/models/{m}"
    return m

def submit_vertex_batch_job_global(input_jsonl_gcs_uri: str, object_name: str, generation: str) -> str:
    if not input_jsonl_gcs_uri:
        return ""

    model_name = _normalize_model_name(VERTEX_GEMINI_MODEL)

    # output은 archive bucket 아래로 고정
    output_prefix = f"gs://{ARCHIVE_BUCKET}/{BATCH_OUTPUT_PREFIX}/{object_name}/{generation}/"

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=VERTEX_LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )

    job = client.batches.create(
        model=model_name,
        src=input_jsonl_gcs_uri,
        config=CreateBatchJobConfig(dest=output_prefix),
    )

    job_name = getattr(job, "name", "") or str(job)
    logger.info(
        "BATCH SUBMITTED location=%s model=%s input=%s output=%s job=%s",
        VERTEX_LOCATION,
        model_name,
        input_jsonl_gcs_uri,
        output_prefix,
        job_name,
    )
    return job_name

# -----------------------------
# Batch output(JSONL) parse -> BQ upsert -> metrics refresh
# -----------------------------
def _extract_text_from_batch_line(obj: dict) -> Tuple[str, str]:
    """
    returns (text, model_version)
    batch output format: each line contains response or error. :contentReference[oaicite:3]{index=3}
    """
    resp = obj.get("response") or {}
    mv = resp.get("modelVersion", "") or resp.get("model_version", "") or ""
    candidates = resp.get("candidates") or []
    if not candidates:
        return "", mv
    content = (candidates[0].get("content") or {})
    parts = content.get("parts") or []
    if not parts:
        return "", mv
    text = parts[0].get("text") or ""
    return text, mv

def _parse_output_jsonl(local_path: str) -> Tuple[List[dict], List[str], List[str]]:
    """
    returns (extract_rows, review_keys, errors)
    """
    extracted_at = _now_utc()
    rows: List[dict] = []
    keys: List[str] = []
    errors: List[str] = []

    with open(local_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                errors.append(f"line{i}: invalid json: {e}")
                continue

            if obj.get("error"):
                errors.append(f"line{i}: error={obj.get('error')}")
                continue

            text, model_version = _extract_text_from_batch_line(obj)
            if not text:
                errors.append(f"line{i}: empty text")
                continue

            # model이 JSON만 출력하도록 시켰으므로 파싱
            try:
                payload = json.loads(text)
            except Exception as e:
                errors.append(f"line{i}: response not json: {e} text={text[:200]}")
                continue

            review_key = str(payload.get("review_key") or "").strip()
            if not review_key:
                errors.append(f"line{i}: missing review_key")
                continue

            signals = payload.get("signals") or {}

            row = {
                "review_key": review_key,
                "extracted_at": _iso_ts(extracted_at),
                "model_name": _normalize_model_name(VERTEX_GEMINI_MODEL),
                "model_version": model_version or "",
                "brand_no": str(payload.get("brand_no") or ""),
                "product_no": str(payload.get("product_no") or ""),
                "issue_category": str(payload.get("issue_category") or ""),
                "severity": str(payload.get("severity") or ""),
                "sentiment": str(payload.get("sentiment") or ""),
                "size_feedback": str(signals.get("size_feedback") or ""),
                "defect_part": str(signals.get("defect_part") or ""),
                "color_mentioned": str(signals.get("color_mentioned") or ""),
                "repurchase_intent": str(signals.get("repurchase_intent") or ""),
                "evidence": str(payload.get("evidence") or ""),
                "raw_json": payload,   # JSON column
                "prompt_version": PROMPT_VERSION,
            }
            rows.append(row)
            keys.append(review_key)

    return rows, keys, errors

def _upsert_review_llm_extract(rows: List[dict]):
    if not rows:
        return

    tmp = f"/tmp/extract_stage_{uuid.uuid4().hex}.ndjson"
    _ndjson_write(tmp, rows)
    _load_ndjson_to_table(STG_EXTRACT, tmp, write_disposition="WRITE_TRUNCATE")

    client = _bq()
    merge_sql = f"""
    MERGE `{TABLE_EXTRACT}` T
    USING (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT
          S.*,
          ROW_NUMBER() OVER (PARTITION BY review_key, prompt_version ORDER BY extracted_at DESC) AS rn
        FROM `{STG_EXTRACT}` S
      )
      WHERE rn = 1
    ) S
    ON T.review_key = S.review_key AND T.prompt_version = S.prompt_version
    WHEN MATCHED THEN UPDATE SET
      extracted_at = S.extracted_at,
      model_name = S.model_name,
      model_version = S.model_version,
      brand_no = S.brand_no,
      product_no = S.product_no,
      issue_category = S.issue_category,
      severity = S.severity,
      sentiment = S.sentiment,
      size_feedback = S.size_feedback,
      defect_part = S.defect_part,
      color_mentioned = S.color_mentioned,
      repurchase_intent = S.repurchase_intent,
      evidence = S.evidence,
      raw_json = S.raw_json
    WHEN NOT MATCHED THEN
      INSERT ROW
    """
    client.query(merge_sql).result()
    logger.info("Upserted review_llm_extract rows=%d", len(rows))

def _refresh_style_daily_metrics(keys: List[str]):
    """
    이번에 새로 적재된 review_key들이 속한 날짜 범위만 재집계하여 MERGE
    """
    if not keys:
        return

    # keys staging
    key_rows = [{"review_key": k} for k in sorted(set(keys))]
    tmp = f"/tmp/keys_{uuid.uuid4().hex}.ndjson"
    _ndjson_write(tmp, key_rows)
    _load_ndjson_to_table(STG_KEYS, tmp, write_disposition="WRITE_TRUNCATE")

    client = _bq()

    # BigQuery script: min/max date 구하고 해당 범위만 재집계
    sql = f"""
    DECLARE min_d DATE;
    DECLARE max_d DATE;

    SET (min_d, max_d) = (
      SELECT AS STRUCT
        MIN(c.write_date) AS min_d,
        MAX(c.write_date) AS max_d
      FROM `{TABLE_CLEAN}` c
      JOIN `{STG_KEYS}` k
        ON c.review_key = k.review_key
    );

    IF min_d IS NULL OR max_d IS NULL THEN
      SELECT "NO_DATES" AS status;
    ELSE
      MERGE `{TABLE_METRICS}` T
      USING (
        SELECT
          c.write_date AS metric_date,
          c.brand_no,
          c.product_no,
          c.channel,
          e.issue_category,
          COUNT(1) AS review_cnt,
          SUM(IF(e.sentiment="불만", 1, 0)) AS neg_cnt,
          SUM(IF(e.severity="중대", 1, 0)) AS severe_cnt,
          AVG(c.review_score) AS avg_rating
        FROM `{TABLE_CLEAN}` c
        JOIN `{TABLE_EXTRACT}` e
          ON c.review_key = e.review_key
        WHERE c.write_date BETWEEN min_d AND max_d
        GROUP BY metric_date, brand_no, product_no, channel, issue_category
      ) S
      ON T.metric_date = S.metric_date
        AND T.brand_no = S.brand_no
        AND T.product_no = S.product_no
        AND T.channel = S.channel
        AND T.issue_category = S.issue_category
      WHEN MATCHED THEN UPDATE SET
        review_cnt = S.review_cnt,
        neg_cnt = S.neg_cnt,
        severe_cnt = S.severe_cnt,
        avg_rating = S.avg_rating
      WHEN NOT MATCHED THEN
        INSERT ROW;
    END IF;
    """
    client.query(sql).result()
    logger.info("Refreshed style_daily_metrics for keys=%d", len(key_rows))

def handle_batch_output_event(bucket: str, name: str, generation: str):
    # output jsonl만 처리
    if bucket != ARCHIVE_BUCKET:
        logger.info("SKIP batch output: bucket mismatch %s", bucket)
        return

    if not name.startswith(f"{BATCH_OUTPUT_PREFIX}/") or not name.lower().endswith(".jsonl"):
        logger.info("SKIP batch output: not target object %s", name)
        return

    # 중복 방지(ingestion_files 재활용)
    if _already_done(bucket, name, generation):
        logger.info("SKIP already DONE output: %s/%s gen=%s", bucket, name, generation)
        return

    _mark_ingestion("STARTED", bucket, name, generation)

    try:
        local_path = _download_to_tmp(bucket, name)
        # output은 jsonl이므로 xlsx 검증 없음

        rows, keys, errors = _parse_output_jsonl(local_path)

        if rows:
            _upsert_review_llm_extract(rows)
            _refresh_style_daily_metrics(keys)

        msg = f"PARSED={len(rows)} ERRORS={len(errors)}"
        if errors:
            msg += " SAMPLE_ERR=" + (errors[0][:500] if errors else "")
        _mark_ingestion("DONE", bucket, name, generation, error_message=msg)
        logger.info("Batch output handled: %s (rows=%d errors=%d)", name, len(rows), len(errors))

    except Exception as e:
        logger.exception("FAILED batch output %s/%s gen=%s", bucket, name, generation)
        _mark_ingestion("FAILED", bucket, name, generation, error_message=str(e)[:5000])
        raise

# -----------------------------
# XLSX upload handler
# -----------------------------
def handle_xlsx_upload_event(bucket: str, name: str, generation: str):
    if bucket != UPLOAD_BUCKET:
        logger.info("SKIP upload: bucket mismatch %s", bucket)
        return

    # 업로드 버킷에서는 XLSX만 처리 (그 외 모두 스킵)
    if not name.lower().endswith(".xlsx"):
        logger.info("SKIP upload: not xlsx %s", name)
        return

    if _already_done(bucket, name, generation):
        logger.info("SKIP already DONE: %s/%s gen=%s", bucket, name, generation)
        return

    _mark_ingestion("STARTED", bucket, name, generation)

    try:
        local_path = _download_to_tmp(bucket, name)
        _assert_xlsx(local_path, name)

        df_std = _load_excel_mapped(local_path)

        if not _raw_already_loaded(bucket, name, generation):
            _append_reviews_raw(df_std, bucket, name, generation)
        else:
            logger.info("SKIP reviews_raw already loaded for %s/%s gen=%s", bucket, name, generation)

        _merge_reviews_clean_fixed_staging(df_std)

        input_uri = make_batch_input_jsonl_and_upload(bucket, name, generation)
        job_name = submit_vertex_batch_job_global(input_uri, name, generation)

        _mark_ingestion(
            "DONE",
            bucket,
            name,
            generation,
            error_message=(f"BATCH_JOB={job_name}" if job_name else "NO_TARGETS"),
        )
        logger.info("Upload handled: %s (batch_job=%s)", name, job_name)

    except Exception as e:
        logger.exception("FAILED upload %s/%s gen=%s", bucket, name, generation)
        _mark_ingestion("FAILED", bucket, name, generation, error_message=str(e)[:5000])
        raise

# -----------------------------
# CloudEvent entrypoint
# -----------------------------
@functions_framework.cloud_event
def ingest_from_gcs(cloud_event: CloudEvent):
    _init_staging_tables()

    data = cloud_event.data or {}
    bucket = data.get("bucket")
    name = data.get("name")
    generation = str(data.get("generation", ""))

    logger.info("EVENT bucket=%s name=%s generation=%s", bucket, name, generation)
    logger.info("EVENT type=%s source=%s id=%s", cloud_event.get("type"), cloud_event.get("source"), cloud_event.get("id"))

    if not bucket or not name:
        logger.warning("Missing bucket/name in event payload. data=%s", data)
        return ("OK", 200)

    try:
        # 1) 업로드 XLSX 처리
        if bucket == UPLOAD_BUCKET:
            handle_xlsx_upload_event(bucket, name, generation)
            return ("OK", 200)

        # 2) Batch output 처리
        if bucket == ARCHIVE_BUCKET:
            # batch_inputs는 스킵(루프 방지)
            if name.startswith(f"{BATCH_INPUT_PREFIX}/"):
                logger.info("SKIP archive batch_inputs object: %s", name)
                return ("OK", 200)
            handle_batch_output_event(bucket, name, generation)
            return ("OK", 200)

        logger.info("SKIP: unrelated bucket=%s name=%s", bucket, name)
        return ("OK", 200)

    except Exception:
        # 이미 _mark_ingestion FAILED 찍고 raise 했지만, Cloud Run 응답도 500으로
        return ("ERROR", 500)
