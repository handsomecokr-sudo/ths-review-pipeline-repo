# main.py
import json
import logging
import os
import random
import time
import uuid
import zipfile
from datetime import datetime, date, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import functions_framework
import google.auth
import pandas as pd
from cloudevents.http import CloudEvent
from google.api_core.exceptions import TooManyRequests
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
        raise RuntimeError("Project ID를 찾지 못했습니다. Cloud Run에 GOOGLE_CLOUD_PROJECT env를 설정하세요.")
    return pid


PROJECT_ID = _get_project_id()
DATASET = (os.getenv("BQ_DATASET") or "ths_review_analytics").strip()

UPLOAD_BUCKET = (os.getenv("UPLOAD_BUCKET") or "ths-review-upload-bkt").strip()
ARCHIVE_BUCKET = (os.getenv("ARCHIVE_BUCKET") or "ths-review-archive-bkt").strip()

# Batch (review-level)
VERTEX_GEMINI_MODEL = (os.getenv("VERTEX_GEMINI_MODEL") or "gemini-2.0-flash-lite").strip()
BATCH_INPUT_PREFIX = (os.getenv("BATCH_INPUT_PREFIX") or "batch_inputs").strip().strip("/")
BATCH_OUTPUT_PREFIX = (os.getenv("BATCH_OUTPUT_PREFIX") or "batch_outputs").strip().strip("/")
PROMPT_VERSION = (os.getenv("PROMPT_VERSION") or "v1").strip()

# Batch (product-level)
BATCH_INPUT_PREFIX_PROD_DAILY = (os.getenv("BATCH_INPUT_PREFIX_PROD_DAILY") or "batch_inputs_product_daily").strip().strip("/")
BATCH_OUTPUT_PREFIX_PROD_DAILY = (os.getenv("BATCH_OUTPUT_PREFIX_PROD_DAILY") or "batch_outputs_product_daily").strip().strip("/")
BATCH_INPUT_PREFIX_PROD_TOTAL = (os.getenv("BATCH_INPUT_PREFIX_PROD_TOTAL") or "batch_inputs_product_total").strip().strip("/")
BATCH_OUTPUT_PREFIX_PROD_TOTAL = (os.getenv("BATCH_OUTPUT_PREFIX_PROD_TOTAL") or "batch_outputs_product_total").strip().strip("/")

MAX_REVIEWS_PER_PRODUCT_DAY = int(os.getenv("MAX_REVIEWS_PER_PRODUCT_DAY") or "2000")
MAX_REVIEW_TEXT_CHARS_PRODUCT = int(os.getenv("MAX_REVIEW_TEXT_CHARS_PRODUCT") or "500")
TOTAL_SUMMARY_LOOKBACK_DAYS = int(os.getenv("TOTAL_SUMMARY_LOOKBACK_DAYS") or "30")

# -----------------------------
# Exclusions (analysis skip)
# - E0 브랜드 전체 제외
# - 특정 상품코드 E02A5ZZZ999M 제외
# -----------------------------
EXCLUDED_BRAND_NOS = set(
    [s.strip() for s in (os.getenv("EXCLUDED_BRAND_NOS") or "E0").split(",") if s.strip()]
)
EXCLUDED_PRODUCT_NOS = set(
    [s.strip() for s in (os.getenv("EXCLUDED_PRODUCT_NOS") or "E02A5ZZZ999M").split(",") if s.strip()]
)

def _is_excluded_brand_product(brand_no: Any, product_no: Any) -> bool:
    b = str(brand_no or "").strip()
    p = str(product_no or "").strip()
    return (b in EXCLUDED_BRAND_NOS) or (p in EXCLUDED_PRODUCT_NOS)

def _sql_not_in_clause(field_expr: str, values: set) -> str:
    """
    values가 비어있으면 항상 TRUE가 되도록 반환.
    BigQuery 표준SQL에서 사용할 수 있는 boolean expression을 만든다.
    """
    if not values:
        return "TRUE"
    # 단순 안전 처리: ' -> \'
    vals = ", ".join([("'" + str(v).replace("'", "\\'") + "'") for v in sorted(values)])
    return f"IFNULL({field_expr}, '') NOT IN ({vals})"


# Tables
TABLE_INGEST = f"{PROJECT_ID}.{DATASET}.ingestion_files"
TABLE_RAW = f"{PROJECT_ID}.{DATASET}.reviews_raw"
TABLE_CLEAN = f"{PROJECT_ID}.{DATASET}.reviews_clean"
TABLE_LLM = f"{PROJECT_ID}.{DATASET}.review_llm_extract"
TABLE_METRICS = f"{PROJECT_ID}.{DATASET}.style_daily_metrics"

# Product-level tables
TABLE_PROD_DAILY = f"{PROJECT_ID}.{DATASET}.product_daily_feedback_llm"
TABLE_PROD_TOTAL = f"{PROJECT_ID}.{DATASET}.product_total_feedback_summary_llm"

# Fixed staging tables
STG_CLEAN = f"{PROJECT_ID}.{DATASET}.staging_reviews_clean"
# NOTE: review-level STG_LLM은 더 이상 "고정 1개"로 쓰지 않고, run마다 임시 테이블로 전환합니다.

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

    if s.isdigit() and len(s) == 8:
        y, m, d = s[0:4], s[4:6], s[6:8]
        return f"{y}-{m}-{d}"

    s2 = s.replace(".", "").replace("-", "").replace("/", "")
    if s2.isdigit() and len(s2) == 8:
        y, m, d = s2[0:4], s2[4:6], s2[6:8]
        return f"{y}-{m}-{d}"

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
        or n.startswith(BATCH_INPUT_PREFIX_PROD_DAILY + "/")
        or n.startswith(BATCH_OUTPUT_PREFIX_PROD_DAILY + "/")
        or n.startswith(BATCH_INPUT_PREFIX_PROD_TOTAL + "/")
        or n.startswith(BATCH_OUTPUT_PREFIX_PROD_TOTAL + "/")
        or n.lower().endswith(".jsonl")
        or n.lower().endswith(".ndjson")
    )

def _basename(path: str) -> str:
    p = (path or "").rstrip("/")
    if "/" not in p:
        return p
    return p.split("/")[-1]

def _safe_json_extract(text: str) -> Optional[Any]:
    """
    model이 준 JSON 텍스트(단일/배열)를 안전하게 파싱.
    fence(```json) 또는 잡문이 섞여도 최대한 복구.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    if s.startswith("```"):
        s = s.strip("`").strip()
        if "\n" in s:
            s = "\n".join(s.split("\n")[1:]).strip()
        if s.endswith("```"):
            s = s[:-3].strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    first_candidates = [s.find("{"), s.find("[")]
    first_candidates = [i for i in first_candidates if i >= 0]
    if not first_candidates:
        return None
    start = min(first_candidates)

    end_obj = s.rfind("}")
    end_arr = s.rfind("]")
    end = max(end_obj, end_arr)
    if end <= start:
        return None

    sub = s[start : end + 1].strip()
    try:
        return json.loads(sub)
    except Exception:
        return None

def _parse_extracted_at_from_line(line_obj: dict) -> str:
    """
    batch output 라인에 processed_time 같은 값이 있으면 사용하고,
    없으면 now를 사용. 반환은 ISO 문자열.
    """
    for key in ["processed_time", "createTime", "create_time", "timestamp"]:
        v = line_obj.get(key)
        if v:
            try:
                dt = pd.to_datetime(v, errors="coerce")
                if pd.isna(dt):
                    continue
                if getattr(dt, "tzinfo", None) is None:
                    dt = dt.tz_localize("UTC")
                return dt.to_pydatetime().astimezone(timezone.utc).isoformat()
            except Exception:
                continue
    return _now_utc().isoformat()

def _extract_object_and_generation_from_archive_path(path: str, output_prefix: str) -> Tuple[str, str]:
    """
    path 예: batch_outputs/<object_name...>/<generation>/predictions_0000.jsonl
    object_name에 '/'가 포함될 수 있어 generation(숫자) 세그먼트를 기준으로 분리한다.
    """
    n = (path or "").lstrip("/")
    prefix = output_prefix.strip("/")

    if not n.startswith(prefix + "/"):
        return ("", "")

    tail = n[len(prefix) + 1 :]
    parts = [p for p in tail.split("/") if p != ""]
    if len(parts) < 2:
        return ("", "")

    gen_idx = -1
    for i, p in enumerate(parts):
        if p.isdigit():
            gen_idx = i
            break

    if gen_idx <= 0:
        return ("", "")

    object_name = "/".join(parts[:gen_idx])
    generation = parts[gen_idx]
    return (object_name, generation)

def _batch_run_key(object_name: str, generation: str) -> str:
    return f"{object_name}::{generation}"

def _is_low_quality_review_text(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    if len(s) < 8:
        return True
    junk_exact = {"ㅎㅎ", "ㅋㅋ", "ㅠㅠ", "...", "??", "!!", "굿", "좋아요", "별로"}
    if s in junk_exact:
        return True
    return False

def _truncate_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "…"

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
def _ensure_ingestion_table():
    client = _bq()
    client.query(
        f"""
    CREATE TABLE IF NOT EXISTS `{TABLE_INGEST}` (
      bucket STRING,
      object_name STRING,
      generation STRING,
      received_at TIMESTAMP,
      status STRING,
      error_message STRING
    )
    """
    ).result()

def _load_ndjson_to_table(table_id: str, local_ndjson_path: str, write_disposition: str):
    """
    NDJSON(local) -> BQ Load job
    + (선택) 429(TooManyRequests) 중에서도 table.write rateLimitExceeded일 때만 아주 얇게 재시도.
    """
    client = _bq()
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=write_disposition,
        ignore_unknown_values=True,
        max_bad_records=0,
    )

    with open(local_ndjson_path, "rb") as f:
        job = client.load_table_from_file(f, table_id, job_config=job_config)

    max_attempts = 6  # small retries
    for attempt in range(max_attempts):
        try:
            job.result()
            return
        except TooManyRequests:
            reason = None
            location = None
            try:
                if getattr(job, "error_result", None):
                    reason = job.error_result.get("reason")
                    location = job.error_result.get("location")
            except Exception:
                reason = None
                location = None

            is_table_write_ratelimit = (reason == "rateLimitExceeded" and location == "table.write")
            if not is_table_write_ratelimit:
                logger.error("BQ LOAD FAILED(non-retriable 429) table=%s file=%s", table_id, local_ndjson_path)
                logger.error("BQ job_id=%s state=%s error_result=%s", job.job_id, job.state, job.error_result)
                logger.error("BQ errors=%s", job.errors)
                raise

            sleep = min((2**attempt) + random.random(), 30.0)
            logger.warning(
                "BQ LOAD 429 rateLimitExceeded(table.write). retry attempt=%d/%d sleep=%.2fs table=%s file=%s job_id=%s",
                attempt + 1,
                max_attempts,
                sleep,
                table_id,
                local_ndjson_path,
                job.job_id,
            )
            time.sleep(sleep)
            continue
        except Exception:
            logger.error("BQ LOAD FAILED table=%s file=%s", table_id, local_ndjson_path)
            logger.error("BQ job_id=%s state=%s error_result=%s", job.job_id, job.state, job.error_result)
            logger.error("BQ errors=%s", job.errors)
            raise

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

def _mark_ingestion(
    status: str,
    bucket: str,
    object_name: str,
    generation: str,
    error_message: Optional[str] = None,
):
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

def _ensure_product_tables():
    """
    product_daily/product_total 테이블이 없으면 생성한다.
    (이미 존재하면 아무 것도 하지 않음)
    """
    client = _bq()

    client.query(
        f"""
    CREATE TABLE IF NOT EXISTS `{TABLE_PROD_DAILY}` (
      product_daily_key STRING,
      batch_run_key STRING,
      metric_date DATE,
      brand_no STRING,
      product_no STRING,
      channel STRING,
      review_cnt INT64,

      weekly_feedback STRING,
      severity STRING,
      sentiment STRING,
      size_feedback STRING,
      defect_part STRING,
      color_mentioned STRING,
      repurchase_intent STRING,

      evidence_list STRING,

      extracted_at TIMESTAMP,
      model_name STRING,
      model_version STRING,
      prompt_version STRING,
      raw_json JSON
    )
    """
    ).result()

    client.query(
        f"""
    CREATE TABLE IF NOT EXISTS `{TABLE_PROD_TOTAL}` (
      product_total_key STRING,
      as_of_date DATE,
      days_lookback INT64,
      brand_no STRING,
      product_no STRING,
      channel STRING,
      daily_summary_cnt INT64,

      total_feedback_summary STRING,
      severity STRING,
      sentiment STRING,
      size_feedback STRING,
      defect_part STRING,
      color_mentioned STRING,
      repurchase_intent STRING,

      evidence_list STRING,

      extracted_at TIMESTAMP,
      model_name STRING,
      model_version STRING,
      prompt_version STRING,
      raw_json JSON
    )
    """
    ).result()

def _ensure_style_metrics_product_label_columns():
    """
    style_daily_metrics에 상품 총평 라벨 컬럼이 없으면 추가한다.
    - INFORMATION_SCHEMA로 존재 여부 확인 후, 누락 컬럼만 1번 ALTER로 추가 (rate limit 회피)
    - 실패해도 파이프라인은 계속 진행 (non-fatal)
    """
    client = _bq()

    desired = [
        ("prod_weekly_feedback", "STRING"),
        ("prod_severity", "STRING"),
        ("prod_sentiment", "STRING"),
        ("prod_size_feedback", "STRING"),
        ("prod_defect_part", "STRING"),
        ("prod_color_mentioned", "STRING"),
        ("prod_repurchase_intent", "STRING"),
        ("prod_extracted_at", "TIMESTAMP"),
        ("prod_model_name", "STRING"),
        ("prod_model_version", "STRING"),
        ("prod_prompt_version", "STRING"),
    ]

    sql = f"""
    SELECT column_name
    FROM `{PROJECT_ID}.{DATASET}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = 'style_daily_metrics'
    """
    existing = {r["column_name"] for r in client.query(sql).result()}
    missing = [(c, t) for (c, t) in desired if c not in existing]
    if not missing:
        return

    alter_parts = [f"ADD COLUMN IF NOT EXISTS {c} {t}" for (c, t) in missing]
    alter_sql = f"ALTER TABLE `{TABLE_METRICS}`\n  " + ",\n  ".join(alter_parts)

    try:
        client.query(alter_sql).result()
        logger.info("style_daily_metrics columns added: %s", [c for c, _ in missing])
    except Exception as e:
        logger.warning("Failed to ALTER style_daily_metrics (non-fatal). err=%s", str(e)[:500])

def _merge_style_daily_metrics_with_product_daily_labels_from_stg(stg_prod_daily_table: str):
    """
    product_daily 결과(이번 이벤트로 처리된 STG 테이블 기준)를
    style_daily_metrics의 동일 (metric_date, brand_no, product_no, channel, issue_category) 행에 붙여 업데이트한다.
    """
    client = _bq()
    _ensure_style_metrics_product_label_columns()

    sql = f"""
    MERGE `{TABLE_METRICS}` T
    USING (
      WITH combos AS (
        SELECT DISTINCT
          SAFE_CAST(NULLIF(CAST(metric_date AS STRING), '') AS DATE) AS metric_date,
          IFNULL(CAST(brand_no AS STRING), '') AS brand_no,
          IFNULL(CAST(product_no AS STRING), '') AS product_no,
          IFNULL(CAST(channel AS STRING), '') AS channel
        FROM `{stg_prod_daily_table}`
        WHERE metric_date IS NOT NULL AND CAST(metric_date AS STRING) != ''
      ),
      prod AS (
        SELECT
          p.metric_date,
          IFNULL(p.brand_no, '') AS brand_no,
          IFNULL(p.product_no, '') AS product_no,
          IFNULL(p.channel, '') AS channel,

          p.weekly_feedback,
          p.severity,
          p.sentiment,
          p.size_feedback,
          p.defect_part,
          p.color_mentioned,
          p.repurchase_intent,

          p.extracted_at,
          p.model_name,
          p.model_version,
          p.prompt_version
        FROM `{TABLE_PROD_DAILY}` p
        JOIN combos c
          ON p.metric_date = c.metric_date
         AND IFNULL(p.brand_no,'') = c.brand_no
         AND IFNULL(p.product_no,'') = c.product_no
         AND IFNULL(p.channel,'') = c.channel
        QUALIFY ROW_NUMBER() OVER (
          PARTITION BY p.metric_date, IFNULL(p.brand_no,''), IFNULL(p.product_no,''), IFNULL(p.channel,'')
          ORDER BY p.extracted_at DESC
        ) = 1
      )
      SELECT
        m.metric_date,
        m.brand_no,
        m.product_no,
        m.channel,
        m.issue_category,

        prod.weekly_feedback    AS prod_weekly_feedback,
        prod.severity           AS prod_severity,
        prod.sentiment          AS prod_sentiment,
        prod.size_feedback      AS prod_size_feedback,
        prod.defect_part        AS prod_defect_part,
        prod.color_mentioned    AS prod_color_mentioned,
        prod.repurchase_intent  AS prod_repurchase_intent,
        prod.extracted_at       AS prod_extracted_at,
        prod.model_name         AS prod_model_name,
        prod.model_version      AS prod_model_version,
        prod.prompt_version     AS prod_prompt_version
      FROM `{TABLE_METRICS}` m
      JOIN prod
        ON m.metric_date = prod.metric_date
       AND m.brand_no = prod.brand_no
       AND m.product_no = prod.product_no
       AND m.channel = prod.channel
      WHERE m.metric_date IS NOT NULL
    ) S
    ON T.metric_date = S.metric_date
    AND T.brand_no = S.brand_no
    AND T.product_no = S.product_no
    AND T.channel = S.channel
    AND T.issue_category = S.issue_category
    WHEN MATCHED THEN UPDATE SET
      prod_weekly_feedback = S.prod_weekly_feedback,
      prod_severity = S.prod_severity,
      prod_sentiment = S.prod_sentiment,
      prod_size_feedback = S.prod_size_feedback,
      prod_defect_part = S.prod_defect_part,
      prod_color_mentioned = S.prod_color_mentioned,
      prod_repurchase_intent = S.prod_repurchase_intent,
      prod_extracted_at = S.prod_extracted_at,
      prod_model_name = S.prod_model_name,
      prod_model_version = S.prod_model_version,
      prod_prompt_version = S.prod_prompt_version
    """

    try:
        client.query(sql).result()
        logger.info("style_daily_metrics enriched with product_daily labels (stg=%s)", stg_prod_daily_table)
    except Exception as e:
        logger.warning("Failed to MERGE product labels into style_daily_metrics (non-fatal). err=%s", str(e)[:500])

# -----------------------------
# reviews_raw append (NDJSON load)
# -----------------------------
def _append_reviews_raw_ndjson(df_std: pd.DataFrame, bucket: str, name: str, generation: str):
    loaded_at = _now_utc().isoformat()
    base_ingest_id = uuid.uuid4().hex

    tmp = f"/tmp/raw_{uuid.uuid4().hex}.ndjson"
    rows_written = 0

    with open(tmp, "w", encoding="utf-8") as f:
        for idx, r in df_std.iterrows():
            row = {
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
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1

    _load_ndjson_to_table(TABLE_RAW, tmp, write_disposition="WRITE_APPEND")
    logger.info("BQ append reviews_raw rows=%d ingest_id=%s", rows_written, base_ingest_id)

# -----------------------------
# reviews_clean MERGE (review_seq STRING) + Exclusion 적용
# -----------------------------
def _merge_reviews_clean_fixed_staging(df_std: pd.DataFrame):
    """
    reviews_clean.review_seq 가 STRING인 버전.
    - staging은 STRING 적재
    - write_date=YYYYMMDD도 YYYY-MM-DD로 정규화
    - MERGE source 중복 review_key는 ROW_NUMBER로 1개만 남김
    - ✅ EXCLUDED_BRAND_NOS / EXCLUDED_PRODUCT_NOS 는 clean부터 제외 (분석 파이프라인 전체 제외)
    """
    client = _bq()
    loaded_at = _now_utc().isoformat()

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
    df["brand_no_str"] = df["brand_no"].astype(str).str.strip()
    df["product_no_str"] = df["product_no"].astype(str).str.strip()
    df["write_date_str"] = df["write_date"].apply(_normalize_write_date_to_ymd)

    # ✅ Exclusion filter (clean & downstream)
    before_excl = len(df)
    mask_keep = (~df["brand_no_str"].isin(EXCLUDED_BRAND_NOS)) & (~df["product_no_str"].isin(EXCLUDED_PRODUCT_NOS))
    df = df[mask_keep].copy()
    dropped_excl = before_excl - len(df)
    if dropped_excl > 0:
        logger.warning(
            "EXCLUDE rows from reviews_clean (brand/product excluded) count=%d excluded_brands=%s excluded_products=%s",
            dropped_excl,
            sorted(list(EXCLUDED_BRAND_NOS))[:50],
            sorted(list(EXCLUDED_PRODUCT_NOS))[:50],
        )

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

    df["review_key"] = df["review_no"] + "-" + df["review_seq"]
    df["review_text_masked"] = df["review_contents"].astype(str)
    df["normalized_text"] = df["review_contents"].astype(str)

    tmp = f"/tmp/stg_clean_{uuid.uuid4().hex}.ndjson"
    rows_written = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            row = {
                "review_key": str(r.get("review_key", "")),
                "review_no": str(r.get("review_no", "")),
                "review_seq": str(r.get("review_seq", "")),
                "channel": str(r.get("channel", "")),
                "brand_no": str(r.get("brand_no_str", "")),
                "product_no": str(r.get("product_no_str", "")),
                "write_date": str(r.get("write_date_str", "")),
                "review_score": str(r.get("review_score", "")),
                "review_1depth": str(r.get("review_1depth", "")),
                "review_2depth": str(r.get("review_2depth", "")),
                "review_contents": str(r.get("review_contents", "")),
                "review_text_masked": str(r.get("review_text_masked", "")),
                "normalized_text": str(r.get("normalized_text", "")),
                "legacy_review_ai_score": str(r.get("review_ai_score", "")),
                "legacy_review_ai_contents": str(r.get("review_ai_contents", "")),
                "loaded_at": loaded_at,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1

    _load_ndjson_to_table(STG_CLEAN, tmp, write_disposition="WRITE_TRUNCATE")

    merge_sql = f"""
    MERGE `{TABLE_CLEAN}` T
    USING (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT
          S.*,
          ROW_NUMBER() OVER (PARTITION BY review_key ORDER BY CAST(loaded_at AS STRING) DESC) AS rn
        FROM `{STG_CLEAN}` S
      )
      WHERE rn = 1
    ) S
    ON T.review_key = S.review_key
    WHEN MATCHED THEN UPDATE SET
      review_no = CAST(S.review_no AS STRING),
      review_seq = CAST(S.review_seq AS STRING),
      channel = CAST(S.channel AS STRING),
      brand_no = CAST(S.brand_no AS STRING),
      product_no = CAST(S.product_no AS STRING),

      write_date = SAFE_CAST(NULLIF(CAST(S.write_date AS STRING), '') AS DATE),
      review_score = SAFE_CAST(NULLIF(CAST(S.review_score AS STRING), '') AS INT64),

      review_1depth = CAST(S.review_1depth AS STRING),
      review_2depth = CAST(S.review_2depth AS STRING),

      review_contents = CAST(S.review_contents AS STRING),
      review_text_masked = CAST(S.review_text_masked AS STRING),
      normalized_text = CAST(S.normalized_text AS STRING),

      legacy_review_ai_score = SAFE_CAST(NULLIF(CAST(S.legacy_review_ai_score AS STRING), '') AS FLOAT64),
      legacy_review_ai_contents = CAST(S.legacy_review_ai_contents AS STRING),

      loaded_at = SAFE_CAST(NULLIF(CAST(S.loaded_at AS STRING), '') AS TIMESTAMP)

    WHEN NOT MATCHED THEN
      INSERT (
        review_key, review_no, review_seq, channel, brand_no, product_no,
        write_date, review_score, review_1depth, review_2depth,
        review_contents, review_text_masked, normalized_text,
        legacy_review_ai_score, legacy_review_ai_contents, loaded_at
      )
      VALUES (
        CAST(S.review_key AS STRING),
        CAST(S.review_no AS STRING),
        CAST(S.review_seq AS STRING),
        CAST(S.channel AS STRING),
        CAST(S.brand_no AS STRING),
        CAST(S.product_no AS STRING),

        SAFE_CAST(NULLIF(CAST(S.write_date AS STRING), '') AS DATE),
        SAFE_CAST(NULLIF(CAST(S.review_score AS STRING), '') AS INT64),
        CAST(S.review_1depth AS STRING),
        CAST(S.review_2depth AS STRING),

        CAST(S.review_contents AS STRING),
        CAST(S.review_text_masked AS STRING),
        CAST(S.normalized_text AS STRING),

        SAFE_CAST(NULLIF(CAST(S.legacy_review_ai_score AS STRING), '') AS FLOAT64),
        CAST(S.legacy_review_ai_contents AS STRING),

        SAFE_CAST(NULLIF(CAST(S.loaded_at AS STRING), '') AS TIMESTAMP)
      )
    """
    client.query(merge_sql).result()
    logger.info("BQ MERGE reviews_clean done rows=%d (seq=STRING)", rows_written)

# -----------------------------
# “이번 파일 신규 대상” SQL (STRING 조인) + Exclusion 적용
# -----------------------------
_SQL_EXCL_BRAND_C = _sql_not_in_clause("c.brand_no", EXCLUDED_BRAND_NOS)
_SQL_EXCL_PROD_C = _sql_not_in_clause("c.product_no", EXCLUDED_PRODUCT_NOS)

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
    c.review_text_masked,
    c.review_score
  FROM `{PROJECT_ID}.{DATASET}.reviews_clean` c
  JOIN file_rows r
    ON c.review_no = r.review_no
   AND c.review_seq = r.review_seq
  WHERE {_SQL_EXCL_BRAND_C}
    AND {_SQL_EXCL_PROD_C}
)
SELECT t.*
FROM targets t
LEFT JOIN `{PROJECT_ID}.{DATASET}.review_llm_extract` e
  ON e.review_key = t.review_key
WHERE e.review_key IS NULL
"""

# -----------------------------
# Review-level prompt (Upgraded)
# -----------------------------
def _build_prompt(row: dict) -> str:
    review_text = (row.get("review_text_masked") or "").strip()
    if len(review_text) > 1200:
        review_text = review_text[:1200] + "…"

    rating = str(row.get("review_score") or "").strip()

    return f"""
너는 패션/의류 상품평을 생산/디자인/QC 관점에서 "라벨링"하는 분석기다.
반드시 JSON 한 개만 출력한다. (설명/문장/코드블록/마크다운 금지)

중요 규칙:
1) 리뷰가 너무 짧거나 의미가 모호하면 무리하게 추정하지 말고 "불명" 또는 "중립"을 사용하라.
   - 예: "굿", "좋아요", "별로", "ㅠㅠ", "ㅎㅎ", 이모지, 점(...) 등
2) 단, 리뷰 텍스트가 부실해도 "평점"이 있으면 sentiment/severity는 '평점 기반'으로 낮은 확신으로 추정할 수 있다.
   - 이 경우 evidence에 "평점 기반 추정"이라고 명시한다.
3) 제품이 아닌 배송/포장/응대 중심이면 issue_category는 "배송포장" 또는 "서비스"를 우선한다.
4) 출력은 반드시 아래 스키마의 키 이름을 그대로 사용한다(누락/추가 금지).

허용값(정확히 이 값만):
- issue_category: ["봉제","원단","색상","사이즈핏","착용감","디자인","마감","내구성","냄새","오염","배송포장","가격가치","서비스","기타","불명"]
- severity: ["없음","경미","보통","심각","치명"]
- sentiment: ["강한불만","불만","중립","만족","강한칭찬"]
- size_feedback: ["매우작다","작다","정사이즈","크다","매우크다","혼합","불명"]
- defect_part: ["봉제/박음질","단추/지퍼","밑단","소매","카라/넥","허리/밴딩","포켓","안감","원단올/올풀림","보풀","프린트/자수/로고","가죽/합피","기타","불명"]
- color_mentioned: ["없음","색상차이","변색","이염","비침","광택/톤","불명"]
- repurchase_intent: ["확실히있음","있음","불명","없음","절대없음"]

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

판정 가이드(요약):
- severity:
  - 치명: 착용 불가/파손/심한 이염·변색/심각한 피부자극, 환불 강력 의사
  - 심각: 교환/환불 언급, 기능 저해, 반복 불만
  - 보통: 불편하지만 사용 가능
  - 경미: 약간 아쉬움/개인차
  - 없음: 문제 언급 없음
- sentiment:
  - 강한불만: "최악", "다신", "환불", "버림", 심한 욕설/분노
  - 불만: 불편/실망/아쉬움 명확
  - 만족/강한칭찬: 만족/추천/재구매/최고
  - 중립: 정보성/애매/근거 부족
- size_feedback:
  - 작다/크다 혼재 시 "혼합"
  - 단서 없으면 "불명"
- defect_part:
  - 언급 없으면 "불명"
- color_mentioned:
  - 색상 관련 문제 언급 없으면 "없음" 또는 "불명"
- repurchase_intent:
  - 재구매/추천/또살래 => 있음/확실히있음
  - 다신/재구매없음 => 없음/절대없음
  - 단서 없으면 불명

리뷰 메타:
- rating(있으면): {rating}

리뷰 텍스트:
{review_text}
""".strip()

# -----------------------------
# Batch Input builder (review-level)
# -----------------------------
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
    gcs.bucket(ARCHIVE_BUCKET).blob(dest_blob).upload_from_filename(tmp_path, content_type="application/jsonl")

    input_uri = f"gs://{ARCHIVE_BUCKET}/{dest_blob}"
    logger.info("BATCH INPUT uploaded: %s (rows=%d)", input_uri, rows_written)
    return input_uri

def _normalize_model_name(model: str) -> str:
    """
    Batch(batches.create)에서 alias가 거부되는 케이스가 있어 stable version(-001)로 보정.
    최종 반환은 publishers/google/models/... 형태.
    """
    m = (model or "").strip()
    if not m:
        m = "gemini-2.0-flash-lite-001"

    alias_to_version = {
        "gemini-2.0-flash-lite": "gemini-2.0-flash-lite-001",
        "gemini-2.0-flash": "gemini-2.0-flash-001",
    }
    m = alias_to_version.get(m, m)

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

def _submit_vertex_batch_job_global_custom(input_jsonl_gcs_uri: str, output_prefix: str) -> str:
    if not input_jsonl_gcs_uri:
        return ""

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
    logger.info("BATCH SUBMITTED model=%s input=%s output=%s job=%s", model_name, input_jsonl_gcs_uri, output_prefix, job_name)
    return job_name

# =========================================================
# REVIEW-LEVEL (ARCHIVE): 임시 stg 테이블 + 스트리밍 파서/NDJSON writer
# =========================================================
def _create_review_llm_temp_stg_table(stg_table: str):
    """
    review-level staging table을 "run마다 임시 테이블"로 생성.
    - 고정 STG + ALTER 반복을 제거해서 table metadata update 폭탄을 줄임
    """
    client = _bq()
    client.query(
        f"""
    CREATE TABLE `{stg_table}` (
      review_key STRING,
      extracted_at STRING,
      model_name STRING,
      model_version STRING,
      brand_no STRING,
      product_no STRING,
      issue_category STRING,
      severity STRING,
      sentiment STRING,
      size_feedback STRING,
      defect_part STRING,
      color_mentioned STRING,
      repurchase_intent STRING,
      evidence STRING,
      raw_json_str STRING,
      prompt_version STRING
    )
    """
    ).result()

def _stream_review_predictions_to_stg_ndjson(local_predictions_path: str, out_ndjson_path: str) -> Tuple[int, List[str]]:
    """
    Vertex Batch predictions.jsonl을 스트리밍으로 읽어서,
    BQ staging table schema에 맞춘 NDJSON을 스트리밍으로 생성.
    반환: (rows_written, distinct_review_keys_list)
    """
    model_name = _normalize_model_name(VERTEX_GEMINI_MODEL)
    review_keys_set = set()
    rows_written = 0

    def _to_str_or_none(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s != "" else None

    with open(local_predictions_path, "r", encoding="utf-8") as fin, open(out_ndjson_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            extracted_at = _parse_extracted_at_from_line(obj)

            resp = obj.get("response") or {}
            model_version = str(resp.get("modelVersion") or resp.get("model_version") or "")

            text = ""
            try:
                cands = resp.get("candidates") or []
                if cands:
                    content = (cands[0].get("content") or {})
                    parts = content.get("parts") or []
                    if parts and isinstance(parts[0], dict):
                        text = str(parts[0].get("text") or "")
            except Exception:
                text = ""

            parsed = _safe_json_extract(text)
            if parsed is None:
                continue

            items = parsed if isinstance(parsed, list) else [parsed]
            for item in items:
                if not isinstance(item, dict):
                    continue

                review_key = str(item.get("review_key") or "").strip()
                if not review_key:
                    continue

                # ✅ EXCLUDE (brand/product) - review-level 결과도 저장/집계 제외
                if _is_excluded_brand_product(item.get("brand_no"), item.get("product_no")):
                    continue

                signals = item.get("signals") or {}
                if not isinstance(signals, dict):
                    signals = {}

                row = {
                    "review_key": review_key,
                    "extracted_at": extracted_at,
                    "model_name": _to_str_or_none(model_name),
                    "model_version": _to_str_or_none(model_version),
                    "brand_no": _to_str_or_none(item.get("brand_no")),
                    "product_no": _to_str_or_none(item.get("product_no")),
                    "issue_category": _to_str_or_none(item.get("issue_category")),
                    "severity": _to_str_or_none(item.get("severity")),
                    "sentiment": _to_str_or_none(item.get("sentiment")),
                    "size_feedback": _to_str_or_none(signals.get("size_feedback")),
                    "defect_part": _to_str_or_none(signals.get("defect_part")),
                    "color_mentioned": _to_str_or_none(signals.get("color_mentioned")),
                    "repurchase_intent": _to_str_or_none(signals.get("repurchase_intent")),
                    "evidence": _to_str_or_none(item.get("evidence")),
                    "raw_json_str": json.dumps(item, ensure_ascii=False),
                    "prompt_version": PROMPT_VERSION,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1
                review_keys_set.add(review_key)

    return rows_written, sorted(review_keys_set)

def _merge_review_llm_extract_from_staging(stg_table: str):
    """
    stg_table -> review_llm_extract 로 MERGE
    """
    client = _bq()

    merge_sql = f"""
    MERGE `{TABLE_LLM}` T
    USING (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT
          S.*,
          ROW_NUMBER() OVER (PARTITION BY review_key ORDER BY CAST(extracted_at AS STRING) DESC) AS rn
        FROM `{stg_table}` S
        WHERE review_key IS NOT NULL AND review_key != ''
      )
      WHERE rn = 1
    ) S
    ON T.review_key = S.review_key
    WHEN MATCHED THEN UPDATE SET
      extracted_at = SAFE_CAST(NULLIF(CAST(S.extracted_at AS STRING), '') AS TIMESTAMP),
      model_name = CAST(S.model_name AS STRING),
      model_version = CAST(S.model_version AS STRING),
      brand_no = CAST(S.brand_no AS STRING),
      product_no = CAST(S.product_no AS STRING),
      issue_category = CAST(S.issue_category AS STRING),
      severity = CAST(S.severity AS STRING),
      sentiment = CAST(S.sentiment AS STRING),
      size_feedback = CAST(S.size_feedback AS STRING),
      defect_part = CAST(S.defect_part AS STRING),
      color_mentioned = CAST(S.color_mentioned AS STRING),
      repurchase_intent = CAST(S.repurchase_intent AS STRING),
      evidence = CAST(S.evidence AS STRING),
      raw_json = SAFE.PARSE_JSON(CAST(S.raw_json_str AS STRING)),
      prompt_version = CAST(S.prompt_version AS STRING)
    WHEN NOT MATCHED THEN
      INSERT (
        review_key, extracted_at, model_name, model_version,
        brand_no, product_no, issue_category, severity, sentiment,
        size_feedback, defect_part, color_mentioned, repurchase_intent,
        evidence, raw_json, prompt_version
      )
      VALUES (
        CAST(S.review_key AS STRING),
        SAFE_CAST(NULLIF(CAST(S.extracted_at AS STRING), '') AS TIMESTAMP),
        CAST(S.model_name AS STRING),
        CAST(S.model_version AS STRING),
        CAST(S.brand_no AS STRING),
        CAST(S.product_no AS STRING),
        CAST(S.issue_category AS STRING),
        CAST(S.severity AS STRING),
        CAST(S.sentiment AS STRING),
        CAST(S.size_feedback AS STRING),
        CAST(S.defect_part AS STRING),
        CAST(S.color_mentioned AS STRING),
        CAST(S.repurchase_intent AS STRING),
        CAST(S.evidence AS STRING),
        SAFE.PARSE_JSON(CAST(S.raw_json_str AS STRING)),
        CAST(S.prompt_version AS STRING)
      )
    """
    client.query(merge_sql).result()

def _merge_style_daily_metrics_for_staged_keys(stg_table: str):
    """
    style_daily_metrics 갱신:
    - 대상: 이번 stg_table에 들어온 review_key들
    - 집계: (write_date, brand_no, product_no, channel, issue_category)
    """
    client = _bq()

    sql = f"""
    MERGE `{TABLE_METRICS}` T
    USING (
      WITH keys AS (
        SELECT DISTINCT review_key
        FROM `{stg_table}`
        WHERE review_key IS NOT NULL AND review_key != ''
      ),
      joined AS (
        SELECT
          c.write_date AS metric_date,
          e.brand_no,
          e.product_no,
          c.channel,
          e.issue_category,
          c.review_score,
          e.sentiment,
          e.severity
        FROM `{TABLE_CLEAN}` c
        JOIN `{TABLE_LLM}` e
          ON e.review_key = c.review_key
        JOIN keys k
          ON k.review_key = e.review_key
        WHERE c.write_date IS NOT NULL
      ),
      agg AS (
        SELECT
          metric_date,
          brand_no,
          product_no,
          channel,
          issue_category,
          COUNT(1) AS review_cnt,
          SUM(CASE WHEN sentiment IN ('불만','강한불만') THEN 1 ELSE 0 END) AS neg_cnt,
          SUM(CASE WHEN severity IN ('중대','심각','치명') THEN 1 ELSE 0 END) AS severe_cnt,
          AVG(SAFE_CAST(review_score AS FLOAT64)) AS avg_rating
        FROM joined
        GROUP BY metric_date, brand_no, product_no, channel, issue_category
      )
      SELECT * FROM agg
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
      INSERT (metric_date, brand_no, product_no, channel, issue_category, review_cnt, neg_cnt, severe_cnt, avg_rating)
      VALUES (S.metric_date, S.brand_no, S.product_no, S.channel, S.issue_category, S.review_cnt, S.neg_cnt, S.severe_cnt, S.avg_rating)
    """
    client.query(sql).result()

# =========================================================
# PRODUCT-LEVEL: prompts + build inputs + parse + merge
# =========================================================
def _build_product_daily_prompt(
    batch_run_key: str,
    metric_date: str,
    brand_no: str,
    product_no: str,
    channel: str,
    reviews: List[Dict[str, Any]],
) -> str:
    reviews_json = json.dumps(reviews, ensure_ascii=False)

    return f"""
너는 패션/의류 "상품 단위 총평" 작성기다.
입력은 특정 상품(product_no)에 대해 특정 날짜(metric_date)에 작성된 "원문 리뷰 목록 전체"다.
반드시 JSON 한 개만 출력한다. (설명/문장/코드블록/마크다운 금지)

중요 규칙:
1) 리뷰가 짧거나 의미가 모호한 경우가 많다. is_low_quality=true 인 리뷰는 근거(evidence)로 거의 사용하지 말고, 전체 흐름 판단에만 약하게 반영하라.
2) 의견이 혼재하면(예: 작다/크다 혼재) 단일 결론으로 뭉개지 말고 size_feedback="혼합"으로 출력하고 evidence_list에 혼재 근거를 2개 이상 제시하라.
3) defect_part, color_mentioned는 "대표 1개"만 출력하되 근거 부족 시 "불명"을 사용하라.
4) evidence_list에는 반드시 리뷰 원문에서 직접 따온 짧은 인용(최대 40자) 2~5개를 넣고, 각각 review_key를 포함하라.
5) 절대 임의로 사실을 만들어내지 말고, 근거 부족 시 보수적으로 "불명"/"중립"을 사용하라.

허용값(정확히 이 값만):
- severity: ["없음","경미","보통","심각","치명"]
- sentiment: ["강한불만","불만","중립","만족","강한칭찬"]
- size_feedback: ["매우작다","작다","정사이즈","크다","매우크다","혼합","불명"]
- defect_part: ["봉제/박음질","단추/지퍼","밑단","소매","카라/넥","허리/밴딩","포켓","안감","원단올/올풀림","보풀","프린트/자수/로고","가죽/합피","기타","불명"]
- color_mentioned: ["없음","색상차이","변색","이염","비침","광택/톤","불명"]
- repurchase_intent: ["확실히있음","있음","불명","없음","절대없음"]

출력 스키마(키 이름 고정):
{{
  "batch_run_key": "{batch_run_key}",
  "metric_date": "{metric_date}",
  "brand_no": "{brand_no}",
  "product_no": "{product_no}",
  "channel": "{channel}",
  "review_cnt": {len(reviews)},

  "weekly_feedback": "",
  "severity": "",
  "sentiment": "",
  "size_feedback": "",
  "defect_part": "",
  "color_mentioned": "",
  "repurchase_intent": "",

  "evidence_list": [
    {{"review_key":"","quote":""}}
  ]
}}

입력(상품 리뷰 전체 목록):
REVIEWS_JSON_ARRAY:
{reviews_json}
""".strip()

def _build_product_total_prompt(
    as_of_date: str,
    days_lookback: int,
    brand_no: str,
    product_no: str,
    channel: str,
    daily_summaries: List[Dict[str, Any]],
) -> str:
    daily_json = json.dumps(daily_summaries, ensure_ascii=False)

    return f"""
너는 패션/의류 "상품 누적 총평(요약의 요약)" 작성기다.
입력은 특정 상품의 일자별 총평 리스트다. 반드시 JSON 한 개만 출력한다. (설명/문장/코드블록/마크다운 금지)

중요 규칙:
1) 일자별 총평을 종합해서, 반복되는 핵심 이슈/칭찬 포인트를 요약하라.
2) 단일 라벨로 단정하기 어려우면 보수적으로 "중립"/"불명"을 사용하라.
3) evidence_list는 일자별 총평 문구에서 짧게 인용하여 2~5개 넣고 metric_date를 함께 적어라.

허용값(정확히 이 값만):
- severity: ["없음","경미","보통","심각","치명"]
- sentiment: ["강한불만","불만","중립","만족","강한칭찬"]
- size_feedback: ["매우작다","작다","정사이즈","크다","매우크다","혼합","불명"]
- defect_part: ["봉제/박음질","단추/지퍼","밑단","소매","카라/넥","허리/밴딩","포켓","안감","원단올/올풀림","보풀","프린트/자수/로고","가죽/합피","기타","불명"]
- color_mentioned: ["없음","색상차이","변색","이염","비침","광택/톤","불명"]
- repurchase_intent: ["확실히있음","있음","불명","없음","절대없음"]

출력 스키마(키 이름 고정):
{{
  "as_of_date": "{as_of_date}",
  "days_lookback": {days_lookback},
  "brand_no": "{brand_no}",
  "product_no": "{product_no}",
  "channel": "{channel}",
  "daily_summary_cnt": {len(daily_summaries)},

  "total_feedback_summary": "",
  "severity": "",
  "sentiment": "",
  "size_feedback": "",
  "defect_part": "",
  "color_mentioned": "",
  "repurchase_intent": "",

  "evidence_list": [
    {{"metric_date":"","quote":""}}
  ]
}}

입력(일자별 총평 리스트):
DAILY_SUMMARIES_JSON_ARRAY:
{daily_json}
""".strip()

def make_product_daily_batch_input_jsonl_and_upload(object_name: str, generation: str, review_keys: List[str]) -> str:
    """
    리뷰단위 결과(이번 invocation에서 처리한 review_keys) 기반으로,
    상품×작성일자(metric_date) 그룹을 뽑고, 해당 그룹의 원문 리뷰 "전부"를 넣어 product_daily batch input 생성.
    """
    if not review_keys:
        return ""

    bq = _bq()
    gcs = _gcs()

    excl_brand = _sql_not_in_clause("brand_no", EXCLUDED_BRAND_NOS)
    excl_prod = _sql_not_in_clause("product_no", EXCLUDED_PRODUCT_NOS)

    sql_groups = f"""
    SELECT
      CAST(write_date AS DATE) AS metric_date,
      IFNULL(brand_no, '') AS brand_no,
      IFNULL(product_no, '') AS product_no,
      IFNULL(channel, '') AS channel
    FROM `{TABLE_CLEAN}`
    WHERE review_key IN UNNEST(@keys)
      AND write_date IS NOT NULL
      AND {excl_brand}
      AND {excl_prod}
    GROUP BY metric_date, brand_no, product_no, channel
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("keys", "STRING", review_keys)]
    )
    groups = list(bq.query(sql_groups, job_config=job_config).result())
    if not groups:
        return ""

    batch_run_key = _batch_run_key(object_name, generation)

    tmp_path = f"/tmp/prod_daily_input_{uuid.uuid4().hex}.jsonl"
    rows_written = 0

    with open(tmp_path, "w", encoding="utf-8") as f:
        for g in groups:
            metric_date = g["metric_date"]
            brand_no = g["brand_no"]
            product_no = g["product_no"]
            channel = g["channel"]

            # 최종 방어: 제외 대상이면 SKIP
            if _is_excluded_brand_product(brand_no, product_no):
                continue

            sql_reviews = f"""
            SELECT
              review_key,
              CAST(review_score AS STRING) AS rating,
              CAST(review_text_masked AS STRING) AS text
            FROM `{TABLE_CLEAN}`
            WHERE write_date = @metric_date
              AND IFNULL(brand_no,'') = @brand_no
              AND IFNULL(product_no,'') = @product_no
              AND IFNULL(channel,'') = @channel
              AND {excl_brand}
              AND {excl_prod}
            ORDER BY review_key
            """
            qcfg = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("metric_date", "DATE", metric_date),
                    bigquery.ScalarQueryParameter("brand_no", "STRING", brand_no),
                    bigquery.ScalarQueryParameter("product_no", "STRING", product_no),
                    bigquery.ScalarQueryParameter("channel", "STRING", channel),
                ]
            )

            reviews: List[Dict[str, Any]] = []
            for r in bq.query(sql_reviews, job_config=qcfg).result(page_size=5000):
                text_full = str(r.get("text") or "")
                reviews.append(
                    {
                        "review_key": str(r.get("review_key") or ""),
                        "rating": str(r.get("rating") or ""),
                        "text": _truncate_text(text_full, MAX_REVIEW_TEXT_CHARS_PRODUCT),
                        "is_low_quality": _is_low_quality_review_text(text_full),
                    }
                )

            if not reviews:
                continue

            if len(reviews) > MAX_REVIEWS_PER_PRODUCT_DAY:
                logger.warning(
                    "Too many reviews for product_daily; truncating %d -> %d (product=%s date=%s)",
                    len(reviews),
                    MAX_REVIEWS_PER_PRODUCT_DAY,
                    product_no,
                    str(metric_date),
                )
                reviews = reviews[:MAX_REVIEWS_PER_PRODUCT_DAY]

            prompt = _build_product_daily_prompt(
                batch_run_key=batch_run_key,
                metric_date=str(metric_date),
                brand_no=brand_no,
                product_no=product_no,
                channel=channel,
                reviews=reviews,
            )

            line = {
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 512,
                        "responseMimeType": "application/json",
                    },
                }
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            rows_written += 1

    if rows_written == 0:
        return ""

    dest_blob = f"{BATCH_INPUT_PREFIX_PROD_DAILY}/{object_name}/{generation}/product_daily_input.jsonl"
    gcs.bucket(ARCHIVE_BUCKET).blob(dest_blob).upload_from_filename(tmp_path, content_type="application/jsonl")
    input_uri = f"gs://{ARCHIVE_BUCKET}/{dest_blob}"
    logger.info("PRODUCT_DAILY INPUT uploaded: %s (requests=%d)", input_uri, rows_written)
    return input_uri

def make_product_total_batch_input_jsonl_and_upload(as_of_date: date, products: List[Tuple[str, str, str]]) -> str:
    """
    products: [(brand_no, product_no, channel), ...]
    최근 N일 product_daily를 모아 product_total 요약 batch input 생성.
    """
    if not products:
        return ""

    bq = _bq()
    gcs = _gcs()

    days_lookback = TOTAL_SUMMARY_LOOKBACK_DAYS
    start_date = as_of_date - timedelta(days=days_lookback)

    tmp_path = f"/tmp/prod_total_input_{uuid.uuid4().hex}.jsonl"
    rows_written = 0

    with open(tmp_path, "w", encoding="utf-8") as f:
        for brand_no, product_no, channel in products:
            # 제외 대상이면 product_total도 생성/제출 안 함
            if _is_excluded_brand_product(brand_no, product_no):
                continue

            sql = f"""
            SELECT
              metric_date,
              weekly_feedback,
              severity,
              sentiment,
              size_feedback,
              defect_part,
              color_mentioned,
              repurchase_intent
            FROM `{TABLE_PROD_DAILY}`
            WHERE metric_date BETWEEN @start_date AND @as_of_date
              AND IFNULL(brand_no,'') = @brand_no
              AND IFNULL(product_no,'') = @product_no
              AND IFNULL(channel,'') = @channel
            ORDER BY metric_date DESC
            """
            qcfg = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
                    bigquery.ScalarQueryParameter("as_of_date", "DATE", as_of_date),
                    bigquery.ScalarQueryParameter("brand_no", "STRING", brand_no),
                    bigquery.ScalarQueryParameter("product_no", "STRING", product_no),
                    bigquery.ScalarQueryParameter("channel", "STRING", channel),
                ]
            )

            daily: List[Dict[str, Any]] = []
            for r in bq.query(sql, job_config=qcfg).result(page_size=1000):
                daily.append(
                    {
                        "metric_date": str(r.get("metric_date") or ""),
                        "weekly_feedback": str(r.get("weekly_feedback") or ""),
                        "severity": str(r.get("severity") or ""),
                        "sentiment": str(r.get("sentiment") or ""),
                        "size_feedback": str(r.get("size_feedback") or ""),
                        "defect_part": str(r.get("defect_part") or ""),
                        "color_mentioned": str(r.get("color_mentioned") or ""),
                        "repurchase_intent": str(r.get("repurchase_intent") or ""),
                    }
                )

            if not daily:
                continue

            prompt = _build_product_total_prompt(
                as_of_date=str(as_of_date),
                days_lookback=days_lookback,
                brand_no=brand_no,
                product_no=product_no,
                channel=channel,
                daily_summaries=daily,
            )

            line = {
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 512,
                        "responseMimeType": "application/json",
                    },
                }
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            rows_written += 1

    if rows_written == 0:
        return ""

    dest_blob = f"{BATCH_INPUT_PREFIX_PROD_TOTAL}/as_of={as_of_date.isoformat()}/product_total_input.jsonl"
    gcs.bucket(ARCHIVE_BUCKET).blob(dest_blob).upload_from_filename(tmp_path, content_type="application/jsonl")
    input_uri = f"gs://{ARCHIVE_BUCKET}/{dest_blob}"
    logger.info("PRODUCT_TOTAL INPUT uploaded: %s (requests=%d)", input_uri, rows_written)
    return input_uri

def _parse_product_daily_predictions_to_rows(local_path: str, object_name: str, generation: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    brk = _batch_run_key(object_name, generation)

    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            extracted_at = _parse_extracted_at_from_line(obj)
            resp = obj.get("response") or {}
            model_version = str(resp.get("modelVersion") or resp.get("model_version") or "")
            model_name = _normalize_model_name(VERTEX_GEMINI_MODEL)

            text = ""
            try:
                cands = resp.get("candidates") or []
                if cands:
                    content = (cands[0].get("content") or {})
                    parts = content.get("parts") or []
                    if parts and isinstance(parts[0], dict):
                        text = str(parts[0].get("text") or "")
            except Exception:
                text = ""

            parsed = _safe_json_extract(text)
            if not isinstance(parsed, dict):
                continue

            metric_date = str(parsed.get("metric_date") or "").strip()
            brand_no = str(parsed.get("brand_no") or "").strip()
            product_no = str(parsed.get("product_no") or "").strip()
            channel = str(parsed.get("channel") or "").strip()
            if not metric_date or not product_no:
                continue

            # ✅ 제외 대상 drop
            if _is_excluded_brand_product(brand_no, product_no):
                continue

            product_daily_key = f"{brk}::{metric_date}::{brand_no}::{product_no}::{channel}"
            evidence_list = parsed.get("evidence_list")
            evidence_str = json.dumps(evidence_list, ensure_ascii=False) if evidence_list is not None else None

            rows.append(
                {
                    "product_daily_key": product_daily_key,
                    "batch_run_key": brk,
                    "metric_date": metric_date,
                    "brand_no": brand_no,
                    "product_no": product_no,
                    "channel": channel,
                    "review_cnt": int(parsed.get("review_cnt") or 0),
                    "weekly_feedback": str(parsed.get("weekly_feedback") or "").strip(),
                    "severity": str(parsed.get("severity") or "").strip(),
                    "sentiment": str(parsed.get("sentiment") or "").strip(),
                    "size_feedback": str(parsed.get("size_feedback") or "").strip(),
                    "defect_part": str(parsed.get("defect_part") or "").strip(),
                    "color_mentioned": str(parsed.get("color_mentioned") or "").strip(),
                    "repurchase_intent": str(parsed.get("repurchase_intent") or "").strip(),
                    "evidence_list": evidence_str,
                    "extracted_at": extracted_at,
                    "model_name": model_name,
                    "model_version": model_version,
                    "prompt_version": PROMPT_VERSION,
                    "raw_json_str": json.dumps(parsed, ensure_ascii=False),
                }
            )

    return rows

def _parse_product_total_predictions_to_rows(local_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            extracted_at = _parse_extracted_at_from_line(obj)
            resp = obj.get("response") or {}
            model_version = str(resp.get("modelVersion") or resp.get("model_version") or "")
            model_name = _normalize_model_name(VERTEX_GEMINI_MODEL)

            text = ""
            try:
                cands = resp.get("candidates") or []
                if cands:
                    content = (cands[0].get("content") or {})
                    parts = content.get("parts") or []
                    if parts and isinstance(parts[0], dict):
                        text = str(parts[0].get("text") or "")
            except Exception:
                text = ""

            parsed = _safe_json_extract(text)
            if not isinstance(parsed, dict):
                continue

            as_of_date = str(parsed.get("as_of_date") or "").strip()
            brand_no = str(parsed.get("brand_no") or "").strip()
            product_no = str(parsed.get("product_no") or "").strip()
            channel = str(parsed.get("channel") or "").strip()
            days_lookback = int(parsed.get("days_lookback") or TOTAL_SUMMARY_LOOKBACK_DAYS)
            if not as_of_date or not product_no:
                continue

            # ✅ 제외 대상 drop
            if _is_excluded_brand_product(brand_no, product_no):
                continue

            product_total_key = f"{as_of_date}::{days_lookback}::{brand_no}::{product_no}::{channel}"
            evidence_list = parsed.get("evidence_list")
            evidence_str = json.dumps(evidence_list, ensure_ascii=False) if evidence_list is not None else None

            rows.append(
                {
                    "product_total_key": product_total_key,
                    "as_of_date": as_of_date,
                    "days_lookback": days_lookback,
                    "brand_no": brand_no,
                    "product_no": product_no,
                    "channel": channel,
                    "daily_summary_cnt": int(parsed.get("daily_summary_cnt") or 0),
                    "total_feedback_summary": str(parsed.get("total_feedback_summary") or "").strip(),
                    "severity": str(parsed.get("severity") or "").strip(),
                    "sentiment": str(parsed.get("sentiment") or "").strip(),
                    "size_feedback": str(parsed.get("size_feedback") or "").strip(),
                    "defect_part": str(parsed.get("defect_part") or "").strip(),
                    "color_mentioned": str(parsed.get("color_mentioned") or "").strip(),
                    "repurchase_intent": str(parsed.get("repurchase_intent") or "").strip(),
                    "evidence_list": evidence_str,
                    "extracted_at": extracted_at,
                    "model_name": model_name,
                    "model_version": model_version,
                    "prompt_version": PROMPT_VERSION,
                    "raw_json_str": json.dumps(parsed, ensure_ascii=False),
                }
            )

    return rows

def _merge_product_daily_from_staging(stg_table: str):
    client = _bq()
    merge_sql = f"""
    MERGE `{TABLE_PROD_DAILY}` T
    USING (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT
          S.*,
          ROW_NUMBER() OVER (PARTITION BY product_daily_key ORDER BY CAST(extracted_at AS STRING) DESC) AS rn
        FROM `{stg_table}` S
        WHERE product_daily_key IS NOT NULL AND product_daily_key != ''
      )
      WHERE rn = 1
    ) S
    ON T.product_daily_key = S.product_daily_key
    WHEN MATCHED THEN UPDATE SET
      batch_run_key = CAST(S.batch_run_key AS STRING),
      metric_date = SAFE_CAST(NULLIF(CAST(S.metric_date AS STRING), '') AS DATE),
      brand_no = CAST(S.brand_no AS STRING),
      product_no = CAST(S.product_no AS STRING),
      channel = CAST(S.channel AS STRING),
      review_cnt = SAFE_CAST(NULLIF(CAST(S.review_cnt AS STRING), '') AS INT64),

      weekly_feedback = CAST(S.weekly_feedback AS STRING),
      severity = CAST(S.severity AS STRING),
      sentiment = CAST(S.sentiment AS STRING),
      size_feedback = CAST(S.size_feedback AS STRING),
      defect_part = CAST(S.defect_part AS STRING),
      color_mentioned = CAST(S.color_mentioned AS STRING),
      repurchase_intent = CAST(S.repurchase_intent AS STRING),

      evidence_list = CAST(S.evidence_list AS STRING),

      extracted_at = SAFE_CAST(NULLIF(CAST(S.extracted_at AS STRING), '') AS TIMESTAMP),
      model_name = CAST(S.model_name AS STRING),
      model_version = CAST(S.model_version AS STRING),
      prompt_version = CAST(S.prompt_version AS STRING),
      raw_json = SAFE.PARSE_JSON(CAST(S.raw_json_str AS STRING))
    WHEN NOT MATCHED THEN
      INSERT (
        product_daily_key, batch_run_key, metric_date, brand_no, product_no, channel, review_cnt,
        weekly_feedback, severity, sentiment, size_feedback, defect_part, color_mentioned, repurchase_intent,
        evidence_list, extracted_at, model_name, model_version, prompt_version, raw_json
      )
      VALUES (
        CAST(S.product_daily_key AS STRING),
        CAST(S.batch_run_key AS STRING),
        SAFE_CAST(NULLIF(CAST(S.metric_date AS STRING), '') AS DATE),
        CAST(S.brand_no AS STRING),
        CAST(S.product_no AS STRING),
        CAST(S.channel AS STRING),
        SAFE_CAST(NULLIF(CAST(S.review_cnt AS STRING), '') AS INT64),

        CAST(S.weekly_feedback AS STRING),
        CAST(S.severity AS STRING),
        CAST(S.sentiment AS STRING),
        CAST(S.size_feedback AS STRING),
        CAST(S.defect_part AS STRING),
        CAST(S.color_mentioned AS STRING),
        CAST(S.repurchase_intent AS STRING),

        CAST(S.evidence_list AS STRING),

        SAFE_CAST(NULLIF(CAST(S.extracted_at AS STRING), '') AS TIMESTAMP),
        CAST(S.model_name AS STRING),
        CAST(S.model_version AS STRING),
        CAST(S.prompt_version AS STRING),
        SAFE.PARSE_JSON(CAST(S.raw_json_str AS STRING))
      )
    """
    client.query(merge_sql).result()

def _merge_product_total_from_staging(stg_table: str):
    client = _bq()
    merge_sql = f"""
    MERGE `{TABLE_PROD_TOTAL}` T
    USING (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT
          S.*,
          ROW_NUMBER() OVER (PARTITION BY product_total_key ORDER BY CAST(extracted_at AS STRING) DESC) AS rn
        FROM `{stg_table}` S
        WHERE product_total_key IS NOT NULL AND product_total_key != ''
      )
      WHERE rn = 1
    ) S
    ON T.product_total_key = S.product_total_key
    WHEN MATCHED THEN UPDATE SET
      as_of_date = SAFE_CAST(NULLIF(CAST(S.as_of_date AS STRING), '') AS DATE),
      days_lookback = SAFE_CAST(NULLIF(CAST(S.days_lookback AS STRING), '') AS INT64),
      brand_no = CAST(S.brand_no AS STRING),
      product_no = CAST(S.product_no AS STRING),
      channel = CAST(S.channel AS STRING),
      daily_summary_cnt = SAFE_CAST(NULLIF(CAST(S.daily_summary_cnt AS STRING), '') AS INT64),

      total_feedback_summary = CAST(S.total_feedback_summary AS STRING),
      severity = CAST(S.severity AS STRING),
      sentiment = CAST(S.sentiment AS STRING),
      size_feedback = CAST(S.size_feedback AS STRING),
      defect_part = CAST(S.defect_part AS STRING),
      color_mentioned = CAST(S.color_mentioned AS STRING),
      repurchase_intent = CAST(S.repurchase_intent AS STRING),

      evidence_list = CAST(S.evidence_list AS STRING),

      extracted_at = SAFE_CAST(NULLIF(CAST(S.extracted_at AS STRING), '') AS TIMESTAMP),
      model_name = CAST(S.model_name AS STRING),
      model_version = CAST(S.model_version AS STRING),
      prompt_version = CAST(S.prompt_version AS STRING),
      raw_json = SAFE.PARSE_JSON(CAST(S.raw_json_str AS STRING))
    WHEN NOT MATCHED THEN
      INSERT (
        product_total_key, as_of_date, days_lookback, brand_no, product_no, channel, daily_summary_cnt,
        total_feedback_summary, severity, sentiment, size_feedback, defect_part, color_mentioned, repurchase_intent,
        evidence_list, extracted_at, model_name, model_version, prompt_version, raw_json
      )
      VALUES (
        CAST(S.product_total_key AS STRING),
        SAFE_CAST(NULLIF(CAST(S.as_of_date AS STRING), '') AS DATE),
        SAFE_CAST(NULLIF(CAST(S.days_lookback AS STRING), '') AS INT64),
        CAST(S.brand_no AS STRING),
        CAST(S.product_no AS STRING),
        CAST(S.channel AS STRING),
        SAFE_CAST(NULLIF(CAST(S.daily_summary_cnt AS STRING), '') AS INT64),

        CAST(S.total_feedback_summary AS STRING),
        CAST(S.severity AS STRING),
        CAST(S.sentiment AS STRING),
        CAST(S.size_feedback AS STRING),
        CAST(S.defect_part AS STRING),
        CAST(S.color_mentioned AS STRING),
        CAST(S.repurchase_intent AS STRING),

        CAST(S.evidence_list AS STRING),

        SAFE_CAST(NULLIF(CAST(S.extracted_at AS STRING), '') AS TIMESTAMP),
        CAST(S.model_name AS STRING),
        CAST(S.model_version AS STRING),
        CAST(S.prompt_version AS STRING),
        SAFE.PARSE_JSON(CAST(S.raw_json_str AS STRING))
      )
    """
    client.query(merge_sql).result()

# =========================================================
# ARCHIVE ROUTE: unified handler (review outputs + product outputs)
# =========================================================
def handle_archive_event(bucket: str, name: str, generation: str) -> Tuple[str, str]:
    """
    ARCHIVE_BUCKET 이벤트 처리 (확장):
    1) 리뷰 단위: batch_outputs/**/predictions*.jsonl -> review_llm_extract + metrics -> product_daily batch 제출
    2) 상품 일자 단위: batch_outputs_product_daily/**/predictions*.jsonl -> product_daily_feedback_llm
       -> style_daily_metrics에 라벨 자동 MERGE -> product_total batch 제출
    3) 상품 누적 단위: batch_outputs_product_total/**/predictions*.jsonl -> product_total_feedback_summary_llm
    """
    if bucket != ARCHIVE_BUCKET:
        return ("SKIP", f"not archive bucket: {bucket}")

    n = (name or "").lstrip("/")
    base = _basename(n).lower()
    if not base.endswith(".jsonl") or "prediction" not in base:
        return ("SKIP", f"archive route: not a predictions jsonl name={name}")

    _ensure_product_tables()

    # ---- product_total outputs
    if n.startswith(BATCH_OUTPUT_PREFIX_PROD_TOTAL + "/"):
        local = _download_from_gcs(bucket, name, suffix=".jsonl")
        rows = _parse_product_total_predictions_to_rows(local)
        if not rows:
            return ("DONE", "PROD_TOTAL_NO_RECORDS")

        stg_table = f"{PROJECT_ID}.{DATASET}.stg_prod_total_{uuid.uuid4().hex}"
        client = _bq()
        client.query(
            f"""
        CREATE TABLE `{stg_table}` (
          product_total_key STRING,
          as_of_date STRING,
          days_lookback INT64,
          brand_no STRING,
          product_no STRING,
          channel STRING,
          daily_summary_cnt INT64,

          total_feedback_summary STRING,
          severity STRING,
          sentiment STRING,
          size_feedback STRING,
          defect_part STRING,
          color_mentioned STRING,
          repurchase_intent STRING,

          evidence_list STRING,

          extracted_at STRING,
          model_name STRING,
          model_version STRING,
          prompt_version STRING,
          raw_json_str STRING
        )
        """
        ).result()

        tmp = f"/tmp/stg_prod_total_{uuid.uuid4().hex}.ndjson"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        _load_ndjson_to_table(stg_table, tmp, write_disposition="WRITE_TRUNCATE")
        _merge_product_total_from_staging(stg_table)
        client.query(f"DROP TABLE `{stg_table}`").result()

        logger.info("ARCHIVE processed product_total rows=%d file=%s", len(rows), name)
        return ("DONE", f"PROD_TOTAL_ROWS={len(rows)}")

    # ---- product_daily outputs
    if n.startswith(BATCH_OUTPUT_PREFIX_PROD_DAILY + "/"):
        obj_name, obj_gen = _extract_object_and_generation_from_archive_path(n, BATCH_OUTPUT_PREFIX_PROD_DAILY)
        if not obj_name or not obj_gen:
            return ("SKIP", f"product_daily route: cannot parse object/gen name={name}")

        local = _download_from_gcs(bucket, name, suffix=".jsonl")
        rows = _parse_product_daily_predictions_to_rows(local, object_name=obj_name, generation=obj_gen)
        if not rows:
            return ("DONE", "PROD_DAILY_NO_RECORDS")

        stg_table = f"{PROJECT_ID}.{DATASET}.stg_prod_daily_{uuid.uuid4().hex}"
        client = _bq()
        client.query(
            f"""
        CREATE TABLE `{stg_table}` (
          product_daily_key STRING,
          batch_run_key STRING,
          metric_date STRING,
          brand_no STRING,
          product_no STRING,
          channel STRING,
          review_cnt INT64,

          weekly_feedback STRING,
          severity STRING,
          sentiment STRING,
          size_feedback STRING,
          defect_part STRING,
          color_mentioned STRING,
          repurchase_intent STRING,

          evidence_list STRING,

          extracted_at STRING,
          model_name STRING,
          model_version STRING,
          prompt_version STRING,
          raw_json_str STRING
        )
        """
        ).result()

        tmp = f"/tmp/stg_prod_daily_{uuid.uuid4().hex}.ndjson"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        _load_ndjson_to_table(stg_table, tmp, write_disposition="WRITE_TRUNCATE")
        _merge_product_daily_from_staging(stg_table)

        # ✅ product_daily 라벨을 style_daily_metrics에 자동으로 붙여넣기
        _merge_style_daily_metrics_with_product_daily_labels_from_stg(stg_table)

        client.query(f"DROP TABLE `{stg_table}`").result()

        # product_total 제출: 이번 product_daily에서 나온 (brand, product, channel)만 대상으로
        products = list({(str(r.get("brand_no") or ""), str(r.get("product_no") or ""), str(r.get("channel") or "")) for r in rows})
        # 제외 대상은 다시 한번 제거
        products = [(b, p, c) for (b, p, c) in products if not _is_excluded_brand_product(b, p)]

        as_of = _now_utc().date()
        input_uri = make_product_total_batch_input_jsonl_and_upload(as_of_date=as_of, products=products)
        if input_uri:
            out_prefix = f"gs://{ARCHIVE_BUCKET}/{BATCH_OUTPUT_PREFIX_PROD_TOTAL}/as_of={as_of.isoformat()}/"
            job = _submit_vertex_batch_job_global_custom(input_uri, output_prefix=out_prefix)
            logger.info("ARCHIVE processed product_daily rows=%d file=%s", len(rows), name)
            return ("DONE", f"PROD_DAILY_ROWS={len(rows)} TOTAL_JOB={job}")

        logger.info("ARCHIVE processed product_daily rows=%d file=%s", len(rows), name)
        return ("DONE", f"PROD_DAILY_ROWS={len(rows)} TOTAL_JOB=SKIP_NO_TARGETS")

    # ---- review-level outputs
    if not n.startswith(BATCH_OUTPUT_PREFIX + "/"):
        return ("SKIP", f"archive route: not under known output prefixes name={name}")

    obj_name, obj_gen = _extract_object_and_generation_from_archive_path(n, BATCH_OUTPUT_PREFIX)
    if not obj_name or not obj_gen:
        return ("SKIP", f"review-level route: cannot parse object/gen name={name}")

    local = _download_from_gcs(bucket, name, suffix=".jsonl")

    client = _bq()
    stg_table = f"{PROJECT_ID}.{DATASET}.stg_llm_{uuid.uuid4().hex}"
    _create_review_llm_temp_stg_table(stg_table)

    try:
        tmp = f"/tmp/stg_llm_{uuid.uuid4().hex}.ndjson"
        rows_written, review_keys = _stream_review_predictions_to_stg_ndjson(local, tmp)
        if rows_written == 0:
            client.query(f"DROP TABLE `{stg_table}`").result()
            return ("DONE", "NO_RECORDS")

        _load_ndjson_to_table(stg_table, tmp, write_disposition="WRITE_TRUNCATE")

        _merge_review_llm_extract_from_staging(stg_table)
        _merge_style_daily_metrics_for_staged_keys(stg_table)

        # NEW: product_daily batch 제출 (이번 invocation에서 처리한 review_keys 기반)
        input_uri = make_product_daily_batch_input_jsonl_and_upload(object_name=obj_name, generation=obj_gen, review_keys=review_keys)
        if input_uri:
            out_prefix = f"gs://{ARCHIVE_BUCKET}/{BATCH_OUTPUT_PREFIX_PROD_DAILY}/{obj_name}/{obj_gen}/"
            job = _submit_vertex_batch_job_global_custom(input_uri, output_prefix=out_prefix)
            logger.info("ARCHIVE processed review-level rows=%d file=%s", rows_written, name)
            return ("DONE", f"LLM_ROWS={rows_written} PROD_DAILY_JOB={job}")

        logger.info("ARCHIVE processed review-level rows=%d file=%s", rows_written, name)
        return ("DONE", f"LLM_ROWS={rows_written} PROD_DAILY_JOB=SKIP_NO_TARGETS")
    finally:
        try:
            client.query(f"DROP TABLE `{stg_table}`").result()
        except Exception as e:
            logger.warning("Failed to DROP temp stg table (non-fatal) table=%s err=%s", stg_table, str(e)[:300])

# =========================================================
# UPLOAD ROUTE: UPLOAD_BUCKET .xlsx -> raw/clean -> batch submit
# =========================================================
def handle_xlsx_upload_event(bucket: str, name: str, generation: str) -> Tuple[str, str]:
    logger.info("CONFIG UPLOAD_BUCKET=%s ARCHIVE_BUCKET=%s DATASET=%s", UPLOAD_BUCKET, ARCHIVE_BUCKET, DATASET)
    logger.info("CONFIG EXCLUDED_BRANDS=%s EXCLUDED_PRODUCTS=%s", sorted(list(EXCLUDED_BRAND_NOS)), sorted(list(EXCLUDED_PRODUCT_NOS)))

    if bucket != UPLOAD_BUCKET:
        return ("SKIP", f"not target upload bucket: {bucket}")

    if _is_internal_object(name):
        return ("SKIP", f"internal object event: {name}")

    if not name.lower().endswith(".xlsx"):
        return ("SKIP", f"not xlsx: {name}")

    local_path = _download_from_gcs(bucket, name, suffix=".xlsx")
    _assert_xlsx(local_path, name)

    df_std = _load_excel_mapped(local_path)

    # raw는 "원본 보관" 목적이라 제외하지 않음 (원하면 여기서도 필터 가능)
    if not _raw_already_loaded(bucket, name, generation):
        _append_reviews_raw_ndjson(df_std, bucket, name, generation)
    else:
        logger.info("SKIP reviews_raw already loaded for %s/%s gen=%s", bucket, name, generation)

    # clean부터 제외 -> 이후 전체 분석 파이프라인 자동 제외
    _merge_reviews_clean_fixed_staging(df_std)

    input_uri = make_batch_input_jsonl_and_upload(bucket, name, generation)
    job_name = submit_vertex_batch_job_global(input_uri, name, generation)

    return ("DONE", f"BATCH_JOB={job_name}" if job_name else "NO_TARGETS")

# -----------------------------
# CloudEvent entrypoint (single service, two routes)
# -----------------------------
@functions_framework.cloud_event
def ingest_from_gcs(cloud_event: CloudEvent):
    _ensure_ingestion_table()

    data = cloud_event.data or {}

    bucket = (data.get("bucket") or "").strip()
    name = (data.get("name") or "").strip()
    generation = str(data.get("generation", "")).strip()

    logger.info("EVENT bucket=%s name=%s generation=%s", bucket, name, generation)
    logger.info("EVENT type=%s source=%s id=%s", cloud_event.get("type"), cloud_event.get("source"), cloud_event.get("id"))

    if not bucket or not name:
        logger.warning("Missing bucket/name in event payload. data=%s", data)
        return ("OK", 200)

    # idempotency (per object + generation)
    if _already_done(bucket, name, generation):
        logger.info("SKIP already DONE: %s/%s gen=%s", bucket, name, generation)
        return ("OK", 200)

    _mark_ingestion("STARTED", bucket, name, generation)

    try:
        if bucket == UPLOAD_BUCKET:
            status, msg = handle_xlsx_upload_event(bucket, name, generation)
        elif bucket == ARCHIVE_BUCKET:
            status, msg = handle_archive_event(bucket, name, generation)
        else:
            status, msg = ("SKIP", f"unknown bucket: {bucket}")

        _mark_ingestion("DONE", bucket, name, generation, error_message=f"{status}: {msg}")
        if status == "SKIP":
            logger.info("SKIP %s/%s gen=%s reason=%s", bucket, name, generation, msg)
        return ("OK", 200)

    except Exception as e:
        logger.exception("FAILED processing %s/%s gen=%s", bucket, name, generation)
        _mark_ingestion("FAILED", bucket, name, generation, error_message=str(e)[:5000])
        return ("ERROR", 500)
