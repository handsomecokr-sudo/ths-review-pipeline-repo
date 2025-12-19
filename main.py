import json
import logging
import os
import uuid
import zipfile
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional, Tuple

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
        raise RuntimeError("Project ID를 찾지 못했습니다. Cloud Run에 GOOGLE_CLOUD_PROJECT env를 설정하세요.")
    return pid


PROJECT_ID = _get_project_id()
DATASET = (os.getenv("BQ_DATASET") or "ths_review_analytics").strip()

UPLOAD_BUCKET = (os.getenv("UPLOAD_BUCKET") or "ths-review-upload-bkt").strip()
ARCHIVE_BUCKET = (os.getenv("ARCHIVE_BUCKET") or "ths-review-archive-bkt").strip()

# Batch
VERTEX_GEMINI_MODEL = (os.getenv("VERTEX_GEMINI_MODEL") or "gemini-2.0-flash-lite").strip()
BATCH_INPUT_PREFIX = (os.getenv("BATCH_INPUT_PREFIX") or "batch_inputs").strip().strip("/")
BATCH_OUTPUT_PREFIX = (os.getenv("BATCH_OUTPUT_PREFIX") or "batch_outputs").strip().strip("/")

PROMPT_VERSION = (os.getenv("PROMPT_VERSION") or "v1").strip()

# Tables
TABLE_INGEST = f"{PROJECT_ID}.{DATASET}.ingestion_files"
TABLE_RAW = f"{PROJECT_ID}.{DATASET}.reviews_raw"
TABLE_CLEAN = f"{PROJECT_ID}.{DATASET}.reviews_clean"
TABLE_LLM = f"{PROJECT_ID}.{DATASET}.review_llm_extract"
TABLE_METRICS = f"{PROJECT_ID}.{DATASET}.style_daily_metrics"

# Fixed staging tables
STG_CLEAN = f"{PROJECT_ID}.{DATASET}.staging_reviews_clean"
STG_LLM = f"{PROJECT_ID}.{DATASET}.staging_review_llm_extract"

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


def _basename(path: str) -> str:
    p = (path or "").rstrip("/")
    if "/" not in p:
        return p
    return p.split("/")[-1]


def _safe_json_extract(text: str) -> Optional[Any]:
    """
    model이 준 JSON 텍스트(단일/배열)를 안전하게 파싱.
    가끔 ```json ... ``` 같은 fence나 앞뒤 잡문이 섞여도 최대한 복구.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    # remove fenced code block
    if s.startswith("```"):
        s = s.strip("`").strip()
        # ```json\n...\n``` 형태 대응(최대한)
        if "\n" in s:
            s = "\n".join(s.split("\n")[1:]).strip()
        if s.endswith("```"):
            s = s[:-3].strip()

    # direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # try substring between first {/[ and last }/]
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
    # 예시: line["processed_time"] = "2025-12-18T07:38:49.616359+00:00"
    for key in ["processed_time", "createTime", "create_time", "timestamp"]:
        v = line_obj.get(key)
        if v:
            try:
                dt = pd.to_datetime(v, errors="coerce")
                if pd.isna(dt):
                    continue
                # timezone-aware로 맞춤
                if getattr(dt, "tzinfo", None) is None:
                    dt = dt.tz_localize("UTC")
                return dt.to_pydatetime().astimezone(timezone.utc).isoformat()
            except Exception:
                continue
    return _now_utc().isoformat()


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


# -----------------------------
# reviews_raw append (NDJSON load)
# -----------------------------
def _append_reviews_raw_ndjson(df_std: pd.DataFrame, bucket: str, name: str, generation: str):
    loaded_at = _now_utc().isoformat()
    base_ingest_id = uuid.uuid4().hex

    rows: List[Dict[str, Any]] = []
    for idx, r in df_std.iterrows():
        rows.append(
            {
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
        )

    tmp = f"/tmp/raw_{uuid.uuid4().hex}.ndjson"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _load_ndjson_to_table(TABLE_RAW, tmp, write_disposition="WRITE_APPEND")
    logger.info("BQ append reviews_raw rows=%d ingest_id=%s", len(rows), base_ingest_id)


# -----------------------------
# reviews_clean MERGE (review_seq STRING)
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

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "review_key": str(r.get("review_key", "")),
                "review_no": str(r.get("review_no", "")),
                "review_seq": str(r.get("review_seq", "")),
                "channel": str(r.get("channel", "")),
                "brand_no": str(r.get("brand_no", "")),
                "product_no": str(r.get("product_no", "")),
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
        )

    tmp = f"/tmp/stg_clean_{uuid.uuid4().hex}.ndjson"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
    logger.info("BQ MERGE reviews_clean done rows=%d (seq=STRING)", len(rows))


# -----------------------------
# “이번 파일 신규 대상” SQL (STRING 조인)
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


# =========================================================
# ARCHIVE ROUTE: batch_outputs/* predictions.jsonl -> BQ 적재 + metrics 갱신
# =========================================================
def _ensure_llm_staging_table():
    client = _bq()

    # 1) 테이블이 없으면 생성
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS `{STG_LLM}` (
      review_key STRING
    )
    """
    client.query(create_sql).result()

    # 2) 컬럼이 빠져있을 수 있으니, 없으면 추가(스키마 보정)
    #    (IF NOT EXISTS 지원)
    alter_columns = [
        ("extracted_at", "STRING"),
        ("model_name", "STRING"),
        ("model_version", "STRING"),
        ("brand_no", "STRING"),
        ("product_no", "STRING"),
        ("issue_category", "STRING"),
        ("severity", "STRING"),
        ("sentiment", "STRING"),
        ("size_feedback", "STRING"),
        ("defect_part", "STRING"),
        ("color_mentioned", "STRING"),
        ("repurchase_intent", "STRING"),
        ("evidence", "STRING"),
        ("raw_json_str", "STRING"),     # ✅ 이번에 문제난 컬럼
        ("prompt_version", "STRING"),
    ]

    for col, typ in alter_columns:
        try:
            client.query(f"ALTER TABLE `{STG_LLM}` ADD COLUMN IF NOT EXISTS {col} {typ}").result()
        except Exception as e:
            # 리전/계정에 따라 IF NOT EXISTS 미지원/권한 이슈 등 예외 대비
            logger.warning("ALTER failed col=%s typ=%s err=%s", col, typ, str(e)[:300])



def _parse_predictions_jsonl_to_rows(local_path: str) -> List[Dict[str, Any]]:
    """
    Vertex Batch predictions.jsonl:
    - 각 줄: {"request":..., "response": {...}} 형태
    - 모델 출력은 보통 response.candidates[0].content.parts[0].text 안에 JSON(단일/배열)
    """
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
            # model_name은 batches에 준 모델명을 우선
            model_name = _normalize_model_name(VERTEX_GEMINI_MODEL)

            # 후보 텍스트
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
                # JSON 파싱 실패해도 raw_json_str는 남기고 싶으면 여기서 저장 가능(원하면)
                continue

            items = parsed if isinstance(parsed, list) else [parsed]
            for item in items:
                if not isinstance(item, dict):
                    continue

                review_key = str(item.get("review_key") or "").strip()
                if not review_key:
                    continue

                signals = item.get("signals") or {}
                if not isinstance(signals, dict):
                    signals = {}

                def _to_str_or_none(v: Any) -> Optional[str]:
                    if v is None:
                        return None
                    s = str(v).strip()
                    return s if s != "" else None

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
                    # JSON 타입 컬럼 안정화를 위해 staging에는 string으로
                    "raw_json_str": json.dumps(item, ensure_ascii=False),
                    "prompt_version": PROMPT_VERSION,
                }
                rows.append(row)

    return rows


def _merge_review_llm_extract_from_staging():
    """
    STG_LLM(STRING) -> review_llm_extract(JSON 포함)로 MERGE
    - raw_json은 SAFE.PARSE_JSON(raw_json_str)
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
        FROM `{STG_LLM}` S
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


def _merge_style_daily_metrics_for_staged_keys():
    """
    style_daily_metrics 갱신:
    - 대상: 이번 STG_LLM에 들어온 review_key들
    - 집계: (write_date, brand_no, product_no, channel, issue_category)
      review_cnt / neg_cnt / severe_cnt / avg_rating
    """
    client = _bq()

    sql = f"""
    MERGE `{TABLE_METRICS}` T
    USING (
      WITH keys AS (
        SELECT DISTINCT review_key
        FROM `{STG_LLM}`
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
          SUM(CASE WHEN sentiment = '불만' THEN 1 ELSE 0 END) AS neg_cnt,
          SUM(CASE WHEN severity = '중대' THEN 1 ELSE 0 END) AS severe_cnt,
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


def handle_archive_event(bucket: str, name: str, generation: str) -> Tuple[str, str]:
    """
    ARCHIVE_BUCKET 이벤트 처리:
    - batch_outputs/**/predictions*.jsonl 만 처리
    - 그 외(batch_inputs 등)는 SKIP
    """
    if bucket != ARCHIVE_BUCKET:
        return ("SKIP", f"not archive bucket: {bucket}")

    n = (name or "").lstrip("/")
    if not n.startswith(BATCH_OUTPUT_PREFIX + "/"):
        return ("SKIP", f"archive route: not under {BATCH_OUTPUT_PREFIX}/ name={name}")

    base = _basename(n).lower()
    # predictions.jsonl 또는 predictions_0000.jsonl 같은 형태를 넓게 허용
    if not base.endswith(".jsonl") or "prediction" not in base:
        return ("SKIP", f"archive route: not a predictions jsonl name={name}")

    local = _download_from_gcs(bucket, name, suffix=".jsonl")
    _ensure_llm_staging_table()

    rows = _parse_predictions_jsonl_to_rows(local)
    if not rows:
        return ("DONE", "NO_RECORDS")

    tmp = f"/tmp/stg_llm_{uuid.uuid4().hex}.ndjson"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _load_ndjson_to_table(STG_LLM, tmp, write_disposition="WRITE_TRUNCATE")

    _merge_review_llm_extract_from_staging()
    _merge_style_daily_metrics_for_staged_keys()

    logger.info("ARCHIVE processed predictions rows=%d file=%s", len(rows), name)
    return ("DONE", f"LLM_ROWS={len(rows)}")


# =========================================================
# UPLOAD ROUTE: UPLOAD_BUCKET .xlsx -> raw/clean -> batch submit
# =========================================================
def handle_xlsx_upload_event(bucket: str, name: str, generation: str) -> Tuple[str, str]:
    logger.info("CONFIG UPLOAD_BUCKET=%s ARCHIVE_BUCKET=%s DATASET=%s", UPLOAD_BUCKET, ARCHIVE_BUCKET, DATASET)

    if bucket != UPLOAD_BUCKET:
        return ("SKIP", f"not target upload bucket: {bucket}")

    if _is_internal_object(name):
        return ("SKIP", f"internal object event: {name}")

    if not name.lower().endswith(".xlsx"):
        return ("SKIP", f"not xlsx: {name}")

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

    return ("DONE", f"BATCH_JOB={job_name}" if job_name else "NO_TARGETS")


# -----------------------------
# CloudEvent entrypoint (single service, two routes)
# -----------------------------
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

    # idempotency (per object + generation)
    if _already_done(bucket, name, generation):
        logger.info("SKIP already DONE: %s/%s gen=%s", bucket, name, generation)
        return ("OK", 200)

    _mark_ingestion("STARTED", bucket, name, generation)

    try:
        # Route by bucket
        if bucket == UPLOAD_BUCKET:
            status, msg = handle_xlsx_upload_event(bucket, name, generation)
        elif bucket == ARCHIVE_BUCKET:
            status, msg = handle_archive_event(bucket, name, generation)
        else:
            status, msg = ("SKIP", f"unknown bucket: {bucket}")

        # Always close ingestion record
        _mark_ingestion("DONE", bucket, name, generation, error_message=f"{status}: {msg}")
        if status == "SKIP":
            logger.info("SKIP %s/%s gen=%s reason=%s", bucket, name, generation, msg)
        return ("OK", 200)

    except Exception as e:
        logger.exception("FAILED processing %s/%s gen=%s", bucket, name, generation)
        _mark_ingestion("FAILED", bucket, name, generation, error_message=str(e)[:5000])
        return ("ERROR", 500)
