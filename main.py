import json
import logging
import os
import re
import uuid
import zipfile
from datetime import datetime, timezone, date
from typing import Dict, Any, Optional, Iterable

import functions_framework
import pandas as pd
from cloudevents.http import CloudEvent
import google.auth

from google.cloud import bigquery
from google.cloud import storage

# Vertex AI Batch (google-genai)
from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("review-pipeline")

# -----------------------------
# Project ID (안전하게 가져오기)
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
            "Project ID를 찾지 못했습니다. Cloud Run에 GOOGLE_CLOUD_PROJECT(또는 PROJECT_ID) env를 설정하세요."
        )
    return pid

PROJECT_ID = _get_project_id()
DATASET = os.getenv("BQ_DATASET", "ths_review_analytics")

# -----------------------------
# Buckets / Prefix
# -----------------------------
UPLOAD_BUCKET = os.getenv("UPLOAD_BUCKET", "ths-review-upload-bkt")     # xlsx 업로드용
ARCHIVE_BUCKET = os.getenv("ARCHIVE_BUCKET", "ths-review-archive-bkt")  # batch input/output 저장용

BATCH_INPUT_PREFIX = os.getenv("BATCH_INPUT_PREFIX", "batch_inputs")
BATCH_OUTPUT_PREFIX = os.getenv("BATCH_OUTPUT_PREFIX", "batch_outputs")

# -----------------------------
# BigQuery tables
# -----------------------------
TABLE_INGEST = f"{PROJECT_ID}.{DATASET}.ingestion_files"
TABLE_RAW = f"{PROJECT_ID}.{DATASET}.reviews_raw"
TABLE_CLEAN = f"{PROJECT_ID}.{DATASET}.reviews_clean"

# 고정 staging 테이블(폭증 방지)
STG_CLEAN = f"{PROJECT_ID}.{DATASET}.staging_reviews_clean"

# -----------------------------
# Vertex model (기본값은 안전한 것으로)
# - 너 환경에서 2.5-flash-lite가 404였으니 기본값을 2.0-flash로 둠.
# - 필요하면 Cloud Run env VERTEX_GEMINI_MODEL로 바꿔서 사용.
# -----------------------------
VERTEX_GEMINI_MODEL = os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.0-flash")

# prompt 버전(추후 결과 추적)
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")

# -----------------------------
# Excel header mapping (한글 -> 표준)
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
# Utilities
# -----------------------------
def _is_xlsx_object(name: str) -> bool:
    return name.lower().endswith(".xlsx")

def _is_internal_object(name: str) -> bool:
    """
    Cloud Storage 이벤트가 batch_inputs/batch_outputs에 대해서도 올라오면
    이 함수로 무시해서 엑셀 파이프라인이 깨지는 걸 방지.
    """
    n = name.lower()
    return (
        n.endswith(".jsonl")
        or n.startswith(f"{BATCH_INPUT_PREFIX.lower()}/")
        or n.startswith(f"{BATCH_OUTPUT_PREFIX.lower()}/")
    )

def _assert_xlsx(local_path: str, object_name: str):
    if not object_name.lower().endswith(".xlsx"):
        raise ValueError(f"Not an .xlsx file: {object_name}")

    # xlsx는 zip 컨테이너이므로 zip signature "PK"로 빠른 검사
    with open(local_path, "rb") as f:
        sig = f.read(2)
    if sig != b"PK":
        raise ValueError("Uploaded file is not a valid .xlsx (zip header 'PK' not found)")

    # zip 자체가 정상인지도 확인
    try:
        with zipfile.ZipFile(local_path, "r") as zf:
            _ = zf.namelist()[:1]
    except zipfile.BadZipFile:
        raise ValueError("Uploaded file is not a valid .xlsx (bad zip archive)")

def _normalize_write_date_to_ymd(val) -> str:
    """
    write_date가
    - 20251215 (YYYYMMDD)
    - 2025-12-15 / 2025.12.15 / 2025/12/15 (+시간 포함)
    - datetime/date/Timestamp
    - 엑셀 serial number
    로 들어와도 항상 'YYYY-MM-DD' 반환. 실패하면 "".
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""

    if isinstance(val, (datetime, date, pd.Timestamp)):
        d = val.date() if hasattr(val, "date") else val
        return d.strftime("%Y-%m-%d")

    s = str(val).strip()
    if not s:
        return ""

    # YYYYMMDD
    if re.fullmatch(r"\d{8}", s):
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"

    # YYYY-MM-DD / YYYY.MM.DD / YYYY/MM/DD (앞 10자리만)
    m = re.match(r"^(\d{4})[./-](\d{2})[./-](\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # 엑셀 serial number (대략 1~60000)
    try:
        f = float(s)
        if 1 <= f <= 60000:
            d = pd.to_datetime(f, unit="D", origin="1899-12-30", errors="coerce")
            if pd.notna(d):
                return d.date().strftime("%Y-%m-%d")
    except Exception:
        pass

    return ""

# -----------------------------
# Ingestion idempotency
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
    """재시도 시 reviews_raw 중복 append 방지"""
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
# GCS download + Excel load
# -----------------------------
def _download_to_tmp(bucket: str, name: str) -> str:
    # 원본 확장자 유지(디버깅 편의)
    ext = os.path.splitext(name)[-1] or ".bin"
    local_path = f"/tmp/{uuid.uuid4().hex}{ext}"
    gcs = _gcs()
    gcs.bucket(bucket).blob(name).download_to_filename(local_path)
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
# BigQuery Load (NDJSON) - pyarrow 불필요
# -----------------------------
def _load_ndjson_to_table(table_fqn: str, ndjson_path: str, write_disposition: str):
    client = _bq()
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=write_disposition,
        ignore_unknown_values=True,
        max_bad_records=0,
    )
    with open(ndjson_path, "rb") as f:
        job = client.load_table_from_file(f, table_fqn, job_config=job_config)
    job.result()

# -----------------------------
# reviews_raw append (NDJSON)
# -----------------------------
def _append_reviews_raw(df_std: pd.DataFrame, bucket: str, name: str, generation: str):
    client = _bq()
    loaded_at = _now_utc()
    ingest_id = uuid.uuid4().hex

    df = df_std.copy()

    # raw는 "가능한 그대로" (STRING 위주)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "ingest_id": ingest_id,
            "source_bucket": bucket,
            "source_object": name,
            "source_generation": generation,
            "loaded_at": loaded_at.isoformat(),

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
    logger.info("BQ append reviews_raw rows=%d ingest_id=%s", len(rows), ingest_id)

# -----------------------------
# ✅ 핵심: reviews_clean fixed staging MERGE (YYYYMMDD 대응)
# -----------------------------
def _merge_reviews_clean_fixed_staging(df_std: pd.DataFrame):
    """
    - 고정 staging 테이블 1개 사용(폭증 방지)
    - staging 적재는 전부 STRING
    - write_date=20251215(YYYYMMDD)도 YYYY-MM-DD로 정규화
    - MERGE는 SAFE_CAST로 타입 변환
    - source 중복 review_key는 ROW_NUMBER로 1개만 남겨 MERGE 오류 방지
    """
    client = _bq()
    loaded_at = _now_utc()

    # 0) staging 고정 테이블 없으면 생성 (전부 STRING + loaded_at TIMESTAMP)
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
      loaded_at TIMESTAMP
    )
    """
    client.query(create_sql).result()

    # 1) 파이썬에서 정규화/필터링
    df = df_std.copy()

    df["review_no"] = df["review_no"].astype(str).str.strip()
    df["review_seq"] = df["review_seq"].astype(str).str.strip()

    # ✅ 20251215 -> 2025-12-15
    df["write_date_str"] = df["write_date"].apply(_normalize_write_date_to_ymd)

    # 필수키 누락 제거
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

    # MVP: DLP 전이면 원문 그대로 사용
    df["review_text_masked"] = df["review_contents"].astype(str)
    df["normalized_text"] = df["review_contents"].astype(str)

    # 2) staging rows(전부 STRING) 생성
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "review_key": str(r.get("review_key", "")),
            "review_no": str(r.get("review_no", "")),
            "review_seq": str(r.get("review_seq", "")),
            "channel": str(r.get("channel", "")),
            "brand_no": str(r.get("brand_no", "")),
            "product_no": str(r.get("product_no", "")),
            "write_date": str(r.get("write_date_str", "")),   # ✅ YYYY-MM-DD
            "review_score": str(r.get("review_score", "")),
            "review_1depth": str(r.get("review_1depth", "")),
            "review_2depth": str(r.get("review_2depth", "")),
            "review_contents": str(r.get("review_contents", "")),
            "review_text_masked": str(r.get("review_text_masked", "")),
            "normalized_text": str(r.get("normalized_text", "")),
            "legacy_review_ai_score": str(r.get("review_ai_score", "")),
            "legacy_review_ai_contents": str(r.get("review_ai_contents", "")),
            "loaded_at": loaded_at.isoformat(),
        })

    # 3) staging WRITE_TRUNCATE 적재(NDJSON)
    tmp = f"/tmp/stg_clean_{uuid.uuid4().hex}.ndjson"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _load_ndjson_to_table(STG_CLEAN, tmp, write_disposition="WRITE_TRUNCATE")

    # 4) MERGE (source dedup + SAFE_CAST)
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
      review_seq = SAFE_CAST(NULLIF(TRIM(S.review_seq), '') AS INT64),
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
      loaded_at = S.loaded_at
    WHEN NOT MATCHED THEN
      INSERT (
        review_key, review_no, review_seq, channel, brand_no, product_no,
        write_date, review_score, review_1depth, review_2depth,
        review_contents, review_text_masked, normalized_text,
        legacy_review_ai_score, legacy_review_ai_contents, loaded_at
      )
      VALUES (
        S.review_key, S.review_no,
        SAFE_CAST(NULLIF(TRIM(S.review_seq), '') AS INT64),
        S.channel, S.brand_no, S.product_no,
        SAFE_CAST(NULLIF(TRIM(S.write_date), '') AS DATE),
        SAFE_CAST(NULLIF(TRIM(S.review_score), '') AS INT64),
        S.review_1depth, S.review_2depth,
        S.review_contents, S.review_text_masked, S.normalized_text,
        SAFE_CAST(NULLIF(TRIM(S.legacy_review_ai_score), '') AS FLOAT64),
        S.legacy_review_ai_contents,
        S.loaded_at
      )
    """
    client.query(merge_sql).result()

    logger.info("BQ MERGE reviews_clean done rows=%d (staging=%s)", len(rows), STG_CLEAN)

# -----------------------------
# Batch input JSONL (이번 파일 신규 review_key만)
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

def _build_prompt(row: Dict[str, Any]) -> str:
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

    # ✅ batch input은 ARCHIVE_BUCKET에 저장(UPLOAD_BUCKET 트리거 오염 방지)
    dest_blob = f"{BATCH_INPUT_PREFIX}/{object_name}/{generation}/batch_input.jsonl"
    gcs.bucket(ARCHIVE_BUCKET).blob(dest_blob).upload_from_filename(
        tmp_path, content_type="application/jsonl"
    )

    input_uri = f"gs://{ARCHIVE_BUCKET}/{dest_blob}"
    logger.info("BATCH INPUT uploaded: %s (rows=%d)", input_uri, rows_written)
    return input_uri

# -----------------------------
# Vertex AI Batch submit (global 강제)
# -----------------------------
def submit_vertex_batch_job_global(input_jsonl_gcs_uri: str, object_name: str, generation: str) -> str:
    if not input_jsonl_gcs_uri:
        return ""

    output_prefix = f"gs://{ARCHIVE_BUCKET}/{BATCH_OUTPUT_PREFIX}/{object_name}/{generation}/"

    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    job = client.batches.create(
        model=VERTEX_GEMINI_MODEL,
        src=input_jsonl_gcs_uri,
        config=CreateBatchJobConfig(dest=output_prefix),
    )

    job_name = getattr(job, "name", "") or str(job)
    logger.info(
        "BATCH SUBMITTED location=global model=%s input=%s output=%s job=%s",
        VERTEX_GEMINI_MODEL,
        input_jsonl_gcs_uri,
        output_prefix,
        job_name,
    )
    return job_name

# -----------------------------
# Main CloudEvent handler (Eventarc -> Cloud Run)
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

    # ✅ (중요) 내부 산출물(jsonl 등) 이벤트는 무시
    if _is_internal_object(name):
        logger.info("SKIP internal object event: %s/%s", bucket, name)
        return ("OK", 200)

    # ✅ (중요) 업로드 버킷 + xlsx만 처리 (다른 버킷/파일 무시)
    if bucket != UPLOAD_BUCKET or not _is_xlsx_object(name):
        logger.info("SKIP not a target upload: bucket=%s name=%s", bucket, name)
        return ("OK", 200)

    # 중복 처리 방지
    if _already_done(bucket, name, generation):
        logger.info("SKIP already DONE: %s/%s gen=%s", bucket, name, generation)
        return ("OK", 200)

    _mark_ingestion("STARTED", bucket, name, generation)

    try:
        # 1) 다운로드
        local_path = _download_to_tmp(bucket, name)

        # 2) XLSX 검증
        _assert_xlsx(local_path, name)

        # 3) 엑셀 로드 & 매핑
        df_std = _load_excel_mapped(local_path)

        # 4) raw 적재(재시도 중복 방지)
        if not _raw_already_loaded(bucket, name, generation):
            _append_reviews_raw(df_std, bucket, name, generation)
        else:
            logger.info("SKIP reviews_raw already loaded for %s/%s gen=%s", bucket, name, generation)

        # 5) clean merge (YYYYMMDD 대응 + 고정 staging)
        _merge_reviews_clean_fixed_staging(df_std)

        # 6) batch input 생성 + 업로드(ARCHIVE_BUCKET)
        input_uri = make_batch_input_jsonl_and_upload(bucket, name, generation)

        # 7) batch 제출(global)
        job_name = submit_vertex_batch_job_global(input_uri, name, generation)

        # 8) DONE
        _mark_ingestion(
            "DONE",
            bucket,
            name,
            generation,
            error_message=(f"BATCH_JOB={job_name}" if job_name else "NO_TARGETS"),
        )
        return ("OK", 200)

    except Exception as e:
        logger.exception("FAILED upload %s/%s gen=%s", bucket, name, generation)
        _mark_ingestion("FAILED", bucket, name, generation, error_message=str(e)[:5000])
        return ("ERROR", 500)
