import json
import logging
import os
import uuid
from datetime import datetime, timezone

import functions_framework
import pandas as pd
from cloudevents.http import CloudEvent
import google.auth  # ✅ 추가

from google.cloud import bigquery
from google.cloud import storage

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("review-pipeline")

# -----------------------------
# Config (Project ID 안전하게 가져오기)
# -----------------------------
def _get_project_id() -> str:
    # 1) 환경변수 우선 (어떤 환경에서든 유연하게)
    env_pid = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("PROJECT_ID")
        or os.getenv("BQ_PROJECT_ID")
    )
    if env_pid:
        return env_pid

    # 2) Cloud Run/ADC(Application Default Credentials)에서 project id 얻기
    _, pid = google.auth.default()
    if not pid:
        raise RuntimeError(
            "Project ID를 찾지 못했습니다. Cloud Run에 PROJECT_ID(또는 GOOGLE_CLOUD_PROJECT) env를 설정하세요."
        )
    return pid

PROJECT_ID = _get_project_id()  # ✅ 교체
DATASET = os.getenv("BQ_DATASET", "ths_review_analytics")

TABLE_INGEST = f"{PROJECT_ID}.{DATASET}.ingestion_files"
TABLE_RAW = f"{PROJECT_ID}.{DATASET}.reviews_raw"
TABLE_CLEAN = f"{PROJECT_ID}.{DATASET}.reviews_clean"

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

    # 한글 헤더 -> 표준 컬럼명
    df = df.rename(columns=EXCEL_TO_STD)

    missing = [c for c in EXPECTED_STD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"컬럼 매핑 후 누락: {missing}. 현재 컬럼={list(df.columns)}")

    # 표준 컬럼만 유지
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

    # 타입 파싱(안전)
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
    logger.info("EVENT data=%s", json.dumps(data, ensure_ascii=False))

    if not bucket or not name:
        logger.warning("Missing bucket/name in event payload. data=%s", data)
        return ("OK", 200)

    # 중복 처리 방지
    if _already_done(bucket, name, generation):
        logger.info("SKIP already DONE: %s/%s gen=%s", bucket, name, generation)
        return ("OK", 200)

    _mark_ingestion("STARTED", bucket, name, generation)

    try:
        local_path = _download_xlsx(bucket, name)
        df_std = _load_excel_mapped(local_path)

        _append_reviews_raw(df_std, bucket, name, generation)
        _merge_reviews_clean(df_std)

        _mark_ingestion("DONE", bucket, name, generation)
        return ("OK", 200)

    except Exception as e:
        logger.exception("FAILED processing %s/%s gen=%s", bucket, name, generation)
        _mark_ingestion("FAILED", bucket, name, generation, error_message=str(e)[:5000])
        return ("ERROR", 500)
