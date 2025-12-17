import json
import logging

import functions_framework
from cloudevents.http import CloudEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("review-pipeline")

@functions_framework.cloud_event
def ingest_from_gcs(cloud_event: CloudEvent):
    # Cloud Storage object finalized 이벤트 payload(data)
    data = cloud_event.data or {}

    # ✅ 올바른 키: bucket / name / generation
    bucket = data.get("bucket")
    name = data.get("name")  # 예: "review_test.xlsx"
    generation = str(data.get("generation", ""))

    # ✅ 로그 (stdout으로 남김)
    logger.info("EVENT bucket=%s name=%s generation=%s", bucket, name, generation)
    logger.info("EVENT type=%s source=%s id=%s",
                cloud_event.get("type"), cloud_event.get("source"), cloud_event.get("id"))

    # payload 전체도 필요하면 확인 (너무 길면 나중에 제거)
    logger.info("EVENT data=%s", json.dumps(data, ensure_ascii=False))

    # TODO: 다음 단계에서 여기부터 구현
    # - ingestion_files STARTED 기록 (중복이면 종료)
    # - GCS에서 name 다운로드
    # - pandas로 엑셀 읽기
    # - BigQuery raw/clean 적재
    # - DONE/FAILED 업데이트

    return ("OK", 200)
