import functions_framework
from cloudevents.http import CloudEvent

@functions_framework.cloud_event
def ingest_from_gcs(cloud_event: CloudEvent):
    # Cloud Storage object finalized 이벤트 payload(data)에서 파일 정보 추출
    data = cloud_event.data

    bucket = data.get("ths-review-upload-bkt")
    name = data.get("review_test.xlsx")                 
    generation = str(data.get("generation", ""))

    # 로그로 확인 (Cloud Run Logs Explorer에서 보임)
    print(f"[EVENT] bucket={bucket}, name={name}, generation={generation}")
    print(f"[EVENT] type={cloud_event['type']} source={cloud_event['source']} id={cloud_event['id']}")

    # TODO: 여기부터
    # - ingestion_files STARTED 기록
    # - GCS에서 name 다운로드
    # - pandas로 엑셀 읽기
    # - BigQuery raw/clean 적재
    # - DONE/FAILED 업데이트

    return ("OK", 200)
