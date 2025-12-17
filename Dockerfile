FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8080
CMD ["functions-framework", "--target=ingest_from_gcs", "--port=8080"]
ENV PYTHONUNBUFFERED=1
