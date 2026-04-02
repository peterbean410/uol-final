FROM python:3.12-slim

WORKDIR /app

COPY marketdata/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY marketdata/ /app/marketdata/
COPY commons/ /app/commons/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "marketdata/usecases/download-interval-price-data.py"]
