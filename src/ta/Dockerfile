FROM python:3.12-slim

WORKDIR /app

COPY ta/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ta/ /app/ta/
COPY commons/ /app/commons/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "ta/usecases/create-eoi-ta-snapshot.py"]
