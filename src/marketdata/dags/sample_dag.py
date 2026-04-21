"""Sample DAG: Market Data Health Check (v2.1)."""

from datetime import datetime, timedelta
from airflow.sdk import DAG, task

with DAG(
    dag_id="marketdata_health_check",
    default_args={"owner": "fintech", "retries": 1, "retry_delay": timedelta(minutes=5)},
    description="Daily health check for market data pipeline",
    schedule="0 6 * * 1-5",
    start_date=datetime(2026, 3, 31),
    catchup=False,
    tags=["marketdata", "health"],
) as dag:

    @task
    def check_data_source():
        print("Checking market data source connectivity...")
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    @task
    def validate_latest_data(source_status: dict):
        print(f"Source status: {source_status}")
        return {"records_checked": 100, "status": "healthy"}

    @task
    def report_status(validation_result: dict):
        print(f"Pipeline health: {validation_result}")

    report_status(validate_latest_data(check_data_source()))
