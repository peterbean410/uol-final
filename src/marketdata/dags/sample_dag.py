"""Sample DAG: Market Data Health Check.

A simple DAG that runs daily to verify market data pipeline connectivity.
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG, task

default_args = {
    "owner": "fintech",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="marketdata_health_check",
    default_args=default_args,
    description="Daily health check for market data pipeline",
    schedule="0 6 * * 1-5",
    start_date=datetime(2026, 3, 31),
    catchup=False,
    tags=["marketdata", "health"],
) as dag:

    @task
    def check_data_source():
        """Verify market data source is reachable."""
        print("Checking market data source connectivity...")
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    @task
    def validate_latest_data(source_status: dict):
        """Validate that recent data exists."""
        print(f"Source status: {source_status}")
        print("Validating latest market data...")
        return {"records_checked": 100, "status": "healthy"}

    @task
    def report_status(validation_result: dict):
        """Report pipeline health status."""
        print(f"Pipeline health: {validation_result}")

    source = check_data_source()
    validation = validate_latest_data(source)
    report_status(validation)
