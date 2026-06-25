"""Anti-starvation guard for serial catchup DAGs.

On a serial DAG (``max_active_runs=1`` + ``depends_on_past=True``), if two
non-success runs are ever in flight at once, e.g. after a manual clear/re-link
leaves more than one run queued; the scheduler can leave the **later** run
holding the single ``max_active_runs`` slot while it is blocked by
``depends_on_past`` on an **earlier** run it depends on. The earlier run then
stays ``queued`` forever and can never get the slot: a starvation deadlock.

It is invisible to the usual failure scans because **no task fails**; every task
in the wrongly-active later run just sits in ``NONE`` (blocked), and the earlier
run's tasks never schedule. We hit this repeatedly on the 2012 aggregate/snapshot
backfill chains after clearing runs.

This guard runs every 10 minutes and heals it deterministically: for any unpaused
``max_active_runs=1`` DAG whose ``running`` run is NOT its earliest non-success
run, it demotes the later run(s) back to ``queued`` and activates the earliest,
so the slot always goes to the run that is actually runnable. In steady state
(one run in flight) it is a no-op.
"""

import pendulum
from airflow.decorators import dag, task


@dag(
    dag_id="anti_starvation_guard",
    schedule="*/10 * * * *",
    start_date=pendulum.datetime(2026, 6, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["maintenance", "scheduler"],
)
def anti_starvation_guard():
    @task
    def heal() -> int:
        from airflow.models.dag import DagModel
        from airflow.models.dagrun import DagRun
        from airflow.utils.session import create_session
        from airflow.utils.state import DagRunState
        from sqlalchemy import select

        healed = []
        with create_session() as session:
            serial_dag_ids = list(
                session.scalars(
                    select(DagModel.dag_id).where(
                        DagModel.is_paused.is_(False),
                        DagModel.max_active_runs == 1,
                    )
                )
            )
            for dag_id in serial_dag_ids:
                try:
                    runs = list(
                        session.scalars(
                            select(DagRun)
                            .where(
                                DagRun.dag_id == dag_id,
                                DagRun.state != DagRunState.SUCCESS,
                            )
                            .order_by(DagRun.logical_date)
                        )
                    )
                    if len(runs) < 2:
                        continue  # at most one run in flight -> cannot starve
                    earliest = runs[0]
                    later_running = [
                        r for r in runs[1:] if r.state == DagRunState.RUNNING
                    ]
                    if not later_running or earliest.state == DagRunState.RUNNING:
                        continue  # the running run already IS the earliest -> healthy
                    for r in later_running:
                        r.state = DagRunState.QUEUED
                    earliest.state = DagRunState.RUNNING
                    if earliest.start_date is None:
                        earliest.start_date = pendulum.now("UTC")
                    healed.append(
                        f"{dag_id}: activated {earliest.run_id}; demoted "
                        + ", ".join(r.run_id for r in later_running)
                    )
                except Exception as exc:  # never let one bad DAG break the sweep
                    print(f"anti-starvation: skipped {dag_id}: {exc}")
            session.commit()

        if healed:
            print(f"anti-starvation: healed {len(healed)} starved DAG(s):")
            for line in healed:
                print("  ", line)
        else:
            print("anti-starvation: nothing to heal")
        return len(healed)

    heal()


anti_starvation_guard()
