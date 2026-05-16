"""Airflow integration for triggering Kubeflow dqnpf-intraday pipeline runs.

GitDagBundle entry (for the Airflow HelmRelease in the FluxCD repo):

    dag_bundle:
      name: forex-dqnpf-intraday
      type: git
      kwargs:
        tracking_ref: master
        subdir: tradingmodel/intraday/dqnpf/kubeflow/airflow
"""
