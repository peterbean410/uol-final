# FluxCD: Register forex-forecaster DAG Bundle in Airflow HelmRelease

## Overview

This document describes the change required in the **FluxCD repo** to register the
`forex-forecaster` DAG bundle with the Airflow 3 DAG processor. This enables Airflow
to automatically discover and load the forecaster DAGs from this Git repository.

**Validates: Requirement 7.7**, DAG files synced via GitDagBundle entry in the Airflow
HelmRelease.

## Target File

**Repository:** `https://git.peterbean.net/Platform/fluxcd.git`

**File to modify:** The Airflow HelmRelease values file in the FluxCD repo. This is the
HelmRelease resource for:
- **Namespace:** `airflow`
- **HelmRelease name:** `airflow`
- **Chart:** `airflow@1.20.0` from the `apache-airflow` HelmRepository

The exact file path depends on the FluxCD repo structure, but it is typically located at:
```
platform/airflow/base/helmrelease.yaml
```
or within the values overlay for the Airflow HelmRelease.

## Change Required

Append the following JSON entry to the **existing** `AIRFLOW__DAG_PROCESSOR__DAG_BUNDLE_CONFIG_LIST`
environment variable in the Airflow HelmRelease `extraEnv` section.

### JSON Entry to Add

```json
{
  "name": "forex-forecaster",
  "classpath": "airflow.providers.git.bundles.git.GitDagBundle",
  "kwargs": {
    "repo_url": "http://gitea-http.gitea.svc.cluster.local:3000/Fintech/forex.git",
    "tracking_ref": "master",
    "subdir": "probabilisticforecaster/kubeflow/airflow",
    "git_conn_id": "gitea"
  }
}
```

### Bundle Configuration Details

| Field          | Value                                                                              |
|----------------|------------------------------------------------------------------------------------|
| `name`         | `forex-forecaster`                                                                 |
| `classpath`    | `airflow.providers.git.bundles.git.GitDagBundle`                                   |
| `repo_url`     | `http://gitea-http.gitea.svc.cluster.local:3000/Fintech/forex.git`                 |
| `tracking_ref` | `master`                                                                           |
| `subdir`       | `probabilisticforecaster/kubeflow/airflow`                                         |
| `git_conn_id`  | `gitea`                                                                            |

## How to Apply

The `AIRFLOW__DAG_PROCESSOR__DAG_BUNDLE_CONFIG_LIST` env var holds a JSON array of bundle
configurations. The existing array already contains entries for:
- `forex` → `marketdata/dags` (market data DAGs)
- `forex-ta` → `ta/dags` (technical analysis DAGs)

### Example: Updated extraEnv Section

In the Airflow HelmRelease values, locate the `extraEnv` section and update the
`AIRFLOW__DAG_PROCESSOR__DAG_BUNDLE_CONFIG_LIST` value to include the new entry:

```yaml
extraEnv:
  - name: AIRFLOW__DAG_PROCESSOR__DAG_BUNDLE_CONFIG_LIST
    value: |
      [
        {
          "name": "forex",
          "classpath": "airflow.providers.git.bundles.git.GitDagBundle",
          "kwargs": {
            "repo_url": "http://gitea-http.gitea.svc.cluster.local:3000/Fintech/forex.git",
            "tracking_ref": "master",
            "subdir": "marketdata/dags",
            "git_conn_id": "gitea"
          }
        },
        {
          "name": "forex-ta",
          "classpath": "airflow.providers.git.bundles.git.GitDagBundle",
          "kwargs": {
            "repo_url": "http://gitea-http.gitea.svc.cluster.local:3000/Fintech/forex.git",
            "tracking_ref": "master",
            "subdir": "ta/dags",
            "git_conn_id": "gitea"
          }
        },
        {
          "name": "forex-forecaster",
          "classpath": "airflow.providers.git.bundles.git.GitDagBundle",
          "kwargs": {
            "repo_url": "http://gitea-http.gitea.svc.cluster.local:3000/Fintech/forex.git",
            "tracking_ref": "master",
            "subdir": "probabilisticforecaster/kubeflow/airflow",
            "git_conn_id": "gitea"
          }
        }
      ]
```

## Prerequisites

- The `gitea` Airflow connection must be configured (already exists for the other bundles).
- The `probabilisticforecaster/kubeflow/airflow/` directory must contain valid Airflow DAG
  files (e.g., `kfp_trigger_dag.py`).
- The `airflow-provider-git` package must be installed in the Airflow image (already
  required by existing bundles).

## Verification

After applying the change and FluxCD reconciles:
1. Check the Airflow webserver UI → Admin → DAG Bundles, `forex-forecaster` should appear.
2. The DAGs defined in `probabilisticforecaster/kubeflow/airflow/` (e.g.,
   `forecaster_weekly_finetune`, `forecaster_drift_retrain`, `forecaster_katib_tuning`)
   should be visible in the Airflow DAG list.
3. Confirm the DAG processor logs show successful sync from the Git repository.
