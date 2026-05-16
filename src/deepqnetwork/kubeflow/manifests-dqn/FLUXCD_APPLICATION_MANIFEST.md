# FluxCD: DQN Advisor Application Manifest

## Overview

This document describes the FluxCD application manifest required in the **FluxCD repo**
to reconcile the DQN trading agent Kubernetes manifests from this repository.

**Validates: Requirement DQN-R25**; THE DQN FluxCD application SHALL be reconciled from
`deepqnetwork/kubeflow/manifests-dqn/` with prune and health check enabled.

## Target File

**Repository:** `https://git.peterbean.net/Platform/fluxcd.git`

**File to create:** `app/base/DQNAdvisor.yaml`

## Manifest Content

Place the following YAML in `app/base/DQNAdvisor.yaml` in the FluxCD repo:

```yaml
---
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: fintech-forex-dqn
  namespace: flux-system
spec:
  interval: 1m
  url: http://gitea-http.gitea.svc.cluster.local:3000/Fintech/forex.git
  ref:
    branch: master
---
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: dqn-advisor
  namespace: flux-system
spec:
  interval: 1m
  path: ./deepqnetwork/kubeflow/manifests-dqn
  sourceRef:
    kind: GitRepository
    name: fintech-forex-dqn
  prune: true
  wait: true
  timeout: 10m
```

## Update to `app/base/kustomization.yaml`

Add `DQNAdvisor.yaml` to the resources list in `app/base/kustomization.yaml` in the
FluxCD repo:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  # ... existing entries ...
  - ProbabilisticForecaster.yaml
  - DQNAdvisor.yaml          # <-- add this line
```

## Resource Details

| Resource         | Kind           | Name                | Namespace    |
|------------------|----------------|---------------------|--------------|
| Git source       | GitRepository  | `fintech-forex-dqn` | `flux-system`|
| App kustomization| Kustomization  | `dqn-advisor`       | `flux-system`|

### GitRepository

- **URL:** `http://gitea-http.gitea.svc.cluster.local:3000/Fintech/forex.git`
- **Branch:** `master`
- **Interval:** 1m (polls for changes every minute)

### Kustomization

- **Path:** `./deepqnetwork/kubeflow/manifests-dqn` (relative to repo root)
- **Source:** References the `fintech-forex-dqn` GitRepository above
- **Prune:** `true`, removes resources from the cluster when they are removed from Git
- **Wait:** `true`, waits for all resources to become ready before reporting success
- **Timeout:** `10m`, maximum time to wait for reconciliation

## What Gets Deployed

The Kustomization at `deepqnetwork/kubeflow/manifests-dqn/kustomization.yaml` deploys:
- `namespace.yaml`; the `forecaster-workloads` namespace (shared with Forecaster)
- `resource_quotas.yaml`, resource quotas for DQN workloads
- `../monitoring/dqn_prometheus_rules.yaml`, Prometheus alerting rules for DQN

## Prerequisites

- The FluxCD platform must be operational on the cluster.
- The `flux-system` namespace must exist with FluxCD controllers running.
- The Gitea instance must be accessible from the cluster at the internal service URL.

## Verification

After applying the change and FluxCD reconciles:
1. Check `flux get sources git fintech-forex-dqn`, should show `Ready` with latest revision.
2. Check `flux get kustomizations dqn-advisor`, should show `Ready` and `Applied`.
3. Verify the `forecaster-workloads` namespace exists with the expected resource quotas.
4. Confirm Prometheus rules are loaded: `kubectl get prometheusrules -n forecaster-workloads`.

## Local Reference

The actual YAML content is also available in this directory as `fluxcd-dqnadvisor.yaml`
for reference and CI validation purposes.
