# FluxCD Application: DqnpfIntraday

This directory contains the FluxCD application manifest for the dqnpf-intraday combined predictor InferenceService.

## Deployment Instructions

### 1. Place the manifest in the FluxCD repo

Copy `DqnpfIntraday.yaml` to `app/base/DqnpfIntraday.yaml` in the FluxCD repo:

```
https://git.peterbean.net/Platform/fluxcd.git
```

### 2. Add to kustomization

Add the following entry to `app/base/kustomization.yaml` in the FluxCD repo:

```yaml
resources:
  - DqnpfIntraday.yaml
```

### 3. Remove deprecated entries

Remove the following entries from `app/base/kustomization.yaml`:

- `ProbabilisticForecaster.yaml` (previously deployed the standalone Forecaster InferenceService
- `DQNAdvisor.yaml`) previously deployed the standalone DQN InferenceService

These standalone InferenceServices are deprecated. The dqnpf-intraday combined predictor
is now the canonical (and only) KServe InferenceService for inference.

The parent pipelines (Probabilistic Forecaster and DQN) retain their own FluxCD entries
for non-serving manifests only (Katib experiments, training operator resources, model registry).

## What this manifest does

- Defines a `GitRepository` source (`fintech-forex-dqnpf`) pointing at the forex repo
  (`http://gitea-http.gitea.svc.cluster.local:3000/Fintech/forex.git`), branch `master`
- Defines a Flux `Kustomization` (`dqnpf-intraday`) that reconciles from
  `tradingmodel/intraday/dqnpf/kubeflow/manifests/` with `prune: true`, `wait: true`,
  `timeout: 10m`

## Spec references

- Requirement 24.6: FluxCD application reconciled from `tradingmodel/intraday/dqnpf/kubeflow/manifests/`
- Requirement 24.7: Replaces deprecated `ProbabilisticForecaster.yaml` and `DQNAdvisor.yaml`
