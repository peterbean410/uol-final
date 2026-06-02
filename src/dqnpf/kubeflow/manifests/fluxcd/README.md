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

## Modelenv image automation

`ModelenvImageAutomation.yaml` makes a merged `modelenv/**` change deploy itself,
instead of leaving the predictor pod on a stale `modelenv:latest`.

It defines three objects in `flux-system`:

- `ImageRepository/modelenv`, scans the ECR `modelenv` repo every 5m
  (`provider: aws`, no static creds).
- `ImagePolicy/modelenv`, selects the newest CI build tag (filters to the
  12-digit `YYYYMMDDHHMM` tags that `modelenv-build.yml` emits, numerical order).
- `ImageUpdateAutomation/modelenv`, writes the selected tag into the
  `$imagepolicy` setter in `inferenceservice.yaml` and pushes the commit to
  `master`, which the `dqnpf-intraday` Kustomization then reconciles → pod rollout.

### Bootstrap (Platform/fluxcd repo)

Copy `ModelenvImageAutomation.yaml` to `app/base/` alongside `DqnpfIntraday.yaml`
and add it to `app/base/kustomization.yaml`:

```yaml
resources:
  - DqnpfIntraday.yaml
  - ModelenvImageAutomation.yaml
```

### Cluster prerequisites (one-time)

- **ECR read for image-reflector-controller.** With `provider: aws`, the
  controller authenticates with its own IAM identity. Grant that identity (IRSA
  on its service account, or the node role) `ecr:GetAuthorizationToken` and read
  on the `modelenv` repository. Without this the `ImageRepository` stays
  `NotReady` and no tags are scanned.
- **Git write for ImageUpdateAutomation.** The `flux-system` secret referenced by
  the `fintech-forex-dqnpf` `GitRepository` must have **push** access to the gitea
  repo (it only needs read for source reconciliation). Without write access the
  automation scans but cannot commit the tag bump.

> Verify after rollout: `flux get image repository modelenv`,
> `flux get image policy modelenv`, `flux get image update modelenv`.

> API versions assume image-automation/image-reflector controllers at the Flux 2.x
> `v1beta2`/`v1beta1` releases shipped in this cluster; adjust if `flux check`
> reports different served versions.

## Spec references

- Requirement 24.6: FluxCD application reconciled from `tradingmodel/intraday/dqnpf/kubeflow/manifests/`
- Requirement 24.7: Replaces deprecated `ProbabilisticForecaster.yaml` and `DQNAdvisor.yaml`
