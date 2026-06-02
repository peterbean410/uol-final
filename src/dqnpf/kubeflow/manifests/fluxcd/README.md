# dqnpf-intraday FluxCD wiring

The cluster-side Flux objects for this InferenceService are **not** stored
here; they live in the `Platform/fluxcd` repo, which is what Flux reconciles:

- `app/base/DqnpfIntraday.yaml`; the `dqnpf-intraday` `Kustomization` (source:
  the `fintech-forex` `GitRepository`) that reconciles this `manifests/`
  directory, plus an `ImageUpdateAutomation` that writes the newest `modelenv`
  tag back into `inferenceservice.yaml`'s `$imagepolicy` setter so a fresh
  `modelenv` CI build rolls the sidecar.
- `app/base/ModelEnv.yaml`, owns the shared `flux-system:modelenv`
  `ImageRepository` / `ImagePolicy` that the setter in `inferenceservice.yaml`
  resolves against, and the ECR-credential refresh CronJob.

## Deployment target

The InferenceService is reconciled into the existing **`peterbean`** Kubeflow
Profile namespace (set by `../kustomization.yaml`), which already provides istio
injection, the `peterbean-ecr-credentials` image pull secret (via a PodDefault),
and the `mlpipeline-minio-artifact` secret the predictor reads. No dedicated
namespace or new secret wiring is required.

## Prerequisite

The `flux-system` git secret used by the `fintech-forex` `GitRepository` must
have **push** access to `Fintech/forex` for the `ImageUpdateAutomation` to commit
tag bumps (read access alone suffices for source reconciliation).

> Verify after rollout:
> `kubectl -n flux-system get kustomization dqnpf-intraday`,
> `kubectl -n flux-system get imageupdateautomation dqnpf-intraday`,
> `kubectl -n peterbean get inferenceservice dqnpf-intraday-predictor`.
