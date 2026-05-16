# Migration Runbook: Remove Deprecated Forecaster & DQN InferenceServices

**Scope:** Removes the standalone `forecaster-predictor` and `dqn-predictor`
KServe InferenceServices from the cluster. After this migration, the only
KServe serving target for the combined trading system is
`dqnpf-intraday-predictor` (deployed by the FluxCD application created in
Task 31.6).

**Spec references:** Requirements 5.4, 15.4, 24.7. Tasks.md Task 27.6.

**Prerequisites:**
- The `dqnpf-intraday-predictor` InferenceService is healthy and reachable
  (Task 30.5 manifest deployed; smoke-tested per Task 32).
- No live trading client is configured to call the deprecated endpoints.
- The DQN and Forecaster Model Registry entries remain published (the
  combined predictor loads checkpoints from them in-process).

---

## Step 1, Confirm zero live traffic on the deprecated endpoints

For each deprecated InferenceService, query KServe / Knative metrics over
the last 24h to confirm zero successful requests:

```bash
# Replace <namespace> with the actual namespace (forecaster / deepqnetwork).
kubectl get inferenceservice forecaster-predictor -n <namespace> \
  -o jsonpath='{.status.url}'
kubectl get inferenceservice dqn-predictor -n <namespace> \
  -o jsonpath='{.status.url}'
```

Then check the Prometheus dashboards (or your equivalent):

- `kserve_request_count{inference_service="forecaster-predictor"}` over 24h → expect 0.
- `kserve_request_count{inference_service="dqn-predictor"}` over 24h → expect 0.

If either shows non-zero traffic, **stop here**: identify the caller(s),
update them to point at `dqnpf-intraday-predictor`, and re-verify after a
24h cooldown.

---

## Step 2, Update the FluxCD repository

Edit the FluxCD repo (`https://git.peterbean.net/Platform/fluxcd.git`) on a
new branch:

1. **Delete the standalone InferenceService Flux applications**, if they
   exist as separate files:

   - `app/base/ProbabilisticForecaster.yaml`, only the `InferenceService`
     resource. **Retain** the GitRepository / Kustomization entries that
     reconcile non-serving manifests (Katib experiments, registry
     resources). If those non-serving manifests live alongside the
     InferenceService in the same `kustomization.yaml`, strip the
     `inferenceservice.yaml` reference from that kustomization, do **not**
     delete the whole file.

   - `app/base/DQNAdvisor.yaml`, same treatment.

2. **Update `app/base/kustomization.yaml`** in the FluxCD repo:

   ```diff
    resources:
      - Oxigraph.yaml
   -  - ProbabilisticForecaster.yaml   # was: standalone Forecaster InferenceService
   -  - DQNAdvisor.yaml                 # was: standalone DQN InferenceService
   +  - DqnpfIntraday.yaml              # combined predictor (Task 31.6)
   ```

   If `ProbabilisticForecaster.yaml` / `DQNAdvisor.yaml` still need to be
   reconciled for non-serving resources, **keep** those lines and instead
   ensure the underlying `manifests/` directory in this repo no longer
   contains `inferenceservice.yaml` (see Step 3).

3. **Commit and open a PR** in the FluxCD repo. PR description should
   reference this runbook and the matching `dqnpf-intraday-predictor`
   deployment.

---

## Step 3, Remove the InferenceService manifests from this repo

In the forex repo, remove only the `InferenceService` resources from each
parent pipeline's manifests directory. The Forecaster and DQN training
pipelines retain their Katib experiments, PyTorchJob templates, registry
resources, and namespaces.

- `probabilisticforecaster/kubeflow/manifests/serving/inferenceservice.yaml`
, delete this file. Update the parent `kustomization.yaml` to no longer
  reference it.

- `deepqnetwork/kubeflow/manifests-dqn/serving/inferenceservice.yaml`,
  delete this file. Update the parent `kustomization.yaml` accordingly.

Open a PR in the forex repo (`Fintech/forex`) tagged
`dqnpf-intraday-migration` and link to the FluxCD PR from Step 2.

---

## Step 4, Roll out via FluxCD

Once both PRs are merged:

1. Watch the FluxCD reconciliation:

   ```bash
   flux get kustomizations --watch
   ```

2. Confirm the deprecated InferenceServices are deleted from the cluster:

   ```bash
   kubectl get inferenceservice -A | grep -E 'forecaster-predictor|dqn-predictor'
   # expect no output
   ```

3. Verify the `dqnpf-intraday-predictor` is still healthy:

   ```bash
   kubectl get inferenceservice dqnpf-intraday-predictor -n dqnpf-intraday
   # READY column should report True
   ```

---

## Step 5, Update the Model Registry KServe trigger paths

Reqs 6.4 and 16.4 now route promotion events through the dqnpf-intraday
predictor's hot-reload path instead of patching standalone InferenceService
manifests. After Step 4, audit any registry-side webhook configuration:

- Remove webhook subscribers that previously targeted
  `forecaster-predictor` or `dqn-predictor`.
- Confirm a subscriber exists for `dqnpf-intraday-predictor` hot-reload
  (created in Task 30.3).

---

## Rollback

If the combined predictor regresses post-migration, redeploy the deprecated
InferenceServices by reverting the FluxCD repo PR from Step 2 and the forex
repo PR from Step 3. The historical YAMLs are recoverable from git history
and the prior FluxCD revision will reconcile them back onto the cluster.

The Model Registry entries for the parent models are unchanged by this
migration, so the deprecated predictors can come back up with the same
checkpoint paths they previously used.

---

## Verification Checklist

- [ ] 24h-zero-traffic check completed for both deprecated endpoints
- [ ] FluxCD PR merged (`kustomization.yaml` no longer references the
      deprecated InferenceService Flux applications)
- [ ] Forex repo PR merged (deprecated `inferenceservice.yaml` files removed)
- [ ] `kubectl get inferenceservice -A` does not list
      `forecaster-predictor` or `dqn-predictor`
- [ ] `dqnpf-intraday-predictor` reports READY=True
- [ ] Registry webhook subscribers point at the combined predictor only
- [ ] Migration PR descriptions cross-reference this runbook
