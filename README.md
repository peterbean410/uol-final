# An Uncertainty-Gated Reinforcement-Learning Advisor Bot for USD/JPY

**University of London CM3070 Final Project, Template 4.2, "Financial advisor bot"**

Code repository for the final project. This repository is **code only**; the written
report and the project specifications are submitted separately.

---

## What this is

An end-to-end financial advisor bot for the USD/JPY currency market. It treats
intraday trading as decision-making under uncertainty: a **Double Deep Q-Network**
learns a trading policy, a **causal-Transformer probabilistic forecaster** predicts a
Gaussian distribution over the next return, and an **integration layer** screens the
policy's actions using the forecaster's uncertainty, with a self-correcting
*profitability gate* that switches the screen off automatically when it stops earning
its keep. A full MLOps pipeline trains, gates and serves the models, so the live bot
only ever runs checkpoints that have passed an automatic out-of-sample test.

Roughly 83,000 lines of first-party code across seven components, running on a
Kubernetes cluster: scheduled ingestion to S3, gated training pipelines, a model
registry, a hot-reloading predictor, and live execution against a real broker account.

## The headline result is negative, and that is the contribution

The system was delivered in full and behaves as designed. The **science** did not
support the hypothesis, and establishing that rigorously is what the project is about:

- **No durable out-of-sample edge in any regime.** A six-quarter walk-forward across
  two models returned **11 of 12 cells negative**. The single positive cell is regime
  luck; the model that *memorised* that quarter loses on it.
- **More data does not help.** A full-history retrain over 14 years (≈15,650 episodes)
  lost −4,339 out of sample. An in-sample overfit probe (thirty passes memorising an
  obvious one-month crash) reached only Sharpe **+0.18**, so the constraint is the RL
  *formulation*, not the sample.
- **The forecaster is a degenerate constant.** Predicted σ spans **5.819–5.822 bps**
  across all 24 hours while realised volatility swings ×1.69 (correlation **0.08**),
  and directional accuracy is indistinguishable from chance (Pesaran–Timmermann
  **p = 0.72**) across four independent retrains.
- **The one profitable-looking run dissolves on decomposition.** A +3,472 quarter had a
  *median losing session*, 36% of sessions profitable, and five crisis sessions
  supplying 202% of the total.

The engineering, by contrast, met its specification: every failing candidate was
blocked by its promotion gate, and no unvalidated model ever reached the served
predictor.

## Layout

```
src/
  modelenv/                 Rust gRPC market environment (replay + live cTrader back-ends)
  ta/                       technical-indicator library (Rust + Python)
  deepqnetwork/             Double-DQN agent, KFP training/backtest pipeline, model registry
  probabilisticforecaster/  causal Transformer, distributed training, KServe predictor
  dqnpf/                    integration layer, profitability gate, combined predictor
  marketdata/               cTrader/news ingestion, aggregation, S3 snapshots
  commons/                  shared configuration and credential handling
  feature-prototype/        the isolated integration-layer prototype + its Telegram advisor
tools/                      export_public.sh, scan_secrets.sh
```

## Where to start reading

| if you want to see… | read |
|---|---|
| the environment's reward and observation | `src/modelenv/core/src/environment.rs` |
| the trading action, netting and fill model | `src/modelenv/core/src/environment.rs`, `position.rs` |
| the Double-DQN learning step | `src/deepqnetwork/agent.py` |
| the forecaster's heads and NLL loss | `src/probabilisticforecaster/model.py`, `training.py` |
| **the project's novel component** | `src/dqnpf/integration.py`, `screen()` and `_recompute_gate()` |
| how promotion is gated | `src/deepqnetwork/kubeflow/components/dqn_backtest/component.py` |
| the user-facing advisor | `src/feature-prototype/telegram_advisor_bot.py`, `llm.py` |

## Running the code

Verified from a clean clone on macOS with Python 3.11 and Rust 1.91.

**Rust; the market environment.** No setup beyond a toolchain:

```bash
cd src/modelenv && cargo test --workspace     # 500 tests pass
```

**Python.** The gRPC stubs are build output and are not committed, so generate
them once, then install whichever component you want to run:

```bash
./tools/gen_protos.sh                          # -> src/environment_pb2*.py
pip install -r src/deepqnetwork/requirements.txt
pip install -r src/dqnpf/requirements.txt
cd src && python -m pytest deepqnetwork dqnpf -q
```

Everything is imported from `src/`, which is the package root.

**What runs standalone:** the environment, the agent and network, the forecaster
model and its training/evaluation code, the integration layer and its
profitability gate, the backtest metric and threshold helpers, and the feature
prototype, 141 of 191 modules import with no external services.

**What does not:** the Airflow DAGs (`*/dags/`, `*/airflow/`) are definitions
deployed *into* an Airflow instance rather than pip-installed, so they need one
to import; the KServe predictors and registry clients need `kserve` and
`model-registry`, pinned in the per-component requirements under
`kubeflow/serving/` and `kubeflow/base/`; and anything that resolves a
checkpoint, reads a market-data snapshot or places an order needs the S3 bucket,
the model registry and the broker account, none of which ship here.

**Known test failures**, 5 of 523, all pre-existing and unrelated to the
code discussed in the report. Four in `deepqnetwork` are stale expectations in
tests that were never updated when a hyperparameter or CLI flag changed
(`gamma`, `epsilon_end`, `--num-episodes-per-range`); one in `dqnpf` asserts a
pipeline precondition against the Model Registry, which does not ship here.
On macOS the Rust *doctests* additionally need a working Xcode SDK; the unit
and integration tests do not.

## Notes on this repository

This is a **curated export** of a larger private working repository, produced by
`tools/export_public.sh`. Build trees, downloaded market-data caches and all credential
files are excluded; `tools/scan_secrets.sh` runs afterwards and fails the export if
anything credential-shaped survives.

The git history is the *real* development history of the exported paths, from
**1 April 2026** onward, brought across with `git-filter-repo` and re-rooted into the
layout above, so authorship, dates and messages are the original ones rather than a
squashed import. The feature prototype appears in June and is developed forward from
there.

The system runs against a private Kubernetes cluster and a broker account, so the
deployment manifests here are illustrative rather than directly runnable by a third
party.

## Licence

MIT, see [LICENSE](LICENSE).
