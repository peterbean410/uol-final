# Feature prototype, DQN→PF integration layer & profitability gate

This prototype isolates and exercises the project's most distinctive technical
feature: the **integration layer** that screens a DQN trading agent's actions
using a probabilistic forecaster's uncertainty (σ) plus risk budgets, and the
**profitability gate** that disengages the screen automatically when it stops
earning its keep.

It reuses the **production code unchanged**,
`tradingmodel.intraday.dqnpf.integration.IntegrationLayer` (screen + gate),
the `tradingmodel.intraday.dqnpf.backtest` pure helpers (`compare_results`,
`validate_thresholds`, `forecaster_position`, `StepRecord`), the action mapper and
config, and the production `deepqnetwork.network.QNetwork` architecture for the
agent. Only the *replay loop* (`engine.py`) and the compact DQN trainer
(`dqn.py`) are new; they stand in for the Rust `modelenv` gRPC environment so the
whole thing runs on a laptop without the K8s/gRPC/S3 stack.

## Layout
- `dukascopy.py`, **download** stage: fetches real USD/JPY ticks from Dukascopy (free, no auth; stdlib only), per-hour disk cache.
- `data.py`, **aggregate** stage: turns ticks into M5 OHLCV bars (`aggregate_m5`); loader resolves cache → Dukascopy download+aggregate → synthetic fallback.
- `signals.py`, two inference-only forecaster σ regimes: `informative` (EWMA-vol σ) and `collapsed` (constant σ, reproducing the production forecaster's failure mode). The real causal-Transformer forecaster is a production component (Ch3), not reproduced here.
- `policy.py`; an **inference-only** trading policy (a fixed momentum/mean-reversion rule, no training) + a lightweight replay env that stands in for the Rust `modelenv`. The real DQN is a production component (Ch3).
- `engine.py`, drives 3 arms (combined / DQN-only / forecaster-only) through the **real** `IntegrationLayer` + gate, scored by the **real** `compare_results`/`validate_thresholds`.
- `run.py`, orchestrates download+aggregate and the three configurations; writes metrics + figures. No training (pure inference), so it runs in seconds.

## Run
From the **forex repo root** (so the production packages import):

```bash
python finalreport/preliminaryreport/prototype/run.py
```

The first run downloads ~500 hourly tick files (~2.5M real ticks) and aggregates
them to ~4,300 M5 bars; both the per-hour `.bi5` files and the aggregated parquet
are cached, so subsequent runs are instant. There is no training, so the run
finishes in seconds. Pass `--no-network` to force the offline synthetic fallback.

Outputs:
- `preliminaryreport/figures/fig1..5_*.png`, price/σ, equity curves, suppression-by-σ, gate timeline, σ-calibration.
- `preliminaryreport/prototype/results/metrics.json`, full metrics + Req-14 gate verdicts per configuration.

## Telegram advisor bot (`telegram_advisor_bot.py`)

A non-technical interface (Template 4.2): message the bot and get the
recommendation, computed through the **real** `IntegrationLayer` screen + gate.
Uses Telegram long-polling (`getUpdates` to receive, `sendMessage` to send).

```bash
cp .env.example .env            # then put your BotFather token in .env
python telegram_advisor_bot.py            # start the polling bot
python telegram_advisor_bot.py --dry-run  # print one advice to stdout (no Telegram)
python telegram_advisor_bot.py --check    # verify the token (getMe) and exit
```

In Telegram, send `/advice` to get e.g.:

```
USD/JPY advice, bar 2024-01-28 23:55 UTC
Recommendation: HOLD
Forecast: mu=+0.30 bp, sigma=2.16 bp (normal regime)
Screen: pass | profitability gate: active
Last price: 148.184
```

The token is read from env / a gitignored `.env` (never committed). Set
`TELEGRAM_ALLOWED_CHAT_IDS` to restrict who may use the bot.

## The three configurations
- **C1 informative + gate**, feasibility: an informative σ gives the screen something real to act on.
- **C2 collapsed + gate**, honest failure: a constant (collapsed) σ degenerates the screen into a global on/off switch; the profitability gate detects this and self-deactivates.
- **C3 collapsed + gate OFF**; the control: with no gate, the collapsed screen runs unchecked and makes the combined system strictly worse than the DQN-only baseline.

## Reproducibility
All randomness is seeded (`--seed`, default 0). The data window is fixed
(`2024-01-08 .. 2024-01-29`, ~3 weeks of real USD/JPY), downloaded from Dukascopy
and aggregated to M5, then cached to `cache/usdjpy_m5.parquet`; the raw hourly
ticks are cached under `cache/dukascopy/`. The data source actually used is
recorded in `metrics.json` under `meta.data_source` (e.g. `dukascopy:2024-01-08..2024-01-29`).
A seeded synthetic fallback keeps the prototype runnable fully offline. The real
system's historical-data OOS verdicts (collapsed forecaster; 2016 ~100% trade
suppression) are the additional real-market corroboration discussed in the report.
