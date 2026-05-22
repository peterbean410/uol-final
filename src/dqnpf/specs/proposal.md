# Proposal: Intraday Trading Agent with DQN + ProbabilisticForecaster

## Summary

Build an intraday trading agent that combines:

- **DQN** for discrete action selection and policy learning
- **ProbabilisticForecaster** for forward-return mean/variance estimates and separate risk screening
- **modelenv** (existing Rust gRPC service) as the single source of truth for market data, state, and execution

Both individual models are already implemented and trained. modelenv already serves the 53-feature state vector the DQN consumes, and already loads M5 OHLC bars that can feed the forecaster's feature engine. This proposal covers the integration layer: wiring the two models together via modelenv, including the execution/risk layer, system flow, and deployment.

## Problem

Both the DQN trading agent and the ProbabilisticForecaster are already implemented and trained independently. modelenv already provides:

- `Reset()` / `Step()`, episode lifecycle with 53-feature state vectors, rewards, and done signals
- `RecentBars(symbol)`, completed 5-min OHLC bars usable by the forecaster's feature engine
- Training and live modes, bar loading from S3 parquet, broker connectivity, position reconciliation

The challenge is combining DQN and the forecaster into a coherent intraday trading system where:

1. **DQN** selects discrete trading actions from the modelenv observation state (unchanged)
2. **ProbabilisticForecaster** consumes completed M5 bars from modelenv and provides return distribution estimates that screen execution
3. The integration layer orchestrates both, using modelenv as the data hub

## Proposed Approach

Wire the two models together through an integration layer that sits between the DQN output and the `Step()` call:

1. **modelenv** (existing)
   - Provides the 53-feature state vector via `Reset()` / `Step()` for DQN consumption
   - Provides completed M5 OHLC bars via `RecentBars(symbol)` for forecaster consumption
   - Executes trades via `Step(action)` in training mode, or broker gateway in live mode
   - step_size_seconds set to 60 (decision frequency). The DQN must be retrained at this step size; the existing checkpoint was trained at 5s, and the Q-values encode state-transition dynamics at that interval. Running at 60s with a 5s-trained checkpoint would feed the network out-of-distribution states.

2. **ProbabilisticForecaster** (existing, pre-trained)
   - Produces predicted return mean, variance, and confidence-derived signals
   - Runs in parallel with DQN, not inside the observation state
   - Consumes completed M5 bars from `RecentBars`, runs its own feature engine to produce the (36, 16) input tensor
   - Flags low-confidence periods where the execution layer may reduce activity

3. **DQN agent** (existing, pre-trained)
   - Selects discrete trading actions from the modelenv observation state only
   - Observation state unchanged, forecaster outputs are not injected

4. **Integration / Execution Layer** (new work)
   - Receives the DQN action recommendation
   - Queries modelenv `RecentBars(symbol)` to obtain completed 5-min OHLC bars
   - Runs the forecaster's feature engine and `predict()` to obtain (mu, sigma)
   - Applies risk rules to decide whether to execute, reduce size, or skip
   - Submits the screened action to modelenv via `Step()`

## System Flow

```
modelenv ──Reset()/Step()──→ 53-feature state ──→ DQN ──→ action
    │                                                       │
    └──RecentBars(M5)──→ forecaster feature engine ──→ (mu, sigma)
                                                            │
                                                    integration layer
                                                    (screen action)
                                                            │
                                                    Step(screened_action)
```

1. Call `Reset(symbol, episode_start_ts, episode_end_ts, step_size_seconds=60)` to initialise the episode.
2. Populate the forecaster's historical window (1440 bars) from the same parquet data modelenv loads at reset. The forecaster is warm from the first decision.
3. At each timestep:
   a. Build the 53-feature state from the current Observation (DQN input, unchanged).
   b. Call `RecentBars(symbol)` and extract the `"M5"` key from the returned `map<string, BarList>` to get completed 5-min OHLC bars.
   c. Run the forecaster's `compute_features()` on the M5 bar series → `(36, 16)` tensor.
   d. Call `ForecasterInference.predict(features)` → `(mu, sigma)`.
   e. Call `DQNAdvisor.recommend_action(state)` → `ActionResult`.
   f. The integration layer screens the action using (mu, sigma).
   g. Call `Step(screened_action)` and receive the next Observation.
4. Evaluate the combined strategy with walk-forward backtesting.

## Integration Layer Design

The integration layer is the primary new work. It sits between the DQN output and the `Step()` call.

### Interface

```python
@dataclass
class IntegrationConfig:
    variance_threshold: float = 4.5     # sigma above which positions count against risk budget (bps)
    max_risk_long_units: int = 2        # max long units allowed when sigma > threshold
    max_risk_short_units: int = 1       # max short units allowed when sigma > threshold
    directional_disagreement: bool = False  # whether to suppress when sign(mu) != sign(dqn_direction)
    directional_tolerance: float = 1.0  # mu magnitude (bps) below which directional check is skipped
    forecast_horizon: int = 1           # 1, 3, 6, or 12 bars (5/15/30/60 min)
    min_bars_warmup: int = 1440         # bars needed before forecaster is valid

class IntegrationLayer:
    def __init__(self, dqn: DQNAdvisor, forecaster: ForecasterInference,
                 config: IntegrationConfig): ...
        self._risk_long_units: int = 0   # long units opened while sigma > threshold
        self._risk_short_units: int = 0  # short units opened while sigma > threshold

    def screen(self, dqn_action: ActionResult, mu: float, sigma: float) -> ScreenedAction:
        """Apply risk rules and return the screened action.

        Only positions opened when sigma > variance_threshold consume
        the risk budget. Positions opened in low-variance regimes
        (sigma <= threshold) pass through without counting.
        """
        ...

    def on_position_closed(self, side: str, units: int) -> None:
        """Release risk budget when a high-sigma position is closed."""
        ...

@dataclass
class ScreenedAction:
    action: int           # original or overridden action index
    action_name: str
    screened: bool        # True if the action was modified by a risk rule
    reason: str           # "budget_exhausted", "directional_conflict", "pass"
    sigma: float          # sigma at decision time (caller can see regime)
    risk_long_used: int   # cumulative risk-budget long units after this action
    risk_short_used: int  # cumulative risk-budget short units after this action
```

### Risk Rules

Rules are evaluated in order. The first rule that triggers determines the screened action.

1. **Risk budget**: If sigma > `variance_threshold` AND the DQN action is BUY_1/BUY_2 AND `risk_long_units + action_size > max_risk_long_units` → override to HOLD (high-sigma long budget exhausted). Same for SELL_1/SELL_2 against `max_risk_short_units`. Positions opened when sigma <= threshold do NOT consume budget and are never blocked by this rule.
2. **Directional conflict**: If `directional_disagreement` is enabled AND `abs(mu) > directional_tolerance` AND `sign(mu) != sign(dqn_action_direction)` → override to HOLD.
3. **Pass-through**: Otherwise → execute the DQN action as-is. If sigma > threshold at the time the position is opened, increment the corresponding risk budget counter.

There is no blanket variance suppression; the forecaster's sigma gates exposure via the risk budget, not via a hard block. In low-variance regimes the DQN trades freely. In high-variance regimes the DQN can still trade, but cumulative exposure is capped at `max_risk_long_units` / `max_risk_short_units`. Budgets only reset when positions are closed (via `on_position_closed`).

Budgets are denominated in DQN action units. The integration layer maps action indices to risk units explicitly:

| ActionType | Index | Direction | Risk Units |
|------------|-------|-----------|-------------|
| HOLD       | 0     |, | 0           |
| BUY_1      | 1     | long      | 1           |
| BUY_2      | 2     | long      | 2           |
| SELL_1     | 3     | short     | 1           |
| SELL_2     | 4     | short     | 2           |

The modelenv position sizing maps these units to actual notional amounts. The raw action index is never used directly as a unit count; the mapping is explicit.

### Feature Bridge

The integration layer calls modelenv's `RecentBars(symbol)`, which returns a `map<string, BarList>` keyed by interval string. The layer extracts the `"M5"` key to obtain completed 5-min OHLC bars, then runs the forecaster's `compute_features()` to produce the (36, 16) input tensor. This is the same feature engine used during forecaster training; no duplication, just reuse.

modelenv already loads M5 bars from S3 parquet in training mode and from the broker in live mode. `RecentBars` returns the most recent completed bars across all configured intervals (not the forming bar at the cursor position), which matches what the forecaster was trained on.

**Signal caching**: A new completed M5 bar arrives every 5 minutes. At 60s step size, 4 out of 5 timesteps see the same bar set. The integration layer caches the forecaster's `(mu, sigma)` and only recomputes when `RecentBars` returns a bar with a new timestamp. This avoids redundant gRPC calls, feature computation, and forward passes on ~80% of steps. The variance screen applies the same (mu, sigma) across all decisions within a 5-min bar; the DQN can change actions between steps, but the forecaster's opinion stays fixed until the next completed bar.

### Cold-Start / Warm-Up

The forecaster requires 1440 bars (~5 days) of history for z-score/volatility normalisation, plus 36 bars (~3 hours) for the lookback window. During initialisation:

1. The integration layer reads the M5 bar series from the same data source modelenv loaded at `Reset()` time (S3 parquet in training mode, broker bar cache in live mode).
2. It runs `compute_features()` on the full history, producing valid z-scores from the first decision.
3. If fewer than 1440 bars are available, the integration layer raises an error (consistent with the forecaster's requirement 2.4).

No unscreened fallback period is needed; the forecaster is warm from t=0.

## Deployment Plan

The deployed agent exposes a single decision interface backed by modelenv:

```
IntegrationAgent
├── DQNAdvisor (stateless, loaded from checkpoint)
├── ForecasterInference (stateless, loaded from checkpoint)
├── IntegrationLayer (screening logic + risk budget tracking)
└── EnvironmentClient (gRPC to modelenv)
```

At runtime, in both training and live modes, modelenv is the single source of truth for all market data. The integration agent:

- Gets state from `Reset()` / `Step()` observations → DQN
- Gets M5 bars from `RecentBars(symbol)` → forecaster feature engine
- Screens DQN actions via the integration layer
- Submits screened actions via `Step()`

This means the integration agent works identically in training and live modes; the mode switch is handled entirely by modelenv.

## Backtesting Methodology

The combined system must be evaluated without leaking the forecaster's training data into the DQN's test episodes.

### Data Partitioning

The forecaster was trained on 2012-2022 and tested on 2023-2026-04-30. The DQN trains on modelenv episodes drawn from configurable date ranges. For a clean walk-forward evaluation:

1. **Train the DQN** (at step_size_seconds=60) on episodes drawn from 2012-2022 only; no overlap with the forecaster's test set.
2. **Train the forecaster** on 2012-2022, matching its existing split.
3. **Tune integration thresholds** on a validation period entirely within 2012-2022 (e.g., the last 12 months).
4. **Evaluate the combined system** on 2023-01-01 to 2026-04-30, unseen by both models.

This ensures neither model has seen the evaluation data. The integration thresholds must be frozen before evaluation begins; no threshold adjustment based on evaluation-period results.

### Metrics

Compare the combined system against the DQN-only baseline on the same episodes:

- Total return and Sharpe ratio
- Proportion of DQN actions suppressed by the integration layer (should be concentrated in high-uncertainty regimes)
- Conditional performance: DQN-only PnL on steps where the forecaster's sigma was above threshold (these are the steps the integration layer would have skipped)
- Number of trades per episode (combined system should have fewer)

## Testing Strategy

### Property-Based Tests

| Property | Description |
|----------|-------------|
| Screened action validity | For any valid DQN action and any (mu, sigma) where sigma > 0, the screened action index is in [0, 4] |
| Low-sigma pass-through | For any valid action, sigma <= variance_threshold, and directional_disagreement disabled; the screened action equals the input action and risk budget is NOT incremented |
| Risk budget only counts high-sigma positions | risk_long_units and risk_short_units only increment when a position is opened with sigma > variance_threshold |
| Risk budget never exceeded | For any sequence of actions, cumulative risk_long_units never exceeds max_risk_long_units and risk_short_units never exceeds max_risk_short_units |
| Risk budget exhaustion | When sigma > threshold and risk_long_units == max_risk_long_units, any BUY_1/BUY_2 is screened to HOLD with reason "budget_exhausted". Same for short side |
| Risk budget release | After on_position_closed("buy", n), risk_long_units decreases by n and previously blocked buy actions pass through again (assuming sigma still > threshold) |
| Sigma edge cases | sigma=0 produces a valid screened action (not NaN, not a crash) |
| Signal cache validity | Cached (mu, sigma) is only replaced when the latest completed bar timestamp advances |
| Directional conflict rule | When enabled, sign(mu) != sign(action_direction) and abs(mu) > tolerance → HOLD |

### Unit Tests

- IntegrationConfig validation (thresholds non-negative, budgets non-negative, forecast_horizon in {1, 3, 6, 12})
- ScreenedAction dataclass construction, immutability, and sigma field
- Budget tracking: only increments when sigma > threshold; BUY_2 increments risk_long_units by 2, SELL_1 increments risk_short_units by 1
- Low-sigma actions do not consume budget: sigma=3.0 with threshold=4.5, BUY_2 passes through and risk_long_units unchanged
- Feature bridge: verify compute_features output shape is (N, 16) given mock M5 OHLC DataFrame
- Signal cache: verify recompute triggers on new bar timestamp, skip on same timestamp

### Integration Tests (with mock modelenv)

- End-to-end: Reset → state → DQN action + RecentBars → forecaster signal → screen → Step
- Warm-up: verify forecaster is ready after 1440+ bars loaded
- Insufficient history: verify error raised when < 1440 bars available
- Cached signal: verify 4 out of 5 steps at 60s use the cached (mu, sigma)

## Open Design Choices

- **DQN retraining**: The DQN must be retrained at `step_size_seconds=60`. The existing checkpoint (trained at 5s) encodes state-transition dynamics at 5-second granularity; running it at 60s would feed out-of-distribution states. Retraining at 60s also lets the DQN learn reward structure that aligns with the integration layer's decision cadence.
- **Forecaster horizon vs. DQN reward horizon**: At `forecast_horizon=1` (5-min), the forecaster predicts the return over the next 5 minutes. But the DQN's reward at `Step()` is the delta in portfolio value over the last 60 seconds. The forecaster screens a 5-min outlook against a 60s reward signal. If the DQN finds a profitable 60s action that the forecaster's 5-min view dislikes, the screen blocks it unnecessarily. The mitigation is to keep variance_threshold loose enough that only genuinely noisy regimes (not direction disagreements) trigger suppression. If this trade-off proves problematic, the alternative is to run the DQN at `step_size_seconds=300` (5-min) so both models operate on the same horizon, at the cost of lower decision frequency.
- **variance_threshold**: default 4.5 bps of return. Sigma above which opened positions count against the risk budget. Does NOT suppress trading outright; the DQN can still act during high-variance regimes, but cumulative exposure is capped by `max_risk_long_units` / `max_risk_short_units`. Tune via walk-forward validation.
- **directional_disagreement**: whether sign conflict between DQN and forecaster triggers suppression. Start with this disabled, variance suppression alone is the simpler, safer first step. Directional disagreement can be added later if variance screening proves insufficient.
- **transaction cost model**: for backtesting the combined system. The forecaster's backtest assumes frictionless execution; at 60s steps on major FX pairs this remains defensible, but spread should be accounted for in live mode
- **per-symbol vs shared integration**: one integration layer per symbol (USDJPY, AUDJPY) or a common one

## Success Criteria

The integration is successful if the combined system:

- improves risk-adjusted returns over the DQN-only baseline
- reduces trading in high-uncertainty regimes identified by the forecaster
- maintains stable intraday performance across walk-forward periods

## Recommendation

Start with a **separate-signal** design using modelenv as the data hub:

- modelenv serves state to DQN and M5 bars to the forecaster
- DQN and forecaster remain independent; forecaster outputs are not injected into DQN state
- lightweight integration layer screens DQN actions using the forecaster's variance signal
- all data flows through modelenv; no separate data pipelines needed at runtime
- step_size_seconds = 60 (aligns with 5-min bar cadence)
- forecast_horizon = 1 (5-min, natural match for 60s decision steps)
- conservative thresholds validated through walk-forward backtesting

This is the safest path because it keeps DQN and forecaster independent, reuses modelenv's existing data loading and execution infrastructure, makes the screening logic easy to inspect and tune, and keeps rollback simple. The DQN must be retrained at step_size_seconds=60, but no architectural changes to either model are needed.
