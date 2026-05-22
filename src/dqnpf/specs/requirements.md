# Requirements Document: Intraday Trading Agent (DQN + ProbabilisticForecaster)

## Introduction

This document specifies requirements for an intraday trading agent that combines a Deep Q-Network (DQN) action-selection model with a Transformer-based ProbabilisticForecaster for risk screening. Both models are already implemented and trained independently. The new work is an integration layer that wires them together via the existing modelenv gRPC service, which serves as the single source of truth for market data, state, and execution.

The integration layer sits between the DQN output and the modelenv `Step()` call. It screens DQN actions using the forecaster's return distribution estimates (mu, sigma), applies risk budget constraints that only count positions opened during high-uncertainty regimes, and submits the screened action to modelenv.

## Glossary

- **Integration_Layer**: The new Python component that screens DQN actions using forecaster signals and risk budgets
- **DQN_Advisor**: The existing `DQNAdvisor` class (`deepqnetwork/advisor.py`), stateless action recommendation from a trained Q-Network
- **Forecaster_Inference**: The existing `ForecasterInference` class (`probabilisticforecaster/inference.py`), produces (mu, sigma) predictions from a (36, 16) feature tensor
- **modelenv**: The existing Rust gRPC service providing `Reset()`, `Step()`, `RecentBars()`, and other RPCs
- **Feature_Engine**: The existing `compute_features()` function (`probabilisticforecaster/features.py`), computes 16 features from 5-min OHLC bars
- **Risk_Budget**: A cumulative cap on exposure opened during high-uncertainty regimes (sigma > variance_threshold)
- **sigma**: The forecaster's predicted standard deviation of forward return, in basis points
- **mu**: The forecaster's predicted mean forward return, in basis points
- **Action_Space**: The DQN's discrete 5-action set: HOLD(0), BUY_1(1), BUY_2(2), SELL_1(3), SELL_2(4)
- **variance_threshold**: The sigma value above which opened positions consume the risk budget (default 4.5 bps)

## Requirements

### Requirement 1: Integration Layer, Core Screening

**User Story:** As a trading system operator, I want the integration layer to screen DQN actions against forecaster signals and risk budgets, so that the combined system trades with uncertainty-aware exposure control.

#### Acceptance Criteria

1. THE Integration_Layer SHALL accept a DQN action recommendation (ActionResult), a forecaster prediction (mu, sigma as floats), and an IntegrationConfig, and return a ScreenedAction
2. THE Integration_Layer SHALL evaluate risk rules in a fixed priority order: risk budget first, then directional conflict, then pass-through
3. THE Integration_Layer SHALL return a ScreenedAction containing the final action index, action name, whether screening modified the action, the reason string, the sigma value at decision time, and the current risk budget utilisation
4. THE Integration_Layer SHALL be stateless except for tracking cumulative risk budget counters (`_risk_long_units`, `_risk_short_units`)
5. WHEN the screened action is HOLD and the reason is not "pass", THE screened flag SHALL be set to true

### Requirement 2: Action-to-Unit Mapping

**User Story:** As a developer, I want the integration layer to explicitly map DQN action indices to risk units, so that the raw integer values are never used directly as unit counts.

#### Acceptance Criteria

1. THE Integration_Layer SHALL map action index 0 (HOLD) to 0 risk units on either side
2. THE Integration_Layer SHALL map action index 1 (BUY_1) to 1 long risk unit
3. THE Integration_Layer SHALL map action index 2 (BUY_2) to 2 long risk units
4. THE Integration_Layer SHALL map action index 3 (SELL_1) to 1 short risk unit
5. THE Integration_Layer SHALL map action index 4 (SELL_2) to 2 short risk units
6. THE Integration_Layer SHALL raise an error if an action index outside [0, 4] is received

### Requirement 3: Risk Budget, High-Sigma Gating

**User Story:** As a risk manager, I want the risk budget to only constrain positions opened during high-uncertainty regimes, so that the DQN trades freely in low-variance conditions while cumulative exposure is capped when the forecaster signals elevated uncertainty.

#### Acceptance Criteria

1. THE Integration_Layer SHALL track two budget counters: `risk_long_units` and `risk_short_units`, both initialised to zero at construction
2. WHEN sigma > variance_threshold AND the DQN action is BUY_1 or BUY_2, AND `risk_long_units + action_units > max_risk_long_units`, THE Integration_Layer SHALL override the action to HOLD with reason "budget_exhausted"
3. WHEN sigma > variance_threshold AND the DQN action is SELL_1 or SELL_2, AND `risk_short_units + action_units > max_risk_short_units`, THE Integration_Layer SHALL override the action to HOLD with reason "budget_exhausted"
4. WHEN sigma > variance_threshold AND the DQN action is BUY_1/BUY_2/SELL_1/SELL_2 AND the corresponding budget is NOT exhausted, THE Integration_Layer SHALL pass the action through and increment the corresponding budget counter by the action's risk units
5. WHEN sigma <= variance_threshold, THE Integration_Layer SHALL NOT increment either budget counter regardless of the DQN action
6. WHEN sigma <= variance_threshold AND the directional conflict rule does not trigger, THE Integration_Layer SHALL pass the DQN action through unchanged with reason "pass"
7. THE Integration_Layer SHALL expose an `on_position_closed(side: str, units: int)` method that decrements the corresponding budget counter (never below zero)

### Requirement 4: Directional Conflict Screening

**User Story:** As a trading system operator, I want optional suppression of DQN actions when the forecaster's directional opinion contradicts the DQN with sufficient conviction, so that I can experiment with signal alignment constraints.

#### Acceptance Criteria

1. THE Integration_Layer SHALL support a configurable `directional_disagreement` flag (default: false)
2. WHERE `directional_disagreement` is enabled, AND `abs(mu) > directional_tolerance` (default: 1.0 bps), AND `sign(mu) != sign(dqn_action_direction)`, THE Integration_Layer SHALL override the action to HOLD with reason "directional_conflict"
3. WHERE `directional_disagreement` is enabled AND `abs(mu) <= directional_tolerance`, THE directional conflict rule SHALL be skipped regardless of sign disagreement
4. WHERE `directional_disagreement` is disabled, THE directional conflict rule SHALL be skipped entirely

### Requirement 5: Variance Threshold Configuration

**User Story:** As a quantitative researcher, I want the variance threshold to be configurable, so that I can tune the boundary between low-variance and high-variance regimes through backtesting.

#### Acceptance Criteria

1. THE `variance_threshold` SHALL default to 4.5 (basis points of return)
2. THE `variance_threshold` SHALL be configurable via IntegrationConfig at construction time
3. THE `variance_threshold` SHALL be non-negative
4. THE Integration_Layer SHALL use the same threshold value for both long and short budget gating

### Requirement 6: Forecaster Signal via modelenv RecentBars

**User Story:** As a system integrator, I want the integration layer to obtain completed 5-min OHLC bars from modelenv and feed them through the forecaster's feature engine, so that the forecaster produces predictions from the same data source used at training time.

#### Acceptance Criteria

1. THE Integration_Layer SHALL call modelenv's `RecentBars(symbol)` to obtain completed bars for all configured intervals
2. THE Integration_Layer SHALL extract the `"M5"` key from the returned `map<string, BarList>` to obtain the 5-min OHLC bar series
3. THE Integration_Layer SHALL run `compute_features()` from `probabilisticforecaster.features` on the M5 bar series to produce a (N, 16) feature DataFrame
4. THE Integration_Layer SHALL take the most recent 36 rows (lookback window) and convert them to a (36, 16) float32 tensor
5. THE Integration_Layer SHALL pass the tensor to `ForecasterInference.predict()` to obtain (mu, sigma)
6. IF the M5 key is missing from the RecentBarsResponse, THEN THE Integration_Layer SHALL raise an error
7. IF the M5 bar series has fewer than 36 bars, THEN THE Integration_Layer SHALL raise an error

### Requirement 7: Signal Caching

**User Story:** As a performance-conscious developer, I want the forecaster signal to be cached between M5 bar completions, so that redundant gRPC calls and forward passes are avoided on ~80% of timesteps.

#### Acceptance Criteria

1. THE Integration_Layer SHALL cache the most recent (mu, sigma) tuple and the timestamp of the latest completed M5 bar used to compute it
2. WHEN `RecentBars(symbol)` returns a bar series whose latest bar timestamp equals the cached timestamp, THE Integration_Layer SHALL reuse the cached (mu, sigma) without recomputing
3. WHEN `RecentBars(symbol)` returns a bar series whose latest bar timestamp is newer than the cached timestamp, THE Integration_Layer SHALL recompute (mu, sigma) and update the cache
4. THE Integration_Layer SHALL initialise the cache as empty and recompute on the first call

### Requirement 8: Cold-Start / Warm-Up

**User Story:** As a trading system operator, I want the forecaster to be fully warmed up before the first trading decision, so that z-score normalisation is valid from the start and no unscreened fallback period occurs.

#### Acceptance Criteria

1. WHEN the integration layer is initialised, THE Integration_Layer SHALL load at least 1440 M5 bars of history from the same data source modelenv uses at Reset() time
2. THE Integration_Layer SHALL run `compute_features()` on the full history to establish valid rolling z-score and volatility normalisation windows
3. IF fewer than 1440 bars are available, THEN THE Integration_Layer SHALL raise an error and refuse to operate
4. THE Integration_Layer SHALL be ready to produce screened actions from the first timestep after Reset()

### Requirement 9: DQN Retraining at step_size_seconds=60

**User Story:** As a quantitative researcher, I want the DQN to be trained at the same step interval used by the integration layer, so that the Q-network learns state-transition dynamics at the correct temporal granularity.

#### Acceptance Criteria

1. THE DQN SHALL be trained with `step_size_seconds = 60` in the ResetRequest
2. THE existing DQN checkpoint trained at 5s SHALL NOT be used with the integration layer
3. WHERE the DQN is in live mode loaded from a 60s-trained checkpoint, THE DQN SHALL use the same step_size_seconds for consistency with training

### Requirement 10: modelenv Interaction Contract

**User Story:** As a system integrator, I want the integration layer to interact with modelenv through a well-defined gRPC contract, so that the combined system works identically in training and live modes.

#### Acceptance Criteria

1. THE Integration_Layer SHALL call `Reset(symbol, episode_start_ts, episode_end_ts, step_size_seconds=60)` to begin each episode
2. THE Integration_Layer SHALL call `Step(screened_action)` to submit the screened action and receive the next Observation
3. THE Integration_Layer SHALL call `RecentBars(symbol)` to obtain completed bars for forecaster feature computation
4. THE Integration_Layer SHALL operate identically in training and live modes; the mode switch is handled by modelenv
5. THE Integration_Layer SHALL NOT call any gRPC methods on modelenv beyond Reset, Step, and RecentBars
6. THE Integration_Layer SHALL be bound to a single symbol for its lifetime; the symbol is set at construction and cannot be changed
7. THE risk budget counters (`_risk_long_units`, `_risk_short_units`) SHALL track exposure for the bound symbol only
8. WHERE multiple symbols are traded, each symbol SHALL have its own Integration_Layer instance with independent configuration and budget state

### Requirement 11: Configuration Management

**User Story:** As a quantitative researcher, I want all integration layer parameters to be configurable, so that I can tune thresholds and budgets through walk-forward validation.

#### Acceptance Criteria

1. THE IntegrationConfig SHALL contain the following fields with their default values:
   - `variance_threshold`: float = 4.5
   - `max_risk_long_units`: int = 2
   - `max_risk_short_units`: int = 1
   - `directional_disagreement`: bool = False
   - `directional_tolerance`: float = 1.0
   - `forecast_horizon`: int = 1
   - `min_bars_warmup`: int = 1440
2. THE Integration_Layer SHALL validate that `variance_threshold >= 0`, `max_risk_long_units >= 0`, `max_risk_short_units >= 0`, `directional_tolerance >= 0`, and `forecast_horizon` is one of {1, 3, 6, 12}
3. THE Integration_Layer SHALL log the full resolved configuration at initialisation
4. THE Integration_Layer SHALL support loading configuration from a YAML file with CLI overrides

### Requirement 12: Backtesting Methodology, Data Partitioning

**User Story:** As a quantitative researcher, I want a clean data partitioning scheme, so that the combined system is evaluated on data unseen by either model during training.

#### Acceptance Criteria

1. THE DQN SHALL be trained on episodes drawn from 2012-01-01 to 2022-12-31 only
2. THE ProbabilisticForecaster SHALL be trained on 2012-01-01 to 2022-12-31, matching its existing split
3. THE integration thresholds SHALL be tuned on a validation period entirely within 2012-2022 (the last 12 months)
4. THE combined system SHALL be evaluated on 2023-01-01 to 2026-04-30, unseen by both models
5. THE integration thresholds SHALL be frozen before evaluation begins; no threshold adjustment based on evaluation-period results

### Requirement 13: Backtesting Metrics

**User Story:** As a quantitative researcher, I want standardised metrics comparing the combined system against the DQN-only baseline, so that I can objectively assess the integration layer's contribution.

#### Acceptance Criteria

1. THE evaluation SHALL compute total return and Sharpe ratio for both the combined system and DQN-only baseline on identical episodes
2. THE evaluation SHALL compute the proportion of DQN actions suppressed by the integration layer, grouped by reason (budget_exhausted, directional_conflict)
3. THE evaluation SHALL compute conditional performance: DQN-only PnL on steps where the forecaster's sigma exceeded the variance threshold
4. THE evaluation SHALL compute the number of trades per episode for both systems
5. THE evaluation SHALL compute the proportion of time spent in high-sigma regimes and the average budget utilisation during those periods

### Requirement 14: Comparative Performance Thresholds

**User Story:** As a quantitative researcher, I want the combined system to demonstrate measurable improvement over the DQN-only baseline, so that the integration layer is proven to add value beyond either model alone.

#### Acceptance Criteria

1. THE combined system SHALL achieve a higher Sharpe ratio than the DQN-only baseline when evaluated on the same out-of-sample episodes (2023-01-01 to 2026-04-30)
2. THE combined system SHALL execute fewer trades per episode than the DQN-only baseline, and the reduction SHALL be concentrated in steps where sigma > variance_threshold
3. THE combined system SHALL have a lower proportion of negative-PnL steps during high-sigma regimes compared to the DQN-only baseline
4. THE combined system SHALL NOT degrade DQN-only performance during low-sigma regimes (sigma <= variance_threshold) by more than a configurable tolerance (default: 5% of baseline PnL in those steps)
5. THE combined system SHALL demonstrate stable performance across walk-forward periods; no single calendar quarter in the evaluation period SHALL account for more than 50% of total system PnL
6. IF the combined system fails to meet any of AC 1–4, THEN the integration layer configuration SHALL be treated as invalid and require threshold retuning

### Requirement 16: Project Structure

**User Story:** As a developer, I want the integration layer to reside in a well-defined directory, so that it is discoverable and maintainable.

#### Acceptance Criteria

1. THE integration layer project SHALL reside in `tradingmodel/intraday/dqnpf/`
2. THE project SHALL contain the following modules: `__init__.py`, `integration.py` (IntegrationLayer, IntegrationConfig, ScreenedAction), `train.py` (combined training entry point), and `config.yaml` (default configuration)
3. THE project SHALL depend on `deepqnetwork` (for DQNAdvisor), `probabilisticforecaster` (for ForecasterInference and compute_features), and the generated protobuf modules for modelenv gRPC
4. THE `__init__.py` SHALL export `IntegrationLayer`, `IntegrationConfig`, and `ScreenedAction` as the public interface

### Requirement 17: Logging

**User Story:** As a developer, I want structured logging from the integration layer, so that I can monitor screening decisions and budget consumption at runtime.

#### Acceptance Criteria

1. THE Integration_Layer SHALL log each screened action at DEBUG level with action, reason, sigma, mu, and budget state
2. THE Integration_Layer SHALL log budget exhaustion events at INFO level
3. THE Integration_Layer SHALL log signal cache hits and misses at DEBUG level
4. THE Integration_Layer SHALL log warm-up progress at INFO level (bars loaded, features computed)
5. THE Integration_Layer SHALL log the full configuration at initialisation at INFO level
