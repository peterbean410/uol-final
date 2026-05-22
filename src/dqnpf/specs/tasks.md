# Implementation Plan: Intraday Trading Agent (DQN + ProbabilisticForecaster)

## Overview

This plan implements an integration layer that combines a Deep Q-Network (DQN) for action selection with a Transformer-based ProbabilisticForecaster for risk screening, wired together via the existing modelenv gRPC service. The integration layer sits between the DQN output and the modelenv `Step()` call, screening DQN actions using forecaster signals and risk budgets.

The DQN must be retrained at `step_size_seconds=60` before the integration layer can be used. Both parent models are otherwise unchanged.

## Tasks

- [x] 1. Set up project structure, configuration, and action mapper
  - [x] 1.1 Create project skeleton and dependencies
    - Create `tradingmodel/intraday/dqnpf/` directory with `__init__.py`
    - Create empty files: `integration.py`, `action_mapper.py`, `signal_cache.py`, `forecaster_bridge.py`, `warmup.py`, `train.py`, `backtest.py`, `config.py`, `config.yaml`
    - Create `tradingmodel/intraday/dqnpf/tests/` directory with `__init__.py` and `conftest.py`
    - Create `tradingmodel/intraday/dqnpf/requirements.txt` with dependencies: `torch`, `numpy`, `pandas`, `grpcio`, `grpcio-tools`, `pyyaml`, `hypothesis`, `pytest`
    - _Requirements: 16.1, 16.3_

  - [x] 1.2 Implement configuration management (`config.py`)
    - Implement `IntegrationConfig` dataclass with all fields: `symbol`, `variance_threshold` (default 4.5), `max_risk_long_units` (default 2), `max_risk_short_units` (default 1), `directional_disagreement` (default False), `directional_tolerance` (default 1.0), `forecast_horizon` (default 1), `min_bars_warmup` (default 1440), `step_size_seconds` (default 60), `dqn_checkpoint_path`, `forecaster_checkpoint_path`
    - Implement `__post_init__` validation: `variance_threshold >= 0`, `max_risk_long_units >= 0`, `max_risk_short_units >= 0`, `directional_tolerance >= 0`, `forecast_horizon` in {1, 3, 6, 12}
    - Implement YAML loading with `load_config()` function
    - Implement argparse CLI that overrides YAML values (`--config`, `--symbol`, `--dqn-checkpoint`, `--forecaster-checkpoint`, `--variance-threshold`, `--max-risk-long`, `--max-risk-short`, `--device`)
    - Create `config.yaml` with all default values
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [x] 1.3 Implement action mapper (`action_mapper.py`)
    - Define `Direction` enum with values NONE, LONG, SHORT
    - Define frozen `ActionUnit` dataclass with `direction: Direction` and `risk_units: int`
    - Define `ACTION_MAP` static lookup table mapping action indices 0–4 to `ActionUnit` values: HOLD→(NONE,0), BUY_1→(LONG,1), BUY_2→(LONG,2), SELL_1→(SHORT,1), SELL_2→(SHORT,2)
    - Implement `map_action(action_index: int) -> ActionUnit` with ValueError for indices outside [0, 4]
    - Export `ACTION_NAMES` list: ["HOLD", "BUY_1", "BUY_2", "SELL_1", "SELL_2"]
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 1.4 Write property test for action-to-unit mapping correctness
    - **Property 2: Action-to-unit mapping correctness**
    - Test that for any valid action index in [0, 4], `map_action(index)` returns the correct `(Direction, risk_units)` pair
    - Test that for any index outside [0, 4], `map_action` raises ValueError
    - File: `tradingmodel/intraday/dqnpf/tests/test_action_mapper_pbt.py`
    - _Requirements: 2.1–2.6_

  - [x] 1.5 Write property test for config validation
    - **Property 12: Config validation**
    - Test that valid field values construct successfully
    - Test that negative thresholds, invalid forecast_horizon raise ValueError
    - File: `tradingmodel/intraday/dqnpf/tests/test_config_pbt.py`
    - _Requirements: 5.3, 5.4, 11.2_

- [x] 2. Implement signal cache and forecaster bridge
  - [x] 2.1 Implement signal cache (`signal_cache.py`)
    - Define `CachedSignal` dataclass with `mu: float`, `sigma: float`, `bar_timestamp: int`
    - Implement `SignalCache` class with `_cache: CachedSignal | None` field
    - Implement `get_or_compute(latest_bar_ts, compute_fn)` (return cached if timestamp matches, else recompute and update
    - Implement `invalidate()`) clear cache on Reset()
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 2.2 Write property test for signal cache hit
    - **Property 10: Signal cache hit**
    - Test that with unchanged `latest_bar_ts`, `compute_fn` is called at most once and subsequent calls return same values
    - File: `tradingmodel/intraday/dqnpf/tests/test_signal_cache_pbt.py`
    - _Requirements: 7.2, 7.3_

  - [x] 2.3 Write property test for signal cache miss
    - **Property 11: Signal cache miss**
    - Test that with changed `latest_bar_ts`, `compute_fn` is invoked and cache is updated
    - File: `tradingmodel/intraday/dqnpf/tests/test_signal_cache_pbt.py`
    - _Requirements: 7.3_

  - [x] 2.4 Implement forecaster bridge (`forecaster_bridge.py`)
    - Implement `ForecasterBridge` class with `__init__(forecaster, symbol, env_client)`
    - Implement `compute_signal()` (call `RecentBars(symbol)`, extract `"M5"` key from response map, convert `BarList` to DataFrame with columns `[Timestamp, Open, High, Low, Close, Volume]`, call `compute_features()`, select last 36 rows, convert to (36, 16) float32 tensor, call `forecaster.predict(tensor)`, return (mu, sigma)
    - Implement `compute_signal_from_bars(bars_df)`) compute (mu, sigma) from a pre-loaded DataFrame, avoiding gRPC call (for warm-up and testing)
    - Raise KeyError if `"M5"` key missing from `RecentBarsResponse`
    - Raise ValueError if M5 bar series has fewer than 36 bars
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [x] 2.5 Write unit tests for signal cache
    - Test empty cache: first call invokes compute_fn
    - Test cache hit: same timestamp returns cached value
    - Test cache miss: new timestamp invokes compute_fn
    - Test invalidate: clears cache, next call invokes compute_fn
    - _Requirements: 7_

  - [x] 2.6 Write unit tests for forecaster bridge
    - Test RecentBars → features → predict pipeline with mock gRPC client
    - Test KeyError when "M5" key missing from response
    - Test ValueError when < 36 bars in M5 series
    - Test compute_signal_from_bars with mock DataFrame
    - _Requirements: 6_

- [x] 3. Implement integration layer with screening logic and risk budgets
  - [x] 3.1 Implement ScreenedAction dataclass and IntegrationLayer (`integration.py`)
    - Define `ScreenedAction` dataclass with fields: `action: int`, `action_name: str`, `screened: bool`, `reason: str`, `sigma: float`, `risk_long_used: int`, `risk_short_used: int`
    - Implement `IntegrationLayer.__init__(dqn, forecaster_bridge, signal_cache, config)`, initialise `_risk_long_units = 0`, `_risk_short_units = 0`
    - Bind to single symbol from config; store symbol for lifetime
    - Implement `risk_long_used` and `risk_short_used` read-only properties
    - _Requirements: 1.1, 1.3, 1.4, 10.6, 10.7_

  - [x] 3.2 Implement `screen()` method
    - Resolve action index via `map_action()` to get Direction and risk_units
    - **Rule 1 (risk budget):** If sigma > variance_threshold AND direction is LONG AND `risk_long_units + action_units > max_risk_long_units` → return HOLD with reason "budget_exhausted". Symmetric for SHORT.
    - **Rule 2 (directional conflict):** If directional_disagreement enabled AND `abs(mu) > directional_tolerance` AND `sign(mu) != direction` AND direction != NONE → return HOLD with reason "directional_conflict"
    - **Rule 3 (pass-through):** If sigma > threshold, increment the corresponding budget counter. Return action unchanged with reason "pass"
    - When low-sigma (sigma <= threshold), pass through without incrementing budget
    - Rules evaluated in priority order; first triggering rule wins
    - _Requirements: 1.1, 1.2, 1.5, 3.2, 3.3, 3.4, 3.5, 3.6, 4.2, 4.3, 4.4_

  - [x] 3.3 Implement `on_position_closed()` method
    - Accept `side: str` ("buy" or "sell") and `units: int`
    - Decrement corresponding budget counter, clamped to zero
    - Log warning for unknown side, no-op
    - _Requirements: 3.7_

  - [x] 3.4 Write property test for screened action validity
    - **Property 1: Screened action validity**
    - Test that for any valid DQN action and any (mu, sigma) where sigma > 0, screened action is in [0, 4] and reason is valid
    - File: `tradingmodel/intraday/dqnpf/tests/test_integration_pbt.py`
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 3.5 Write property test for low-sigma pass-through
    - **Property 3: Low-sigma pass-through**
    - Test that for sigma <= threshold and directional_disagreement disabled, action passes through unchanged with reason "pass" and budget counters unchanged
    - File: `tradingmodel/intraday/dqnpf/tests/test_integration_pbt.py`
    - _Requirements: 3.5, 3.6_

  - [x] 3.6 Write property test for high-sigma budget consumption
    - **Property 4: High-sigma budget consumption**
    - Test that for sigma > threshold and budget not exhausted, action passes through and correct budget counter increments by action's risk_units, opposite counter unchanged
    - File: `tradingmodel/intraday/dqnpf/tests/test_integration_pbt.py`
    - _Requirements: 3.4_

  - [x] 3.7 Write property test for budget exhaustion
    - **Property 5: Budget exhaustion**
    - Test that when budget would be exceeded, action screened to HOLD with reason "budget_exhausted"
    - File: `tradingmodel/intraday/dqnpf/tests/test_integration_pbt.py`
    - _Requirements: 3.2, 3.3_

  - [x] 3.8 Write property test for budget never exceeded
    - **Property 6: Budget never exceeded**
    - Test that over any sequence of screen() calls, risk_long_units never exceeds max_risk_long_units and risk_short_units never exceeds max_risk_short_units
    - File: `tradingmodel/intraday/dqnpf/tests/test_integration_pbt.py`
    - _Requirements: 3.2, 3.3_

  - [x] 3.9 Write property test for budget release
    - **Property 7: Budget release**
    - Test that on_position_closed decrements correct counter, clamped to zero. After release, previously blocked action becomes pass-through
    - File: `tradingmodel/intraday/dqnpf/tests/test_integration_pbt.py`
    - _Requirements: 3.7_

  - [x] 3.10 Write property test for directional conflict rule
    - **Property 8: Directional conflict rule**
    - Test that when enabled, abs(mu) > tolerance, and sign(mu) != direction → HOLD with reason "directional_conflict". When disabled, rule skipped
    - File: `tradingmodel/intraday/dqnpf/tests/test_integration_pbt.py`
    - _Requirements: 4.2, 4.3, 4.4_

  - [x] 3.11 Write property test for rule priority order
    - **Property 9: Rule priority order**
    - Test that when both budget rule and directional conflict would trigger, budget rule takes precedence (reason "budget_exhausted")
    - File: `tradingmodel/intraday/dqnpf/tests/test_integration_pbt.py`
    - _Requirements: 1.2_

  - [x] 3.12 Write property test for single-symbol isolation
    - **Property 13: Single-symbol isolation**
    - Test that two IntegrationLayer instances with different symbols have independent budget counters and screening results
    - File: `tradingmodel/intraday/dqnpf/tests/test_integration_pbt.py`
    - _Requirements: 10.6, 10.8_

  - [x] 3.13 Write unit tests for integration layer
    - Test budget increment/decrement for each action type
    - Test HOLD bypass: HOLD action never increments budget regardless of sigma
    - Test ScreenedAction dataclass fields and values
    - Test directional conflict with mu exactly at tolerance boundary (abs(mu) == tolerance → skipped)
    - Test directional conflict with mu just above tolerance (abs(mu) == tolerance + epsilon → triggers)
    - Test variance_threshold boundary: sigma exactly at threshold → low-sigma path
    - Test variance_threshold boundary: sigma just above threshold → high-sigma path
    - _Requirements: 1, 3, 4_

- [x] 6. Implement backtesting entry point
  - [x] 6.1 Implement backtest entry point (`backtest.py`)
    - Define `BacktestComparison` dataclass with all comparison fields
    - Implement `run_backtest(config)`, run combined system and DQN-only baseline on identical episodes with same seed
    - Baseline: DQN actions go directly to Step() without integration layer
    - Compute total return and Sharpe ratio for both systems
    - Compute suppression rate: fraction of actions screened, grouped by reason
    - Compute conditional performance: combined vs baseline PnL during high-sigma and low-sigma steps
    - Compute trade counts for both systems
    - Compute quarterly PnL for walk-forward stability check
    - Log all comparison metrics at INFO
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

  - [x] 6.2 Implement comparative threshold validation
    - Verify combined Sharpe > baseline Sharpe (Req 14.1)
    - Verify combined trades < baseline trades, reduction concentrated in high-sigma steps (Req 14.2)
    - Verify combined negative-PnL proportion during high-sigma < baseline (Req 14.3)
    - Verify low-sigma PnL degradation <= 5% (Req 14.4)
    - Verify no single quarter > 50% of total PnL (Req 14.5)
    - If any threshold fails, log failure and mark config invalid (Req 14.6)
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6_

- [x] 7. Write remaining unit tests
  - [x] 7.1 Write unit tests for configuration
    - Test default values match spec
    - Test validation: negative variance_threshold raises ValueError
    - Test validation: negative risk budget caps raise ValueError
    - Test validation: negative directional_tolerance raises ValueError
    - Test validation: invalid forecast_horizon (e.g., 2, 0, 13) raises ValueError
    - Test YAML round-trip: save config to YAML, load back, verify field equality
    - _Requirements: 5, 11_

  - [x] 7.2 Write unit tests for action mapper
    - Test each of the 5 valid indices returns correct (direction, risk_units)
    - Test index -1 raises ValueError
    - Test index 5 raises ValueError
    - _Requirements: 2_

  - [x] 7.3 Write unit tests for ScreenedAction
    - Test dataclass construction with all fields
    - Test screened=True when reason is not "pass"
    - Test screened=False when reason is "pass"
    - _Requirements: 1_

  - [x] 7.4 Write unit tests for budget tracking
    - Test BUY_1 increments risk_long_units by 1 (only when sigma > threshold)
    - Test BUY_2 increments risk_long_units by 2 (only when sigma > threshold)
    - Test SELL_1 increments risk_short_units by 1 (only when sigma > threshold)
    - Test SELL_2 increments risk_short_units by 2 (only when sigma > threshold)
    - Test HOLD increments neither counter
    - Test on_position_closed("buy", 1) decrements risk_long_units by 1
    - Test on_position_closed("sell", 2) decrements risk_short_units by 2
    - Test on_position_closed with units exceeding budget clamps to zero
    - Test on_position_closed with unknown side is no-op
    - _Requirements: 3_

- [x] 8. Write integration tests (with mock modelenv gRPC server)
  - [x] 8.1 Write end-to-end integration test
    - Mock modelenv gRPC server with Reset, Step, RecentBars
    - Full loop: Reset → get Observation → DQNAdvisor.recommend_action → RecentBars → compute_features → ForecasterInference.predict → IntegrationLayer.screen → Step
    - Verify screened action is valid and metrics logged
    - _Requirements: 10_

  - [x] 8.2 Write warm-up integration test
    - Verify forecaster ready after 1440+ bars loaded via warm_up()
    - Verify first screened action uses valid (mu, sigma) from warm-up features
    - _Requirements: 8_

  - [x] 8.3 Write insufficient history integration test
    - Mock RecentBars to return < 1440 M5 bars
    - Verify RuntimeError raised at warm-up
    - _Requirements: 8.3_

  - [x] 8.4 Write signal cache integration test
    - Step through 5 timesteps at 60s with the same completed M5 bar timestamp
    - Verify forecaster bridge called only once, other 4 steps use cache
    - Step with new completed M5 bar timestamp
    - Verify forecaster bridge called again
    - _Requirements: 7_

  - [x] 8.5 Write budget exhaustion integration test
    - Run multi-step episode with sigma > threshold throughout
    - Verify BUY_2 passes (long=2, cap=2), next BUY_1 blocked (would be long=3 > cap=2)
    - Verify SELL_1 passes (short=1, cap=1), next SELL_1 blocked
    - Call on_position_closed, verify previously blocked action now passes
    - _Requirements: 3_

  - [x] 8.6 Write DQN-only baseline comparison integration test
    - Run same episodes with combined system and DQN-only baseline
    - Verify comparison metrics are computed correctly
    - Verify combined system has fewer or equal trades than baseline
    - _Requirements: 13, 14_

  - [x] 8.7 Write multi-instance isolation integration test
    - Create two IntegrationLayer instances for different symbols (e.g., USDJPY, AUDJPY)
    - Open positions on instance 1, verify instance 2 budget counters unchanged
    - Verify screening results on instance 2 unaffected by instance 1 state
    - _Requirements: 10.6, 10.8_

- [ ] 9. Final validation
  - [ ] 9.1 Verify all 17 requirements are covered by at least one test
    - Cross-reference requirements.md acceptance criteria against test coverage
    - _Requirements: All_

  - [ ] 9.2 Verify DQN retrained at step_size_seconds=60 before integration testing
    - Run DQN training with `step_size_seconds=60` on 2012-2022 episodes
    - Save checkpoint for integration layer to load
    - _Requirements: 9.1, 9.2_

  - [ ] 9.3 Verify forecaster trained on 2012-2022 split
    - Confirm existing forecaster checkpoint covers 2012-2022 training range
    - Confirm no data leakage into 2023-2026 evaluation range
    - _Requirements: 12.1, 12.2_

  - [ ] 9.4 Tune integration thresholds on validation period
    - Run validation on last 12 months of training range (2022-01-01 to 2022-12-31)
    - Grid search: variance_threshold over [2.0, 3.0, 4.5, 6.0, 8.0], max_risk_long_units over [1, 2, 3, 4], max_risk_short_units over [1, 2]
    - Select best config by validation Sharpe
    - _Requirements: 12.3_

  - [ ] 9.5 Run final evaluation on out-of-sample period
    - Evaluate combined system on 2023-01-01 to 2026-04-30 with frozen thresholds
    - Compare against DQN-only baseline on identical episodes
    - Verify all comparative thresholds pass (Req 14.1–14.5)
    - _Requirements: 12.4, 12.5, 14_
