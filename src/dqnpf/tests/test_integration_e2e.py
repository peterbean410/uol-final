"""End-to-end integration tests against a mock modelenv gRPC server.

Covers Task 8.1–8.7 from specs/tasks.md by driving ``backtest._run_episode``
with scripted mocks for ``EnvironmentClient``, ``DQNAdvisor``,
``StatePreprocessor``, and ``ForecasterBridge``.
"""

from __future__ import annotations

import logging

import pytest

from dqnpf import backtest
from dqnpf.backtest import (
    StepRecord,
    compare_results,
    validate_thresholds,
)
from dqnpf.config import IntegrationConfig
from dqnpf.forecaster_bridge import ForecasterBridge
from dqnpf.integration import IntegrationLayer
from dqnpf.signal_cache import SignalCache
from dqnpf.tests._mocks import (
    FakeActionResult,
    MockBridge,
    MockDQN,
    MockEnvClient,
    MockObservation,
    MockPreprocessor,
    make_m5_bars,
    make_observation,
    make_response,
)


def _config(
    *,
    symbol: str = "USDJPY",
    variance_threshold: float = 4.5,
    max_long: int = 2,
    max_short: int = 1,
    step_size_seconds: int = 60,
    episode_start_ts: int = 1_700_000_000,
) -> IntegrationConfig:
    return IntegrationConfig(
        symbol=symbol,
        variance_threshold=variance_threshold,
        max_risk_long_units=max_long,
        max_risk_short_units=max_short,
        step_size_seconds=step_size_seconds,
        episode_start_ts=episode_start_ts,
        episode_end_ts=episode_start_ts + 3600,
    )


def _obs_sequence(reward_then_done: list[tuple[float, bool]]) -> list[MockObservation]:
    return [
        make_observation(reward=r, done=d) for r, d in reward_then_done
    ]


def test_end_to_end_loop_drives_all_components(caplog: pytest.LogCaptureFixture) -> None:
    """Full loop: Reset → DQN → RecentBars → bridge → screen → Step (Req 10)."""
    config = _config()

    observations = _obs_sequence([(0.0, False), (0.1, False), (0.2, False), (0.0, True)])
    bars = make_m5_bars(50)
    env_client = MockEnvClient(
        observations=observations,
        bars_responses=[make_response(bars)] * 100,
    )
    dqn = MockDQN(action_script=[1, 2, 3])
    preprocessor = MockPreprocessor()
    bridge = MockBridge(signal_script=[(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)])
    cache = SignalCache()

    with caplog.at_level(logging.INFO, logger="dqnpf.integration"):
        integration = IntegrationLayer(
            dqn=dqn, forecaster_bridge=bridge, signal_cache=cache, config=config
        )
        records = backtest._run_episode(
            env_client=env_client,
            dqn=dqn,
            preprocessor=preprocessor,
            bridge=bridge,
            cache=cache,
            integration=integration,
            config=config,
        )

    assert len(records) == 3
    assert env_client.reset_calls == [
        (config.symbol, config.episode_start_ts, config.episode_end_ts, config.step_size_seconds)
    ]
    assert len(env_client.step_calls) == 3
    for action, order_id in env_client.step_calls:
        assert action in {0, 1, 2, 3, 4}
        assert order_id.startswith(config.symbol)
    for r in records:
        assert r.reason in {"pass", "budget_exhausted", "directional_conflict"}
        assert r.final_action in {0, 1, 2, 3, 4}
    assert any("IntegrationLayer initialised" in rec.message for rec in caplog.records)


def test_end_to_end_records_are_valid_against_screening_rules() -> None:
    config = _config(max_long=2, max_short=1, variance_threshold=1.0)
    bars = make_m5_bars(50)
    env_client = MockEnvClient(
        observations=_obs_sequence([(0.0, False), (0.0, False), (0.0, True)]),
        bars_responses=[make_response(bars)] * 100,
    )
    dqn = MockDQN(action_script=[2, 1])
    bridge = MockBridge(signal_script=[(0.0, 5.0), (0.0, 5.0)])
    cache = SignalCache()
    integration = IntegrationLayer(dqn, bridge, cache, config)

    records = backtest._run_episode(
        env_client=env_client,
        dqn=dqn,
        preprocessor=MockPreprocessor(),
        bridge=bridge,
        cache=cache,
        integration=integration,
        config=config,
    )

    assert records[0].final_action == 2
    assert records[0].reason == "pass"
    assert records[1].final_action == 0
    assert records[1].reason == "budget_exhausted"


def test_first_step_uses_valid_mu_sigma_from_real_pipeline() -> None:
    """With 1500 real bars, the bridge produces a finite (mu, sigma) on step 1."""
    config = _config()
    bars = make_m5_bars(1500)

    class _Forecaster:
        def predict(self, tensor):
            assert tensor.shape == (36, 16)
            return 0.42, 2.71

    bridge = ForecasterBridge(
        forecaster=_Forecaster(), symbol=config.symbol, env_client=None  # type: ignore[arg-type]
    )
    import pandas as pd
    bars_df = pd.DataFrame(
        [(b.timestamp_ns, b.open, b.high, b.low, b.close, b.volume) for b in bars],
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )
    mu, sigma = bridge.compute_signal_from_bars(bars_df)

    assert isinstance(mu, float)
    assert isinstance(sigma, float)
    assert (mu, sigma) == (0.42 * 10_000, 2.71 * 10_000)


def test_insufficient_history_raises_value_error_on_first_signal() -> None:
    """With <1440 bars, compute_features inside the bridge raises ValueError."""
    config = _config()
    short_bars = make_m5_bars(100)

    class _Forecaster:
        def predict(self, tensor):  # pragma: no cover - never reached
            raise AssertionError("predict should not be called")

    bridge = ForecasterBridge(
        forecaster=_Forecaster(), symbol=config.symbol, env_client=None  # type: ignore[arg-type]
    )
    import pandas as pd
    bars_df = pd.DataFrame(
        [(b.timestamp_ns, b.open, b.high, b.low, b.close, b.volume) for b in short_bars],
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )
    with pytest.raises(ValueError, match="Insufficient history"):
        bridge.compute_signal_from_bars(bars_df)


def test_cache_hits_for_same_bar_timestamp_across_loop() -> None:
    """Five steps with an unchanged M5 latest timestamp → bridge called once."""
    config = _config()
    fixed_bars = make_m5_bars(50)
    env_client = MockEnvClient(
        observations=_obs_sequence(
            [(0.0, False)] * 4 + [(0.0, True)]
        ),
        bars_responses=[make_response(fixed_bars)] * 10,
    )
    dqn = MockDQN(action_script=[0, 0, 0, 0])
    bridge = MockBridge(signal_script=[(0.1, 0.2)])
    cache = SignalCache()
    integration = IntegrationLayer(dqn, bridge, cache, config)

    records = backtest._run_episode(
        env_client=env_client,
        dqn=dqn,
        preprocessor=MockPreprocessor(),
        bridge=bridge,
        cache=cache,
        integration=integration,
        config=config,
    )

    assert len(records) == 4
    assert bridge.call_count == 1
    assert {(r.mu, r.sigma) for r in records} == {(0.1, 0.2)}


def test_cache_recomputes_when_bar_timestamp_advances() -> None:
    config = _config()
    bars_a = make_m5_bars(50)
    bars_b = make_m5_bars(51)

    responses_per_step = [make_response(bars_a)] * 6 + [make_response(bars_b)] * 2

    env_client = MockEnvClient(
        observations=_obs_sequence(
            [(0.0, False)] * 4 + [(0.0, True)]
        ),
        bars_responses=responses_per_step,
    )
    dqn = MockDQN(action_script=[0, 0, 0, 0])
    bridge = MockBridge(signal_script=[(0.1, 0.2), (0.3, 0.4)])
    cache = SignalCache()
    integration = IntegrationLayer(dqn, bridge, cache, config)

    records = backtest._run_episode(
        env_client=env_client,
        dqn=dqn,
        preprocessor=MockPreprocessor(),
        bridge=bridge,
        cache=cache,
        integration=integration,
        config=config,
    )

    assert len(records) == 4
    assert bridge.call_count == 2
    assert (records[0].mu, records[0].sigma) == (0.1, 0.2)
    assert (records[1].mu, records[1].sigma) == (0.1, 0.2)
    assert (records[2].mu, records[2].sigma) == (0.1, 0.2)
    assert (records[3].mu, records[3].sigma) == (0.3, 0.4)


def test_budget_exhaustion_then_release_across_loop() -> None:
    """BUY_2 passes, next BUY_1 blocked; SELL_1 passes, next SELL_1 blocked;
    release on_position_closed → previously blocked action now passes."""
    config = _config(max_long=2, max_short=1, variance_threshold=1.0)
    bars = make_m5_bars(50)
    dqn = MockDQN(action_script=[2, 1, 3, 3, 1])
    bridge = MockBridge(signal_script=[(0.0, 5.0)])
    cache = SignalCache()
    env_client = MockEnvClient(
        observations=_obs_sequence(
            [(0.0, False)] * 4 + [(0.0, True)]
        ),
        bars_responses=[make_response(bars)] * 20,
    )
    integration = IntegrationLayer(dqn, bridge, cache, config)

    records = backtest._run_episode(
        env_client=env_client,
        dqn=dqn,
        preprocessor=MockPreprocessor(),
        bridge=bridge,
        cache=cache,
        integration=integration,
        config=config,
    )

    assert [r.final_action for r in records] == [2, 0, 3, 0]
    assert [r.reason for r in records] == [
        "pass",
        "budget_exhausted",
        "pass",
        "budget_exhausted",
    ]
    assert integration.risk_long_used == 2
    assert integration.risk_short_used == 1

    integration.on_position_closed("buy", 1)
    assert integration.risk_long_used == 1
    follow_up = integration.screen(
        FakeActionResult(action=1, action_name="BUY_1"), mu=0.0, sigma=5.0
    )
    assert follow_up.reason == "pass"
    assert follow_up.action == 1


def test_combined_vs_baseline_trades_and_metrics() -> None:
    """Combined system trades less than baseline on identical scripted episodes."""
    config = _config(max_long=2, max_short=1, variance_threshold=1.0)
    bars = make_m5_bars(50)
    action_script = [2, 1, 3, 3]
    rewards = [(0.0, False), (0.5, False), (-0.2, False), (0.3, False), (0.1, True)]

    def _new_client() -> MockEnvClient:
        return MockEnvClient(
            observations=_obs_sequence(rewards),
            bars_responses=[make_response(bars)] * 20,
        )

    cache_c = SignalCache()
    bridge_c = MockBridge(signal_script=[(0.0, 5.0)])
    dqn_c = MockDQN(action_script=action_script)
    integration = IntegrationLayer(dqn_c, bridge_c, cache_c, config)
    combined = backtest._run_episode(
        env_client=_new_client(),
        dqn=dqn_c,
        preprocessor=MockPreprocessor(),
        bridge=bridge_c,
        cache=cache_c,
        integration=integration,
        config=config,
    )

    cache_b = SignalCache()
    bridge_b = MockBridge(signal_script=[(0.0, 5.0)])
    dqn_b = MockDQN(action_script=action_script)
    baseline = backtest._run_episode(
        env_client=_new_client(),
        dqn=dqn_b,
        preprocessor=MockPreprocessor(),
        bridge=bridge_b,
        cache=cache_b,
        integration=None,
        config=config,
    )

    comparison = compare_results(combined, baseline)

    assert comparison.trades_combined == 2
    assert comparison.trades_baseline == 4
    assert comparison.trades_combined < comparison.trades_baseline
    assert comparison.suppression_rate == 0.5
    assert comparison.suppression_by_reason == {"budget_exhausted": 2}
    assert comparison.high_sigma_time_fraction == 1.0
    assert comparison.quarterly_pnl_combined
    assert comparison.quarterly_pnl_baseline
    report = validate_thresholds(comparison)
    assert isinstance(report.passed, bool)


def test_two_layers_independent_budgets() -> None:
    config_a = _config(symbol="USDJPY", max_long=2, max_short=1)
    config_b = _config(symbol="AUDJPY", max_long=2, max_short=1)
    layer_a = IntegrationLayer(
        dqn=None,  # type: ignore[arg-type]
        forecaster_bridge=None,  # type: ignore[arg-type]
        signal_cache=None,  # type: ignore[arg-type]
        config=config_a,
    )
    layer_b = IntegrationLayer(
        dqn=None,  # type: ignore[arg-type]
        forecaster_bridge=None,  # type: ignore[arg-type]
        signal_cache=None,  # type: ignore[arg-type]
        config=config_b,
    )

    layer_a.screen(FakeActionResult(action=2), mu=0.0, sigma=5.0)
    layer_a.screen(FakeActionResult(action=3), mu=0.0, sigma=5.0)
    assert layer_a.risk_long_used == 2
    assert layer_a.risk_short_used == 1

    assert layer_b.risk_long_used == 0
    assert layer_b.risk_short_used == 0
    assert layer_a.symbol == "USDJPY"
    assert layer_b.symbol == "AUDJPY"


def test_two_layers_independent_screening() -> None:
    """Screening decisions on layer A don't change layer B outcomes."""
    config_a = _config(symbol="USDJPY", max_long=1, max_short=1)
    config_b = _config(symbol="AUDJPY", max_long=1, max_short=1)
    layer_a = IntegrationLayer(
        dqn=None,  # type: ignore[arg-type]
        forecaster_bridge=None,  # type: ignore[arg-type]
        signal_cache=None,  # type: ignore[arg-type]
        config=config_a,
    )
    layer_b = IntegrationLayer(
        dqn=None,  # type: ignore[arg-type]
        forecaster_bridge=None,  # type: ignore[arg-type]
        signal_cache=None,  # type: ignore[arg-type]
        config=config_b,
    )

    layer_a.screen(FakeActionResult(action=1), mu=0.0, sigma=5.0)
    blocked_a = layer_a.screen(FakeActionResult(action=1), mu=0.0, sigma=5.0)
    assert blocked_a.reason == "budget_exhausted"

    result_b = layer_b.screen(FakeActionResult(action=1), mu=0.0, sigma=5.0)
    assert result_b.reason == "pass"
    assert result_b.action == 1
