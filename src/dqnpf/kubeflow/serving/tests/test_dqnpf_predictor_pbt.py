"""Property-based tests for the dqnpf-intraday combined predictor.

Properties:
  DQNPF-1: Predictor loads both checkpoints from registry
  DQNPF-2: ScreenedAction validity
  DQNPF-3: Budget state persists across requests
  DQNPF-4: Hot-reload swaps in-process model atomically

Validates: Requirements 22.3, 22.5, 22.6, 22.12
"""

from __future__ import annotations

import sys
import threading
import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st


if "kserve" not in sys.modules:

    class _FakeKServeModel:
        """Minimal stand-in for kserve.Model base class."""

        def __init__(self, name: str) -> None:
            self.name = name
            self.ready = False

    _mock_kserve = MagicMock()
    _mock_kserve.Model = _FakeKServeModel
    sys.modules["kserve"] = _mock_kserve

sys.modules.setdefault("model_registry", MagicMock())
sys.modules.setdefault("model_registry.types", MagicMock())


@pytest.fixture(autouse=True)
def _stub_resolve_production_checkpoint(monkeypatch):
    """Replace ``resolve_production_checkpoint`` with a stub that returns a
    fake path for every test in this module. ``monkeypatch`` auto-restores at
    teardown, so other test modules see the real implementation.
    """
    stub = MagicMock(return_value="/fake/checkpoint/path.pt")
    monkeypatch.setattr(
        "dqnpf.kubeflow.registry.registry_client."
        "resolve_production_checkpoint",
        stub,
    )
    yield stub


from dqnpf.config import IntegrationConfig


_SYMBOLS = ["USDJPY", "AUDJPY"]

_valid_action = st.integers(min_value=0, max_value=4)
_finite_float = st.floats(
    min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False
)
_positive_sigma = st.floats(
    min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False
)
_observation_dim = 53

_NANOS_PER_DAY = 86_400_000_000_000
_M5_NS = 300_000_000_000
_REF_TS_NS = 1_700_000_000_000_000_000
_DAY_BASE = (_REF_TS_NS // _NANOS_PER_DAY) * _NANOS_PER_DAY


def _intraday_m5_bars(count: int = 40) -> list[dict]:
    """Build ``count`` M5 bars that all fall within a single UTC day.

    Keeps the latest-bar timestamp (and thus the screen budget-reset day) fixed
    so sequential requests accumulate budget rather than tripping a day reset.
    """
    return [{"timestamp_ns": _DAY_BASE + j * _M5_NS} for j in range(count)]


def _observation_strategy():
    """Generate a random 1-D float32 observation vector."""
    return st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=_observation_dim,
        max_size=_observation_dim,
    )


ACTION_NAMES = ["HOLD", "BUY_1", "BUY_2", "SELL_1", "SELL_2"]


@dataclass
class FakeActionResult:
    """Minimal stand-in for deepqnetwork.advisor.ActionResult."""

    action: int
    action_name: str
    q_values: Any = None
    confidence: float = 0.5


def _make_fake_action_result(action: int) -> FakeActionResult:
    return FakeActionResult(action=action, action_name=ACTION_NAMES[action])


def _build_predictor(
    symbols: list[str] | None = None,
    variance_threshold: float = 4.5,
    max_risk_long_units: int = 2,
    max_risk_short_units: int = 1,
):
    """Build a DqnpfIntradayPredictor with mocked external dependencies.

    Returns (predictor, configs) so tests can configure the mock return values.
    """
    if symbols is None:
        symbols = ["USDJPY"]

    configs = {
        sym: IntegrationConfig(
            symbol=sym,
            variance_threshold=variance_threshold,
            max_risk_long_units=max_risk_long_units,
            max_risk_short_units=max_risk_short_units,
            forecast_horizon=1,
            min_bars_warmup=1440,
            step_size_seconds=60,
        )
        for sym in symbols
    }

    from dqnpf.kubeflow.serving.dqnpf_predictor import (
        DqnpfIntradayPredictor,
    )

    predictor = DqnpfIntradayPredictor(name="test-predictor", configs=configs)
    return predictor, configs


def _load_predictor_with_mocks(
    predictor,
    dqn_action: int = 1,
    mu: float = 0.5,
    sigma: float = 5.0,
):
    """Call predictor.load() with all external dependencies mocked.

    Returns (predictor, mock_dqn, mock_forecaster, mock_env_client).
    """
    mock_dqn = MagicMock()
    mock_dqn.recommend_action.return_value = _make_fake_action_result(dqn_action)

    mock_forecaster = MagicMock()
    mock_forecaster.predict.return_value = (mu, sigma)

    mock_env_client = MagicMock()
    mock_env_client.reference_data.return_value = MagicMock()

    mock_bars_response = MagicMock()
    mock_bar = MagicMock()
    mock_bar.timestamp_ns = 1_700_000_000_000_000_000
    mock_bar.open = 150.0
    mock_bar.high = 150.5
    mock_bar.low = 149.5
    mock_bar.close = 150.2
    mock_bar.volume = 1000.0
    mock_bars_response.bars = {"M5": MagicMock(bars=[mock_bar] * 40)}
    mock_env_client.recent_bars.return_value = mock_bars_response

    with (
        patch(
            "dqnpf.kubeflow.serving.dqnpf_predictor.DQNAdvisor",
            return_value=mock_dqn,
        ),
        patch(
            "dqnpf.kubeflow.serving.dqnpf_predictor.ForecasterInference",
            return_value=mock_forecaster,
        ),
        patch(
            "dqnpf.kubeflow.serving.dqnpf_predictor._ServingEnvClient",
            return_value=mock_env_client,
        ),
        patch(
            "dqnpf.kubeflow.serving.dqnpf_predictor.StatePreprocessor",
        ) as mock_preprocessor_cls,
    ):
        mock_state = np.zeros(_observation_dim, dtype=np.float32)
        mock_preprocessor_cls.return_value.process.return_value = MagicMock(
            numpy=MagicMock(return_value=mock_state)
        )
        predictor.load()

    predictor._dqn = mock_dqn
    predictor._forecaster = mock_forecaster
    predictor._env_client = mock_env_client

    return predictor, mock_dqn, mock_forecaster, mock_env_client


@given(
    symbol=st.sampled_from(_SYMBOLS),
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_predictor_loads_both_checkpoints(symbol: str) -> None:
    """**Validates: Requirements 22.3, 22.5**

    After load(), predictor.ready is True and internal _dqn and _forecaster
    are not None.
    """
    predictor, _ = _build_predictor(symbols=[symbol])
    predictor, mock_dqn, mock_forecaster, _ = _load_predictor_with_mocks(predictor)

    assert predictor.ready is True
    assert predictor._dqn is not None
    assert predictor._forecaster is not None
    assert symbol in predictor._layers
    assert symbol in predictor._bridges
    assert symbol in predictor._caches


_VALID_ACTIONS = {0, 1, 2, 3, 4}
_VALID_REASONS = {"pass", "budget_exhausted", "directional_conflict", "gate_bypassed"}


@given(
    action=_valid_action,
    observation=_observation_strategy(),
    mu=_finite_float,
    sigma=_positive_sigma,
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_screened_action_validity(
    action: int,
    observation: list[float],
    mu: float,
    sigma: float,
) -> None:
    """**Validates: Requirements 22.3, 22.6**

    For any well-formed request, response contains action in [0,4] and
    reason in {"pass", "budget_exhausted", "directional_conflict"}.
    """
    predictor, _ = _build_predictor(symbols=["USDJPY"])
    predictor, mock_dqn, mock_forecaster, mock_env_client = (
        _load_predictor_with_mocks(predictor, dqn_action=action, mu=mu, sigma=sigma)
    )

    for bridge in predictor._bridges.values():
        bridge.compute_signal = MagicMock(return_value=(mu, sigma))

    for cache in predictor._caches.values():
        cache.invalidate()

    payload = {
        "symbol": "USDJPY",
        "observation": observation,
        "recent_bars_m5": [{"timestamp_ns": 1_700_000_000_000_000_000 + i * 300_000_000_000} for i in range(40)],
    }

    response = predictor.predict(payload)

    assert response["action"] in _VALID_ACTIONS
    assert response["reason"] in _VALID_REASONS
    assert isinstance(response["sigma"], float)
    assert isinstance(response["mu"], float)
    assert isinstance(response["risk_long_used"], int)
    assert isinstance(response["risk_short_used"], int)
    assert isinstance(response["screened"], bool)
    assert isinstance(response["action_name"], str)


@given(
    num_requests=st.integers(min_value=1, max_value=10),
    max_risk_long=st.integers(min_value=1, max_value=5),
    max_risk_short=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_budget_state_persists_across_requests(
    num_requests: int,
    max_risk_long: int,
    max_risk_short: int,
) -> None:
    """**Validates: Requirements 22.5, 22.6**

    Sequential requests with BUY_1 actions and high sigma accumulate
    per-symbol risk-budget counters bounded by max_risk_long_units.
    """
    predictor, _ = _build_predictor(
        symbols=["USDJPY"],
        variance_threshold=2.0,
        max_risk_long_units=max_risk_long,
        max_risk_short_units=max_risk_short,
    )
    predictor, mock_dqn, mock_forecaster, mock_env_client = (
        _load_predictor_with_mocks(
            predictor,
            dqn_action=1,
            mu=0.5,
            sigma=10.0,
        )
    )

    for bridge in predictor._bridges.values():
        bridge.compute_signal = MagicMock(return_value=(0.5, 10.0))

    observation = [0.0] * _observation_dim

    for i in range(num_requests):
        for cache in predictor._caches.values():
            cache.invalidate()

        payload = {
            "symbol": "USDJPY",
            "observation": observation,
            "recent_bars_m5": _intraday_m5_bars(40),
        }

        response = predictor.predict(payload)

        assert response["risk_long_used"] <= max_risk_long
        assert response["risk_short_used"] <= max_risk_short

    layer = predictor._layers["USDJPY"]
    assert layer.risk_long_used <= max_risk_long
    assert layer.risk_short_used <= max_risk_short

    if num_requests > max_risk_long:
        assert layer.risk_long_used == max_risk_long


@given(
    num_threads=st.integers(min_value=2, max_value=8),
    requests_per_thread=st.integers(min_value=1, max_value=5),
)
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_hot_reload_atomicity(
    num_threads: int,
    requests_per_thread: int,
) -> None:
    """**Validates: Requirement 22.12**

    Concurrent in-flight predict() calls during a hot-reload swap observe
    either the old model (action=1) or the new model (action=2), never a
    partially-loaded state. Budget state is preserved after the swap.
    """
    predictor, _ = _build_predictor(
        symbols=["USDJPY"],
        variance_threshold=2.0,
        max_risk_long_units=5,
        max_risk_short_units=5,
    )
    predictor, mock_dqn_old, mock_forecaster_old, mock_env_client = (
        _load_predictor_with_mocks(
            predictor,
            dqn_action=1,
            mu=0.5,
            sigma=10.0,
        )
    )

    for bridge in predictor._bridges.values():
        bridge.compute_signal = MagicMock(return_value=(0.5, 10.0))

    predictor._layers["USDJPY"]._risk_long_units = 2
    predictor._layers["USDJPY"]._risk_short_units = 1
    predictor._layers["USDJPY"]._current_day = _DAY_BASE // _NANOS_PER_DAY

    initial_risk_long = 2
    initial_risk_short = 1

    mock_dqn_new = MagicMock()
    mock_dqn_new.recommend_action.return_value = _make_fake_action_result(2)

    mock_forecaster_new = MagicMock()
    mock_forecaster_new.predict.return_value = (0.5, 10.0)

    def _make_mock_bridge(*args, **kwargs):
        mock_bridge = MagicMock()
        mock_bridge.compute_signal = MagicMock(return_value=(0.5, 10.0))
        return mock_bridge

    results: list[dict] = []
    results_lock = threading.Lock()
    errors: list[Exception] = []

    barrier = threading.Barrier(num_threads + 1, timeout=10)

    observation = [0.0] * _observation_dim

    def worker(thread_id: int) -> None:
        """Worker thread that calls predict() multiple times."""
        try:
            barrier.wait()
            for req_idx in range(requests_per_thread):
                for cache in predictor._caches.values():
                    cache.invalidate()

                payload = {
                    "symbol": "USDJPY",
                    "observation": observation,
                    "recent_bars_m5": _intraday_m5_bars(40),
                }

                response = predictor.predict(payload)
                with results_lock:
                    results.append(response)
        except Exception as exc:
            with results_lock:
                errors.append(exc)

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        threads.append(t)
        t.start()

    def do_hot_reload() -> None:
        """Perform hot-reload with mocked model constructors and bridge."""
        with (
            patch(
                "dqnpf.kubeflow.serving.dqnpf_predictor.DQNAdvisor",
                return_value=mock_dqn_new,
            ),
            patch(
                "dqnpf.kubeflow.serving.dqnpf_predictor.ForecasterInference",
                return_value=mock_forecaster_new,
            ),
            patch(
                "dqnpf.kubeflow.serving.dqnpf_predictor.ForecasterBridge",
                side_effect=_make_mock_bridge,
            ),
        ):
            predictor._hot_reload("/new/dqn/path.pt", "/new/fc/path.pt")

    barrier.wait()

    do_hot_reload()

    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Worker threads raised exceptions: {errors}"

    non_hold_actions = {r["action"] for r in results if r["action"] != 0}
    valid_non_hold = {1, 2}
    assert non_hold_actions.issubset(valid_non_hold), (
        f"Unexpected actions observed: {non_hold_actions}. "
        f"Expected only actions from old model (1) or new model (2)."
    )

    layer = predictor._layers["USDJPY"]
    assert layer.risk_long_used >= initial_risk_long, (
        f"Budget state lost: risk_long_used={layer.risk_long_used} < "
        f"initial={initial_risk_long}"
    )
    assert layer.risk_short_used >= initial_risk_short, (
        f"Budget state lost: risk_short_used={layer.risk_short_used} < "
        f"initial={initial_risk_short}"
    )


class _FakeInferOutput:
    """Records the InferOutput constructor args for assertions."""

    def __init__(self, name, shape, datatype, data):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.data = data


class _FakeInferResponse:
    def __init__(self, response_id, model_name, infer_outputs):
        self.response_id = response_id
        self.model_name = model_name
        self.infer_outputs = infer_outputs


def _patch_infer_types(monkeypatch):
    import kserve

    monkeypatch.setattr(kserve, "InferOutput", _FakeInferOutput, raising=False)
    monkeypatch.setattr(
        kserve, "InferResponse", _FakeInferResponse, raising=False
    )


def test_predict_grpc_infer_request_returns_infer_response(monkeypatch):
    """A v2 InferRequest with a BYTES 'symbol' input takes the sidecar-pull
    path and answers with output tensors mirroring the ScreenedAction dict."""
    from types import SimpleNamespace

    _patch_infer_types(monkeypatch)

    predictor, _ = _build_predictor(symbols=["USDJPY"])
    predictor, _, _, _ = _load_predictor_with_mocks(predictor, dqn_action=1)
    for bridge in predictor._bridges.values():
        bridge.compute_signal = MagicMock(return_value=(0.5, 1.0))
    for cache in predictor._caches.values():
        cache.invalidate()

    request = SimpleNamespace(
        id="req-42",
        inputs=[SimpleNamespace(name="symbol", data=[b"USDJPY"])],
    )

    response = predictor.predict(request)

    assert isinstance(response, _FakeInferResponse)
    assert response.response_id == "req-42"
    assert response.model_name == "test-predictor"
    outputs = {output.name: output for output in response.infer_outputs}
    assert set(outputs) == {
        "action",
        "action_name",
        "screened",
        "reason",
        "sigma",
        "risk_long_used",
        "risk_short_used",
        "mu",
    }
    assert outputs["action"].datatype == "INT64"
    assert outputs["action"].data[0] in _VALID_ACTIONS
    assert outputs["action_name"].datatype == "BYTES"
    assert outputs["reason"].data[0] in _VALID_REASONS
    assert all(output.shape == [1] for output in outputs.values())


def test_predict_grpc_infer_request_without_symbol_input_is_rejected(
    monkeypatch,
):
    from types import SimpleNamespace

    _patch_infer_types(monkeypatch)

    predictor, _ = _build_predictor(symbols=["USDJPY"])
    predictor, _, _, _ = _load_predictor_with_mocks(predictor)

    request = SimpleNamespace(
        id="", inputs=[SimpleNamespace(name="other", data=[b"x"])]
    )

    with pytest.raises(ValueError, match="input named 'symbol'"):
        predictor.predict(request)
