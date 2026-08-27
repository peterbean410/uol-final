"""Unit tests for the live execution loop (Reset-once / Step-forever)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tradingmodel.intraday.dqnpf.live_executor import (
    LiveExecutorConfig,
    decode_infer_response,
    parse_screened_action,
    run_live_executor,
)


class FakeEnvClient:
    """Records reset/step calls; optionally fails step a set number of times."""

    def __init__(self, step_failures: int = 0) -> None:
        self.reset_calls: list[dict] = []
        self.step_calls: list[tuple[int, str]] = []
        self._step_failures = step_failures

    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        return object()

    def step(self, action: int, client_order_id: str):
        if self._step_failures > 0:
            self._step_failures -= 1
            raise ConnectionError("sidecar unavailable")
        self.step_calls.append((action, client_order_id))
        return object()


def fast_config(**overrides) -> LiveExecutorConfig:
    defaults = dict(symbol="USDJPY", step_interval_seconds=0.0)
    defaults.update(overrides)
    return LiveExecutorConfig(**defaults)


def test_resets_once_then_steps_screened_actions():
    env = FakeEnvClient()
    responses = iter(
        [
            {"action": 1, "action_name": "BUY_1", "reason": "pass"},
            {"action": 0, "action_name": "HOLD", "reason": "budget_exhausted"},
            {"action": 3, "action_name": "SELL_1", "reason": "pass"},
        ]
    )

    steps = run_live_executor(
        fast_config(),
        env_client=env,
        predict_fn=lambda symbol: next(responses),
        max_steps=3,
    )

    assert steps == 3
    assert len(env.reset_calls) == 1
    assert env.reset_calls[0]["symbol"] == "USDJPY"
    assert [action for action, _ in env.step_calls] == [1, 0, 3]


def test_hold_actions_are_still_stepped():
    """HOLD must reach Step(): modelenv handles it locally but the step still
    re-syncs positions and runs the session-liquidation boundary check."""
    env = FakeEnvClient()

    run_live_executor(
        fast_config(),
        env_client=env,
        predict_fn=lambda symbol: {"action": 0, "action_name": "HOLD"},
        max_steps=2,
    )

    assert [action for action, _ in env.step_calls] == [0, 0]


def test_client_order_ids_are_unique_and_symbol_prefixed():
    env = FakeEnvClient()

    run_live_executor(
        fast_config(),
        env_client=env,
        predict_fn=lambda symbol: {"action": 1},
        max_steps=5,
    )

    order_ids = [order_id for _, order_id in env.step_calls]
    assert len(set(order_ids)) == 5
    assert all(order_id.startswith("USDJPY-") for order_id in order_ids)


def test_aborts_after_max_consecutive_errors():
    env = FakeEnvClient(step_failures=3)

    with pytest.raises(RuntimeError, match="3 consecutive"):
        run_live_executor(
            fast_config(max_consecutive_errors=3),
            env_client=env,
            predict_fn=lambda symbol: {"action": 1},
            max_steps=10,
        )


def test_error_counter_resets_on_success():
    env = FakeEnvClient(step_failures=2)

    steps = run_live_executor(
        fast_config(max_consecutive_errors=3),
        env_client=env,
        predict_fn=lambda symbol: {"action": 1},
        max_steps=4,
    )

    # 2 failed iterations then 4 successful ones; no abort.
    assert steps == 4
    assert len(env.step_calls) == 4


def test_stop_event_exits_cleanly():
    env = FakeEnvClient()
    stop_event = threading.Event()

    def predict_then_stop(symbol):
        stop_event.set()
        return {"action": 1}

    steps = run_live_executor(
        fast_config(),
        env_client=env,
        predict_fn=predict_then_stop,
        stop_event=stop_event,
    )

    assert steps == 1
    assert len(env.step_calls) == 1


def test_parse_screened_action_accepts_raw_and_v1_envelope():
    raw = {"action": 3, "action_name": "SELL_1"}
    assert parse_screened_action(raw)["action"] == 3
    wrapped = {"predictions": [{"action": 2, "action_name": "BUY_2"}]}
    assert parse_screened_action(wrapped)["action"] == 2


def test_parse_screened_action_rejects_actionless_payloads():
    with pytest.raises(ValueError, match="no 'action' field"):
        parse_screened_action({"predictions": []})
    with pytest.raises(ValueError, match="no 'action' field"):
        parse_screened_action({"detail": "error"})


def test_config_defaults_use_grpc_and_60s_interval():
    config = LiveExecutorConfig()

    assert config.predictor_grpc_address == "localhost:8081"
    assert config.model_name == "dqnpf-intraday"
    assert config.step_interval_seconds == 60.0


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("DQNPF_SYMBOL", "EURUSD")
    monkeypatch.setenv("DQNPF_PREDICTOR_GRPC_ADDRESS", "predictor:8081")
    monkeypatch.setenv("DQNPF_MODEL_NAME", "dqnpf-eurusd")
    monkeypatch.setenv("GRPC_ADDRESS", "modelenv:50051")
    monkeypatch.setenv("DQNPF_STEP_INTERVAL_SECONDS", "120")
    monkeypatch.setenv("DQNPF_DRY_RUN", "true")
    monkeypatch.setenv("DQNPF_MAX_CONSECUTIVE_ERRORS", "5")

    config = LiveExecutorConfig.from_env()

    assert config.symbol == "EURUSD"
    assert config.predictor_grpc_address == "predictor:8081"
    assert config.model_name == "dqnpf-eurusd"
    assert config.grpc_address == "modelenv:50051"
    assert config.step_interval_seconds == 120.0
    assert config.max_consecutive_errors == 5


def _output_tensor(name: str, **contents) -> SimpleNamespace:
    base: dict = dict(
        int64_contents=[],
        fp64_contents=[],
        bool_contents=[],
        bytes_contents=[],
        fp32_contents=[],
        int_contents=[],
        uint_contents=[],
        uint64_contents=[],
    )
    base.update(contents)
    return SimpleNamespace(name=name, contents=SimpleNamespace(**base))


def test_decode_infer_response_reads_typed_contents():
    response = SimpleNamespace(
        outputs=[
            _output_tensor("action", int64_contents=[3]),
            _output_tensor("action_name", bytes_contents=[b"SELL_1"]),
            _output_tensor("screened", bool_contents=[False]),
            _output_tensor("reason", bytes_contents=[b"pass"]),
            _output_tensor("sigma", fp64_contents=[0.25]),
        ]
    )

    decoded = decode_infer_response(response)

    assert decoded == {
        "action": 3,
        "action_name": "SELL_1",
        "screened": False,
        "reason": "pass",
        "sigma": 0.25,
    }


def test_decoded_response_feeds_parse_screened_action():
    response = SimpleNamespace(
        outputs=[_output_tensor("action", int64_contents=[1])]
    )

    assert parse_screened_action(decode_infer_response(response))["action"] == 1
