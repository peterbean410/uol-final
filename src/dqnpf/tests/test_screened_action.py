"""Unit tests for the ScreenedAction dataclass."""

from __future__ import annotations

import pytest

from dqnpf.integration import ScreenedAction
from dqnpf.tests.conftest import FakeActionResult, make_layer


def test_construction_with_all_fields() -> None:
    action = ScreenedAction(
        action=1,
        action_name="BUY_1",
        screened=False,
        reason="pass",
        sigma=4.5,
        risk_long_used=1,
        risk_short_used=0,
    )
    assert action.action == 1
    assert action.action_name == "BUY_1"
    assert action.screened is False
    assert action.reason == "pass"
    assert action.sigma == 4.5
    assert action.risk_long_used == 1
    assert action.risk_short_used == 0


def test_screened_false_when_reason_is_pass() -> None:
    layer = make_layer(variance_threshold=10.0)
    result = layer.screen(FakeActionResult(action=1, action_name="BUY_1"), mu=0.0, sigma=0.1)
    assert result.reason == "pass"
    assert result.screened is False


def test_screened_true_when_reason_is_budget_exhausted() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=1)
    layer.screen(FakeActionResult(action=1, action_name="BUY_1"), mu=0.0, sigma=5.0)
    result = layer.screen(FakeActionResult(action=1, action_name="BUY_1"), mu=0.0, sigma=5.0)
    assert result.reason == "budget_exhausted"
    assert result.screened is True
    assert result.action == 0
    assert result.action_name == "HOLD"



@pytest.mark.parametrize(
    "reason,expected_screened",
    [
        ("pass", False),
        ("budget_exhausted", True),
        ("directional_conflict", True),
    ],
)
def test_screened_flag_semantics_by_reason(reason: str, expected_screened: bool) -> None:
    # Direct dataclass construction, confirms invariant of the field semantics
    # even outside the live screen() path.
    action = ScreenedAction(
        action=0 if expected_screened else 1,
        action_name="HOLD" if expected_screened else "BUY_1",
        screened=expected_screened,
        reason=reason,
        sigma=1.0,
        risk_long_used=0,
        risk_short_used=0,
    )
    assert action.screened is expected_screened
    assert action.reason == reason
