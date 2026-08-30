"""Unit tests for the action mapper (deterministic, per-index)."""

from __future__ import annotations

import pytest

from dqnpf.action_mapper import (
    ACTION_NAMES,
    Direction,
    map_action,
)


def test_hold_maps_to_none_zero_units() -> None:
    unit = map_action(0)
    assert unit.direction == Direction.NONE
    assert unit.risk_units == 0


def test_buy_1_maps_to_long_one_unit() -> None:
    unit = map_action(1)
    assert unit.direction == Direction.LONG
    assert unit.risk_units == 1


def test_buy_2_maps_to_long_two_units() -> None:
    unit = map_action(2)
    assert unit.direction == Direction.LONG
    assert unit.risk_units == 2


def test_sell_1_maps_to_short_one_unit() -> None:
    unit = map_action(3)
    assert unit.direction == Direction.SHORT
    assert unit.risk_units == 1


def test_sell_2_maps_to_short_two_units() -> None:
    unit = map_action(4)
    assert unit.direction == Direction.SHORT
    assert unit.risk_units == 2


def test_index_minus_one_raises_value_error() -> None:
    with pytest.raises(ValueError, match="-1"):
        map_action(-1)


def test_index_five_raises_value_error() -> None:
    with pytest.raises(ValueError, match="5"):
        map_action(5)


def test_action_names_align_with_indices() -> None:
    assert ACTION_NAMES == ["HOLD", "BUY_1", "BUY_2", "SELL_1", "SELL_2"]


def test_action_unit_is_frozen() -> None:
    unit = map_action(1)
    with pytest.raises(Exception):
        unit.risk_units = 99  # type: ignore[misc]
