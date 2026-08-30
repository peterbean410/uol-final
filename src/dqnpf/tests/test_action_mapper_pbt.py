"""Property-based tests for the action mapper.

Feature: dqnpf, Property 2, Action-to-unit mapping correctness
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from dqnpf.action_mapper import (
    ACTION_MAP,
    Direction,
    map_action,
)


_EXPECTED: dict[int, tuple[Direction, int]] = {
    0: (Direction.NONE, 0),
    1: (Direction.LONG, 1),
    2: (Direction.LONG, 2),
    3: (Direction.SHORT, 1),
    4: (Direction.SHORT, 2),
}


@given(action_index=st.integers(min_value=0, max_value=4))
def test_valid_index_maps_to_expected_unit(action_index: int) -> None:
    unit = map_action(action_index)
    expected_direction, expected_risk = _EXPECTED[action_index]
    assert unit.direction == expected_direction
    assert unit.risk_units == expected_risk
    assert ACTION_MAP[action_index] is unit


@given(
    action_index=st.integers().filter(lambda i: i < 0 or i > 4),
)
def test_invalid_index_raises_value_error(action_index: int) -> None:
    with pytest.raises(ValueError):
        map_action(action_index)
