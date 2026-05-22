"""Action-to-unit mapping for the integration layer.

DQN action indices are never used directly as unit counts. This module
defines the explicit static lookup from index to (direction, risk_units).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    NONE = "none"
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class ActionUnit:
    direction: Direction
    risk_units: int


ACTION_MAP: dict[int, ActionUnit] = {
    0: ActionUnit(Direction.NONE, 0),
    1: ActionUnit(Direction.LONG, 1),
    2: ActionUnit(Direction.LONG, 2),
    3: ActionUnit(Direction.SHORT, 1),
    4: ActionUnit(Direction.SHORT, 2),
}

ACTION_NAMES: list[str] = ["HOLD", "BUY_1", "BUY_2", "SELL_1", "SELL_2"]


def map_action(action_index: int) -> ActionUnit:
    """Map a DQN action index to its direction and risk units.

    Raises:
        ValueError: If action_index is not in [0, 4].
    """
    if action_index not in ACTION_MAP:
        raise ValueError(f"Invalid action index: {action_index}")
    return ACTION_MAP[action_index]
