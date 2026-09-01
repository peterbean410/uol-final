"""Tests for adaptive reward normalisation (scale-only, sign-preserving)."""

from __future__ import annotations

import numpy as np
import torch

from deepqnetwork.agent import DQNAgent, _RunningRMS
from deepqnetwork.config import DQNConfig


def test_running_rms_matches_numpy():
    rms = _RunningRMS()
    assert rms.rms == 1.0
    a = torch.tensor([3.0, -4.0])
    rms.update(a)
    assert abs(rms.rms - (12.5 ** 0.5)) < 1e-9
    assert rms.count == 2
    b = torch.tensor([0.0, 5.0, -1.0])
    rms.update(b)
    allv = np.array([3.0, -4.0, 0.0, 5.0, -1.0])
    assert abs(rms.rms - float(np.sqrt((allv ** 2).mean()))) < 1e-9
    assert rms.count == 5


def _agent() -> DQNAgent:
    cfg = DQNConfig(
        batch_size=8,
        replay_buffer_size=1000,
        hidden_dims=[16, 16],
    )
    agent = DQNAgent(cfg, torch.device("cpu"), state_dim=53)
    agent._reward_norm_warmup = 8
    return agent


def _fill(agent: DQNAgent, reward: float, n: int = 64) -> None:
    for _ in range(n):
        s = np.zeros(53, dtype=np.float32)
        agent.replay_buffer.push(s, 0, reward, s, False)


def test_update_tracks_reward_rms():
    agent = _agent()
    _fill(agent, reward=5.0)
    for _ in range(3):
        agent.update()
    assert agent.reward_rms.count >= 8
    assert abs(agent.reward_rms.rms - 5.0) < 1e-6

