"""Property-based tests for DQN Backtest component metrics and degradation gate.

Uses Hypothesis to verify correctness properties across randomly generated
episode results and backtest metrics.

**Validates: Requirements DQN-R9, DQN-R10**
"""

from __future__ import annotations

import math

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from deepqnetwork.kubeflow.components.dqn_backtest.component import (
    BacktestMetrics,
    EpisodeResult,
    compute_backtest_metrics,
    degradation_gate,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def episode_results_strategy(draw):
    """Generate a list of EpisodeResult instances with realistic value ranges."""
    num_episodes = draw(st.integers(min_value=1, max_value=50))
    results = []
    for _ in range(num_episodes):
        num_trades = draw(st.integers(min_value=0, max_value=200))
        winning_trades = draw(st.integers(min_value=0, max_value=num_trades))
        results.append(
            EpisodeResult(
                total_reward=draw(
                    st.floats(
                        min_value=-100.0,
                        max_value=100.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                ),
                cumulative_pnl=draw(
                    st.floats(
                        min_value=-10.0,
                        max_value=10.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                ),
                num_steps=draw(st.integers(min_value=1, max_value=30_000)),
                num_trades=num_trades,
                winning_trades=winning_trades,
            )
        )
    return results


@st.composite
def backtest_metrics_strategy(draw):
    """Generate BacktestMetrics instances with arbitrary finite values."""
    return BacktestMetrics(
        cumulative_pnl=draw(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        sharpe_ratio=draw(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        max_drawdown=draw(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        win_rate=draw(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        avg_episode_reward=draw(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        avg_episode_length=draw(
            st.floats(
                min_value=1.0,
                max_value=30_000.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
    )


# ---------------------------------------------------------------------------
# Property DQN-7: Backtest metric bounds
# ---------------------------------------------------------------------------


class TestBacktestMetricBounds:
    """Property DQN-7: Backtest metric bounds.

    For any valid list of EpisodeResult instances, compute_backtest_metrics()
    produces metrics where Sharpe ratio is finite, max drawdown is in [0, 1],
    and win rate is in [0, 1].

    **Validates: Requirements DQN-R9, DQN-R10**
    """

    @given(episode_results=episode_results_strategy())
    @settings(max_examples=100, deadline=None)
    def test_sharpe_ratio_is_finite(self, episode_results: list[EpisodeResult]):
        """For any valid episode results, the computed Sharpe ratio is finite.

        **Validates: Requirements DQN-R10**
        """
        metrics = compute_backtest_metrics(episode_results)
        assert math.isfinite(metrics.sharpe_ratio), (
            f"Sharpe ratio is not finite: {metrics.sharpe_ratio}. "
            f"Episode count: {len(episode_results)}"
        )

    @given(episode_results=episode_results_strategy())
    @settings(max_examples=100, deadline=None)
    def test_max_drawdown_in_bounds(self, episode_results: list[EpisodeResult]):
        """For any valid episode results, max drawdown is in [0, 1].

        **Validates: Requirements DQN-R10**
        """
        metrics = compute_backtest_metrics(episode_results)
        assert 0.0 <= metrics.max_drawdown <= 1.0, (
            f"Max drawdown out of bounds: {metrics.max_drawdown}. "
            f"Expected [0, 1]. Episode count: {len(episode_results)}"
        )

    @given(episode_results=episode_results_strategy())
    @settings(max_examples=100, deadline=None)
    def test_win_rate_in_bounds(self, episode_results: list[EpisodeResult]):
        """For any valid episode results, win rate is in [0, 1].

        **Validates: Requirements DQN-R10**
        """
        metrics = compute_backtest_metrics(episode_results)
        assert 0.0 <= metrics.win_rate <= 1.0, (
            f"Win rate out of bounds: {metrics.win_rate}. "
            f"Expected [0, 1]. Episode count: {len(episode_results)}"
        )


# ---------------------------------------------------------------------------
# Property DQN-8: Degradation gate blocks promotion
# ---------------------------------------------------------------------------


class TestDegradationGateBlocksPromotion:
    """Property DQN-8: Degradation gate blocks promotion.

    When Sharpe degrades beyond relative threshold, P&L flips negative vs
    positive production, Sharpe is below absolute 1.0, or P&L is <= 0 absolute,
    gate_passed is False.

    **Validates: Requirements DQN-R9, DQN-R10**
    """

    @given(
        prod_sharpe=st.floats(
            min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        degradation=st.floats(
            min_value=0.11, max_value=5.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_sharpe_degradation_blocks_promotion(
        self, prod_sharpe: float, degradation: float
    ):
        """(a) When Sharpe degrades beyond relative threshold, gate_passed is False.

        **Validates: Requirements DQN-R9**
        """
        # Current Sharpe is lower than production by more than the threshold (0.1)
        current_sharpe = prod_sharpe - degradation
        assume(degradation > 0.1)  # Ensure it exceeds the default threshold

        current_metrics = BacktestMetrics(
            cumulative_pnl=100.0,  # Profitable to avoid triggering other gates
            sharpe_ratio=current_sharpe,
            max_drawdown=0.1,
            win_rate=0.6,
            avg_episode_reward=1.0,
            avg_episode_length=1000.0,
        )
        production_metrics = {
            "sharpe_ratio": prod_sharpe,
            "cumulative_pnl": 50.0,
        }

        # Use a high absolute threshold that won't trigger (current_sharpe could be high)
        # We specifically test the relative degradation check here
        gate_passed, reason = degradation_gate(
            current_metrics=current_metrics,
            production_metrics=production_metrics,
            sharpe_degradation_threshold=0.1,
            sharpe_absolute_threshold=-100.0,  # Disable absolute check
            pnl_absolute_threshold=-1000.0,  # Disable absolute P&L check
        )

        assert gate_passed is False, (
            f"Gate should have blocked promotion due to Sharpe degradation. "
            f"current_sharpe={current_sharpe:.4f}, prod_sharpe={prod_sharpe:.4f}, "
            f"delta={prod_sharpe - current_sharpe:.4f}, reason={reason}"
        )

    @given(
        current_pnl=st.floats(
            min_value=-100.0, max_value=-0.001, allow_nan=False, allow_infinity=False
        ),
        prod_pnl=st.floats(
            min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_pnl_sign_flip_blocks_promotion(
        self, current_pnl: float, prod_pnl: float
    ):
        """(b) When P&L < 0 while production P&L > 0, gate_passed is False.

        **Validates: Requirements DQN-R9**
        """
        current_metrics = BacktestMetrics(
            cumulative_pnl=current_pnl,
            sharpe_ratio=5.0,  # High Sharpe to avoid triggering other gates
            max_drawdown=0.1,
            win_rate=0.6,
            avg_episode_reward=1.0,
            avg_episode_length=1000.0,
        )
        production_metrics = {
            "sharpe_ratio": 2.0,  # Lower than current to avoid relative check
            "cumulative_pnl": prod_pnl,
        }

        gate_passed, reason = degradation_gate(
            current_metrics=current_metrics,
            production_metrics=production_metrics,
            sharpe_degradation_threshold=0.1,
            sharpe_absolute_threshold=-100.0,  # Disable absolute Sharpe check
            pnl_absolute_threshold=-1000.0,  # Disable absolute P&L check
        )

        assert gate_passed is False, (
            f"Gate should have blocked promotion due to P&L sign flip. "
            f"current_pnl={current_pnl:.6f}, prod_pnl={prod_pnl:.6f}, "
            f"reason={reason}"
        )

    @given(
        current_sharpe=st.floats(
            min_value=-10.0, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_sharpe_below_absolute_threshold_blocks_promotion(
        self, current_sharpe: float
    ):
        """(c) When Sharpe < 1.0 absolute, gate_passed is False.

        **Validates: Requirements DQN-R10**
        """
        assume(current_sharpe < 1.0)

        current_metrics = BacktestMetrics(
            cumulative_pnl=100.0,  # Profitable to avoid triggering P&L gates
            sharpe_ratio=current_sharpe,
            max_drawdown=0.1,
            win_rate=0.6,
            avg_episode_reward=1.0,
            avg_episode_length=1000.0,
        )
        production_metrics = {
            "sharpe_ratio": current_sharpe - 1.0,  # Lower than current to avoid relative check
            "cumulative_pnl": 50.0,
        }

        gate_passed, reason = degradation_gate(
            current_metrics=current_metrics,
            production_metrics=production_metrics,
            sharpe_degradation_threshold=0.1,
            sharpe_absolute_threshold=1.0,
            pnl_absolute_threshold=-1000.0,  # Disable absolute P&L check
        )

        assert gate_passed is False, (
            f"Gate should have blocked promotion due to Sharpe below absolute threshold. "
            f"current_sharpe={current_sharpe:.4f}, threshold=1.0, reason={reason}"
        )

    @given(
        current_pnl=st.floats(
            min_value=-100.0, max_value=0.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_pnl_at_or_below_zero_blocks_promotion(self, current_pnl: float):
        """(d) When P&L <= 0 absolute, gate_passed is False.

        **Validates: Requirements DQN-R10**
        """
        assume(current_pnl <= 0.0)

        current_metrics = BacktestMetrics(
            cumulative_pnl=current_pnl,
            sharpe_ratio=5.0,  # High Sharpe to avoid triggering Sharpe gates
            max_drawdown=0.1,
            win_rate=0.6,
            avg_episode_reward=1.0,
            avg_episode_length=1000.0,
        )
        production_metrics = {
            "sharpe_ratio": 2.0,  # Lower than current to avoid relative check
            "cumulative_pnl": -50.0,  # Negative to avoid P&L sign flip check
        }

        gate_passed, reason = degradation_gate(
            current_metrics=current_metrics,
            production_metrics=production_metrics,
            sharpe_degradation_threshold=0.1,
            sharpe_absolute_threshold=-100.0,  # Disable absolute Sharpe check
            pnl_absolute_threshold=0.0,
        )

        assert gate_passed is False, (
            f"Gate should have blocked promotion due to P&L <= 0 absolute. "
            f"current_pnl={current_pnl:.6f}, threshold=0.0, reason={reason}"
        )
