"""Unit tests for the trading strategy module.

Tests position sizing logic for all strategy classes on known-answer cases.
Validates: Requirements 9.1, 9.2, 9.3, 9.4
"""

import pytest

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.strategy import (
    BuyAndHoldBenchmark,
    DirectionalStrategy,
    MeanVarianceStrategy,
    MovingAverageBenchmark,
    TradingStrategy,
)


@pytest.fixture
def config() -> ForecasterConfig:
    """Default config with 10m position size and γ=0.05."""
    return ForecasterConfig()


class TestDirectionalStrategy:
    """Tests for DirectionalStrategy, Requirement 9.1."""

    def test_positive_mu_returns_long(self, config: ForecasterConfig):
        """Positive μ̂ should produce +10m position."""
        strategy = DirectionalStrategy()
        assert strategy.compute_position(0.001, 0.01, config) == 10_000_000

    def test_negative_mu_returns_short(self, config: ForecasterConfig):
        """Negative μ̂ should produce -10m position."""
        strategy = DirectionalStrategy()
        assert strategy.compute_position(-0.001, 0.01, config) == -10_000_000

    def test_zero_mu_returns_flat(self, config: ForecasterConfig):
        """Zero μ̂ should produce 0 position."""
        strategy = DirectionalStrategy()
        assert strategy.compute_position(0.0, 0.01, config) == 0.0

    def test_ignores_sigma(self, config: ForecasterConfig):
        """Position should not depend on sigma value."""
        strategy = DirectionalStrategy()
        assert strategy.compute_position(0.5, 0.0, config) == 10_000_000
        assert strategy.compute_position(0.5, 100.0, config) == 10_000_000

    def test_is_trading_strategy(self):
        """DirectionalStrategy should be a TradingStrategy."""
        assert issubclass(DirectionalStrategy, TradingStrategy)


class TestMeanVarianceStrategy:
    """Tests for MeanVarianceStrategy, Requirement 9.2."""

    def test_basic_computation(self, config: ForecasterConfig):
        """π* = μ / (σ² × γ) with known values."""
        # μ=0.001, σ=0.01, γ=0.05
        # π* = 0.001 / (0.0001 × 0.05) = 0.001 / 0.000005 = 200 → clipped to 1
        strategy = MeanVarianceStrategy()
        position = strategy.compute_position(0.001, 0.01, config)
        assert position == 10_000_000  # clipped to max

    def test_negative_mu_gives_short(self, config: ForecasterConfig):
        """Negative μ̂ should produce negative position."""
        # μ=-0.001, σ=0.01, γ=0.05
        # π* = -0.001 / (0.0001 × 0.05) = -200 → clipped to -1
        strategy = MeanVarianceStrategy()
        position = strategy.compute_position(-0.001, 0.01, config)
        assert position == -10_000_000

    def test_partial_position(self, config: ForecasterConfig):
        """When π* is between -1 and 1, position should be fractional."""
        # μ=0.0001, σ=0.1, γ=0.05
        # π* = 0.0001 / (0.01 × 0.05) = 0.0001 / 0.0005 = 0.2
        strategy = MeanVarianceStrategy()
        position = strategy.compute_position(0.0001, 0.1, config)
        assert position == pytest.approx(0.2 * 10_000_000)

    def test_sigma_zero_fallback_positive_mu(self, config: ForecasterConfig):
        """When σ=0, should fall back to directional: +position_size for positive μ."""
        strategy = MeanVarianceStrategy()
        position = strategy.compute_position(0.001, 0.0, config)
        assert position == 10_000_000

    def test_sigma_zero_fallback_negative_mu(self, config: ForecasterConfig):
        """When σ=0, should fall back to directional: -position_size for negative μ."""
        strategy = MeanVarianceStrategy()
        position = strategy.compute_position(-0.001, 0.0, config)
        assert position == -10_000_000

    def test_sigma_zero_fallback_zero_mu(self, config: ForecasterConfig):
        """When σ=0 and μ=0, should return 0."""
        strategy = MeanVarianceStrategy()
        position = strategy.compute_position(0.0, 0.0, config)
        assert position == 0.0

    def test_custom_risk_aversion(self, config: ForecasterConfig):
        """Custom γ should override config.risk_aversion."""
        # μ=0.0001, σ=0.1, γ=0.01
        # π* = 0.0001 / (0.01 × 0.01) = 0.0001 / 0.0001 = 1.0
        strategy = MeanVarianceStrategy(risk_aversion=0.01)
        position = strategy.compute_position(0.0001, 0.1, config)
        assert position == pytest.approx(1.0 * 10_000_000)

    def test_risk_aversion_0_1(self, config: ForecasterConfig):
        """γ=0.1 should produce smaller positions than γ=0.05."""
        # μ=0.0001, σ=0.1, γ=0.1
        # π* = 0.0001 / (0.01 × 0.1) = 0.0001 / 0.001 = 0.1
        strategy = MeanVarianceStrategy(risk_aversion=0.1)
        position = strategy.compute_position(0.0001, 0.1, config)
        assert position == pytest.approx(0.1 * 10_000_000)

    def test_clipping_upper_bound(self, config: ForecasterConfig):
        """Position should never exceed +position_size."""
        strategy = MeanVarianceStrategy()
        # Very large mu relative to sigma should clip to +1
        position = strategy.compute_position(1.0, 0.001, config)
        assert position == 10_000_000

    def test_clipping_lower_bound(self, config: ForecasterConfig):
        """Position should never be below -position_size."""
        strategy = MeanVarianceStrategy()
        # Very negative mu relative to sigma should clip to -1
        position = strategy.compute_position(-1.0, 0.001, config)
        assert position == -10_000_000

    def test_is_trading_strategy(self):
        """MeanVarianceStrategy should be a TradingStrategy."""
        assert issubclass(MeanVarianceStrategy, TradingStrategy)


class TestBuyAndHoldBenchmark:
    """Tests for BuyAndHoldBenchmark, Requirement 9.3."""

    def test_always_long(self, config: ForecasterConfig):
        """Should always return +position_size regardless of inputs."""
        strategy = BuyAndHoldBenchmark()
        assert strategy.compute_position(0.001, 0.01, config) == 10_000_000
        assert strategy.compute_position(-0.001, 0.01, config) == 10_000_000
        assert strategy.compute_position(0.0, 0.0, config) == 10_000_000

    def test_ignores_mu_and_sigma(self, config: ForecasterConfig):
        """Position should not depend on mu or sigma."""
        strategy = BuyAndHoldBenchmark()
        assert strategy.compute_position(999.0, 999.0, config) == 10_000_000
        assert strategy.compute_position(-999.0, 0.0, config) == 10_000_000

    def test_is_trading_strategy(self):
        """BuyAndHoldBenchmark should be a TradingStrategy."""
        assert issubclass(BuyAndHoldBenchmark, TradingStrategy)


class TestMovingAverageBenchmark:
    """Tests for MovingAverageBenchmark, Requirement 9.4."""

    def test_close_above_ma_returns_long(self, config: ForecasterConfig):
        """close ≥ MA(20) should produce +position_size."""
        strategy = MovingAverageBenchmark()
        strategy.set_market_data(close=150.50, ma20=150.00)
        assert strategy.compute_position(0.0, 0.0, config) == 10_000_000

    def test_close_equal_ma_returns_long(self, config: ForecasterConfig):
        """close == MA(20) should produce +position_size (≥ condition)."""
        strategy = MovingAverageBenchmark()
        strategy.set_market_data(close=150.00, ma20=150.00)
        assert strategy.compute_position(0.0, 0.0, config) == 10_000_000

    def test_close_below_ma_returns_short(self, config: ForecasterConfig):
        """close < MA(20) should produce -position_size."""
        strategy = MovingAverageBenchmark()
        strategy.set_market_data(close=149.50, ma20=150.00)
        assert strategy.compute_position(0.0, 0.0, config) == -10_000_000

    def test_raises_without_market_data(self, config: ForecasterConfig):
        """Should raise ValueError if set_market_data not called."""
        strategy = MovingAverageBenchmark()
        with pytest.raises(ValueError, match="Market data not set"):
            strategy.compute_position(0.0, 0.0, config)

    def test_compute_position_with_ma_convenience(self, config: ForecasterConfig):
        """compute_position_with_ma should work as a one-call interface."""
        strategy = MovingAverageBenchmark()
        position = strategy.compute_position_with_ma(close=151.0, ma20=150.0, config=config)
        assert position == 10_000_000

    def test_is_trading_strategy(self):
        """MovingAverageBenchmark should be a TradingStrategy."""
        assert issubclass(MovingAverageBenchmark, TradingStrategy)
