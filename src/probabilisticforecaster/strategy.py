"""Trading strategy engine for the Transformer Probabilistic Forex Forecaster.

Implements position sizing strategies that convert probabilistic forecasts (μ̂, σ̂)
into position sizes in base currency units.

Validates: Requirements 9.1, 9.2, 9.3, 9.4
"""

from abc import ABC, abstractmethod

from probabilisticforecaster.config import ForecasterConfig


class TradingStrategy(ABC):
    """Abstract base class for trading strategies.

    All strategies convert model predictions (mu, sigma) into a position size
    denominated in base currency units.
    """

    @abstractmethod
    def compute_position(self, mu: float, sigma: float, config: ForecasterConfig) -> float:
        """Return position size in base currency units.

        Args:
            mu: Predicted mean return (μ̂).
            sigma: Predicted standard deviation (σ̂).
            config: Forecaster configuration with position_size and risk_aversion.

        Returns:
            Position size in base currency units. Positive = long, negative = short.
        """


class DirectionalStrategy(TradingStrategy):
    """Strategy 1: Directional no-scaling.

    Takes a full long position if μ̂ > 0, full short if μ̂ < 0, flat if μ̂ == 0.

    Validates: Requirement 9.1
    """

    def compute_position(self, mu: float, sigma: float, config: ForecasterConfig) -> float:
        """Compute position based on sign of predicted mean.

        Args:
            mu: Predicted mean return (μ̂).
            sigma: Predicted standard deviation (σ̂), unused by this strategy.
            config: Forecaster configuration with position_size.

        Returns:
            +position_size if μ̂ > 0, -position_size if μ̂ < 0, 0 if μ̂ == 0.
        """
        if mu > 0:
            return config.position_size
        elif mu < 0:
            return -config.position_size
        else:
            return 0.0


class MeanVarianceStrategy(TradingStrategy):
    """Strategies 2-4: Mean-variance scaling with configurable risk aversion.

    Computes optimal proportion π* = μ̂ / (σ̂² × γ), then clips to [-1, 1]
    and scales by position_size.

    When σ̂ = 0, falls back to directional strategy to avoid division by zero.

    Validates: Requirement 9.2
    """

    def __init__(self, risk_aversion: float | None = None):
        """Initialize with optional risk aversion override.

        Args:
            risk_aversion: Override γ value. If None, uses config.risk_aversion.
                Common values: 0.01, 0.05, 0.1.
        """
        self._risk_aversion = risk_aversion

    def compute_position(self, mu: float, sigma: float, config: ForecasterConfig) -> float:
        """Compute position using mean-variance optimization.

        Formula: π* = μ̂ / (σ̂² × γ), position = clip(π*, -1, 1) × position_size

        When σ̂ = 0, falls back to directional strategy (±position_size based on sign of μ̂).

        Args:
            mu: Predicted mean return (μ̂).
            sigma: Predicted standard deviation (σ̂).
            config: Forecaster configuration with position_size and risk_aversion.

        Returns:
            Position size in base currency units, clipped to [-position_size, +position_size].
        """
        gamma = self._risk_aversion if self._risk_aversion is not None else config.risk_aversion

        if sigma == 0.0:
            if mu > 0:
                return config.position_size
            elif mu < 0:
                return -config.position_size
            else:
                return 0.0

        pi_star = mu / (sigma**2 * gamma)

        pi_clipped = max(-1.0, min(1.0, pi_star))
        return pi_clipped * config.position_size


class BuyAndHoldBenchmark(TradingStrategy):
    """Benchmark 1: Buy-and-hold.

    Always holds a full long position regardless of model predictions.

    Validates: Requirement 9.3
    """

    def compute_position(self, mu: float, sigma: float, config: ForecasterConfig) -> float:
        """Return full long position always.

        Args:
            mu: Predicted mean return (μ̂), unused.
            sigma: Predicted standard deviation (σ̂), unused.
            config: Forecaster configuration with position_size.

        Returns:
            +position_size always.
        """
        return config.position_size


class MovingAverageBenchmark(TradingStrategy):
    """Benchmark 2: Moving average crossover MA(20).

    Goes long if close ≥ MA(20), short otherwise. The close and ma20 values
    must be provided via the compute_position_with_ma method or passed as
    keyword arguments.

    Validates: Requirement 9.4
    """

    def __init__(self) -> None:
        """Initialize with no stored state."""
        self._close: float | None = None
        self._ma20: float | None = None

    def set_market_data(self, close: float, ma20: float) -> None:
        """Set the current close price and MA(20) value for position computation.

        Args:
            close: Current close price.
            ma20: 20-period moving average of close prices.
        """
        self._close = close
        self._ma20 = ma20

    def compute_position(self, mu: float, sigma: float, config: ForecasterConfig) -> float:
        """Compute position based on close vs MA(20) crossover.

        Requires set_market_data() to be called first with current close and ma20.

        Args:
            mu: Predicted mean return (μ̂), unused by this benchmark.
            sigma: Predicted standard deviation (σ̂), unused by this benchmark.
            config: Forecaster configuration with position_size.

        Returns:
            +position_size if close ≥ MA(20), -position_size otherwise.

        Raises:
            ValueError: If set_market_data() has not been called.
        """
        if self._close is None or self._ma20 is None:
            raise ValueError(
                "Market data not set. Call set_market_data(close, ma20) before compute_position."
            )

        if self._close >= self._ma20:
            return config.position_size
        else:
            return -config.position_size

    def compute_position_with_ma(
        self, close: float, ma20: float, config: ForecasterConfig
    ) -> float:
        """Convenience method to compute position with close and MA(20) directly.

        Args:
            close: Current close price.
            ma20: 20-period moving average of close prices.
            config: Forecaster configuration with position_size.

        Returns:
            +position_size if close ≥ MA(20), -position_size otherwise.
        """
        self.set_market_data(close, ma20)
        return self.compute_position(0.0, 0.0, config)
