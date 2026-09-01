"""Integration tests for the Transformer Probabilistic Forex Forecaster.

Tests end-to-end pipeline flow, model save/load round-trip, and full
strategy backtest integration with synthetic data.

Validates: All Requirements
"""

import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch

from probabilisticforecaster.backtest import BacktestResult, run_backtest
from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.dataset import ForexDataset
from probabilisticforecaster.evaluation import EvaluationMetrics, evaluate_model
from probabilisticforecaster.features import compute_features
from probabilisticforecaster.inference import ForecasterInference
from probabilisticforecaster.model import ProbabilisticTransformer
from probabilisticforecaster.strategy import (
    DirectionalStrategy,
    MeanVarianceStrategy,
)
from probabilisticforecaster.training import train_model


def _generate_synthetic_ohlc(n_bars: int = 1600, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 5-min OHLC data resembling USDJPY.

    Creates a random walk around 150.0 with realistic spreads.

    Args:
        n_bars: Number of bars to generate (must be > 1440 for feature computation).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns [Timestamp, Symbol, Open, High, Low, Close, Volume].
    """
    rng = np.random.default_rng(seed)

    returns = rng.normal(0, 0.0001, size=n_bars)
    close = 150.0 * np.exp(np.cumsum(returns))

    spread = rng.uniform(0.005, 0.02, size=n_bars)
    high = close + spread
    low = close - spread
    open_price = close + rng.normal(0, 0.005, size=n_bars)

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    timestamps = pd.date_range(
        start="2023-01-02 00:00", periods=n_bars, freq="5min", tz="UTC"
    )

    volume = rng.integers(100, 5000, size=n_bars)

    df = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Symbol": "USDJPY",
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )
    return df


class TestEndToEndPipeline:
    """Test the full pipeline: synthetic data → features → model → prediction → strategy → PnL."""

    def test_end_to_end_synthetic_data(self):
        """Full pipeline produces valid predictions and PnL from synthetic data."""
        torch.manual_seed(42)

        ohlc_df = _generate_synthetic_ohlc(n_bars=1600)
        assert len(ohlc_df) == 1600

        features_df = compute_features(ohlc_df, historical_window=1440)
        assert features_df.shape[1] == 16
        assert len(features_df) > 0
        assert len(features_df) == 160

        close_prices = pd.Series(
            ohlc_df.set_index(pd.to_datetime(ohlc_df["Timestamp"], utc=True))["Close"]
        ).reindex(features_df.index)

        config = ForecasterConfig()
        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=config.lookback_window,
            horizon=config.forecast_horizon,
        )
        assert len(dataset) > 0

        features_sample, label_sample = dataset[0]
        assert features_sample.shape == (36, 16)
        assert label_sample.shape == (1,)

        model = ProbabilisticTransformer(config)
        model.eval()

        with torch.no_grad():
            x = features_sample.unsqueeze(0)
            mu, sigma = model(x)

        assert mu.shape == (1, 36, 1)
        assert sigma.shape == (1, 36, 1)

        assert (sigma > 0).all(), "All sigma values must be strictly positive"

        mu_val = mu[0, -1, 0].item()
        sigma_val = sigma[0, -1, 0].item()
        assert sigma_val > 0

        strategy = DirectionalStrategy()
        position = strategy.compute_position(mu_val, sigma_val, config)
        assert position != 0.0 or mu_val == 0.0
        assert abs(position) <= config.position_size

        actual_return = label_sample[0].item()
        pnl = position * actual_return

        assert np.isfinite(pnl), f"PnL must be finite, got {pnl}"

    def test_batch_forward_pass_shapes(self):
        """Model produces correct shapes for batch input from dataset."""
        torch.manual_seed(42)

        ohlc_df = _generate_synthetic_ohlc(n_bars=1600)
        features_df = compute_features(ohlc_df, historical_window=1440)
        close_prices = pd.Series(
            ohlc_df.set_index(pd.to_datetime(ohlc_df["Timestamp"], utc=True))["Close"]
        ).reindex(features_df.index)

        config = ForecasterConfig()
        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=config.lookback_window,
            horizon=config.forecast_horizon,
        )

        batch_size = min(4, len(dataset))
        batch_features = torch.stack([dataset[i][0] for i in range(batch_size)])
        assert batch_features.shape == (batch_size, 36, 16)

        model = ProbabilisticTransformer(config)
        model.eval()

        with torch.no_grad():
            mu, sigma = model(batch_features)

        assert mu.shape == (batch_size, 36, 1)
        assert sigma.shape == (batch_size, 36, 1)
        assert (sigma > 0).all()

    @patch("probabilisticforecaster.training._upload_model_to_s3")
    def test_full_train_evaluate_backtest_pipeline(self, mock_upload, tmp_path):
        """Full pipeline: synthetic data → features → dataset → train → evaluate → backtest."""
        torch.manual_seed(42)

        ohlc_df = _generate_synthetic_ohlc(n_bars=1600, seed=99)

        features_df = compute_features(ohlc_df, historical_window=1440)
        assert features_df.shape[1] == 16

        close_prices = pd.Series(
            ohlc_df.set_index(pd.to_datetime(ohlc_df["Timestamp"], utc=True))["Close"]
        ).reindex(features_df.index)

        config = ForecasterConfig(
            epochs=2,
            batch_size=32,
            model_path=str(tmp_path / "integration_model.pt"),
        )
        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=config.lookback_window,
            horizon=config.forecast_horizon,
        )
        assert len(dataset) > 0

        n_total = len(dataset)
        n_train = int(n_total * 0.8)
        train_dataset = torch.utils.data.Subset(dataset, list(range(n_train)))
        test_indices = list(range(n_train, n_total))

        test_start_idx = n_train
        test_features_df = features_df.iloc[test_start_idx:]
        test_close_prices = close_prices.iloc[test_start_idx:]
        test_dataset = ForexDataset(
            features_df=test_features_df,
            close_prices=test_close_prices,
            lookback=config.lookback_window,
            horizon=config.forecast_horizon,
        )

        model = ProbabilisticTransformer(config)
        history = train_model(
            model=model,
            train_dataset=dataset,
            config=config,
            upload_to_s3=False,
        )

        assert len(history["epoch_loss"]) == config.epochs
        assert all(np.isfinite(loss) for loss in history["epoch_loss"])
        assert len(history["batch_losses"]) > 0

        metrics = evaluate_model(model, test_dataset, config)
        assert isinstance(metrics, EvaluationMetrics)
        assert np.isfinite(metrics.nll)
        assert 0.0 <= metrics.directional_accuracy <= 1.0
        assert 0.0 <= metrics.covered_ratio_95 <= 1.0
        assert metrics.rmse >= 0.0

        model.eval()
        predictions_list = []
        from torch.utils.data import DataLoader

        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        sample_idx = 0
        with torch.no_grad():
            for features_batch, labels_batch in test_loader:
                mu_batch, sigma_batch = model(features_batch)
                mu_last = mu_batch[:, -1, 0]
                sigma_last = sigma_batch[:, -1, 0]
                for i in range(len(mu_last)):
                    ds_idx = test_dataset.valid_indices[sample_idx]
                    ts = test_dataset.timestamps[ds_idx + test_dataset.lookback - 1]
                    predictions_list.append({
                        "timestamp": ts,
                        "mu": mu_last[i].item(),
                        "sigma": sigma_last[i].item(),
                    })
                    sample_idx += 1

        predictions_df = pd.DataFrame(predictions_list)

        price_timestamps = test_features_df.index.tolist()
        prices_df = pd.DataFrame({
            "timestamp": price_timestamps,
            "close": test_close_prices.values,
        })

        strategy = DirectionalStrategy()
        result = run_backtest(predictions_df, prices_df, strategy, config)

        assert isinstance(result, BacktestResult)
        assert np.isfinite(result.annualised_return)
        assert np.isfinite(result.sharpe_ratio)
        assert result.max_drawdown >= 0.0


class TestModelSaveLoadRoundTrip:
    """Test that model save/load round-trip produces identical predictions."""

    def test_save_load_identical_predictions(self, tmp_path):
        """Saved and reloaded model produces bitwise identical predictions."""
        torch.manual_seed(42)

        config = ForecasterConfig()
        model = ProbabilisticTransformer(config)
        model.eval()

        x = torch.randn(4, 36, 16)

        with torch.no_grad():
            mu1, sigma1 = model(x)

        model_path = tmp_path / "test_model.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_features": config.num_features,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "dropout": config.dropout,
            },
            "training_history": {"epoch_loss": [], "batch_losses": []},
            "metadata": {
                "symbol": config.symbol,
                "horizon": config.forecast_horizon,
                "trained_at": "2024-01-01T00:00:00Z",
                "train_nll": 0.0,
            },
        }
        torch.save(checkpoint, model_path)

        new_model = ProbabilisticTransformer(config)
        loaded_checkpoint = torch.load(model_path, weights_only=False)
        new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
        new_model.eval()

        with torch.no_grad():
            mu2, sigma2 = new_model(x)

        assert torch.equal(mu1, mu2), "mu predictions must be identical after load"
        assert torch.equal(sigma1, sigma2), "sigma predictions must be identical after load"

    def test_save_load_different_inputs_consistent(self, tmp_path):
        """Loaded model is consistent across multiple different inputs."""
        torch.manual_seed(42)

        config = ForecasterConfig()
        model = ProbabilisticTransformer(config)
        model.eval()

        model_path = tmp_path / "test_model.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_features": config.num_features,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "dropout": config.dropout,
            },
            "training_history": {"epoch_loss": [], "batch_losses": []},
            "metadata": {},
        }
        torch.save(checkpoint, model_path)

        new_model = ProbabilisticTransformer(config)
        loaded = torch.load(model_path, weights_only=False)
        new_model.load_state_dict(loaded["model_state_dict"])
        new_model.eval()

        for seed in [0, 1, 2, 3]:
            torch.manual_seed(seed)
            x = torch.randn(2, 36, 16)

            with torch.no_grad():
                mu1, sigma1 = model(x)
                mu2, sigma2 = new_model(x)

            assert torch.equal(mu1, mu2), f"mu mismatch for seed {seed}"
            assert torch.equal(sigma1, sigma2), f"sigma mismatch for seed {seed}"

    @patch("probabilisticforecaster.training._upload_model_to_s3")
    def test_save_load_via_forecaster_inference(self, mock_upload, tmp_path):
        """Model saved via train_model and loaded via ForecasterInference produces identical predictions."""
        torch.manual_seed(42)

        config = ForecasterConfig(
            epochs=1,
            batch_size=32,
            model_path=str(tmp_path / "inference_roundtrip.pt"),
        )

        model = ProbabilisticTransformer(config)

        n_bars = 1600
        ohlc_df = _generate_synthetic_ohlc(n_bars=n_bars, seed=77)
        features_df = compute_features(ohlc_df, historical_window=1440)
        close_prices = pd.Series(
            ohlc_df.set_index(pd.to_datetime(ohlc_df["Timestamp"], utc=True))["Close"]
        ).reindex(features_df.index)

        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=config.lookback_window,
            horizon=config.forecast_horizon,
        )

        train_model(model=model, train_dataset=dataset, config=config, upload_to_s3=False)

        model.eval()
        test_input = torch.randn(1, 36, 16)
        with torch.no_grad():
            mu_direct, sigma_direct = model(test_input)

        mu_direct_val = mu_direct[0, -1, 0].item()
        sigma_direct_val = sigma_direct[0, -1, 0].item()

        inference = ForecasterInference(model_path=config.model_path, config=config)

        mu_inf, sigma_inf = inference.predict(test_input.squeeze(0))

        assert abs(mu_direct_val - mu_inf) < 1e-6, (
            f"mu mismatch: direct={mu_direct_val}, inference={mu_inf}"
        )
        assert abs(sigma_direct_val - sigma_inf) < 1e-6, (
            f"sigma mismatch: direct={sigma_direct_val}, inference={sigma_inf}"
        )

        batch_input = torch.randn(4, 36, 16)
        with torch.no_grad():
            mu_batch_direct, sigma_batch_direct = model(batch_input)

        mu_batch_inf, sigma_batch_inf = inference.predict_batch(batch_input)

        mu_batch_direct_last = mu_batch_direct[:, -1, 0]
        sigma_batch_direct_last = sigma_batch_direct[:, -1, 0]

        assert torch.allclose(mu_batch_direct_last, mu_batch_inf, atol=1e-6), (
            "Batch mu predictions must match between direct and inference"
        )
        assert torch.allclose(sigma_batch_direct_last, sigma_batch_inf, atol=1e-6), (
            "Batch sigma predictions must match between direct and inference"
        )


class TestFullStrategyBacktestIntegration:
    """Test full strategy backtest integration with synthetic data."""

    def test_backtest_with_synthetic_predictions(self):
        """run_backtest produces valid BacktestResult from synthetic predictions."""
        torch.manual_seed(42)
        rng = np.random.default_rng(42)

        n_predictions = 100
        timestamps = pd.date_range(
            start="2023-06-01 00:00", periods=n_predictions + 1, freq="5min", tz="UTC"
        )

        predictions_df = pd.DataFrame(
            {
                "timestamp": timestamps[:n_predictions],
                "mu": rng.normal(0, 0.0001, size=n_predictions),
                "sigma": rng.uniform(0.0001, 0.001, size=n_predictions),
            }
        )

        close_prices = 150.0 + np.cumsum(rng.normal(0, 0.01, size=n_predictions + 1))
        prices_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "close": close_prices,
            }
        )

        config = ForecasterConfig()
        strategy = DirectionalStrategy()
        result = run_backtest(predictions_df, prices_df, strategy, config)

        assert isinstance(result, BacktestResult)
        assert np.isfinite(result.sharpe_ratio), "Sharpe ratio must be finite"
        assert result.max_drawdown >= 0, "Max drawdown must be non-negative"
        assert np.isfinite(result.annualised_return), "Annualised return must be finite"
        assert not result.daily_pnl.empty, "Daily PnL should not be empty"

    def test_backtest_pnl_is_finite(self):
        """All PnL values in backtest result are finite numbers."""
        rng = np.random.default_rng(123)

        n_predictions = 200
        timestamps = pd.date_range(
            start="2023-06-01 00:00", periods=n_predictions + 1, freq="5min", tz="UTC"
        )

        predictions_df = pd.DataFrame(
            {
                "timestamp": timestamps[:n_predictions],
                "mu": rng.normal(0, 0.0002, size=n_predictions),
                "sigma": rng.uniform(0.0001, 0.0005, size=n_predictions),
            }
        )

        close_prices = 150.0 + np.cumsum(rng.normal(0, 0.005, size=n_predictions + 1))
        prices_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "close": close_prices,
            }
        )

        config = ForecasterConfig()
        strategy = DirectionalStrategy()
        result = run_backtest(predictions_df, prices_df, strategy, config)

        assert result.daily_pnl.apply(np.isfinite).all(), "All daily PnL values must be finite"
        assert np.isfinite(result.sharpe_ratio)
        assert np.isfinite(result.max_drawdown)
        assert np.isfinite(result.annualised_return)

    def test_backtest_with_mean_variance_strategy(self):
        """Backtest with MeanVarianceStrategy produces valid results."""
        rng = np.random.default_rng(55)

        n_predictions = 150
        timestamps = pd.date_range(
            start="2023-06-01 00:00", periods=n_predictions + 1, freq="5min", tz="UTC"
        )

        predictions_df = pd.DataFrame(
            {
                "timestamp": timestamps[:n_predictions],
                "mu": rng.normal(0, 0.0003, size=n_predictions),
                "sigma": rng.uniform(0.0002, 0.001, size=n_predictions),
            }
        )

        close_prices = 150.0 + np.cumsum(rng.normal(0, 0.008, size=n_predictions + 1))
        prices_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "close": close_prices,
            }
        )

        config = ForecasterConfig(risk_aversion=0.05)
        strategy = MeanVarianceStrategy(risk_aversion=0.05)
        result = run_backtest(predictions_df, prices_df, strategy, config)

        assert isinstance(result, BacktestResult)
        assert np.isfinite(result.annualised_return)
        assert np.isfinite(result.sharpe_ratio)
        assert result.max_drawdown >= 0.0
        assert not result.daily_pnl.empty
