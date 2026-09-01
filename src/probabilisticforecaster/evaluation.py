"""Evaluation module for the Probabilistic Transformer Forecaster.

Computes NLL, Directional Accuracy, 95% Covered Ratio, and RMSE metrics
on test data. Supports both overall evaluation and intraday analysis
grouped by hour of day.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.dataset import ForexDataset
from probabilisticforecaster.model import ProbabilisticTransformer


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics.

    Attributes:
        nll: Mean negative log-likelihood under predicted Gaussian.
        directional_accuracy: Proportion of predictions where sign(μ̂) == sign(actual).
        covered_ratio_95: Proportion of actuals within μ̂ ± 2σ̂.
        rmse: Root mean squared error between μ̂ and actual.
    """

    nll: float
    directional_accuracy: float
    covered_ratio_95: float
    rmse: float


def _compute_nll(mu: np.ndarray, sigma: np.ndarray, actual: np.ndarray) -> float:
    """Compute mean Gaussian negative log-likelihood.

    NLL = mean(0.5 * (log(σ²) + ((x - μ) / σ)² + log(2π)))

    Args:
        mu: Predicted means, shape (N,).
        sigma: Predicted std devs (> 0), shape (N,).
        actual: Realized values, shape (N,).

    Returns:
        Mean NLL as a float.
    """
    variance = sigma**2
    log_variance = np.log(variance)
    squared_error = ((actual - mu) / sigma) ** 2
    log_2pi = math.log(2 * math.pi)

    nll = 0.5 * (log_variance + squared_error + log_2pi)
    return float(np.mean(nll))


def _compute_directional_accuracy(mu: np.ndarray, actual: np.ndarray) -> float:
    """Compute directional accuracy.

    DA = count(sign(μ̂) == sign(actual)) / N

    Args:
        mu: Predicted means, shape (N,).
        actual: Realized values, shape (N,).

    Returns:
        Directional accuracy as a float in [0, 1].
    """
    return float(np.mean(np.sign(mu) == np.sign(actual)))


def _compute_covered_ratio_95(
    mu: np.ndarray, sigma: np.ndarray, actual: np.ndarray
) -> float:
    """Compute 95% covered ratio.

    CR95 = count(|actual - μ̂| ≤ 2σ̂) / N

    Args:
        mu: Predicted means, shape (N,).
        sigma: Predicted std devs (> 0), shape (N,).
        actual: Realized values, shape (N,).

    Returns:
        Covered ratio as a float in [0, 1].
    """
    return float(np.mean(np.abs(actual - mu) <= 2 * sigma))


def _compute_rmse(mu: np.ndarray, actual: np.ndarray) -> float:
    """Compute root mean squared error.

    RMSE = sqrt(mean((μ̂ - actual)²))

    Args:
        mu: Predicted means, shape (N,).
        actual: Realized values, shape (N,).

    Returns:
        RMSE as a float.
    """
    return float(np.sqrt(np.mean((mu - actual) ** 2)))


def _compute_metrics(
    mu: np.ndarray, sigma: np.ndarray, actual: np.ndarray
) -> EvaluationMetrics:
    """Compute all four evaluation metrics from arrays.

    Args:
        mu: Predicted means, shape (N,).
        sigma: Predicted std devs (> 0), shape (N,).
        actual: Realized values, shape (N,).

    Returns:
        EvaluationMetrics with nll, directional_accuracy, covered_ratio_95, rmse.
    """
    return EvaluationMetrics(
        nll=_compute_nll(mu, sigma, actual),
        directional_accuracy=_compute_directional_accuracy(mu, actual),
        covered_ratio_95=_compute_covered_ratio_95(mu, sigma, actual),
        rmse=_compute_rmse(mu, actual),
    )


def _collect_predictions(
    model: ProbabilisticTransformer,
    test_dataset: ForexDataset,
    config: ForecasterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference on the test dataset and collect predictions.

    Args:
        model: Trained ProbabilisticTransformer model.
        test_dataset: Test ForexDataset.
        config: ForecasterConfig with batch_size.

    Returns:
        Tuple of (mu_array, sigma_array, actual_array), each shape (N,).
    """
    device = next(model.parameters()).device
    model.eval()

    loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    all_mu: list[np.ndarray] = []
    all_sigma: list[np.ndarray] = []
    all_actual: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)

            mu, sigma = model(features)

            mu_last = mu[:, -1, 0]
            sigma_last = sigma[:, -1, 0]
            actual = labels[:, 0]

            all_mu.append(mu_last.cpu().numpy())
            all_sigma.append(sigma_last.cpu().numpy())
            all_actual.append(actual.numpy())

    mu_array = np.concatenate(all_mu)
    sigma_array = np.concatenate(all_sigma)
    actual_array = np.concatenate(all_actual)

    return mu_array, sigma_array, actual_array


def evaluate_model(
    model: ProbabilisticTransformer,
    test_dataset: ForexDataset,
    config: ForecasterConfig,
) -> EvaluationMetrics:
    """Evaluate model on the test dataset and compute all metrics.

    Runs inference on the full test set and computes NLL, Directional Accuracy,
    95% Covered Ratio, and RMSE.

    Args:
        model: Trained ProbabilisticTransformer model.
        test_dataset: Test ForexDataset.
        config: ForecasterConfig with batch_size and other settings.

    Returns:
        EvaluationMetrics with all four metrics computed over the test set.
    """
    mu_array, sigma_array, actual_array = _collect_predictions(
        model, test_dataset, config
    )
    return _compute_metrics(mu_array, sigma_array, actual_array)


def evaluate_by_hour(
    model: ProbabilisticTransformer,
    test_dataset: ForexDataset,
    timestamps: pd.Series,
    config: ForecasterConfig,
) -> pd.DataFrame:
    """Evaluate model performance grouped by hour of day.

    Computes all four metrics (NLL, DA, CR95, RMSE) for each hour of the day,
    enabling intraday performance analysis.

    Args:
        model: Trained ProbabilisticTransformer model.
        test_dataset: Test ForexDataset.
        timestamps: Series of timestamps aligned with the test dataset samples.
            Each timestamp corresponds to the last bar in the lookback window
            of the respective sample. Must have the same length as test_dataset.
        config: ForecasterConfig with batch_size and other settings.

    Returns:
        DataFrame with 24 rows (one per hour, indexed 0-23) and columns:
        [hour, nll, directional_accuracy, covered_ratio_95, rmse, count].
    """
    mu_array, sigma_array, actual_array = _collect_predictions(
        model, test_dataset, config
    )

    hours = pd.to_datetime(timestamps).dt.hour.values[: len(mu_array)]

    results: list[dict] = []
    for hour in range(24):
        mask = hours == hour
        count = int(np.sum(mask))

        if count == 0:
            results.append(
                {
                    "hour": hour,
                    "nll": float("nan"),
                    "directional_accuracy": float("nan"),
                    "covered_ratio_95": float("nan"),
                    "rmse": float("nan"),
                    "count": 0,
                }
            )
        else:
            mu_h = mu_array[mask]
            sigma_h = sigma_array[mask]
            actual_h = actual_array[mask]

            metrics = _compute_metrics(mu_h, sigma_h, actual_h)
            results.append(
                {
                    "hour": hour,
                    "nll": metrics.nll,
                    "directional_accuracy": metrics.directional_accuracy,
                    "covered_ratio_95": metrics.covered_ratio_95,
                    "rmse": metrics.rmse,
                    "count": count,
                }
            )

    df = pd.DataFrame(results)
    df = df.set_index("hour")
    return df
