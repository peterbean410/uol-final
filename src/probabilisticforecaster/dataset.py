"""PyTorch Dataset with sliding window and forward return labels for Transformer training.

Implements gap detection to exclude samples spanning weekend/holiday boundaries,
and supports configurable lookback, horizon, and stride parameters.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ForexDataset(Dataset):
    """Sliding-window dataset for Transformer training.

    Each sample: (features[t-35:t+1], label[t+1:t+1+h])
    - features: Tensor of shape (36, 16)
    - label: scalar forward return

    Args:
        features_df: DataFrame with DatetimeIndex and 16 feature columns.
        close_prices: Series of close prices aligned with features_df index.
        lookback: Number of bars in the lookback window (default 36).
        horizon: Number of bars ahead for forward return label (default 1).
        stride: Step size between consecutive samples (default 1).

    Raises:
        ValueError: If feature dimension != 16 or no valid samples after gap exclusion.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        close_prices: pd.Series,
        lookback: int = 36,
        horizon: int = 1,
        stride: int = 1,
    ) -> None:
        n_features = features_df.shape[1]
        if n_features != 16:
            raise ValueError(f"Expected 16 features, got {n_features}")

        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride

        self.features = features_df.values.astype(np.float32)
        self.close_prices = close_prices.values.astype(np.float64)
        self.timestamps = features_df.index

        gap_mask = self._detect_gaps(self.timestamps)

        self.valid_indices = self._build_valid_indices(gap_mask)

        if len(self.valid_indices) == 0:
            raise ValueError("No valid samples after gap exclusion")

    def _detect_gaps(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Detect temporal gaps > 10 minutes between consecutive bars.

        Args:
            timestamps: DatetimeIndex of bar timestamps.

        Returns:
            Boolean array where True indicates a gap AFTER that index.
            gap_mask[i] = True means there is a gap between bar i and bar i+1.
        """
        if len(timestamps) < 2:
            return np.zeros(len(timestamps), dtype=bool)

        time_diffs = pd.Series(timestamps).diff()

        gap_at_bar = (time_diffs > pd.Timedelta(minutes=10)).values

        gap_after = np.zeros(len(timestamps), dtype=bool)
        if len(gap_at_bar) > 1:
            gap_after[:-1] = gap_at_bar[1:]

        return gap_after

    def _build_valid_indices(self, gap_after: np.ndarray) -> list[int]:
        """Build list of valid starting indices for samples.

        A sample starting at index i uses bars [i, i+lookback) for features
        and bar i+lookback-1+horizon for the label. The sample is valid only
        if no gap exists within the range [i, i+lookback-1+horizon].

        Args:
            gap_after: Boolean array where True at index i means gap between
                       bar i and bar i+1.

        Returns:
            List of valid starting indices.
        """
        n = len(self.features)
        total_span = self.lookback + self.horizon

        if n < total_span:
            return []

        cumulative_gaps = np.zeros(n, dtype=np.int64)
        cumulative_gaps[1:] = np.cumsum(gap_after[:-1])

        valid_indices = []
        for i in range(0, n - total_span + 1, self.stride):
            end_idx = i + total_span - 1
            gaps_in_range = cumulative_gaps[end_idx] - cumulative_gaps[i]
            if gaps_in_range == 0:
                valid_indices.append(i)

        return valid_indices

    def __len__(self) -> int:
        """Return the number of valid samples in the dataset."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample by index.

        Args:
            idx: Sample index (0-based into valid_indices).

        Returns:
            Tuple of (features_tensor, label_tensor) where:
            - features_tensor: shape (lookback, 16) = (36, 16)
            - label_tensor: shape (1,) = forward return scalar
        """
        start = self.valid_indices[idx]

        features = self.features[start : start + self.lookback]
        features_tensor = torch.from_numpy(features.copy())

        t = start + self.lookback - 1
        t_h = t + self.horizon
        close_t = self.close_prices[t]
        close_t_h = self.close_prices[t_h]
        label = (close_t_h - close_t) / close_t
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return features_tensor, label_tensor
