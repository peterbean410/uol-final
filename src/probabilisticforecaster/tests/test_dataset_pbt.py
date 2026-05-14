"""Property-based tests for the ForexDataset (dataset.py).

Uses Hypothesis to verify correctness properties across randomly generated inputs.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from probabilisticforecaster.dataset import ForexDataset


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Use smaller parameters for faster PBT execution
PBT_LOOKBACK = 8
PBT_HORIZON = 1


@st.composite
def features_with_gaps(
    draw,
    min_bars: int = 30,
    max_bars: int = 60,
    min_gaps: int = 1,
    max_gaps: int = 3,
    lookback: int = PBT_LOOKBACK,
    horizon: int = PBT_HORIZON,
):
    """Generate a feature DataFrame with intentional temporal gaps.

    Creates a DatetimeIndex-indexed DataFrame with 16 feature columns where
    some consecutive bars have >10 minute separation (temporal gaps).

    The strategy ensures:
    - At least min_gaps temporal gaps are present
    - Gaps are placed at random positions
    - There are enough contiguous bars to produce at least one valid sample
    """
    n_bars = draw(st.integers(min_value=min_bars, max_value=max_bars))
    max_possible_gaps = min(max_gaps, n_bars - lookback - horizon - 1)
    assume(max_possible_gaps >= min_gaps)
    n_gaps = draw(st.integers(min_value=min_gaps, max_value=max_possible_gaps))

    # Generate gap positions: indices where a gap occurs AFTER that bar
    # Avoid placing gaps in the first (lookback + horizon) bars to ensure
    # at least some valid samples exist
    possible_gap_positions = list(range(lookback + horizon, n_bars - 1))
    assume(len(possible_gap_positions) >= n_gaps)

    gap_positions = sorted(
        draw(
            st.lists(
                st.sampled_from(possible_gap_positions),
                min_size=n_gaps,
                max_size=n_gaps,
                unique=True,
            )
        )
    )

    # Build timestamps: start with 5-minute intervals, then insert gaps
    base_start = pd.Timestamp("2023-06-01 09:00:00")
    timestamps = []
    current_time = base_start

    for i in range(n_bars):
        timestamps.append(current_time)
        if i in gap_positions:
            # Insert a gap > 10 minutes (e.g., 15 to 60 minutes)
            gap_minutes = draw(st.integers(min_value=11, max_value=60))
            current_time += pd.Timedelta(minutes=gap_minutes)
        else:
            # Normal 5-minute interval
            current_time += pd.Timedelta(minutes=5)

    index = pd.DatetimeIndex(timestamps)

    # Generate 16 feature columns using numpy (much faster than individual floats)
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**32 - 1)))
    feature_data = rng.uniform(-3.0, 3.0, size=(n_bars, 16)).astype(np.float32)

    features_df = pd.DataFrame(
        feature_data,
        index=index,
        columns=[f"feat_{i}" for i in range(16)],
    )

    # Generate close prices (positive, for label computation)
    close_values = rng.uniform(100.0, 200.0, size=n_bars)
    close_prices = pd.Series(close_values, index=index)

    return features_df, close_prices, gap_positions


# ---------------------------------------------------------------------------
# Property 4: Dataset Gap Exclusion
# ---------------------------------------------------------------------------


class TestDatasetGapExclusion:
    """Property 4: Dataset Gap Exclusion.

    For any feature DataFrame containing temporal gaps (consecutive bars with
    >10 minute separation), no sample in the constructed Dataset SHALL have
    indices that span a gap boundary; i.e., for every sample, all bars in the
    lookback window and the label bar must be temporally contiguous.

    **Validates: Requirements 4.4**
    """

    @given(data=features_with_gaps())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.large_base_example],
    )
    def test_no_sample_spans_gap_boundary(self, data):
        """No sample in the dataset spans a temporal gap boundary.

        For every valid sample index, all bars from the start of the lookback
        window through the label bar must have consecutive timestamps with
        ≤10 minute separation.

        **Validates: Requirements 4.4**
        """
        features_df, close_prices, gap_positions = data

        lookback = PBT_LOOKBACK
        horizon = PBT_HORIZON

        # Construct the dataset
        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=lookback,
            horizon=horizon,
            stride=1,
        )

        timestamps = features_df.index

        # For every sample in the dataset, verify no gap is spanned
        for sample_idx in range(len(dataset)):
            start = dataset.valid_indices[sample_idx]
            # The sample uses bars [start, start+lookback) for features
            # and bar start+lookback-1+horizon for the label
            end = start + lookback - 1 + horizon

            # Check all consecutive pairs in [start, end] have ≤10 min gap
            for bar_idx in range(start, end):
                time_diff = timestamps[bar_idx + 1] - timestamps[bar_idx]
                assert time_diff <= pd.Timedelta(minutes=10), (
                    f"Sample {sample_idx} (start={start}) spans a gap: "
                    f"bar {bar_idx} at {timestamps[bar_idx]} → "
                    f"bar {bar_idx + 1} at {timestamps[bar_idx + 1]} "
                    f"(diff={time_diff})"
                )

    @given(data=features_with_gaps())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.large_base_example],
    )
    def test_gap_positions_excluded_from_samples(self, data):
        """Samples that would span a known gap position are excluded.

        For each known gap position g, no sample should have a range
        [start, start+lookback+horizon-1] that includes g (since gap_after[g]
        means a gap between bar g and bar g+1).

        **Validates: Requirements 4.4**
        """
        features_df, close_prices, gap_positions = data

        lookback = PBT_LOOKBACK
        horizon = PBT_HORIZON

        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=lookback,
            horizon=horizon,
            stride=1,
        )

        total_span = lookback + horizon

        # For each valid sample, its range [start, start+total_span-1] must
        # not contain any gap position (since gap at position g means gap
        # between bar g and bar g+1, and we need contiguity across the range)
        for sample_idx in range(len(dataset)):
            start = dataset.valid_indices[sample_idx]
            end = start + total_span - 1

            for gap_pos in gap_positions:
                # A gap at gap_pos means discontinuity between bar gap_pos
                # and bar gap_pos+1. The sample range [start, end] spans this
                # gap if start <= gap_pos < end (i.e., both gap_pos and
                # gap_pos+1 are within the sample's bar range)
                assert not (start <= gap_pos < end), (
                    f"Sample {sample_idx} (start={start}, end={end}) "
                    f"includes gap position {gap_pos}"
                )
