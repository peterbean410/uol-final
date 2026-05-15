"""Property-based tests for Degradation Gate Logic.

Verifies that the degradation gate correctly blocks model promotion when:
1. Current NLL exceeds production NLL by more than nll_threshold (0.1)
2. Current DA drops below production DA by more than da_threshold (0.05)
3. When both metrics are within thresholds, gate passes (model can be promoted)

Tests the degradation_gate function directly without S3 access.

**Validates: Requirements 9.4**
"""

from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from probabilisticforecaster.kubeflow.components.model_evaluation.component import (
    EvaluationMetrics,
    degradation_gate,
    DEFAULT_NLL_DEGRADATION_THRESHOLD,
    DEFAULT_DA_DEGRADATION_THRESHOLD,
    DEFAULT_NLL_ABSOLUTE_THRESHOLD,
    DEFAULT_DA_ABSOLUTE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Reasonable metric ranges for forex forecasting
nll_strategy = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
da_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
cr95_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
rmse_strategy = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# Threshold strategies (positive values)
threshold_strategy = st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def production_metrics_dict(draw):
    """Generate a production metrics dictionary with nll and directional_accuracy."""
    nll = draw(nll_strategy)
    da = draw(da_strategy)
    return {"nll": nll, "directional_accuracy": da}


@st.composite
def evaluation_metrics(draw):
    """Generate an EvaluationMetrics instance with valid metric values."""
    nll = draw(nll_strategy)
    da = draw(da_strategy)
    cr95 = draw(cr95_strategy)
    rmse = draw(rmse_strategy)
    return EvaluationMetrics(nll=nll, directional_accuracy=da, coverage_ratio_95=cr95, rmse=rmse)


# ---------------------------------------------------------------------------
# Property 20: Metric degradation gates model promotion
# ---------------------------------------------------------------------------


class TestDegradationGateLogic:
    """Property 20: Metric degradation gates model promotion.

    When NLL exceeds threshold or DA drops below threshold vs production,
    model is NOT auto-promoted.

    **Validates: Requirements 9.4**
    """

    @given(
        prod_nll=nll_strategy,
        prod_da=da_strategy,
        nll_excess=st.floats(min_value=0.001, max_value=5.0, allow_nan=False, allow_infinity=False),
        nll_threshold=threshold_strategy,
        da_threshold=threshold_strategy,
        cr95=cr95_strategy,
        rmse=rmse_strategy,
    )
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_nll_exceeds_threshold_blocks_promotion(
        self, prod_nll, prod_da, nll_excess, nll_threshold, da_threshold, cr95, rmse
    ):
        """When current NLL exceeds production NLL by more than nll_threshold,
        gate_passed is False (model is NOT auto-promoted).

        We construct current_nll = prod_nll + nll_threshold + nll_excess
        so that the NLL delta strictly exceeds the threshold.
        Absolute thresholds are set wide to test only relative degradation.

        **Validates: Requirements 9.4**
        """
        # Construct current NLL that exceeds threshold
        current_nll = prod_nll + nll_threshold + nll_excess
        assume(current_nll <= 100.0)  # Keep within reasonable bounds

        # DA is within threshold (so only NLL triggers the gate)
        current_da = prod_da

        current_metrics = EvaluationMetrics(
            nll=current_nll,
            directional_accuracy=current_da,
            coverage_ratio_95=cr95,
            rmse=rmse,
        )
        production_metrics = {"nll": prod_nll, "directional_accuracy": prod_da}

        gate_passed, reason = degradation_gate(
            current_metrics=current_metrics,
            production_metrics=production_metrics,
            nll_threshold=nll_threshold,
            da_threshold=da_threshold,
            nll_absolute_threshold=999.0,  # Wide (only testing relative degradation
            da_absolute_threshold=-1.0,  # Wide) only testing relative degradation
        )

        assert gate_passed is False, (
            f"Gate should FAIL when NLL degrades beyond threshold. "
            f"current_nll={current_nll:.6f}, prod_nll={prod_nll:.6f}, "
            f"delta={current_nll - prod_nll:.6f}, threshold={nll_threshold}"
        )
        assert "NLL" in reason, (
            f"Reason should mention NLL degradation, got: {reason}"
        )

    @given(
        prod_nll=nll_strategy,
        prod_da=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
        da_excess=st.floats(min_value=0.001, max_value=0.5, allow_nan=False, allow_infinity=False),
        nll_threshold=threshold_strategy,
        da_threshold=threshold_strategy,
        cr95=cr95_strategy,
        rmse=rmse_strategy,
    )
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_da_drops_below_threshold_blocks_promotion(
        self, prod_nll, prod_da, da_excess, nll_threshold, da_threshold, cr95, rmse
    ):
        """When current DA drops below production DA by more than da_threshold,
        gate_passed is False (model is NOT auto-promoted).

        We construct current_da = prod_da - da_threshold - da_excess
        so that the DA delta strictly exceeds the threshold.
        Absolute thresholds are set wide to test only relative degradation.

        **Validates: Requirements 9.4**
        """
        # Construct current DA that drops below threshold
        current_da = prod_da - da_threshold - da_excess
        assume(current_da >= 0.0)  # DA must be non-negative

        # NLL is within threshold (so only DA triggers the gate)
        current_nll = prod_nll

        current_metrics = EvaluationMetrics(
            nll=current_nll,
            directional_accuracy=current_da,
            coverage_ratio_95=cr95,
            rmse=rmse,
        )
        production_metrics = {"nll": prod_nll, "directional_accuracy": prod_da}

        gate_passed, reason = degradation_gate(
            current_metrics=current_metrics,
            production_metrics=production_metrics,
            nll_threshold=nll_threshold,
            da_threshold=da_threshold,
            nll_absolute_threshold=999.0,  # Wide (only testing relative degradation
            da_absolute_threshold=-1.0,  # Wide) only testing relative degradation
        )

        assert gate_passed is False, (
            f"Gate should FAIL when DA degrades beyond threshold. "
            f"current_da={current_da:.4f}, prod_da={prod_da:.4f}, "
            f"delta={prod_da - current_da:.4f}, threshold={da_threshold}"
        )
        assert "DA" in reason, (
            f"Reason should mention DA degradation, got: {reason}"
        )

    @given(
        prod_nll=nll_strategy,
        prod_da=da_strategy,
        nll_threshold=threshold_strategy,
        da_threshold=threshold_strategy,
        nll_margin=st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False),
        da_margin=st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False),
        cr95=cr95_strategy,
        rmse=rmse_strategy,
    )
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_metrics_within_thresholds_allows_promotion(
        self, prod_nll, prod_da, nll_threshold, da_threshold, nll_margin, da_margin, cr95, rmse
    ):
        """When both NLL and DA are within their respective relative and absolute
        thresholds, gate_passed is True (model CAN be promoted).

        We construct:
        - current_nll = prod_nll + nll_threshold * nll_margin (within relative threshold)
        - current_da = prod_da - da_threshold * da_margin (within relative threshold)

        Since nll_margin and da_margin are in [0, 0.99], the deltas are
        strictly less than the relative thresholds.
        Absolute thresholds are set wide to test only relative degradation.

        **Validates: Requirements 9.4**
        """
        # Construct metrics within thresholds
        current_nll = prod_nll + nll_threshold * nll_margin
        current_da = prod_da - da_threshold * da_margin

        # Ensure values are reasonable
        assume(current_nll <= 100.0)
        assume(current_da >= 0.0)
        assume(current_da <= 1.0)

        current_metrics = EvaluationMetrics(
            nll=current_nll,
            directional_accuracy=current_da,
            coverage_ratio_95=cr95,
            rmse=rmse,
        )
        production_metrics = {"nll": prod_nll, "directional_accuracy": prod_da}

        gate_passed, reason = degradation_gate(
            current_metrics=current_metrics,
            production_metrics=production_metrics,
            nll_threshold=nll_threshold,
            da_threshold=da_threshold,
            nll_absolute_threshold=999.0,  # Wide (only testing relative degradation
            da_absolute_threshold=-1.0,  # Wide) only testing relative degradation
        )

        assert gate_passed is True, (
            f"Gate should PASS when metrics are within thresholds. "
            f"current_nll={current_nll:.6f}, prod_nll={prod_nll:.6f}, "
            f"nll_delta={current_nll - prod_nll:.6f}, nll_threshold={nll_threshold}, "
            f"current_da={current_da:.4f}, prod_da={prod_da:.4f}, "
            f"da_delta={prod_da - current_da:.4f}, da_threshold={da_threshold}"
        )
        assert "within acceptable thresholds" in reason, (
            f"Reason should indicate metrics are acceptable, got: {reason}"
        )

    # -------------------------------------------------------------------
    # Absolute threshold tests
    # -------------------------------------------------------------------

    @given(
        nll_above_absolute=st.floats(min_value=3.51, max_value=10.0, allow_nan=False, allow_infinity=False),
        prod_nll=nll_strategy,
        prod_da=da_strategy,
        cr95=cr95_strategy,
        rmse=rmse_strategy,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_nll_exceeds_absolute_threshold_blocks_promotion(
        self, nll_above_absolute, prod_nll, prod_da, cr95, rmse
    ):
        """When current NLL exceeds the absolute NLL threshold (3.5),
        gate_passed is False even if relative degradation is within tolerance.

        **Validates: Requirements 9.4 (absolute floor)**
        """
        assume(nll_above_absolute > 3.5)

        # NLL is above absolute floor, DA is fine, relative delta is within bounds
        current_nll = nll_above_absolute
        current_da = prod_da  # No relative degradation

        current_metrics = EvaluationMetrics(
            nll=current_nll,
            directional_accuracy=current_da,
            coverage_ratio_95=cr95,
            rmse=rmse,
        )
        production_metrics = {"nll": prod_nll, "directional_accuracy": prod_da}

        gate_passed, reason = degradation_gate(
            current_metrics=current_metrics,
            production_metrics=production_metrics,
            nll_threshold=999.0,  # Wide (relative gate won't trigger
            da_threshold=999.0,  # Wide) relative gate won't trigger
            nll_absolute_threshold=3.5,
            da_absolute_threshold=0.0,
        )

        assert gate_passed is False, (
            f"Gate should FAIL when NLL exceeds absolute threshold. "
            f"current_nll={current_nll:.6f}, absolute_threshold=3.5"
        )
        assert "absolute" in reason.lower(), (
            f"Reason should mention absolute floor, got: {reason}"
        )

    @given(
        da_below_absolute=st.floats(min_value=0.0, max_value=0.49, allow_nan=False, allow_infinity=False),
        prod_nll=nll_strategy,
        prod_da=da_strategy,
        cr95=cr95_strategy,
        rmse=rmse_strategy,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_da_below_absolute_threshold_blocks_promotion(
        self, da_below_absolute, prod_nll, prod_da, cr95, rmse
    ):
        """When current DA falls below the absolute DA threshold (0.50),
        gate_passed is False even if relative degradation is within tolerance.

        **Validates: Requirements 9.4 (absolute floor)**
        """
        assume(da_below_absolute < 0.50)

        # DA is below absolute floor, NLL is fine, relative delta is within bounds
        current_nll = prod_nll  # No relative degradation
        current_da = da_below_absolute

        current_metrics = EvaluationMetrics(
            nll=current_nll,
            directional_accuracy=current_da,
            coverage_ratio_95=cr95,
            rmse=rmse,
        )
        production_metrics = {"nll": prod_nll, "directional_accuracy": prod_da}

        gate_passed, reason = degradation_gate(
            current_metrics=current_metrics,
            production_metrics=production_metrics,
            nll_threshold=999.0,  # Wide (relative gate won't trigger
            da_threshold=999.0,  # Wide) relative gate won't trigger
            nll_absolute_threshold=999.0,
            da_absolute_threshold=0.50,
        )

        assert gate_passed is False, (
            f"Gate should FAIL when DA falls below absolute threshold. "
            f"current_da={current_da:.4f}, absolute_threshold=0.50"
        )
        assert "absolute" in reason.lower(), (
            f"Reason should mention absolute floor, got: {reason}"
        )
