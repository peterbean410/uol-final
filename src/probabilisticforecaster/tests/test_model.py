"""Unit tests for Transformer model architecture.

Tests model structure (layers, heads, d_model) and input validation.
"""

import pytest
import torch

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.model import ProbabilisticTransformer


class TestModelArchitecture:
    """Tests for model architecture structure, Requirements 5.1, 5.2, 5.4."""

    def test_model_has_3_layers(self):
        """Model should have exactly 3 Transformer layers."""
        config = ForecasterConfig()
        model = ProbabilisticTransformer(config)
        assert len(model.layers) == 3

    def test_model_has_4_heads(self):
        """Each Transformer layer should have 4 attention heads."""
        config = ForecasterConfig()
        model = ProbabilisticTransformer(config)
        for layer in model.layers:
            assert layer.num_heads == 4

    def test_model_d_model_is_16(self):
        """Each Transformer layer should have d_model=16."""
        config = ForecasterConfig()
        model = ProbabilisticTransformer(config)
        for layer in model.layers:
            assert layer.d_model == 16


class TestModelInputValidation:
    """Tests for input feature dimension validation, Requirement 11.3."""

    def test_invalid_feature_dimension_raises_value_error(self):
        """Passing input with wrong feature dimension should raise ValueError."""
        config = ForecasterConfig()
        model = ProbabilisticTransformer(config)
        model.eval()

        bad_input = torch.randn(1, 36, 10)
        with pytest.raises(ValueError, match="Input feature dimension"):
            model(bad_input)

    def test_correct_feature_dimension_does_not_raise(self):
        """Passing input with correct feature dimension should not raise."""
        config = ForecasterConfig()
        model = ProbabilisticTransformer(config)
        model.eval()

        good_input = torch.randn(1, 36, 16)
        mu, sigma = model(good_input)

        assert mu.shape == (1, 36, 1)
        assert sigma.shape == (1, 36, 1)
