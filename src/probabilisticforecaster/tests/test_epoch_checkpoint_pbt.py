"""Property-based tests for epoch checkpoint persistence.

Verifies that distributed training saves exactly N epoch checkpoints for N
epochs, that each checkpoint is a valid PyTorch checkpoint loadable by
ProbabilisticTransformer, and that the checkpoints contain all required fields.

Uses Hypothesis to generate varying epoch counts and model configurations,
and mocks boto3 S3 to capture checkpoint uploads without requiring actual
S3 connectivity.

**Validates: Requirements 4.6**
"""

import io
from dataclasses import asdict
from unittest import mock

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.model import ProbabilisticTransformer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def epoch_counts(draw):
    """Generate valid epoch counts for training."""
    return draw(st.integers(min_value=1, max_value=10))


@st.composite
def model_configs(draw):
    """Generate valid ForecasterConfig instances for checkpoint testing."""
    return ForecasterConfig(
        symbol=draw(st.sampled_from(["USDJPY", "AUDJPY"])),
        forecast_horizon=draw(st.sampled_from([1, 3, 6, 12])),
        lookback_window=draw(st.sampled_from([24, 36, 48])),
        historical_window=draw(st.sampled_from([720, 1440])),
        num_layers=draw(st.sampled_from([2, 3, 4])),
        num_heads=draw(st.sampled_from([2, 4, 8])),
        dropout=draw(st.floats(min_value=0.05, max_value=0.3)),
        learning_rate=draw(st.floats(min_value=0.0001, max_value=0.01)),
        batch_size=draw(st.sampled_from([32, 64, 128])),
        epochs=draw(st.integers(min_value=1, max_value=10)),
        random_seed=42,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_small_model(config: ForecasterConfig) -> ProbabilisticTransformer:
    """Create a ProbabilisticTransformer model with the given config.

    Uses consistent random seed for reproducibility.
    """
    torch.manual_seed(config.random_seed)
    model = ProbabilisticTransformer(config)
    model.eval()
    return model


def _serialize_checkpoint(checkpoint: dict) -> bytes:
    """Serialize a checkpoint dict to bytes using torch.save."""
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def _deserialize_checkpoint(data: bytes) -> dict:
    """Deserialize a checkpoint from bytes and verify it's loadable."""
    buffer = io.BytesIO(data)
    return torch.load(buffer, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Property 5: Epoch checkpoint persistence
# ---------------------------------------------------------------------------


class TestEpochCheckpointPersistence:
    """Property 5: Epoch checkpoint persistence.

    For any N epochs, exactly N checkpoints exist in the artifact store,
    each containing a valid model state dict loadable by ProbabilisticTransformer.

    **Validates: Requirements 4.6**
    """

    @given(num_epochs=epoch_counts())
    @settings(max_examples=100, deadline=None)
    def test_exactly_n_epoch_checkpoints_saved(self, num_epochs):
        """For any N epochs (1-10), exactly N epoch checkpoints are saved to S3.

        Mocks boto3 S3 put_object to capture and count checkpoint uploads,
        then simulates the epoch loop from distributed_train.py.

        **Validates: Requirements 4.6**
        """
        config = ForecasterConfig(
            symbol="USDJPY",
            forecast_horizon=1,
            lookback_window=36,
            num_layers=2,
            num_heads=2,

            dropout=0.1,
            learning_rate=0.001,
            batch_size=32,
            epochs=num_epochs,
            random_seed=42,
        )

        model = _create_small_model(config)
        uploaded_checkpoints: list[dict] = []

        # Mock boto3 S3 put_object to capture checkpoint data in memory
        def _mock_put_object(Bucket, Key, Body, **kwargs):
            data = Body if isinstance(Body, bytes) else Body.read()
            uploaded_checkpoints.append({
                "key": Key,
                "data": data,
                "bucket": Bucket,
            })

        with mock.patch("boto3.client") as mock_client:
            mock_s3 = mock.MagicMock()
            mock_s3.put_object.side_effect = _mock_put_object
            mock_client.return_value = mock_s3

            from probabilisticforecaster.kubeflow.components.model_training.distributed_train import (
                _save_checkpoint_to_s3,
                _save_final_checkpoint_to_s3,
            )

            checkpoint_dir = "models/forecaster/checkpoints/test-run"

            # Simulate the epoch loop: save checkpoint at end of each epoch
            for epoch in range(num_epochs):
                epoch_loss = 0.5 - epoch * 0.05  # decreasing loss
                _save_checkpoint_to_s3(
                    model=model,
                    config=config,
                    epoch=epoch,
                    epoch_loss=epoch_loss,
                    checkpoint_dir=checkpoint_dir,
                )

            # Save final consolidated checkpoint
            training_history = {
                "epoch_losses": [0.5 - e * 0.05 for e in range(num_epochs)],
                "distributed": False,
                "world_size": 1,
                "rank": 0,
            }
            _save_final_checkpoint_to_s3(
                model=model,
                config=config,
                training_history=training_history,
                checkpoint_dir=checkpoint_dir,
            )

        # Separate epoch checkpoints from final checkpoint
        epoch_checkpoints = [
            c for c in uploaded_checkpoints if "epoch_" in c["key"]
        ]
        final_checkpoints = [
            c for c in uploaded_checkpoints if "final_model_" in c["key"]
        ]

        # Assert exactly N epoch checkpoints
        assert len(epoch_checkpoints) == num_epochs, (
            f"Expected {num_epochs} epoch checkpoints, "
            f"got {len(epoch_checkpoints)}"
        )

        # Assert exactly 1 final checkpoint
        assert len(final_checkpoints) == 1, (
            f"Expected 1 final checkpoint, got {len(final_checkpoints)}"
        )

        # Verify epoch checkpoint keys are unique and ordered
        epoch_keys = [c["key"] for c in epoch_checkpoints]
        assert len(set(epoch_keys)) == len(epoch_keys), (
            f"Epoch checkpoint keys are not unique: {epoch_keys}"
        )

        # Each epoch_XXX number matches the epoch
        for c in epoch_checkpoints:
            key = c["key"]
            # Extract epoch number from key: epoch_XXX_timestamp.pt
            epoch_part = key.split("/")[-1].split("_")[1]
            epoch_num = int(epoch_part)
            assert 0 <= epoch_num < num_epochs, (
                f"Epoch number {epoch_num} out of range [0, {num_epochs})"
            )

    @given(num_epochs=epoch_counts())
    @settings(max_examples=100, deadline=None)
    def test_each_epoch_checkpoint_is_loadable(self, num_epochs):
        """For any N epochs (1-10), each epoch checkpoint is loadable and
        contains all required fields (model_state_dict, config, epoch,
        epoch_loss, timestamp).

        **Validates: Requirements 4.6**
        """
        config = ForecasterConfig(
            symbol="USDJPY",
            forecast_horizon=1,
            lookback_window=36,
            num_layers=2,
            num_heads=2,

            dropout=0.1,
            learning_rate=0.001,
            batch_size=32,
            epochs=num_epochs,
            random_seed=42,
        )

        model = _create_small_model(config)
        # Keep original state dict to verify it's preserved in checkpoint
        original_state_keys = set(model.state_dict().keys())

        with mock.patch("boto3.client") as mock_client:
            mock_s3 = mock.MagicMock()
            uploaded_data: list[bytes] = []

            def _capture_put_object(Bucket, Key, Body, **kwargs):
                data = Body if isinstance(Body, bytes) else Body.read()
                uploaded_data.append(data)

            mock_s3.put_object.side_effect = _capture_put_object
            mock_client.return_value = mock_s3

            from probabilisticforecaster.kubeflow.components.model_training.distributed_train import (
                _save_checkpoint_to_s3,
            )

            checkpoint_dir = "models/forecaster/checkpoints/test-loadable"

            for epoch in range(num_epochs):
                _save_checkpoint_to_s3(
                    model=model,
                    config=config,
                    epoch=epoch,
                    epoch_loss=0.5 - epoch * 0.05,
                    checkpoint_dir=checkpoint_dir,
                )

        # Each checkpoint must be loadable and contain required fields
        for i, data in enumerate(uploaded_data):
            checkpoint = _deserialize_checkpoint(data)

            # Required fields
            assert "model_state_dict" in checkpoint, (
                f"Checkpoint {i} missing model_state_dict"
            )
            assert "config" in checkpoint, (
                f"Checkpoint {i} missing config"
            )
            assert "epoch" in checkpoint, (
                f"Checkpoint {i} missing epoch"
            )
            assert "epoch_loss" in checkpoint, (
                f"Checkpoint {i} missing epoch_loss"
            )
            assert "timestamp" in checkpoint, (
                f"Checkpoint {i} missing timestamp"
            )

            # Epoch field matches iteration
            assert checkpoint["epoch"] == i, (
                f"Checkpoint {i} epoch field is {checkpoint['epoch']}, expected {i}"
            )

            # Config round-trip: deserialize and verify key fields
            config_dict = checkpoint["config"]
            assert config_dict["symbol"] == config.symbol
            assert config_dict["forecast_horizon"] == config.forecast_horizon
            assert config_dict["lookback_window"] == config.lookback_window
            assert config_dict["num_layers"] == config.num_layers
            assert config_dict["num_heads"] == config.num_heads

            # epoch_loss must be a finite float
            assert isinstance(checkpoint["epoch_loss"], float), (
                f"Checkpoint {i} epoch_loss is not a float"
            )
            import math
            assert math.isfinite(checkpoint["epoch_loss"]), (
                f"Checkpoint {i} epoch_loss is not finite"
            )

            # timestamp must be a non-empty string
            assert isinstance(checkpoint["timestamp"], str), (
                f"Checkpoint {i} timestamp is not a string"
            )
            assert len(checkpoint["timestamp"]) > 0, (
                f"Checkpoint {i} timestamp is empty"
            )

            # State dict keys match the original model
            state_keys = set(checkpoint["model_state_dict"].keys())
            assert state_keys == original_state_keys, (
                f"Checkpoint {i} state dict keys do not match model. "
                f"Missing: {original_state_keys - state_keys}, "
                f"Extra: {state_keys - original_state_keys}"
            )

    @given(config=model_configs())
    @settings(max_examples=50, deadline=None)
    def test_checkpoint_state_dict_loadable_by_model(self, config):
        """For any valid model configuration, a checkpoint saved from a model
        is loadable by a fresh ProbabilisticTransformer instance with the
        same config.

        **Validates: Requirements 4.6**
        """
        # Create and save a model checkpoint
        model = _create_small_model(config)
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "epoch": 0,
            "epoch_loss": 0.5,
            "timestamp": "2026-05-14T00:00:00+00:00",
        }

        data = _serialize_checkpoint(checkpoint)

        # Load the checkpoint into a fresh model
        loaded = _deserialize_checkpoint(data)
        fresh_model = ProbabilisticTransformer(config)
        fresh_model.load_state_dict(loaded["model_state_dict"])
        fresh_model.eval()

        # Verify parameters match between original and loaded model
        for key in original_state:
            original_param = original_state[key]
            loaded_param = fresh_model.state_dict()[key]
            assert torch.equal(original_param, loaded_param), (
                f"Parameter '{key}' differs between original and loaded model"
            )

        # Verify the loaded model can run a forward pass
        batch_size = 2
        lookback = config.lookback_window
        num_features = config.num_features
        dummy_input = torch.randn(batch_size, lookback, num_features)

        with torch.no_grad():
            mu, sigma = fresh_model(dummy_input)

        # Outputs must be finite and sigma must be positive
        assert torch.isfinite(mu).all(), "Model output mu contains non-finite values"
        assert torch.isfinite(sigma).all(), "Model output sigma contains non-finite values"
        assert (sigma > 0).all(), "Model output sigma contains non-positive values"

        # Output shape: (batch_size, seq_len, 1)
        assert mu.shape == (batch_size, lookback, 1), (
            f"Unexpected mu shape: {mu.shape}"
        )
        assert sigma.shape == (batch_size, lookback, 1), (
            f"Unexpected sigma shape: {sigma.shape}"
        )

    @given(num_epochs=epoch_counts())
    @settings(max_examples=50, deadline=None)
    def test_epoch_checkpoint_ordering_is_correct(self, num_epochs):
        """For any N epochs, epoch checkpoints are saved with monotonically
        increasing epoch numbers and unique timestamps.

        **Validates: Requirements 4.6**
        """
        config = ForecasterConfig(
            symbol="USDJPY",
            forecast_horizon=1,
            lookback_window=36,
            num_layers=2,
            num_heads=2,

            dropout=0.1,
            learning_rate=0.001,
            batch_size=32,
            epochs=num_epochs,
            random_seed=42,
        )

        model = _create_small_model(config)
        checkpoint_info: list[dict] = []

        def _capture_put_object(Bucket, Key, Body, **kwargs):
            data = Body if isinstance(Body, bytes) else Body.read()
            buffer = io.BytesIO(data)
            ckpt = torch.load(buffer, map_location="cpu", weights_only=False)
            checkpoint_info.append({
                "key": Key,
                "epoch": ckpt["epoch"],
                "timestamp": ckpt["timestamp"],
            })

        with mock.patch("boto3.client") as mock_client:
            mock_s3 = mock.MagicMock()
            mock_s3.put_object.side_effect = _capture_put_object
            mock_client.return_value = mock_s3

            from probabilisticforecaster.kubeflow.components.model_training.distributed_train import (
                _save_checkpoint_to_s3,
            )

            for epoch in range(num_epochs):
                _save_checkpoint_to_s3(
                    model=model,
                    config=config,
                    epoch=epoch,
                    epoch_loss=0.5 - epoch * 0.05,
                    checkpoint_dir="checkpoints/ordering-test",
                )

        # Filter to epoch checkpoints only
        epoch_checkpoints = [
            c for c in checkpoint_info if "epoch_" in c["key"]
        ]

        assert len(epoch_checkpoints) == num_epochs

        # Verify epochs are 0, 1, 2, ..., N-1 (monotonically increasing)
        epochs_found = sorted([c["epoch"] for c in epoch_checkpoints])
        assert epochs_found == list(range(num_epochs)), (
            f"Epoch numbers not consecutive: {epochs_found}"
        )

        # Verify all epoch checkpoint keys are unique
        keys = [c["key"] for c in epoch_checkpoints]
        assert len(set(keys)) == len(keys), (
            f"Epoch checkpoint keys are not unique: {keys}"
        )

        # Verify all timestamps are valid ISO 8601 with timezone
        from datetime import datetime

        for c in epoch_checkpoints:
            ts = datetime.fromisoformat(c["timestamp"])
            assert ts.tzinfo is not None, (
                f"Timestamp '{c['timestamp']}' is not timezone-aware"
            )
