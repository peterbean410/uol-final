"""Configuration module for the Transformer Probabilistic Forex Forecaster.

Defines hyperparameters for data preparation, model architecture, training,
trading strategy, and model persistence paths.
"""

from dataclasses import dataclass


S3_BUCKET = "prod-fintech-forex-sg-731833471586"

S3_MODEL_PATH_TEMPLATE = (
    "transformerforecaster/model/"
    "symbol={SYMBOL}/horizon={HORIZON}/version={version}/{timestamp}.pt"
)


@dataclass
class ForecasterConfig:
    """Configuration dataclass for the Transformer Probabilistic Forecaster.

    Groups all hyperparameters needed for data preparation, model architecture,
    training, and trading strategy execution.
    """

    symbol: str = "USDJPY"
    lookback_window: int = 36
    historical_window: int = 1440
    forecast_horizon: int = 1

    num_features: int = 16
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1

    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 5
    random_seed: int = 42

    position_size: float = 10_000_000
    risk_aversion: float = 0.05

    model_path: str = "models/transformer_forecaster.pt"

    def get_s3_model_path(self, version: int, timestamp: str) -> str:
        """Format the S3 model path template with config values.

        Args:
            version: Model version number.
            timestamp: Timestamp string for the model checkpoint.

        Returns:
            Formatted S3 key path for the model artifact.
        """
        return S3_MODEL_PATH_TEMPLATE.format(
            SYMBOL=self.symbol,
            HORIZON=self.forecast_horizon,
            version=version,
            timestamp=timestamp,
        )

    def get_full_s3_uri(self, version: int, timestamp: str) -> str:
        """Get the full S3 URI for a model checkpoint.

        Args:
            version: Model version number.
            timestamp: Timestamp string for the model checkpoint.

        Returns:
            Full s3:// URI for the model artifact.
        """
        key = self.get_s3_model_path(version, timestamp)
        return f"s3://{S3_BUCKET}/{key}"
