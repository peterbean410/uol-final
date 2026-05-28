"""Entry point for the dqnpf-intraday combined KServe predictor.

Loads IntegrationConfig(s) from environment variables or a config file,
instantiates DqnpfIntradayPredictor, calls load(), and starts the kserve
model server on port 8080.

Environment variables:
    SYMBOLS: Comma-separated list of symbols to serve (default: "USDJPY").
    CONFIG_PATH: Path to YAML config file (default: uses IntegrationConfig defaults).
    MODEL_NAME: KServe model name (default: "dqnpf-intraday").
    GRPC_ADDRESS: modelenv sidecar gRPC address (default: "localhost:50051").
    DEVICE: PyTorch device (default: "cpu").
    DQN_MODEL_REGISTRY_NAME: Registry name for DQN model (default: "deepqnetwork-usdjpy").
    FORECASTER_MODEL_REGISTRY_NAME: Registry name for Forecaster model
        (default: "probabilistic-transformer-usdjpy-h1").

Requirements: 20.1, 22.1
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import kserve

from tradingmodel.intraday.dqnpf.config import IntegrationConfig

logger = logging.getLogger(__name__)


def _load_configs() -> dict[str, IntegrationConfig]:
    """Load per-symbol IntegrationConfigs from env vars or config file.

    If CONFIG_PATH is set and the file exists, loads base config from YAML.
    Otherwise uses IntegrationConfig defaults. Then creates one config per
    symbol listed in the SYMBOLS env var.

    Returns:
        Mapping of symbol → IntegrationConfig.
    """
    symbols_str = os.environ.get("SYMBOLS", "USDJPY")
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]

    config_path = os.environ.get("CONFIG_PATH", "")
    base_kwargs: dict = {}

    if config_path and Path(config_path).exists():
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        # Extract only fields that IntegrationConfig accepts
        from dataclasses import fields as dc_fields

        valid_fields = {f.name for f in dc_fields(IntegrationConfig)}
        base_kwargs = {k: v for k, v in data.items() if k in valid_fields}

    # Override with env vars
    grpc_address = os.environ.get("GRPC_ADDRESS", "localhost:50051")
    device = os.environ.get("DEVICE", "cpu")

    configs: dict[str, IntegrationConfig] = {}
    for symbol in symbols:
        kwargs = {**base_kwargs, "symbol": symbol, "grpc_address": grpc_address, "device": device}
        configs[symbol] = IntegrationConfig(**kwargs)

    return configs


def main() -> None:
    """Load predictor and start the KServe model server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    model_name = os.environ.get("MODEL_NAME", "dqnpf-intraday")

    logger.info("Starting dqnpf-intraday predictor: model_name=%s", model_name)

    # Load per-symbol configs
    configs = _load_configs()
    logger.info("Loaded configs for symbols: %s", list(configs.keys()))

    # Instantiate predictor
    from tradingmodel.intraday.dqnpf.kubeflow.serving.dqnpf_predictor import (
        DqnpfIntradayPredictor,
    )

    predictor = DqnpfIntradayPredictor(name=model_name, configs=configs)

    # Load models from registry
    predictor.load()

    # Start KServe model server
    kserve.ModelServer().start([predictor])


if __name__ == "__main__":
    main()
