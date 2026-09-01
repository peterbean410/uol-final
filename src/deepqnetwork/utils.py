"""Shared utilities for the DQN trading agent.

Provides device resolution, logging setup, and order ID generation.
"""

import logging
import uuid
from pathlib import Path

import torch


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to torch.device. Auto-detects best available.

    Args:
        device_str: One of "auto", "cpu", "cuda", "cuda:0", "cuda:1", "mps".
            When "auto", detects CUDA first, then MPS (Apple Silicon), then CPU.

    Returns:
        The resolved torch.device.
    """
    device_str = device_str.strip().lower()

    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    logging.getLogger(__name__).info("Selected device: %s", device)
    return device


def setup_logging(
    level: str = "INFO", csv_path: str | None = None
) -> logging.Logger:
    """Configure root logger with console handler and optional CSV file handler.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        csv_path: Optional path for a CSV file handler. If provided, log records
            are written as comma-separated lines to this file.

    Returns:
        The configured root logger.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    if csv_path is not None:
        csv_file = Path(csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(csv_file), mode="a")
        file_handler.setLevel(log_level)
        csv_fmt = logging.Formatter("%(asctime)s,%(levelname)s,%(name)s,%(message)s")
        file_handler.setFormatter(csv_fmt)
        root_logger.addHandler(file_handler)

    return root_logger


def generate_order_id() -> str:
    """Generate a unique client_order_id for gRPC Step() calls.

    Returns:
        A UUID4 string suitable for use as a unique order identifier.
    """
    return str(uuid.uuid4())
