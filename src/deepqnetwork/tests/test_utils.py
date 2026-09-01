"""Unit tests for deepqnetwork.utils module."""

import logging
import os
import tempfile
from unittest.mock import patch

import torch

from deepqnetwork.utils import generate_order_id, resolve_device, setup_logging


class TestResolveDevice:
    """Tests for resolve_device function."""

    def test_cpu_explicit(self):
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_auto_fallback_to_cpu(self):
        """When neither CUDA nor MPS is available, auto should select CPU."""
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            device = resolve_device("auto")
            assert device == torch.device("cpu")

    def test_auto_selects_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            device = resolve_device("auto")
            assert device == torch.device("cuda")

    def test_auto_selects_mps_when_cuda_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=True
        ):
            device = resolve_device("auto")
            assert device == torch.device("mps")

    def test_explicit_cuda_device_index(self):
        device = resolve_device("cuda:0")
        assert device == torch.device("cuda:0")

    def test_case_insensitive(self):
        device = resolve_device("CPU")
        assert device == torch.device("cpu")

    def test_whitespace_stripped(self):
        device = resolve_device("  cpu  ")
        assert device == torch.device("cpu")

    def test_logs_selected_device(self, caplog):
        with caplog.at_level(logging.INFO):
            resolve_device("cpu")
        assert "Selected device: cpu" in caplog.text


class TestSetupLogging:
    """Tests for setup_logging function."""

    def teardown_method(self):
        """Clean up root logger handlers after each test."""
        root = logging.getLogger()
        root.handlers.clear()

    def test_returns_root_logger(self):
        logger = setup_logging()
        assert logger is logging.getLogger()

    def test_sets_log_level(self):
        logger = setup_logging(level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_console_handler_added(self):
        logger = setup_logging()
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1

    def test_csv_handler_added_when_path_provided(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name
        try:
            logger = setup_logging(csv_path=csv_path)
            file_handlers = [
                h for h in logger.handlers if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) == 1
        finally:
            os.unlink(csv_path)

    def test_no_csv_handler_when_path_none(self):
        logger = setup_logging(csv_path=None)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 0

    def test_csv_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "subdir", "metrics.csv")
            setup_logging(csv_path=csv_path)
            assert os.path.isdir(os.path.join(tmpdir, "subdir"))

    def test_csv_handler_writes_records(self):
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            csv_path = f.name
        try:
            logger = setup_logging(level="INFO", csv_path=csv_path)
            logger.info("test message")
            for h in logger.handlers:
                h.flush()
            with open(csv_path) as f:
                content = f.read()
            assert "test message" in content
            assert "INFO" in content
        finally:
            os.unlink(csv_path)

    def test_clears_existing_handlers(self):
        """Repeated calls should not accumulate handlers."""
        setup_logging()
        setup_logging()
        logger = logging.getLogger()
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1


class TestGenerateOrderId:
    """Tests for generate_order_id function."""

    def test_returns_string(self):
        order_id = generate_order_id()
        assert isinstance(order_id, str)

    def test_returns_valid_uuid_format(self):
        import uuid

        order_id = generate_order_id()
        parsed = uuid.UUID(order_id)
        assert parsed.version == 4

    def test_unique_ids(self):
        ids = {generate_order_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_non_empty(self):
        order_id = generate_order_id()
        assert len(order_id) > 0
