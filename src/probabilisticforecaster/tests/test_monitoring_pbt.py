"""Property-based tests for the structured JSON logging and alerting module.

Uses Hypothesis to verify that the StructuredJsonFormatter always produces
valid JSON output with required fields, regardless of the log message content
(including special characters, unicode, and exception tracebacks).

Also verifies that the alert_webhook function constructs correct payloads
for pipeline failure notifications.
"""

import json
import logging

from hypothesis import given, settings
from hypothesis import strategies as st
from unittest import mock

from probabilisticforecaster.kubeflow.monitoring.metrics import (
    StructuredJsonFormatter,
    alert_webhook,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for arbitrary log messages including special chars and unicode
log_messages = st.text(
    alphabet=st.characters(
        categories=("L", "M", "N", "P", "S", "Z", "C"),
    ),
    min_size=0,
    max_size=500,
)

# Strategy for component names
component_names = st.text(
    alphabet=st.characters(categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=50,
)

# Strategy for logger names
logger_names = st.text(
    alphabet=st.characters(categories=("L", "N"), whitelist_characters="._"),
    min_size=1,
    max_size=100,
)

# Strategy for log levels
log_levels = st.sampled_from(
    [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
)

# Strategy for exception types to simulate tracebacks
exception_types = st.sampled_from(
    [ValueError, TypeError, RuntimeError, KeyError, IOError, ZeroDivisionError]
)


# ---------------------------------------------------------------------------
# Property 3: Structured JSON logging validity
# ---------------------------------------------------------------------------


class TestStructuredJsonLoggingValidity:
    """Property 3: Structured JSON logging validity.

    For any log message (including special chars, unicode, tracebacks),
    output is valid JSON with required fields (timestamp, level, logger,
    message, component).

    **Validates: Requirements 2.5**
    """

    @given(
        message=log_messages,
        component=component_names,
        logger_name=logger_names,
        level=log_levels,
    )
    @settings(max_examples=100, deadline=None)
    def test_output_is_valid_json_with_required_fields(
        self, message, component, logger_name, level
    ):
        """For any log message, formatter outputs valid JSON with required fields.

        **Validates: Requirements 2.5**
        """
        formatter = StructuredJsonFormatter(component=component)

        record = logging.LogRecord(
            name=logger_name,
            level=level,
            pathname="test.py",
            lineno=1,
            msg=message,
            args=None,
            exc_info=None,
        )

        output = formatter.format(record)

        # Must be valid JSON
        parsed = json.loads(output)

        # Must contain all required fields
        required_fields = {"timestamp", "level", "logger", "message", "component"}
        assert required_fields.issubset(
            parsed.keys()
        ), f"Missing fields: {required_fields - set(parsed.keys())}"

        # Field values must match expectations
        assert parsed["level"] == logging.getLevelName(level)
        assert parsed["logger"] == logger_name
        assert parsed["message"] == message
        assert parsed["component"] == component

    @given(
        message=log_messages,
        component=component_names,
        exc_type=exception_types,
        exc_message=log_messages,
    )
    @settings(max_examples=100, deadline=None)
    def test_output_is_valid_json_with_exception_traceback(
        self, message, component, exc_type, exc_message
    ):
        """For any log message with a traceback, output is still valid JSON with required fields.

        **Validates: Requirements 2.5**
        """
        formatter = StructuredJsonFormatter(component=component)

        # Generate a real exception with traceback
        try:
            raise exc_type(exc_message)
        except Exception:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg=message,
            args=None,
            exc_info=exc_info,
        )

        output = formatter.format(record)

        # Must be valid JSON even with exception info
        parsed = json.loads(output)

        # Must contain all required fields
        required_fields = {"timestamp", "level", "logger", "message", "component"}
        assert required_fields.issubset(
            parsed.keys()
        ), f"Missing fields: {required_fields - set(parsed.keys())}"

        # Must also contain the exception field
        assert "exception" in parsed, "Exception field missing when exc_info is set"
        assert isinstance(parsed["exception"], str)
        assert len(parsed["exception"]) > 0

        # Field values must match expectations
        assert parsed["message"] == message
        assert parsed["component"] == component


# ---------------------------------------------------------------------------
# Property 19: Alert webhook invocation on pipeline failure
# ---------------------------------------------------------------------------


class TestAlertWebhookInvocation:
    """Property 19: Alert webhook invocation on pipeline failure.

    For any pipeline failure event (component failure after retries exhausted),
    the monitoring system SHALL invoke the configured alert webhook with a
    payload containing the pipeline run ID, failed component name, and error
    message.

    **Validates: Requirements 9.3**
    """

    @given(
        run_id=st.text(
            alphabet=st.characters(categories=("L", "N"), whitelist_characters="-_"),
            min_size=8,
            max_size=64,
        ),
        failed_component=st.sampled_from(
            [
                "data_preparation",
                "model_training",
                "model_evaluation",
                "backtesting",
            ]
        ),
        error_message=st.text(
            alphabet=st.characters(
                categories=("L", "M", "N", "P", "S", "Z", "C"),
            ),
            min_size=1,
            max_size=500,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_webhook_payload_contains_required_fields(
        self, run_id, failed_component, error_message
    ):
        """For any failure event, webhook payload includes run_id, failed_component, error_message.

        **Validates: Requirements 9.3**
        """
        webhook_url = "https://hooks.example.com/alert"

        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = mock.MagicMock()
            mock_response.status = 200
            mock_urlopen.return_value.__enter__.return_value = mock_response

            status = alert_webhook(
                run_id=run_id,
                failed_component=failed_component,
                error_message=error_message,
                webhook_url=webhook_url,
            )

            assert status == 200

            # Verify the request was sent to the correct URL
            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            assert request.full_url == webhook_url
            assert request.method == "POST"
            assert request.get_header("Content-type") == "application/json"

            # Verify payload contains required fields
            payload = json.loads(request.data)
            assert payload["run_id"] == run_id
            assert payload["failed_component"] == failed_component
            assert payload["error_message"] == error_message
            assert "timestamp" in payload

    @given(
        run_id=st.text(
            alphabet=st.characters(categories=("L", "N"), whitelist_characters="-_"),
            min_size=8,
            max_size=64,
        ),
        failed_component=st.sampled_from(
            ["data_preparation", "model_training", "model_evaluation", "backtesting"]
        ),
        error_message=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_webhook_payload_is_valid_json_with_timestamp(
        self, run_id, failed_component, error_message
    ):
        """For any failure event, the webhook payload is valid JSON with ISO timestamp.

        **Validates: Requirements 9.3**
        """
        webhook_url = "https://hooks.example.com/alert"

        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = mock.MagicMock()
            mock_response.status = 200
            mock_urlopen.return_value.__enter__.return_value = mock_response

            alert_webhook(
                run_id=run_id,
                failed_component=failed_component,
                error_message=error_message,
                webhook_url=webhook_url,
            )

            # Extract and validate the payload
            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            payload = json.loads(request.data)

            # Timestamp must be a valid ISO 8601 string
            from datetime import datetime

            ts = datetime.fromisoformat(payload["timestamp"])
            assert ts.tzinfo is not None, "Timestamp must be timezone-aware"

    @given(
        run_id=st.text(min_size=1, max_size=32),
        failed_component=st.text(min_size=1, max_size=32),
        error_message=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=20, deadline=None)
    def test_webhook_raises_value_error_when_no_url_configured(
        self, run_id, failed_component, error_message
    ):
        """For any failure event, ValueError is raised when no webhook URL is configured.

        **Validates: Requirements 9.3**
        """
        # Test with None URL and no env var
        with mock.patch.dict("os.environ", {}, clear=True):
            from probabilisticforecaster.kubeflow.monitoring.metrics import alert_webhook as aw

            import pytest
            with pytest.raises(ValueError, match="No webhook URL configured"):
                aw(
                    run_id=run_id,
                    failed_component=failed_component,
                    error_message=error_message,
                    webhook_url=None,
                )

    @given(
        run_id=st.text(min_size=1, max_size=32),
        failed_component=st.text(min_size=1, max_size=32),
        error_message=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=20, deadline=None)
    def test_webhook_uses_env_var_fallback(self, run_id, failed_component, error_message):
        """When webhook_url is not passed but ALERT_WEBHOOK_URL is set, it uses the env var.

        **Validates: Requirements 9.3**
        """
        webhook_url = "https://hooks.pagerduty.example.com/trigger"

        with mock.patch.dict(
            "os.environ", {"ALERT_WEBHOOK_URL": webhook_url}, clear=True
        ):
            with mock.patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = mock.MagicMock()
                mock_response.status = 202
                mock_urlopen.return_value.__enter__.return_value = mock_response

                status = alert_webhook(
                    run_id=run_id,
                    failed_component=failed_component,
                    error_message=error_message,
                    webhook_url=None,
                )

                assert status == 202
                call_args = mock_urlopen.call_args
                request = call_args[0][0]
                assert request.full_url == webhook_url
