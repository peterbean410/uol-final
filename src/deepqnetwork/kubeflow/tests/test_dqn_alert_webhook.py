"""Property-based tests for DQN alert webhook invocation on pipeline failure.

Uses Hypothesis to verify that the dqn_alert_webhook function always invokes
the webhook with a valid JSON payload containing run_id, failed_component,
error_message, and timestamp fields for any arbitrary DQN pipeline failure event.

**Property DQN-16: Alert webhook invocation on DQN pipeline failure**, for any
pipeline failure event, webhook is invoked with payload containing run_id,
failed component, error message.
"""

import json
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for run IDs, arbitrary non-empty strings including special chars
run_ids = st.text(
    alphabet=st.characters(
        categories=("L", "M", "N", "P", "S"),
        whitelist_characters="-_",
    ),
    min_size=1,
    max_size=200,
)

# Strategy for DQN component names, realistic DQN pipeline components plus arbitrary
component_names = st.one_of(
    st.sampled_from([
        "dqn_training",
        "dqn_backtest",
        "dqn_registry",
        "dqn_katib",
    ]),
    st.text(
        alphabet=st.characters(
            categories=("L", "M", "N", "P", "S"),
            whitelist_characters="-_/.",
        ),
        min_size=1,
        max_size=200,
    ),
)

# Strategy for error messages, include special characters and unicode
error_messages = st.text(
    alphabet=st.characters(
        categories=("L", "M", "N", "P", "S", "Z", "C"),
    ),
    min_size=0,
    max_size=500,
)

# Strategy for webhook URLs
webhook_urls = st.sampled_from([
    "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
    "https://events.pagerduty.com/v2/enqueue",
    "https://example.com/webhook",
    "https://hooks.slack.com/services/abc/def/ghi",
])


# ---------------------------------------------------------------------------
# Property DQN-16: Alert webhook invocation on DQN pipeline failure
# ---------------------------------------------------------------------------


class TestDqnAlertWebhookInvocation:
    """Property DQN-16: Alert webhook invocation on DQN pipeline failure.

    For any pipeline failure event, the webhook is invoked with a payload
    containing run_id, failed_component, error_message, and timestamp.

    **Validates: Requirements DQN-R24**
    """

    @given(
        run_id=run_ids,
        failed_component=component_names,
        error_message=error_messages,
        webhook_url=webhook_urls,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_webhook_called_exactly_once(
        self, run_id, failed_component, error_message, webhook_url
    ):
        """For any DQN pipeline failure event, the webhook is called exactly once.

        **Validates: Requirements DQN-R24**
        """
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            from deepqnetwork.kubeflow.monitoring.metrics import dqn_alert_webhook

            dqn_alert_webhook(
                run_id=run_id,
                failed_component=failed_component,
                error_message=error_message,
                webhook_url=webhook_url,
            )

            assert mock_urlopen.call_count == 1

    @given(
        run_id=run_ids,
        failed_component=component_names,
        error_message=error_messages,
        webhook_url=webhook_urls,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_payload_is_valid_json(
        self, run_id, failed_component, error_message, webhook_url
    ):
        """For any DQN pipeline failure event, the payload sent is valid JSON.

        **Validates: Requirements DQN-R24**
        """
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            from deepqnetwork.kubeflow.monitoring.metrics import dqn_alert_webhook

            dqn_alert_webhook(
                run_id=run_id,
                failed_component=failed_component,
                error_message=error_message,
                webhook_url=webhook_url,
            )

            # Extract the Request object passed to urlopen
            call_args = mock_urlopen.call_args
            request_obj = call_args.args[0] if call_args.args else call_args[0][0]

            # The data attribute contains the JSON payload bytes
            payload_bytes = request_obj.data
            assert payload_bytes is not None

            # Verify it's valid JSON
            payload = json.loads(payload_bytes.decode("utf-8"))
            assert isinstance(payload, dict)

    @given(
        run_id=run_ids,
        failed_component=component_names,
        error_message=error_messages,
        webhook_url=webhook_urls,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_payload_contains_required_fields_matching_inputs(
        self, run_id, failed_component, error_message, webhook_url
    ):
        """For any DQN pipeline failure event, the payload contains run_id,
        failed_component, and error_message fields matching the inputs.

        **Validates: Requirements DQN-R24**
        """
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            from deepqnetwork.kubeflow.monitoring.metrics import dqn_alert_webhook

            dqn_alert_webhook(
                run_id=run_id,
                failed_component=failed_component,
                error_message=error_message,
                webhook_url=webhook_url,
            )

            # Extract the Request object and parse payload
            call_args = mock_urlopen.call_args
            request_obj = call_args.args[0] if call_args.args else call_args[0][0]
            payload = json.loads(request_obj.data.decode("utf-8"))

            # Verify required fields exist and match inputs
            assert "run_id" in payload, "Payload missing 'run_id' field"
            assert "failed_component" in payload, "Payload missing 'failed_component' field"
            assert "error_message" in payload, "Payload missing 'error_message' field"

            assert payload["run_id"] == run_id
            assert payload["failed_component"] == failed_component
            assert payload["error_message"] == error_message

    @given(
        run_id=run_ids,
        failed_component=component_names,
        error_message=error_messages,
        webhook_url=webhook_urls,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_payload_contains_timestamp(
        self, run_id, failed_component, error_message, webhook_url
    ):
        """For any DQN pipeline failure event, the payload contains a valid
        ISO 8601 timestamp field.

        **Validates: Requirements DQN-R24**
        """
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            from deepqnetwork.kubeflow.monitoring.metrics import dqn_alert_webhook

            dqn_alert_webhook(
                run_id=run_id,
                failed_component=failed_component,
                error_message=error_message,
                webhook_url=webhook_url,
            )

            # Extract the Request object and parse payload
            call_args = mock_urlopen.call_args
            request_obj = call_args.args[0] if call_args.args else call_args[0][0]
            payload = json.loads(request_obj.data.decode("utf-8"))

            # Verify timestamp field exists and is a non-empty string
            assert "timestamp" in payload, "Payload missing 'timestamp' field"
            assert isinstance(payload["timestamp"], str)
            assert len(payload["timestamp"]) > 0

            # Verify it's a valid ISO 8601 timestamp
            from datetime import datetime

            # Should not raise if it's a valid ISO format
            datetime.fromisoformat(payload["timestamp"])

    @given(
        run_id=run_ids,
        failed_component=component_names,
        error_message=error_messages,
        webhook_url=webhook_urls,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_webhook_called_with_correct_url(
        self, run_id, failed_component, error_message, webhook_url
    ):
        """For any DQN pipeline failure event, the webhook is called with the correct URL.

        **Validates: Requirements DQN-R24**
        """
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            from deepqnetwork.kubeflow.monitoring.metrics import dqn_alert_webhook

            dqn_alert_webhook(
                run_id=run_id,
                failed_component=failed_component,
                error_message=error_message,
                webhook_url=webhook_url,
            )

            # Extract the Request object and verify URL
            call_args = mock_urlopen.call_args
            request_obj = call_args.args[0] if call_args.args else call_args[0][0]
            assert request_obj.full_url == webhook_url
