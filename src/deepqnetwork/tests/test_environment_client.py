"""Unit tests for deepqnetwork.environment_client module."""

import time
from concurrent import futures
from unittest.mock import patch

import grpc
import pytest

import environment_pb2
import environment_pb2_grpc
from deepqnetwork.environment_client import EnvironmentClient


# --- Mock gRPC server for testing ---


class MockEnvironmentServicer(environment_pb2_grpc.EnvironmentServicer):
    """Mock implementation of the Environment gRPC service."""

    def __init__(self):
        self.reset_calls = []
        self.step_calls = []
        self.reference_data_calls = []

    def Reset(self, request, context):
        self.reset_calls.append(request)
        return environment_pb2.Observation(
            state_columns=["a", "b", "c"],
            state_data=[environment_pb2.StateRow(values=[1.0, 2.0, 3.0])],
            reward=0.0,
            done=False,
        )

    def Step(self, request, context):
        self.step_calls.append(request)
        return environment_pb2.StepResponse(
            data=environment_pb2.Observation(
                state_columns=["a", "b", "c"],
                state_data=[environment_pb2.StateRow(values=[4.0, 5.0, 6.0])],
                reward=0.5,
                done=False,
            ),
            info="step executed",
        )

    def ReferenceData(self, request, context):
        self.reference_data_calls.append(request)
        return environment_pb2.Reference(
            symbol=request.symbol,
            timestamp_ns=1000000000,
            done=False,
        )


class FailingEnvironmentServicer(environment_pb2_grpc.EnvironmentServicer):
    """Servicer that always returns UNAVAILABLE to test retry logic."""

    def __init__(self):
        self.call_count = 0

    def Reset(self, request, context):
        self.call_count += 1
        context.set_code(grpc.StatusCode.UNAVAILABLE)
        context.set_details("Server unavailable")
        return environment_pb2.Observation()

    def Step(self, request, context):
        self.call_count += 1
        context.set_code(grpc.StatusCode.UNAVAILABLE)
        context.set_details("Server unavailable")
        return environment_pb2.StepResponse()

    def ReferenceData(self, request, context):
        self.call_count += 1
        context.set_code(grpc.StatusCode.UNAVAILABLE)
        context.set_details("Server unavailable")
        return environment_pb2.Reference()


class EventuallySucceedsServicer(environment_pb2_grpc.EnvironmentServicer):
    """Servicer that fails N times then succeeds."""

    def __init__(self, fail_count: int = 2):
        self.call_count = 0
        self.fail_count = fail_count

    def Reset(self, request, context):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Server unavailable")
            return environment_pb2.Observation()
        return environment_pb2.Observation(
            state_columns=["x"],
            state_data=[environment_pb2.StateRow(values=[42.0])],
            reward=0.0,
            done=False,
        )


@pytest.fixture
def mock_server():
    """Start a mock gRPC server and return (address, servicer)."""
    servicer = MockEnvironmentServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    environment_pb2_grpc.add_EnvironmentServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    yield f"localhost:{port}", servicer
    server.stop(grace=0)


@pytest.fixture
def failing_server():
    """Start a mock gRPC server that always fails."""
    servicer = FailingEnvironmentServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    environment_pb2_grpc.add_EnvironmentServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    yield f"localhost:{port}", servicer
    server.stop(grace=0)


@pytest.fixture
def eventually_succeeds_server():
    """Start a mock gRPC server that fails twice then succeeds."""
    servicer = EventuallySucceedsServicer(fail_count=2)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    environment_pb2_grpc.add_EnvironmentServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    yield f"localhost:{port}", servicer
    server.stop(grace=0)


class TestEnvironmentClientInit:
    """Tests for EnvironmentClient initialisation."""

    def test_default_parameters(self):
        client = EnvironmentClient()
        assert client._address == "localhost:50051"
        assert client._timeout == 30.0
        assert client._max_retries == 5
        client.close()

    def test_custom_parameters(self):
        client = EnvironmentClient(
            address="myhost:9090", timeout=10.0, max_retries=3
        )
        assert client._address == "myhost:9090"
        assert client._timeout == 10.0
        assert client._max_retries == 3
        client.close()


class TestEnvironmentClientReset:
    """Tests for EnvironmentClient.reset method."""

    def test_reset_returns_observation(self, mock_server):
        """Req 1.2: Reset returns an Observation."""
        address, servicer = mock_server
        client = EnvironmentClient(address=address, timeout=5.0)
        try:
            obs = client.reset(
                symbol="USDJPY",
                episode_start_ts=1000,
                episode_end_ts=2000,
                step_size_seconds=5,
            )
            assert len(obs.state_columns) == 3
            assert list(obs.state_data[0].values) == [1.0, 2.0, 3.0]
            assert obs.reward == 0.0
            assert obs.done is False
        finally:
            client.close()

    def test_reset_sends_correct_request(self, mock_server):
        """Req 1.2: Reset sends correct ResetRequest fields."""
        address, servicer = mock_server
        client = EnvironmentClient(address=address, timeout=5.0)
        try:
            client.reset(
                symbol="USDJPY",
                episode_start_ts=1000,
                episode_end_ts=2000,
                step_size_seconds=5,
            )
            assert len(servicer.reset_calls) == 1
            req = servicer.reset_calls[0]
            assert req.symbol == "USDJPY"
            assert req.episode_start_ts == 1000
            assert req.episode_end_ts == 2000
            assert req.step_size_seconds == 5
        finally:
            client.close()


class TestEnvironmentClientStep:
    """Tests for EnvironmentClient.step method."""

    def test_step_returns_step_response(self, mock_server):
        """Req 1.3: Step returns a StepResponse."""
        address, servicer = mock_server
        client = EnvironmentClient(address=address, timeout=5.0)
        try:
            response = client.step(action=1, client_order_id="order-123")
            assert response.data.reward == 0.5
            assert response.info == "step executed"
            assert list(response.data.state_data[0].values) == [4.0, 5.0, 6.0]
        finally:
            client.close()

    def test_step_sends_correct_request(self, mock_server):
        """Req 1.3: Step sends correct Action fields."""
        address, servicer = mock_server
        client = EnvironmentClient(address=address, timeout=5.0)
        try:
            client.step(action=2, client_order_id="order-456")
            assert len(servicer.step_calls) == 1
            req = servicer.step_calls[0]
            assert req.action == environment_pb2.ACTION_BUY_2
            assert req.client_order_id == "order-456"
        finally:
            client.close()


class TestEnvironmentClientReferenceData:
    """Tests for EnvironmentClient.reference_data method."""

    def test_reference_data_returns_reference(self, mock_server):
        """Req 1.4: ReferenceData returns a Reference."""
        address, servicer = mock_server
        client = EnvironmentClient(address=address, timeout=5.0)
        try:
            ref = client.reference_data(symbol="USDJPY")
            assert ref.symbol == "USDJPY"
            assert ref.timestamp_ns == 1000000000
        finally:
            client.close()

    def test_reference_data_sends_correct_request(self, mock_server):
        """Req 1.4: ReferenceData sends correct ObserveRequest."""
        address, servicer = mock_server
        client = EnvironmentClient(address=address, timeout=5.0)
        try:
            client.reference_data(symbol="EURUSD")
            assert len(servicer.reference_data_calls) == 1
            req = servicer.reference_data_calls[0]
            assert req.symbol == "EURUSD"
        finally:
            client.close()


class TestEnvironmentClientRetry:
    """Tests for retry logic with exponential backoff."""

    @patch("deepqnetwork.environment_client.time.sleep")
    def test_raises_connection_error_on_exhaustion(self, mock_sleep, failing_server):
        """Req 1.5, 1.6: Raises ConnectionError after max retries."""
        address, servicer = failing_server
        client = EnvironmentClient(address=address, timeout=5.0, max_retries=3)
        try:
            with pytest.raises(ConnectionError, match=address):
                client.reset(
                    symbol="USDJPY",
                    episode_start_ts=1000,
                    episode_end_ts=2000,
                    step_size_seconds=5,
                )
            # Should have attempted exactly max_retries times
            assert servicer.call_count == 3
        finally:
            client.close()

    @patch("deepqnetwork.environment_client.time.sleep")
    def test_retries_with_exponential_backoff(self, mock_sleep, failing_server):
        """Req 1.5: Retries use exponential backoff with jitter."""
        address, servicer = failing_server
        client = EnvironmentClient(address=address, timeout=5.0, max_retries=4)
        try:
            with pytest.raises(ConnectionError):
                client.reset(
                    symbol="USDJPY",
                    episode_start_ts=1000,
                    episode_end_ts=2000,
                    step_size_seconds=5,
                )
            # Should have slept (max_retries - 1) times
            assert mock_sleep.call_count == 3
            # Verify exponential backoff pattern (base=1, factor=2, jitter ±0.5)
            delays = [call.args[0] for call in mock_sleep.call_args_list]
            # delay[0] ≈ 1.0 ± 0.5 → [0.5, 1.5]
            assert 0.0 <= delays[0] <= 1.5
            # delay[1] ≈ 2.0 ± 0.5 → [1.5, 2.5]
            assert 1.5 <= delays[1] <= 2.5
            # delay[2] ≈ 4.0 ± 0.5 → [3.5, 4.5]
            assert 3.5 <= delays[2] <= 4.5
        finally:
            client.close()

    @patch("deepqnetwork.environment_client.time.sleep")
    def test_succeeds_after_retries(self, mock_sleep, eventually_succeeds_server):
        """Req 1.5: Succeeds after transient failures."""
        address, servicer = eventually_succeeds_server
        client = EnvironmentClient(address=address, timeout=5.0, max_retries=5)
        try:
            obs = client.reset(
                symbol="USDJPY",
                episode_start_ts=1000,
                episode_end_ts=2000,
                step_size_seconds=5,
            )
            # Should have succeeded on 3rd attempt
            assert servicer.call_count == 3
            assert list(obs.state_data[0].values) == [42.0]
        finally:
            client.close()

    @patch("deepqnetwork.environment_client.time.sleep")
    def test_connection_error_includes_last_error(self, mock_sleep, failing_server):
        """Req 1.6: ConnectionError includes address and last error."""
        address, servicer = failing_server
        client = EnvironmentClient(address=address, timeout=5.0, max_retries=2)
        try:
            with pytest.raises(ConnectionError) as exc_info:
                client.reset(
                    symbol="USDJPY",
                    episode_start_ts=1000,
                    episode_end_ts=2000,
                    step_size_seconds=5,
                )
            error_msg = str(exc_info.value)
            assert address in error_msg
            assert "2 attempts" in error_msg
        finally:
            client.close()


class TestEnvironmentClientClose:
    """Tests for EnvironmentClient.close method."""

    def test_close_does_not_raise(self, mock_server):
        """Close completes without error."""
        address, _ = mock_server
        client = EnvironmentClient(address=address, timeout=5.0)
        client.close()  # Should not raise
