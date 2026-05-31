"""gRPC client wrapper for the Model Environment service.

Provides a Python interface to the Rust gRPC model environment, handling
connection management, retries with exponential backoff, and clean shutdown.
"""

import logging
import random
import time

import grpc

import environment_pb2
import environment_pb2_grpc

logger = logging.getLogger(__name__)


class EnvironmentClient:
    """gRPC client wrapping the Environment service.

    Connects to the model environment server and exposes Reset, Step, and
    ReferenceData RPCs with configurable timeout and exponential backoff retry.

    Args:
        address: gRPC server address (host:port).
        timeout: Timeout in seconds for each RPC call.
        max_retries: Maximum number of retry attempts before raising.
    """

    def __init__(
        self,
        address: str = "localhost:50051",
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        """Initialise the environment client.

        Args:
            address: gRPC server address (default: localhost:50051).
            timeout: Timeout in seconds for each RPC call (default: 30.0).
            max_retries: Maximum retry attempts with exponential backoff (default: 5).
        """
        self._address = address
        self._timeout = timeout
        self._max_retries = max_retries

        self._channel = grpc.insecure_channel(address)
        self._stub = environment_pb2_grpc.EnvironmentStub(self._channel)

        logger.info(
            "EnvironmentClient initialised: address=%s, timeout=%.1fs, max_retries=%d",
            address,
            timeout,
            max_retries,
        )

    def reset(
        self,
        symbol: str,
        episode_start_ts: int,
        episode_end_ts: int,
        step_size_seconds: int,
    ):
        """Call Reset() RPC to start a new episode.

        Args:
            symbol: Trading symbol (e.g. "USDJPY").
            episode_start_ts: Episode start timestamp (unix seconds).
            episode_end_ts: Episode end timestamp (unix seconds).
            step_size_seconds: Step size in seconds between observations.

        Returns:
            An Observation protobuf message with the initial state.

        Raises:
            ConnectionError: If all retry attempts are exhausted.
        """
        request = environment_pb2.ResetRequest(
            symbol=symbol,
            episode_start_ts=episode_start_ts,
            episode_end_ts=episode_end_ts,
            step_size_seconds=step_size_seconds,
        )
        return self._call_with_retry("Reset", request)

    def step(self, action: int, client_order_id: str):
        """Call Step() RPC to advance the environment by one timestep.

        Args:
            action: Action index (0-4) corresponding to ActionType enum.
            client_order_id: Unique identifier for this order.

        Returns:
            A StepResponse protobuf message containing the next observation.

        Raises:
            ConnectionError: If all retry attempts are exhausted.
        """
        request = environment_pb2.Action(
            action=action,
            client_order_id=client_order_id,
        )
        return self._call_with_retry("Step", request)

    def reference_data(self, symbol: str):
        """Call ReferenceData() RPC to retrieve full reference observation.

        Args:
            symbol: Trading symbol (e.g. "USDJPY").

        Returns:
            A Reference protobuf message with full market reference data.

        Raises:
            ConnectionError: If all retry attempts are exhausted.
        """
        request = environment_pb2.ObserveRequest(symbol=symbol)
        return self._call_with_retry("ReferenceData", request)

    def recent_bars(self, symbol: str, count: int = 0):
        """Call RecentBars() RPC to retrieve recent completed bars per interval.

        Args:
            symbol: Trading symbol (e.g. "USDJPY").
            count: Number of most-recent completed bars to request per interval.
                0 (default) lets the server use its default RECENT_WINDOW.
                Deep-history consumers (e.g. the forecaster's 1440-bar feature
                window) pass a larger value.

        Returns:
            A RecentBarsResponse protobuf message whose ``bars`` field maps an
            interval label (e.g. "M5") to a BarList of completed Bars.

        Raises:
            ConnectionError: If all retry attempts are exhausted.
        """
        request = environment_pb2.RecentBarsRequest(symbol=symbol, count=count)
        return self._call_with_retry("RecentBars", request)

    def close(self) -> None:
        """Close the gRPC channel cleanly."""
        self._channel.close()
        logger.info("EnvironmentClient channel closed: address=%s", self._address)

    def _call_with_retry(self, method_name: str, request):
        """Execute an RPC call with exponential backoff retry.

        Retry strategy: base delay 1s, factor 2, jitter ±0.5s, max attempts
        as configured by max_retries.

        Args:
            method_name: Name of the RPC method on the stub (e.g. "Reset").
            request: The protobuf request message.

        Returns:
            The protobuf response message from the server.

        Raises:
            ConnectionError: If all retry attempts are exhausted, with the
                server address and last error details.
        """
        last_error: Exception | None = None
        base_delay = 1.0
        factor = 2.0

        for attempt in range(self._max_retries):
            try:
                rpc_method = getattr(self._stub, method_name)
                response = rpc_method(request, timeout=self._timeout)
                return response
            except grpc.RpcError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = base_delay * (factor ** attempt)
                    jitter = random.uniform(-0.5, 0.5)
                    sleep_time = max(0.0, delay + jitter)
                    logger.warning(
                        "%s RPC failed (attempt %d/%d): %s. "
                        "Retrying in %.2fs...",
                        method_name,
                        attempt + 1,
                        self._max_retries,
                        e,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        "%s RPC failed after %d attempts: %s",
                        method_name,
                        self._max_retries,
                        e,
                    )

        raise ConnectionError(
            f"Failed to connect to environment server at {self._address} "
            f"after {self._max_retries} attempts. Last error: {last_error}"
        )
