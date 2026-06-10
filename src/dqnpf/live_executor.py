"""Live execution loop: act on screened DQNPF actions through modelenv Step().

This is the component that makes DQNPF *act* in production. The serving
predictor only recommends (it deliberately exposes no ``Reset``/``Step``), and
the modelenv live sidecar only actuates when something drives its ``Step()``
RPC. This executor closes the loop:

1. ``Reset()`` once at startup, modelenv syncs the broker book, builds the
   initial observation, and seeds the session-liquidation wall-clock anchor.
2. Every ``step_interval_seconds``: POST the symbol to the predictor's REST
   endpoint (the predictor pulls fresh features from the same sidecar via
   ``ReferenceData``, so the policy, hot-reload, and screening stay in one
   place), then submit the screened action via ``Step()``.

HOLD actions are still stepped: modelenv handles HOLD locally (no broker
order) but each ``Step()`` also re-syncs/reconciles positions and runs the
end-of-session liquidation boundary check, so the executor cadence bounds the
liquidation latency.

The ``client_order_id`` sent with each step is unique per attempt
(``{symbol}-{epoch_ns}``) and is reused across the client's internal gRPC
retries, so a broker that deduplicates by client order id will not
double-fill a retried submission.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

_HOLD_ACTION = 0


@dataclass
class LiveExecutorConfig:
    """Configuration for the live execution loop.

    Attributes:
        symbol: Trading symbol (e.g. "USDJPY").
        predictor_url: Full URL of the predictor's predict endpoint.
        grpc_address: modelenv live sidecar gRPC address.
        step_interval_seconds: Wall-clock pause between steps. The DQN was
            trained on a fixed step size; keep this aligned with it.
        request_timeout: Timeout (seconds) for each predictor HTTP call.
        dry_run: When True, log the screened action but never call Step();
            no broker orders and no session liquidation are triggered.
        max_consecutive_errors: Abort the loop (non-zero exit) after this many
            back-to-back failed iterations, letting the orchestrator restart
            the pod with a fresh Reset().
    """

    symbol: str = "USDJPY"
    predictor_url: str = "http://localhost:8080/v1/models/dqnpf-intraday:predict"
    grpc_address: str = "localhost:50051"
    step_interval_seconds: float = 300.0
    request_timeout: float = 30.0
    dry_run: bool = False
    max_consecutive_errors: int = 10

    @classmethod
    def from_env(cls) -> "LiveExecutorConfig":
        """Build a config from ``DQNPF_*`` environment variables."""
        defaults = cls()
        return cls(
            symbol=os.environ.get("DQNPF_SYMBOL", defaults.symbol),
            predictor_url=os.environ.get(
                "DQNPF_PREDICTOR_URL", defaults.predictor_url
            ),
            grpc_address=os.environ.get(
                "GRPC_ADDRESS", defaults.grpc_address
            ),
            step_interval_seconds=float(
                os.environ.get(
                    "DQNPF_STEP_INTERVAL_SECONDS",
                    defaults.step_interval_seconds,
                )
            ),
            request_timeout=float(
                os.environ.get(
                    "DQNPF_REQUEST_TIMEOUT", defaults.request_timeout
                )
            ),
            dry_run=os.environ.get("DQNPF_DRY_RUN", "").strip().lower()
            in ("1", "true", "yes", "on"),
            max_consecutive_errors=int(
                os.environ.get(
                    "DQNPF_MAX_CONSECUTIVE_ERRORS",
                    defaults.max_consecutive_errors,
                )
            ),
        )


def _post_predict(url: str, symbol: str, timeout: float) -> dict[str, Any]:
    """POST the sidecar-pull payload form to the predictor and decode JSON."""
    body = json.dumps({"instances": [{"symbol": symbol}]}).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_screened_action(response: dict[str, Any]) -> dict[str, Any]:
    """Extract the ScreenedAction dict from a predictor response.

    Accepts both the raw dict the predictor's ``predict`` returns and a
    KServe v1 ``{"predictions": [...]}`` envelope.

    Raises:
        ValueError: If no ``action`` field can be located.
    """
    candidate: Any = response
    if "action" not in candidate and "predictions" in candidate:
        predictions = candidate["predictions"]
        if isinstance(predictions, list) and predictions:
            candidate = predictions[0]
    if not isinstance(candidate, dict) or "action" not in candidate:
        raise ValueError(
            f"Predictor response has no 'action' field: {response!r}"
        )
    return candidate


def run_live_executor(
    config: LiveExecutorConfig,
    env_client: Any | None = None,
    predict_fn: Callable[[str], dict[str, Any]] | None = None,
    stop_event: threading.Event | None = None,
    max_steps: int | None = None,
) -> int:
    """Run the Reset-once / Step-forever live execution loop.

    Args:
        config: Loop configuration.
        env_client: modelenv gRPC client exposing ``reset``/``step``; built
            from ``config.grpc_address`` when omitted. Injectable for tests.
        predict_fn: ``symbol -> predictor response`` callable; defaults to an
            HTTP POST against ``config.predictor_url``. Injectable for tests.
        stop_event: Loop exits cleanly when set (SIGTERM/SIGINT handling);
            also interrupts the inter-step pause immediately.
        max_steps: Stop after this many iterations (None = run until stopped).

    Returns:
        The number of executed steps.

    Raises:
        RuntimeError: After ``config.max_consecutive_errors`` back-to-back
            failed iterations.
    """
    if env_client is None:
        from deepqnetwork.environment_client import EnvironmentClient

        env_client = EnvironmentClient(address=config.grpc_address)
    if predict_fn is None:
        predict_fn = lambda symbol: _post_predict(  # noqa: E731
            config.predictor_url, symbol, config.request_timeout
        )
    if stop_event is None:
        stop_event = threading.Event()

    # One-time bootstrap: broker position sync + initial observation. The
    # episode fields are training-only and ignored by modelenv in Live mode.
    env_client.reset(
        symbol=config.symbol,
        episode_start_ts=0,
        episode_end_ts=0,
        step_size_seconds=0,
    )
    logger.info(
        "live executor started: symbol=%s, interval=%.0fs, dry_run=%s",
        config.symbol,
        config.step_interval_seconds,
        config.dry_run,
    )

    steps = 0
    consecutive_errors = 0
    while not stop_event.is_set():
        if max_steps is not None and steps >= max_steps:
            break
        try:
            screened = parse_screened_action(predict_fn(config.symbol))
            action = int(screened["action"])
            order_id = f"{config.symbol}-{time.time_ns()}"
            if config.dry_run:
                logger.info(
                    "dry-run: would step action=%s (%s, reason=%s) order_id=%s",
                    action,
                    screened.get("action_name"),
                    screened.get("reason"),
                    order_id,
                )
            else:
                env_client.step(action, order_id)
                logger.info(
                    "stepped action=%s (%s, reason=%s, screened=%s) order_id=%s",
                    action,
                    screened.get("action_name"),
                    screened.get("reason"),
                    screened.get("screened"),
                    order_id,
                )
            steps += 1
            consecutive_errors = 0
        except Exception:
            consecutive_errors += 1
            logger.exception(
                "live executor iteration failed (%d/%d consecutive)",
                consecutive_errors,
                config.max_consecutive_errors,
            )
            if consecutive_errors >= config.max_consecutive_errors:
                raise RuntimeError(
                    f"Aborting after {consecutive_errors} consecutive "
                    "failed iterations"
                )
        # wait() doubles as the inter-step pause and the shutdown latch: a
        # SIGTERM during the pause exits immediately instead of after a full
        # interval.
        stop_event.wait(config.step_interval_seconds)

    logger.info("live executor stopped after %d steps", steps)
    return steps


def main() -> int:
    """Entry point: ``python -m tradingmodel.intraday.dqnpf.live_executor``."""
    import signal

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = LiveExecutorConfig.from_env()
    stop_event = threading.Event()

    def _request_stop(signum: int, _frame: Any) -> None:
        logger.info("received signal %d, stopping after current step", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    run_live_executor(config, stop_event=stop_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
