"""KServe predictor: combined DQN + Forecaster integration layer.

Single Python process loads DQNAdvisor and ForecasterInference from the
production-stage checkpoints, instantiates IntegrationLayer per symbol, and
returns ScreenedAction responses. modelenv runs as a sidecar in the same pod.

Supports registry-driven hot-reload: when a new DQN or Forecaster production
checkpoint is registered, the predictor atomically swaps in new model instances
without restarting the pod, preserving per-symbol budget state.

Requirements: 22.1, 22.2, 22.3, 22.5, 22.6, 22.9, 22.12, 23.4
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import grpc
import kserve
import numpy as np

import environment_pb2
import environment_pb2_grpc

from deepqnetwork.advisor import DQNAdvisor
from deepqnetwork.environment_client import EnvironmentClient
from deepqnetwork.preprocessor import StatePreprocessor
from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.inference import ForecasterInference

from dqnpf.config import IntegrationConfig
from dqnpf.forecaster_bridge import ForecasterBridge
from dqnpf.integration import IntegrationLayer, ScreenedAction
from dqnpf.signal_cache import SignalCache

logger = logging.getLogger(__name__)

_REGISTRY_POLL_INTERVAL_SECONDS = int(
    os.environ.get("REGISTRY_POLL_INTERVAL_SECONDS", "30")
)


def _structured_log(event: str, **fields: Any) -> None:
    """Emit a structured JSON log line for hot-reload events."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "component": "dqnpf-intraday-predictor",
        "event": event,
        **fields,
    }
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def _download_uri_to_local(uri: str) -> str:
    """Ensure ``uri`` is available on the local filesystem and return the path.

    DQNAdvisor and ForecasterInference call ``torch.load()`` (and
    ``os.path.exists``), which require a local path. The Model Registry,
    however, stores the artifact URI in whatever scheme KFP wrote it with
    (``minio://...`` or ``s3://...``), so we have to materialise it locally
    first.

    For URIs with a scheme we route through
    ``probabilisticforecaster.artifact_io.get_object_bytes`` (which handles
    MinIO via env-mounted credentials and AWS S3 via IRSA), writing the
    payload to a NamedTemporaryFile. For paths without a scheme we assume
    the file is already local.
    """
    parsed = urlparse(uri)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("minio", "s3"):
        # Already a local path, nothing to do.
        return uri

    # Lazy import so unit tests / dev environments that lack the helper still
    # work for local paths.
    from probabilisticforecaster.artifact_io import get_object_bytes

    data = get_object_bytes(uri)
    fd, local_path = tempfile.mkstemp(suffix=".pt", prefix="dqnpf-ckpt-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except Exception:
        # Best-effort cleanup on write failure.
        try:
            os.unlink(local_path)
        except OSError:
            pass
        raise
    return local_path


# ScreenedAction-dict key -> KServe v2 output tensor datatype. Single source
# of truth for the gRPC response shape; the executor decodes by tensor name.
_INFER_OUTPUT_DATATYPES: dict[str, str] = {
    "action": "INT64",
    "action_name": "BYTES",
    "screened": "BOOL",
    "reason": "BYTES",
    "sigma": "FP64",
    "risk_long_used": "INT64",
    "risk_short_used": "INT64",
    "mu": "FP64",
}


def _payload_from_infer_request(request: Any) -> dict[str, Any]:
    """Map a KServe v2 ``InferRequest`` onto the dict payload form.

    Only the sidecar-pull form is supported over gRPC: a single BYTES input
    named ``symbol`` with one element.

    Raises:
        ValueError: If the request carries no usable ``symbol`` input.
    """
    for tensor in getattr(request, "inputs", None) or []:
        if tensor.name == "symbol":
            data = list(tensor.data or [])
            if not data:
                break
            value = data[0]
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8")
            return {"symbol": str(value)}
    raise ValueError(
        "InferRequest must carry a BYTES input named 'symbol' with one element"
    )


def _screened_to_infer_response(
    model_name: str, infer_request: Any, response: dict[str, Any]
) -> Any:
    """Build a KServe v2 ``InferResponse`` from the ScreenedAction dict.

    Each field becomes a shape-[1] output tensor named after the dict key, so
    gRPC clients (the live executor) can decode by name without a shared
    schema beyond ``_INFER_OUTPUT_DATATYPES``.
    """
    outputs = [
        kserve.InferOutput(
            name=key, shape=[1], datatype=datatype, data=[response[key]]
        )
        for key, datatype in _INFER_OUTPUT_DATATYPES.items()
    ]
    return kserve.InferResponse(
        response_id=getattr(infer_request, "id", "") or "",
        model_name=model_name,
        infer_outputs=outputs,
    )


def _m5_close_from_reference(reference: Any) -> float | None:
    """Latest M5 close from a ``Reference``'s live_bars, or ``None``."""
    live_bars = getattr(reference, "live_bars", None)
    if not live_bars or "M5" not in live_bars:
        return None
    bar = live_bars["M5"]
    close = getattr(bar, "close", None)
    return float(close) if close else None


class _ServingEnvClient:
    """Thin wrapper around EnvironmentClient adding ``recent_bars`` support.

    The base ``EnvironmentClient`` exposes Reset, Step, and ReferenceData.
    The predictor additionally needs the ``RecentBars`` RPC for the
    ForecasterBridge. This wrapper delegates standard calls to the base
    client and adds ``recent_bars`` via the gRPC stub directly.
    """

    def __init__(self, address: str = "localhost:50051", timeout: float = 30.0) -> None:
        self._base = EnvironmentClient(address=address, timeout=timeout)
        self._channel = grpc.insecure_channel(address)
        self._stub = environment_pb2_grpc.EnvironmentStub(self._channel)
        self._timeout = timeout

    def reference_data(self, symbol: str):
        """Delegate to base EnvironmentClient."""
        return self._base.reference_data(symbol)

    def recent_bars(self, symbol: str, count: int = 0):
        """Call RecentBars RPC on the modelenv sidecar.

        ``count`` is forwarded so the ForecasterBridge can request its deep
        feature-window history (default 0 -> server's RECENT_WINDOW).
        """
        request = environment_pb2.RecentBarsRequest(symbol=symbol, count=count)
        return self._stub.RecentBars(request, timeout=self._timeout)

    def close(self) -> None:
        """Close both channels."""
        self._base.close()
        self._channel.close()


class DqnpfIntradayPredictor(kserve.Model):
    """Combined predictor returning a screened trading action.

    Loads both parent models in-process, runs the integration screening rules,
    and serves a single REST endpoint. Maintains per-symbol IntegrationLayer
    instances with independent risk-budget state.

    Args:
        name: KServe model name.
        configs: Mapping of symbol → IntegrationConfig for each symbol served.
    """

    def __init__(self, name: str, configs: dict[str, IntegrationConfig]) -> None:
        super().__init__(name)
        self._configs = configs
        self._grpc_address = os.environ.get("GRPC_ADDRESS", "localhost:50051")
        self._device = os.environ.get("DEVICE", "cpu")
        self._dqn_registry_name = os.environ.get(
            "DQN_MODEL_REGISTRY_NAME", "deepqnetwork-usdjpy"
        )
        self._forecaster_registry_name = os.environ.get(
            "FORECASTER_MODEL_REGISTRY_NAME", "probabilistic-transformer-usdjpy-h1"
        )

        self._env_client: _ServingEnvClient | None = None
        self._preprocessor: StatePreprocessor | None = None
        self._dqn: DQNAdvisor | None = None
        self._forecaster: ForecasterInference | None = None
        self._layers: dict[str, IntegrationLayer] = {}
        self._bridges: dict[str, ForecasterBridge] = {}
        self._caches: dict[str, SignalCache] = {}
        # Profit-gate: last modelenv session-start timestamp seen per symbol.
        # The live session boundary is the authoritative `session_start_ts` that
        # modelenv surfaces on the Reference (the same anchor that drives
        # end-of-session liquidation); when it advances we roll the gate via
        # IntegrationLayer.begin_session().
        self._session_start_ts: dict[str, int] = {}

        # Hot-reload state
        self._model_lock = threading.RLock()
        self._current_dqn_path: str | None = None
        self._current_fc_path: str | None = None
        self._watcher_thread: threading.Thread | None = None
        self._watcher_stop_event = threading.Event()

    def load(self) -> bool:
        """Resolve checkpoints from Model Registry and instantiate per-symbol layers.

        Resolves ``dqn_model_registry_name`` and ``forecaster_model_registry_name``
        to production checkpoint paths, instantiates DQNAdvisor and
        ForecasterInference in-process, and builds per-symbol ForecasterBridge,
        SignalCache, and IntegrationLayer.

        Starts the registry watcher thread for hot-reload on completion.

        Returns:
            True when the model is ready to serve.
        """
        from dqnpf.kubeflow.registry.registry_client import (
            resolve_production_checkpoint,
        )

        logger.info(
            "Loading dqnpf-intraday predictor: dqn_registry=%s, "
            "forecaster_registry=%s, device=%s, grpc=%s",
            self._dqn_registry_name,
            self._forecaster_registry_name,
            self._device,
            self._grpc_address,
        )

        # Resolve production checkpoint paths from Model Registry.
        # resolve_production_checkpoint's 2nd positional arg is registry_url,
        # not a lifecycle stage; it always resolves the production-stage
        # version. Passing "production" here made it the URL (is_secure=True →
        # "user token required"). Omit it so the in-cluster default URL is used.
        dqn_path = resolve_production_checkpoint(self._dqn_registry_name)
        fc_path = resolve_production_checkpoint(self._forecaster_registry_name)

        logger.info(
            "Resolved checkpoints: dqn=%s, forecaster=%s", dqn_path, fc_path
        )

        # Instantiate shared components
        self._env_client = _ServingEnvClient(
            address=self._grpc_address, timeout=30.0
        )
        self._preprocessor = StatePreprocessor(device=self._device)

        # Resolve URIs (minio://, s3://) to local files before handing to
        # torch.load-based loaders, which only accept local paths.
        local_dqn_path = _download_uri_to_local(dqn_path)
        local_fc_path = _download_uri_to_local(fc_path)

        self._dqn = DQNAdvisor(checkpoint_path=local_dqn_path, device=self._device)

        # Use forecast_horizon from the first config (all symbols share the same
        # forecaster checkpoint; horizon is a model-level parameter)
        first_config = next(iter(self._configs.values()))
        self._forecaster = ForecasterInference(
            model_path=local_fc_path,
            config=ForecasterConfig(forecast_horizon=first_config.forecast_horizon),
        )

        # Build per-symbol integration layers
        for symbol, cfg in self._configs.items():
            cache = SignalCache()
            bridge = ForecasterBridge(self._forecaster, symbol, self._env_client)
            layer = IntegrationLayer(self._dqn, bridge, cache, cfg)
            self._layers[symbol] = layer
            self._bridges[symbol] = bridge
            self._caches[symbol] = cache

        # Track current checkpoint paths for hot-reload comparison
        self._current_dqn_path = dqn_path
        self._current_fc_path = fc_path

        self.ready = True
        logger.info(
            "dqnpf-intraday predictor loaded: symbols=%s",
            list(self._configs.keys()),
        )

        # Start registry watcher for hot-reload
        self._start_registry_watcher()

        return True

    def predict(
        self, payload: Any, headers: dict[str, str] | None = None
    ) -> Any:
        """Run combined DQN + Forecaster screening and return ScreenedAction.

        Accepts three payload formats:
        1. ``{"instances": [{"symbol": "USDJPY"}]}``, pulls state and bars
           from the modelenv sidecar (REST v1).
        2. ``{"symbol": "USDJPY", "observation": [...], "recent_bars_m5": [...]}``
, uses provided observation and bars directly (REST v1).
        3. A KServe v2 ``InferRequest`` (gRPC Open Inference Protocol) with a
           single BYTES input named ``symbol``, pulls state and bars from the
           sidecar and answers with an ``InferResponse`` whose output tensors
           carry the ScreenedAction fields plus ``mu``.

        If observation/bars are not provided, they are pulled from the modelenv
        sidecar via EnvironmentClient.

        Args:
            payload: JSON request body (dict) or a v2 InferRequest.
            headers: Optional HTTP headers (unused).

        Returns:
            Dict containing ScreenedAction fields plus ``mu`` (REST), or an
            ``InferResponse`` mirroring those fields (gRPC).

        Raises:
            ValueError: On unknown symbol, malformed observation, or
                insufficient bars (mapped to HTTP 400 by kserve).
        """
        # KServe v2 gRPC path: decode the InferRequest into the dict payload
        # form so both protocols share the screening logic below.
        infer_request = None
        if not isinstance(payload, dict):
            infer_request = payload
            payload = _payload_from_infer_request(payload)

        # Parse symbol from payload
        symbol = self._extract_symbol(payload)

        if symbol not in self._layers:
            raise ValueError(
                f"Unknown symbol: {symbol!r}. "
                f"Available symbols: {list(self._configs.keys())}"
            )

        # Acquire lock to ensure we see a consistent model snapshot
        with self._model_lock:
            # Get state vector
            state = self._resolve_state(payload, symbol)

            # Get DQN action recommendation
            dqn_result = self._dqn.recommend_action(state)  # type: ignore[union-attr]

            # Get (mu, sigma) from forecaster via cache
            bridge = self._bridges[symbol]
            cache = self._caches[symbol]
            latest_ts = self._resolve_latest_bar_ts(payload, symbol)
            mu, sigma = cache.get_or_compute(latest_ts, bridge.compute_signal)

            layer = self._layers[symbol]

            # Profit gate (when enabled): drive the session boundary off the
            # authoritative `session_start_ts` modelenv surfaces on the Reference
            # (the same anchor as end-of-session liquidation) rather than a
            # client-side heuristic. When it advances, roll the prior session's
            # screen counterfactual and re-decide whether the screen stays
            # active; pass the latest M5 close (from the same Reference) so the
            # gate can mark its next-bar counterfactual.
            reference = self._env_client.reference_data(symbol)  # type: ignore[union-attr]
            sess_ts = int(getattr(reference, "session_start_ts", 0) or 0)
            if sess_ts and self._session_start_ts.get(symbol) != sess_ts:
                layer.begin_session()
                self._session_start_ts[symbol] = sess_ts
            screen_price = _m5_close_from_reference(reference)

            # Screen the action through the integration layer. Pass the latest
            # bar timestamp so the risk budget resets per UTC day in production
            # (without it the budget accumulates for the pod lifetime); the same
            # timestamp already keys the signal cache above.
            screened = layer.screen(
                dqn_result, mu, sigma, timestamp_ns=latest_ts, price=screen_price
            )

        # Build response
        response = asdict(screened)
        response["mu"] = mu
        if infer_request is not None:
            return _screened_to_infer_response(self.name, infer_request, response)
        return response

    # -------------------------------------------------------------------------
    # Hot-reload: registry-driven model swap
    # -------------------------------------------------------------------------

    def _start_registry_watcher(self) -> None:
        """Start a background daemon thread that polls the Model Registry.

        The watcher periodically resolves the latest production checkpoint
        paths for both parent models. When a change is detected, it triggers
        ``_hot_reload()`` to atomically swap in new model instances.
        """
        self._watcher_thread = threading.Thread(
            target=self._registry_watcher_loop,
            name="registry-watcher",
            daemon=True,
        )
        self._watcher_thread.start()
        logger.info(
            "Registry watcher started: poll_interval=%ds",
            _REGISTRY_POLL_INTERVAL_SECONDS,
        )

    def _registry_watcher_loop(self) -> None:
        """Background loop that polls the Model Registry for version changes."""
        while not self._watcher_stop_event.is_set():
            self._watcher_stop_event.wait(timeout=_REGISTRY_POLL_INTERVAL_SECONDS)
            if self._watcher_stop_event.is_set():
                break

            try:
                from dqnpf.kubeflow.registry.registry_client import (
                    resolve_production_checkpoint,
                )

                new_dqn_path = resolve_production_checkpoint(self._dqn_registry_name)
                new_fc_path = resolve_production_checkpoint(
                    self._forecaster_registry_name
                )

                dqn_changed = new_dqn_path != self._current_dqn_path
                fc_changed = new_fc_path != self._current_fc_path

                if dqn_changed or fc_changed:
                    _structured_log(
                        "hot_reload_triggered",
                        old_dqn_path=self._current_dqn_path,
                        new_dqn_path=new_dqn_path,
                        old_fc_path=self._current_fc_path,
                        new_fc_path=new_fc_path,
                        dqn_changed=dqn_changed,
                        fc_changed=fc_changed,
                    )
                    self._hot_reload(new_dqn_path, new_fc_path)

            except Exception as exc:
                logger.debug(
                    "Registry watcher poll error (non-fatal): %s", exc
                )

    def _hot_reload(self, new_dqn_path: str, new_fc_path: str) -> None:
        """Atomically swap in new DQNAdvisor and ForecasterInference instances.

        Preserves per-symbol budget state (``_risk_long_units``,
        ``_risk_short_units`` in each IntegrationLayer) across the swap.
        On failure, logs the error and triggers a pod restart via sys.exit(1).

        Args:
            new_dqn_path: S3 path to the new DQN production checkpoint.
            new_fc_path: S3 path to the new Forecaster production checkpoint.
        """
        start_time = time.monotonic()

        try:
            # Load new model instances outside the lock (IO-heavy)
            local_dqn_path = _download_uri_to_local(new_dqn_path)
            local_fc_path = _download_uri_to_local(new_fc_path)

            new_dqn = DQNAdvisor(checkpoint_path=local_dqn_path, device=self._device)
            first_config = next(iter(self._configs.values()))
            new_forecaster = ForecasterInference(
                model_path=local_fc_path,
                config=ForecasterConfig(forecast_horizon=first_config.forecast_horizon),
            )

            # Build new per-symbol bridges and layers, preserving budget state
            new_layers: dict[str, IntegrationLayer] = {}
            new_bridges: dict[str, ForecasterBridge] = {}

            for symbol, cfg in self._configs.items():
                new_bridge = ForecasterBridge(
                    new_forecaster, symbol, self._env_client
                )

                # Preserve existing SignalCache (will be invalidated on next bar)
                cache = self._caches[symbol]

                new_layer = IntegrationLayer(new_dqn, new_bridge, cache, cfg)

                # Preserve budget state from the old layer. _current_day must be
                # carried over alongside the counters: the layer resets the
                # budget on each UTC day boundary, so a fresh (None) day on the
                # new layer would zero the just-preserved counters on the very
                # next screen, defeating the preservation.
                old_layer = self._layers.get(symbol)
                if old_layer is not None:
                    new_layer._risk_long_units = old_layer._risk_long_units
                    new_layer._risk_short_units = old_layer._risk_short_units
                    new_layer._current_day = old_layer._current_day
                    # Preserve the profit gate's rolling counterfactual memory
                    # across the swap; otherwise a hot-reload would reset the
                    # trailing-window history and re-arm the screen from scratch.
                    new_layer._cf_history = old_layer._cf_history
                    new_layer._session_cf = old_layer._session_cf
                    new_layer._pending = old_layer._pending
                    new_layer._gate_active = old_layer._gate_active
                    new_layer._session_started = old_layer._session_started

                new_layers[symbol] = new_layer
                new_bridges[symbol] = new_bridge

            # Atomic swap under the lock
            with self._model_lock:
                self._dqn = new_dqn
                self._forecaster = new_forecaster
                self._layers = new_layers
                self._bridges = new_bridges
                self._current_dqn_path = new_dqn_path
                self._current_fc_path = new_fc_path

            duration_ms = (time.monotonic() - start_time) * 1000
            _structured_log(
                "hot_reload_success",
                duration_ms=round(duration_ms, 1),
                new_dqn_path=new_dqn_path,
                new_fc_path=new_fc_path,
            )
            logger.info(
                "Hot-reload completed in %.1fms: dqn=%s, forecaster=%s",
                duration_ms,
                new_dqn_path,
                new_fc_path,
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            _structured_log(
                "hot_reload_failure",
                error=str(exc),
                duration_ms=round(duration_ms, 1),
                new_dqn_path=new_dqn_path,
                new_fc_path=new_fc_path,
                fallback="pod_restart",
            )
            logger.error(
                "Hot-reload failed after %.1fms: %s. Falling back to pod restart.",
                duration_ms,
                exc,
            )
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _extract_symbol(self, payload: dict[str, Any]) -> str:
        """Extract symbol from either payload format.

        Supports:
        - ``{"symbol": "USDJPY", ...}``
        - ``{"instances": [{"symbol": "USDJPY"}]}``
        """
        if "symbol" in payload:
            return str(payload["symbol"])

        instances = payload.get("instances")
        if instances and isinstance(instances, list) and len(instances) > 0:
            instance = instances[0]
            if isinstance(instance, dict) and "symbol" in instance:
                return str(instance["symbol"])

        raise ValueError(
            "Missing 'symbol' field. Provide either "
            '{"symbol": "USDJPY"} or {"instances": [{"symbol": "USDJPY"}]}'
        )

    def _resolve_state(
        self, payload: dict[str, Any], symbol: str
    ) -> np.ndarray:
        """Resolve the state vector from payload or modelenv sidecar.

        If ``observation`` is provided in the payload, validates and uses it
        directly. Otherwise, pulls state from the modelenv sidecar via
        ``EnvironmentClient.reference_data``.

        Raises:
            ValueError: If the observation is malformed (wrong type or shape).
        """
        observation = payload.get("observation")
        if observation is None:
            # Check instances format
            instances = payload.get("instances")
            if instances and isinstance(instances, list) and len(instances) > 0:
                instance = instances[0]
                if isinstance(instance, dict):
                    observation = instance.get("observation")

        if observation is not None:
            # Validate and convert provided observation
            try:
                state = np.asarray(observation, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed observation: cannot convert to float array. "
                    f"Detail: {exc}"
                ) from exc

            if state.ndim != 1:
                raise ValueError(
                    f"Observation must be a 1-D array, got shape {state.shape}"
                )
            return state

        # Pull from modelenv sidecar
        obs = self._env_client.reference_data(symbol)  # type: ignore[union-attr]
        return self._preprocessor.process(obs).numpy()  # type: ignore[union-attr]

    def _resolve_latest_bar_ts(
        self, payload: dict[str, Any], symbol: str
    ) -> int:
        """Resolve the latest M5 bar timestamp from payload or modelenv sidecar.

        If ``recent_bars_m5`` is provided in the payload, extracts the last
        bar's timestamp. Otherwise, pulls from the modelenv sidecar.

        Raises:
            ValueError: If insufficient bars are provided (< 36).
        """
        recent_bars = payload.get("recent_bars_m5")
        if recent_bars is None:
            instances = payload.get("instances")
            if instances and isinstance(instances, list) and len(instances) > 0:
                instance = instances[0]
                if isinstance(instance, dict):
                    recent_bars = instance.get("recent_bars_m5")

        if recent_bars is not None:
            # Validate bar count
            if not isinstance(recent_bars, list) or len(recent_bars) < 36:
                bar_count = len(recent_bars) if isinstance(recent_bars, list) else 0
                raise ValueError(
                    f"Insufficient M5 bars: got {bar_count}, need at least 36"
                )
            # Extract timestamp from last bar
            last_bar = recent_bars[-1]
            if isinstance(last_bar, dict):
                ts = last_bar.get("timestamp_ns") or last_bar.get("Timestamp", 0)
                return int(ts)
            return int(last_bar)

        # Pull from modelenv sidecar
        response = self._env_client.recent_bars(symbol)  # type: ignore[union-attr]
        bars = getattr(response, "bars", None)
        if bars is None or "M5" not in bars:
            raise ValueError(
                f"RecentBars response for {symbol} missing 'M5' key; "
                f"cannot determine latest bar timestamp"
            )
        series = bars["M5"].bars
        if not series:
            raise ValueError(
                f"RecentBars M5 series for {symbol} is empty"
            )
        return int(series[-1].timestamp_ns)
