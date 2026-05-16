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
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

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

from tradingmodel.intraday.dqnpf.config import IntegrationConfig
from tradingmodel.intraday.dqnpf.forecaster_bridge import ForecasterBridge
from tradingmodel.intraday.dqnpf.integration import IntegrationLayer, ScreenedAction
from tradingmodel.intraday.dqnpf.signal_cache import SignalCache

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

    def recent_bars(self, symbol: str):
        """Call RecentBars RPC on the modelenv sidecar."""
        request = environment_pb2.RecentBarsRequest(symbol=symbol)
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
            "FORECASTER_MODEL_REGISTRY_NAME", "probabilisticforecaster-usdjpy"
        )

        self._env_client: _ServingEnvClient | None = None
        self._preprocessor: StatePreprocessor | None = None
        self._dqn: DQNAdvisor | None = None
        self._forecaster: ForecasterInference | None = None
        self._layers: dict[str, IntegrationLayer] = {}
        self._bridges: dict[str, ForecasterBridge] = {}
        self._caches: dict[str, SignalCache] = {}

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
        from tradingmodel.intraday.dqnpf.kubeflow.registry.registry_client import (
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

        # Resolve production checkpoint paths from Model Registry
        dqn_path = resolve_production_checkpoint(self._dqn_registry_name, "production")
        fc_path = resolve_production_checkpoint(
            self._forecaster_registry_name, "production"
        )

        logger.info(
            "Resolved checkpoints: dqn=%s, forecaster=%s", dqn_path, fc_path
        )

        # Instantiate shared components
        self._env_client = _ServingEnvClient(
            address=self._grpc_address, timeout=30.0
        )
        self._preprocessor = StatePreprocessor(device=self._device)
        self._dqn = DQNAdvisor(checkpoint_path=dqn_path, device=self._device)

        # Use forecast_horizon from the first config (all symbols share the same
        # forecaster checkpoint; horizon is a model-level parameter)
        first_config = next(iter(self._configs.values()))
        self._forecaster = ForecasterInference(
            model_path=fc_path,
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
        self, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Run combined DQN + Forecaster screening and return ScreenedAction.

        Accepts two payload formats:
        1. ``{"instances": [{"symbol": "USDJPY"}]}``, pulls state and bars
           from the modelenv sidecar.
        2. ``{"symbol": "USDJPY", "observation": [...], "recent_bars_m5": [...]}``
, uses provided observation and bars directly.

        If observation/bars are not provided, they are pulled from the modelenv
        sidecar via EnvironmentClient.

        Args:
            payload: JSON request body.
            headers: Optional HTTP headers (unused).

        Returns:
            Dict containing ScreenedAction fields plus ``mu``.

        Raises:
            ValueError: On unknown symbol, malformed observation, or
                insufficient bars (mapped to HTTP 400 by kserve).
        """
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

            # Screen the action through the integration layer
            screened = self._layers[symbol].screen(dqn_result, mu, sigma)

        # Build response
        response = asdict(screened)
        response["mu"] = mu
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
                from tradingmodel.intraday.dqnpf.kubeflow.registry.registry_client import (
                    resolve_production_checkpoint,
                )

                new_dqn_path = resolve_production_checkpoint(
                    self._dqn_registry_name, "production"
                )
                new_fc_path = resolve_production_checkpoint(
                    self._forecaster_registry_name, "production"
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
            new_dqn = DQNAdvisor(checkpoint_path=new_dqn_path, device=self._device)
            first_config = next(iter(self._configs.values()))
            new_forecaster = ForecasterInference(
                model_path=new_fc_path,
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

                # Preserve budget state from the old layer
                old_layer = self._layers.get(symbol)
                if old_layer is not None:
                    new_layer._risk_long_units = old_layer._risk_long_units
                    new_layer._risk_short_units = old_layer._risk_short_units

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
