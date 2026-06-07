"""KFP pipeline definition for dqnpf-intraday.

A deliberately small pipeline: one container component (``dqnpf_backtest``)
runs the integration-layer backtest against the production-stage DQN and
Forecaster checkpoints resolved from the Model Registry, then emits a single
JSON artifact containing the ``BacktestComparison`` and ``ThresholdReport``.

The pipeline has no training step; the integration layer has no learnable
parameters. Threshold tuning happens offline via
``tradingmodel/intraday/dqnpf/backtest.py``; the resulting frozen thresholds
are committed to ``config/dqnpf_pipeline_config.yaml``.

Requirements: 21.1
"""

from pathlib import Path
from typing import Union

from kfp import compiler, dsl, kubernetes
from kfp.dsl import Dataset, Output

# ECR registry base matches the rest of the platform's images.
ECR_BASE = "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com"
COMPONENT_IMAGE = f"{ECR_BASE}/dqnpf-intraday-backtest:latest"

# The in-pod modelenv sidecar runs in Training mode (modelenv's default) and
# preloads market data on startup. The deepqnetwork warmup DAG populates this
# RWX PVC; mounting it at modelenv's DEFAULT_LOCAL_CACHE_DIR lets the preload
# read from the warm cache instead of pulling everything from S3 (which blows
# past the sidecar healthcheck timeout). Matches the DQN training/backtest
# components (see deepqnetwork/kubeflow/pipeline/dqn_pipeline.py).
MODELENV_CACHE_PVC = "modelenv-cache"
MODELENV_CACHE_MOUNT = "/tmp/modelenv-cache"

PIPELINE_NAME = "dqnpf-intraday-pipeline"
PIPELINE_DESCRIPTION = (
    "Combined DQN + ProbabilisticForecaster intraday integration: runs the "
    "screening backtest against production-stage parent checkpoints from the "
    "Model Registry and emits a BacktestComparison + ThresholdReport artifact."
)

DEFAULT_IR_PATH = Path(__file__).resolve().parent / "dqnpf_pipeline.yaml"


# ---------------------------------------------------------------------------
# Container component
# ---------------------------------------------------------------------------


@dsl.container_component
def dqnpf_backtest(
    integration_config_yaml: str,
    dqn_model_registry_name: str,
    forecaster_model_registry_name: str,
    backtest_report: Output[Dataset],
    trade_log: Output[Dataset],
    # Strings (not floats) so they pass straight into the container args; the
    # __main__ CLI parses them with argparse type=float. "0.0" = no financing.
    # no_swap defaults "true": backtests are swap-free unless explicitly enabled.
    no_swap: str = "true",
    swap_rate_long: str = "0.0",
    swap_rate_short: str = "0.0",
):
    """Run the dqnpf-intraday backtest with the modelenv sidecar in-pod.

    Mirrors ``components/dqnpf_backtest/component.yaml``. The container
    ENTRYPOINT is the ``__main__.py`` CLI in this same package, which
    invokes :func:`run_dqnpf_backtest` with the inputs below.
    """
    return dsl.ContainerSpec(
        image=COMPONENT_IMAGE,
        command=[
            "python",
            "-m",
            "tradingmodel.intraday.dqnpf.kubeflow.components.dqnpf_backtest",
        ],
        args=[
            "--integration-config-yaml",
            integration_config_yaml,
            "--dqn-model-registry-name",
            dqn_model_registry_name,
            "--forecaster-model-registry-name",
            forecaster_model_registry_name,
            "--output-artifact-path",
            backtest_report.uri,
            "--trade-log-output-path",
            trade_log.uri,
            "--no-swap",
            no_swap,
            "--swap-rate-long",
            swap_rate_long,
            "--swap-rate-short",
            swap_rate_short,
        ],
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dsl.pipeline(name=PIPELINE_NAME, description=PIPELINE_DESCRIPTION)
def dqnpf_intraday_pipeline(
    integration_config_yaml: str = (
        "/etc/dqnpf-intraday/config/dqnpf_pipeline_config.yaml"
    ),
    dqn_model_registry_name: str = "deepqnetwork-usdjpy",
    forecaster_model_registry_name: str = "probabilistic-transformer-usdjpy-h1",
    no_swap: str = "true",
    swap_rate_long: str = "0.0",
    swap_rate_short: str = "0.0",
):
    """Single-step DAG: dqnpf_backtest with retry(3).

    Resource limits and modelenv sidecar are declared at the pod level via the
    InferenceService-adjacent manifests (Task 30.5 / Task 31.5); KFP-level
    resource requests live in the component image's spec.
    """
    task = dqnpf_backtest(
        integration_config_yaml=integration_config_yaml,
        dqn_model_registry_name=dqn_model_registry_name,
        forecaster_model_registry_name=forecaster_model_registry_name,
        no_swap=no_swap,
        swap_rate_long=swap_rate_long,
        swap_rate_short=swap_rate_short,
    )
    task.set_retry(num_retries=3)
    # Disable caching: every backtest run resolves fresh production checkpoints
    # and emits a versioned artifact, cached output would be a foot-gun.
    task.set_caching_options(enable_caching=False)
    # Mount the warm modelenv cache so the in-pod sidecar's training-data
    # preload reads locally instead of downloading from S3 on every run.
    kubernetes.mount_pvc(
        task,
        pvc_name=MODELENV_CACHE_PVC,
        mount_path=MODELENV_CACHE_MOUNT,
    )
    # The parent checkpoints resolve to minio:// KFP artifact URIs; the
    # component downloads them via deepqnetwork.artifact_io, which needs the
    # in-cluster MinIO credentials. Mirrors the DQN pipeline's _mount_minio_creds.
    kubernetes.use_secret_as_env(
        task,
        secret_name="mlpipeline-minio-artifact",
        secret_key_to_env={
            "accesskey": "MINIO_ACCESS_KEY",
            "secretkey": "MINIO_SECRET_KEY",
        },
    )


# ---------------------------------------------------------------------------
# Compilation helper
# ---------------------------------------------------------------------------


def compile_pipeline(output_path: Union[str, Path] = DEFAULT_IR_PATH) -> Path:
    """Compile the pipeline to KFP IR YAML.

    Calling this twice with the same KFP version SHALL produce byte-identical
    output (Property DQNPF-PIPE-1). The caller passes ``output_path`` to choose
    where the IR YAML is written; by default it lands next to this module.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=dqnpf_intraday_pipeline,
        package_path=str(output),
    )
    return output


if __name__ == "__main__":
    path = compile_pipeline()
    print(f"compiled IR YAML written to {path}")
