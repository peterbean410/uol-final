"""Pipeline definition and configuration for the dqnpf-intraday KFP pipeline."""

from tradingmodel.intraday.dqnpf.kubeflow.pipeline.config_schema import (
    DqnpfPipelineConfig,
)

__all__ = ["DqnpfPipelineConfig"]
