# dqnpf_backtest KFP component image.
# Extends the dqnpf-intraday-base image which already provides
# DQNAdvisor, ForecasterInference, IntegrationLayer, and the modelenv binary.
# Requirements: 20.1, 21.1, 21.2

ARG DQNPF_INTRADAY_BASE=731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/dqnpf-intraday-base:latest
FROM ${DQNPF_INTRADAY_BASE}

LABEL component="dqnpf-intraday-backtest"

# The component module is shipped inside the base image (installed from
# dqnpf), so no extra COPY is needed for code. The
# component CLI entry point lives at
# dqnpf/kubeflow/components/dqnpf_backtest/__main__.py.
#
# Pipeline-level config: the KFP pipeline DSL passes
# `--integration-config-yaml /etc/dqnpf-intraday/config/dqnpf_pipeline_config.yaml`
# as the default. Bake the source-controlled config to that path so the
# default works out-of-the-box. Run-time overrides (e.g. mounting a
# ConfigMap at the same path, or passing a different --integration-config-yaml)
# still take precedence.
COPY dqnpf/kubeflow/pipeline/config/dqnpf_pipeline_config.yaml \
     /etc/dqnpf-intraday/config/dqnpf_pipeline_config.yaml

ENTRYPOINT ["python", "-m", "dqnpf.kubeflow.components.dqnpf_backtest"]
