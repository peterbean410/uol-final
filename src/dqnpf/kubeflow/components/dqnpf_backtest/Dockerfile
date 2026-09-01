
ARG DQNPF_INTRADAY_BASE=731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/dqnpf-intraday-base:latest
FROM ${DQNPF_INTRADAY_BASE}

LABEL component="dqnpf-intraday-backtest"

COPY dqnpf/kubeflow/pipeline/config/dqnpf_pipeline_config.yaml \
     /etc/dqnpf-intraday/config/dqnpf_pipeline_config.yaml

ENTRYPOINT ["python", "-m", "dqnpf.kubeflow.components.dqnpf_backtest"]
