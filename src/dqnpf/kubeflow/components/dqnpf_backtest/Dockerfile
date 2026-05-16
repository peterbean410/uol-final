# dqnpf_backtest KFP component image.
# Extends the dqnpf-intraday-base image which already provides
# DQNAdvisor, ForecasterInference, IntegrationLayer, and the modelenv binary.
# Requirements: 20.1, 21.1, 21.2

ARG DQNPF_INTRADAY_BASE=731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/dqnpf-intraday-base:latest
FROM ${DQNPF_INTRADAY_BASE}

LABEL component="dqnpf-intraday-backtest"

# The component module is shipped inside the base image (installed from
# tradingmodel/intraday/dqnpf), so no extra COPY is needed here. The component
# CLI entry point lives at tradingmodel/intraday/dqnpf/kubeflow/components/
# dqnpf_backtest/__main__.py.

ENTRYPOINT ["python", "-m", "tradingmodel.intraday.dqnpf.kubeflow.components.dqnpf_backtest"]
