
ARG DQN_BASE_IMAGE=731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/dqn/base:latest
FROM ${DQN_BASE_IMAGE}

LABEL component="dqn_backtest"
LABEL description="Backtest DQN agent on unseen date ranges via modelenv sidecar, compute metrics and degradation gate"

WORKDIR /app

COPY deepqnetwork/kubeflow/components/dqn_backtest/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY deepqnetwork/ /app/deepqnetwork/

COPY probabilisticforecaster/kubeflow/monitoring/ /app/probabilisticforecaster/kubeflow/monitoring/

COPY deepqnetwork/kubeflow/components/dqn_backtest/component.py /app/component.py


ENTRYPOINT ["python", "component.py"]
