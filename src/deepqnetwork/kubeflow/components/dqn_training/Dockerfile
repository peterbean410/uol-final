
ARG DQN_BASE_IMAGE=731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/dqn/base:latest
FROM ${DQN_BASE_IMAGE}

LABEL component="dqn_training"
LABEL description="Train DQN agent with modelenv gRPC sidecar, supports scratch and finetune modes"

WORKDIR /app

COPY deepqnetwork/kubeflow/components/dqn_training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY deepqnetwork/ /app/deepqnetwork/

COPY probabilisticforecaster/kubeflow/monitoring/ /app/probabilisticforecaster/kubeflow/monitoring/

COPY deepqnetwork/kubeflow/components/dqn_training/component.py /app/component.py


ENTRYPOINT ["python", "component.py"]
