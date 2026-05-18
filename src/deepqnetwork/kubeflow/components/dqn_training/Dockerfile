# DQN Training component for the DQN KFP pipeline
# Extends the shared DQN base image with training-specific dependencies
# Requirements: DQN-R5, DQN-R6, DQN-R7, DQN-R8

FROM 731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/dqn/base:latest

LABEL component="dqn_training"
LABEL description="Train DQN agent with modelenv gRPC sidecar, supports scratch and finetune modes"

WORKDIR /app

# Install component-specific dependencies
COPY deepqnetwork/kubeflow/components/dqn_training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the deepqnetwork package (agent, train, config, network, replay_buffer, etc.)
COPY deepqnetwork/ /app/deepqnetwork/

# Copy the component script
COPY deepqnetwork/kubeflow/components/dqn_training/component.py /app/component.py

# S3 authentication via IRSA; no credentials embedded (Requirement DQN-R5)
# The pod's service account is annotated with an IAM role ARN

ENTRYPOINT ["python", "component.py"]
