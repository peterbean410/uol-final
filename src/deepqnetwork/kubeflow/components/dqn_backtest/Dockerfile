# DQN Backtest component for the DQN KFP pipeline
# Extends the shared DQN base image with backtest-specific dependencies
# Requirements: DQN-R9, DQN-R10

FROM 731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/dqn-base:latest

LABEL component="dqn_backtest"
LABEL description="Backtest DQN agent on unseen date ranges via modelenv sidecar, compute metrics and degradation gate"

WORKDIR /app

# Install component-specific dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the deepqnetwork package (advisor, network, preprocessor, utils, config, etc.)
COPY deepqnetwork/ /app/deepqnetwork/

# Copy the probabilisticforecaster monitoring module for structured logging
COPY probabilisticforecaster/kubeflow/monitoring/ /app/probabilisticforecaster/kubeflow/monitoring/

# Copy the component script
COPY deepqnetwork/kubeflow/components/dqn_backtest/component.py /app/component.py

# S3 authentication via IRSA; no credentials embedded (Requirement DQN-R5)
# The pod's service account is annotated with an IAM role ARN

ENTRYPOINT ["python", "component.py"]
