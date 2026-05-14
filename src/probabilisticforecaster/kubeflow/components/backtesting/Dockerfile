# Backtesting component for the Forecaster KFP pipeline
# Extends the shared base image with component-specific dependencies
# Requirements: 1.6, 2.1, 2.2, 2.3, 2.4, 2.5

FROM 731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/forecaster-base:latest

LABEL component="backtesting"
LABEL description="Load model checkpoint and test dataset, run all trading strategies, output performance metrics"

WORKDIR /app

# Install component-specific dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the probabilisticforecaster package (model, strategy, backtest, config modules)
COPY probabilisticforecaster/ /app/probabilisticforecaster/

# Copy the component script
COPY probabilisticforecaster/kubeflow/components/backtesting/component.py /app/component.py

# S3 authentication via IRSA; no credentials embedded (Requirement 2.3)
# The pod's service account is annotated with an IAM role ARN

ENTRYPOINT ["python", "component.py"]
