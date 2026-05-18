# Data Preparation component for the Forecaster KFP pipeline
# Extends the shared base image with component-specific dependencies
# Requirements: 2.1, 2.2, 2.3, 2.4, 2.5

FROM 731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/forecaster/base:latest

LABEL component="data_preparation"
LABEL description="Load S3 parquet data, compute 16 features, build train/test ForexDataset splits"

WORKDIR /app

# Install component-specific dependencies
COPY probabilisticforecaster/kubeflow/components/data_preparation/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the probabilisticforecaster package (features, dataset, config modules)
COPY probabilisticforecaster/ /app/probabilisticforecaster/

# Copy the component script
COPY probabilisticforecaster/kubeflow/components/data_preparation/component.py /app/component.py

# S3 authentication via IRSA; no credentials embedded (Requirement 2.3)
# The pod's service account is annotated with an IAM role ARN

ENTRYPOINT ["python", "component.py"]
