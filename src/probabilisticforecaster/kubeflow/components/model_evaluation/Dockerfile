# Model Evaluation component for the Forecaster KFP pipeline
# Extends the shared base image with evaluation-specific dependencies
# Requirements: 1.5, 9.4

FROM 731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/forecaster-base:latest

LABEL component="model_evaluation"
LABEL description="Evaluate model: compute NLL, DA, CR95, RMSE; forgetting check; degradation gate"

WORKDIR /app

# Install component-specific dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the probabilisticforecaster package (model, evaluation, config, dataset modules)
COPY probabilisticforecaster/ /app/probabilisticforecaster/

# Copy the component script
COPY probabilisticforecaster/kubeflow/components/model_evaluation/component.py /app/component.py

# S3 authentication via IRSA; no credentials embedded (Requirement 2.3)
# The pod's service account is annotated with an IAM role ARN

ENTRYPOINT ["python", "component.py"]
