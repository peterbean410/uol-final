# Model Training component for the Forecaster KFP pipeline
# Extends the shared base image with training-specific dependencies
# Requirements: 2.1, 2.2, 2.3, 2.4, 2.5

FROM 731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/forecaster-base:latest

LABEL component="model_training"
LABEL description="Train ProbabilisticTransformer with Gaussian NLL loss, supports scratch and finetune modes"

WORKDIR /app

# Install component-specific dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the probabilisticforecaster package (model, training, config, dataset modules)
COPY probabilisticforecaster/ /app/probabilisticforecaster/

# Copy the component script
COPY probabilisticforecaster/kubeflow/components/model_training/component.py /app/component.py

# S3 authentication via IRSA; no credentials embedded (Requirement 2.3)
# The pod's service account is annotated with an IAM role ARN

ENTRYPOINT ["python", "component.py"]
