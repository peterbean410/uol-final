
ARG FORECASTER_BASE_IMAGE=731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/forecaster/base:latest
FROM ${FORECASTER_BASE_IMAGE}

LABEL component="model_training"
LABEL description="Train ProbabilisticTransformer with Gaussian NLL loss, supports scratch and finetune modes"

WORKDIR /app

COPY probabilisticforecaster/kubeflow/components/model_training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY probabilisticforecaster/ /app/probabilisticforecaster/

COPY probabilisticforecaster/kubeflow/components/model_training/component.py /app/component.py


ENTRYPOINT ["python", "component.py"]
