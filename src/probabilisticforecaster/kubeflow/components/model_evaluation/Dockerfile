
ARG FORECASTER_BASE_IMAGE=731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/forecaster/base:latest
FROM ${FORECASTER_BASE_IMAGE}

LABEL component="model_evaluation"
LABEL description="Evaluate model: compute NLL, DA, CR95, RMSE; forgetting check; degradation gate"

WORKDIR /app

COPY probabilisticforecaster/kubeflow/components/model_evaluation/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY probabilisticforecaster/ /app/probabilisticforecaster/

COPY probabilisticforecaster/kubeflow/components/model_evaluation/component.py /app/component.py


ENTRYPOINT ["python", "component.py"]
