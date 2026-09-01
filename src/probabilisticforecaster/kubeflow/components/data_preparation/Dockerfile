
ARG FORECASTER_BASE_IMAGE=731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/forecaster/base:latest
FROM ${FORECASTER_BASE_IMAGE}

LABEL component="data_preparation"
LABEL description="Load S3 parquet data, compute 16 features, build train/test ForexDataset splits"

WORKDIR /app

COPY probabilisticforecaster/kubeflow/components/data_preparation/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY probabilisticforecaster/ /app/probabilisticforecaster/

COPY probabilisticforecaster/kubeflow/components/data_preparation/component.py /app/component.py


ENTRYPOINT ["python", "component.py"]
