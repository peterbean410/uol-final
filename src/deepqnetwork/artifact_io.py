"""URI-aware artifact I/O for KFP pipeline components (DQN side).

Mirrors ``probabilisticforecaster.artifact_io``. KFP populates artifact
``.uri`` fields with a scheme that names the backing store
(``minio://mlpipeline/v2/artifacts/...`` for the cluster's MinIO,
``s3://bucket/key`` for AWS S3). Earlier DQN component code treated the
URI string as a raw S3 key against AWS S3 (no ``endpoint_url``), so the
checkpoint and backtest metrics landed at e.g.
``s3://prod-fintech-forex-sg-.../minio://mlpipeline/v2/artifacts/...``
, unreadable by the KFP UI and by any consumer that resolves the URI
correctly (model registration, dqnpf-intraday backtest, etc.).

This module parses the URI and routes through the correct store.
Components import the public helpers (``put_object_bytes``,
``put_file``, ``download_file``, ``get_object_bytes``) and pass the URI
they received from KFP unchanged.

URI conventions
---------------
- ``minio://<bucket>/<key>``, cluster MinIO. Endpoint is fixed to
  ``http://minio-service.kubeflow:9000``; credentials read from env
  ``MINIO_ACCESS_KEY``/``MINIO_SECRET_KEY`` (mounted from the
  ``mlpipeline-minio-artifact`` Secret via DSL ``use_secret_as_env``).
- ``s3://<bucket>/<key>``, AWS S3. Credentials come from the pod's
  IRSA role (no explicit creds passed to boto3).
- bare key (no scheme), legacy callers; routed to AWS S3 at the
  default bucket. Callers either pass ``default_bucket`` or set the
  ``S3_BUCKET`` env var.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import boto3

MINIO_ENDPOINT = "http://minio-service.kubeflow:9000"
MINIO_REGION = "us-east-1"  # boto3 still requires a region; MinIO ignores it


@dataclass(frozen=True)
class _Location:
    client: object
    bucket: str
    key: str
    scheme: str  # "minio" | "s3" | "bare"


def _build_minio_client():
    access_key = os.environ.get("MINIO_ACCESS_KEY", "")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "")
    if not access_key or not secret_key:
        raise RuntimeError(
            "MINIO_ACCESS_KEY/MINIO_SECRET_KEY env vars not set; the component "
            "needs the mlpipeline-minio-artifact secret mounted as env via "
            "kfp.kubernetes.use_secret_as_env(...) in the pipeline DSL."
        )
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=MINIO_REGION,
    )


def _build_aws_s3_client():
    return boto3.client("s3")


def resolve(uri: str, default_bucket: Optional[str] = None) -> _Location:
    """Resolve a KFP artifact URI to a (client, bucket, key) tuple."""
    parsed = urlparse(uri)
    scheme = (parsed.scheme or "").lower()

    if scheme == "minio":
        if not parsed.netloc:
            raise ValueError(f"minio:// URI missing bucket: {uri!r}")
        return _Location(
            client=_build_minio_client(),
            bucket=parsed.netloc,
            key=parsed.path.lstrip("/"),
            scheme="minio",
        )

    if scheme == "s3":
        if not parsed.netloc:
            raise ValueError(f"s3:// URI missing bucket: {uri!r}")
        return _Location(
            client=_build_aws_s3_client(),
            bucket=parsed.netloc,
            key=parsed.path.lstrip("/"),
            scheme="s3",
        )

    if default_bucket is None:
        default_bucket = os.environ.get(
            "S3_BUCKET", "prod-fintech-forex-sg-731833471586"
        )
    return _Location(
        client=_build_aws_s3_client(),
        bucket=default_bucket,
        key=uri.lstrip("/"),
        scheme="bare",
    )


def get_object_bytes(uri: str, default_bucket: Optional[str] = None) -> bytes:
    loc = resolve(uri, default_bucket=default_bucket)
    obj = loc.client.get_object(Bucket=loc.bucket, Key=loc.key)
    return obj["Body"].read()


def get_object_fileobj(uri: str, default_bucket: Optional[str] = None) -> io.BytesIO:
    return io.BytesIO(get_object_bytes(uri, default_bucket=default_bucket))


def put_object_bytes(
    uri: str,
    data: bytes,
    *,
    content_type: Optional[str] = None,
    default_bucket: Optional[str] = None,
) -> None:
    loc = resolve(uri, default_bucket=default_bucket)
    extra: dict = {}
    if content_type:
        extra["ContentType"] = content_type
    loc.client.put_object(Bucket=loc.bucket, Key=loc.key, Body=data, **extra)


def put_file(
    uri: str,
    local_path: str,
    *,
    default_bucket: Optional[str] = None,
) -> None:
    """Upload a local file to the store named by the URI scheme.

    Uses multipart-aware ``upload_file`` for large checkpoints (~hundreds of MB).
    """
    loc = resolve(uri, default_bucket=default_bucket)
    loc.client.upload_file(local_path, loc.bucket, loc.key)


def download_file(
    uri: str,
    local_path: str,
    *,
    default_bucket: Optional[str] = None,
) -> None:
    """Download an object to a local file path."""
    loc = resolve(uri, default_bucket=default_bucket)
    loc.client.download_file(loc.bucket, loc.key, local_path)
