"""URI-aware artifact I/O for KFP pipeline components.

KFP populates artifact `.uri` fields with a scheme that names the backing
store (e.g. `minio://mlpipeline/v2/artifacts/...` for the cluster's MinIO,
or `s3://bucket/key` for AWS S3). The store the component must read/write
to is determined by that scheme; the URI is not just a key.

Earlier component code treated the full URI string as a raw S3 key against
AWS S3 (no `endpoint_url`), so files landed at e.g.
`s3://prod-fintech-forex-sg-.../minio://mlpipeline/v2/artifacts/...`. That
silently broke the KFP UI's "Visualize" and "Download" buttons (which fetch
from the URI's actual store) and produced unreadable keys in S3.

This module fixes that by parsing the URI and routing through the correct
store. Components import the public helpers (`get_object_bytes`,
`put_object_bytes`, ...) and pass the URI they received from KFP unchanged.

URI conventions
---------------
- ``minio://<bucket>/<key>``, cluster MinIO. Endpoint is fixed to
  ``http://minio-service.kubeflow:9000``; credentials are read from env
  ``MINIO_ACCESS_KEY``/``MINIO_SECRET_KEY`` (mounted from the
  ``mlpipeline-minio-artifact`` Secret via DSL ``use_secret_as_env``).
- ``s3://<bucket>/<key>``, AWS S3. Credentials come from the pod's IRSA
  role (no explicit creds passed to boto3).
- bare key (no scheme), legacy callers; routed to AWS S3 at the default
  bucket ``probabilisticforecaster.config.S3_BUCKET``.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import boto3

MINIO_ENDPOINT = "http://minio-service.kubeflow:9000"
MINIO_REGION = "us-east-1"


@dataclass(frozen=True)
class _Location:
    """A resolved (client, bucket, key) tuple for a given URI."""

    client: object
    bucket: str
    key: str
    scheme: str


def _build_minio_client():
    """Build a boto3 S3 client configured for the in-cluster MinIO."""
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
    """Build a boto3 S3 client for AWS S3 (credentials via IRSA / env)."""
    return boto3.client("s3")


def resolve(uri: str, default_bucket: Optional[str] = None) -> _Location:
    """Resolve a KFP artifact URI to a (client, bucket, key) tuple.

    Args:
        uri: One of ``minio://b/k``, ``s3://b/k``, or a bare key.
        default_bucket: Bucket to use when ``uri`` is a bare key. If ``None``,
            falls back to ``probabilisticforecaster.config.S3_BUCKET``.
    """
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
        from probabilisticforecaster.config import S3_BUCKET

        default_bucket = S3_BUCKET
    return _Location(
        client=_build_aws_s3_client(),
        bucket=default_bucket,
        key=uri.lstrip("/"),
        scheme="bare",
    )


def get_object_bytes(uri: str, default_bucket: Optional[str] = None) -> bytes:
    """Fetch an object's bytes from the store named by the URI scheme."""
    loc = resolve(uri, default_bucket=default_bucket)
    obj = loc.client.get_object(Bucket=loc.bucket, Key=loc.key)
    return obj["Body"].read()


def put_object_bytes(
    uri: str,
    data: bytes,
    *,
    content_type: Optional[str] = None,
    default_bucket: Optional[str] = None,
) -> None:
    """Write bytes to the store named by the URI scheme."""
    loc = resolve(uri, default_bucket=default_bucket)
    extra: dict = {}
    if content_type:
        extra["ContentType"] = content_type
    loc.client.put_object(Bucket=loc.bucket, Key=loc.key, Body=data, **extra)


def get_object_fileobj(uri: str, default_bucket: Optional[str] = None) -> io.BytesIO:
    """Fetch an object and return it wrapped in a BytesIO for streamed reads.

    Convenience for callers that previously did ``io.BytesIO(obj.read())``.
    """
    return io.BytesIO(get_object_bytes(uri, default_bucket=default_bucket))
