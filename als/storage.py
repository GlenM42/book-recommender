"""
S3-compatible storage helpers for model artifact upload/download.

Reads configuration from environment variables:
    S3_ENDPOINT_URL  — e.g. http://host.docker.internal:9100
    S3_ACCESS_KEY
    S3_SECRET_KEY
    S3_BUCKET        — default: book-recommender

If any of the three required vars are absent, all operations are no-ops.
"""

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_ARTIFACT_FILES = ["als_model.npz", "item_lookup.parquet", "training_info.json"]
_S3_PREFIX = "models/latest"


def s3_configured() -> bool:
    return all(os.environ.get(k) for k in ("S3_ENDPOINT_URL", "S3_ACCESS_KEY", "S3_SECRET_KEY"))


def _client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    )


def upload_artifacts(models_dir: Path) -> None:
    bucket = os.environ.get("S3_BUCKET", "book-recommender")
    client = _client()
    for fname in _ARTIFACT_FILES:
        key = f"{_S3_PREFIX}/{fname}"
        log.info("Uploading %s → s3://%s/%s …", models_dir / fname, bucket, key)
        client.upload_file(str(models_dir / fname), bucket, key)
    log.info("Artifacts uploaded to s3://%s/%s/", bucket, _S3_PREFIX)


def download_artifacts(models_dir: Path) -> None:
    bucket = os.environ.get("S3_BUCKET", "book-recommender")
    client = _client()
    models_dir.mkdir(exist_ok=True)
    for fname in _ARTIFACT_FILES:
        key = f"{_S3_PREFIX}/{fname}"
        local = models_dir / fname
        log.info("Downloading s3://%s/%s → %s …", bucket, key, local)
        client.download_file(bucket, key, str(local))
    log.info("Artifacts downloaded to %s/", models_dir)
