import os
import tempfile

import boto3
from dotenv import load_dotenv

load_dotenv()

HETZNER_ENDPOINT = os.environ.get("HETZNER_STORAGE_ENDPOINT", "https://hel1.your-objectstorage.com")
HETZNER_BUCKET = os.environ.get("HETZNER_STORAGE_BUCKET", "soundfragments")
HETZNER_ACCESS_KEY = os.environ.get("HETZNER_STORAGE_ACCESS_KEY")
HETZNER_SECRET_KEY = os.environ.get("HETZNER_STORAGE_SECRET_KEY")

_client = None


def get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=HETZNER_ENDPOINT,
            aws_access_key_id=HETZNER_ACCESS_KEY,
            aws_secret_access_key=HETZNER_SECRET_KEY,
        )
    return _client


def download_to_temp(file_key: str) -> str:
    """Download an object from Hetzner storage to a local temp file, return its path.
    Caller is responsible for deleting the file once done with it."""
    suffix = os.path.splitext(file_key)[1] or ".audio"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="spectra_")
    os.close(fd)
    get_client().download_file(HETZNER_BUCKET, file_key, path)
    return path
