"""The download itself, in a node with the weights mount: no provider. ``miles_ext.platform.weights`` runs it."""

import os
from pathlib import Path


def download(hf_repo: str, revision: str, path: str) -> dict:
    from huggingface_hub import snapshot_download

    snapshot_download(hf_repo, revision=revision, local_dir=path, token=os.environ.get("HF_TOKEN") or None)
    return {"bytes": sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())}
