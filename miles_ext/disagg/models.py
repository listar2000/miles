"""Fill the models volume with the HF checkpoints and datasets the disagg runs use (CPU only).

    modal run -m miles_ext.disagg.models
"""

import modal

MODELS_VOLUME = "miles-disagg-models"
MOUNT = "/models"
CHECKPOINTS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B")
DATASETS = ("zhuzilin/dapo-math-17k", "zhuzilin/aime-2024")

app = modal.App("miles-disagg-models")
volume = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]==0.35.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, cpu=4, memory=8192, volumes={MOUNT: volume}, timeout=3600)
def download() -> dict:
    from huggingface_hub import snapshot_download

    out = {}
    for repo in CHECKPOINTS:
        out[repo] = snapshot_download(repo, local_dir=f"{MOUNT}/{repo.split('/')[1]}")
    for repo in DATASETS:
        out[repo] = snapshot_download(repo, repo_type="dataset", local_dir=f"{MOUNT}/datasets/{repo.split('/')[1]}")
    volume.commit()
    return out


@app.local_entrypoint()
def main():
    print(download.remote())
