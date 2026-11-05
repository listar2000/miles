"""The run config's records to token shards on a Modal CPU container."""

from __future__ import annotations

from pathlib import Path

import modal

from miles_ext.sft import config as sft_config
from miles_ext.sft import job
from miles_ext.sft import modal_common as M

app = modal.App("miles-rl-sft-prepare")
image = M.image()
weights_volume = M.volume("MILES_WEIGHTS_VOLUME")
data_volume = M.data_volume()


@app.function(
    image=image,
    cpu=16,
    memory=64 * 1024,
    timeout=3 * 3600,
    volumes={str(sft_config.WEIGHTS_MOUNT): weights_volume.read_only(), str(sft_config.DATA_MOUNT): data_volume},
)
def prepare(config_yaml: str, tests: bool) -> dict[str, str]:
    return job.prepare(config_yaml, tests, data_volume.commit)


@app.local_entrypoint()
def main(tests: bool = False):
    for split, manifest in prepare.remote(Path(M.CONFIG_PATH).read_text(), tests).items():
        print(split, manifest)
