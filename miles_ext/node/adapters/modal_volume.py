"""``MILES_STORAGE=modal-volume``: a Modal v2 volume keeps only what a node commits."""

import json
import os
from pathlib import Path

import modal

VOLUMES_ENV = "MILES_MODAL_VOLUMES"  # {"<mount>": "<volume name>"}, set by the caller that mounted them


class ModalVolumeStorage:
    live = False

    def __init__(self, mount: Path):
        name = json.loads(os.environ[VOLUMES_ENV])[str(mount)]
        self.volume = modal.Volume.from_name(name)

    def commit(self) -> None:
        self.volume.commit()

    def refresh(self) -> None:
        self.volume.reload()

    def remove(self, rel: str) -> None:
        try:
            self.volume.remove_file(rel, recursive=True)
        except modal.exception.NotFoundError:
            pass
