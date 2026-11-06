"""Storage adapters for durable writes and cross-node visibility (ext-062)."""

import importlib
import json
import os
from pathlib import Path
from typing import Protocol

from miles_ext.node.contract import GCS_MOUNTS_ENV, STORAGE_ENV

MODAL_VOLUME = "modal-volume"
POSIX = "posix"


class Storage(Protocol):
    live: bool  # every node sees writes as they happen, and an open file blocks nothing

    def commit(self) -> None:
        """Make this node's writes under the mount durable and visible to other nodes' next ``refresh``."""

    def refresh(self) -> None:
        """See what other nodes committed."""

    def remove(self, rel: str) -> None:
        """Remove ``rel``, already deleted in this node's view and committed, from every node's view. A missing
        path is fine."""


class PosixStorage:
    """A filesystem every node sees alike: nothing to flush or refresh, and a delete is already global."""

    live = True

    def __init__(self, mount: Path):
        self.mount = mount

    def commit(self) -> None:
        pass

    def refresh(self) -> None:
        pass

    def remove(self, rel: str) -> None:
        pass


def _adapter(name: str):
    if name == POSIX:
        return PosixStorage
    if name == MODAL_VOLUME:
        from miles_ext.node.adapters.modal_volume import ModalVolumeStorage

        return ModalVolumeStorage
    module, _, attr = name.rpartition(".")
    if not module:
        raise ValueError(f"{STORAGE_ENV}={name!r}: expected {POSIX}, {MODAL_VOLUME} or a dotted path")
    return getattr(importlib.import_module(module), attr)


def for_mount(mount: str | Path) -> Storage:
    if uri := json.loads(os.environ.get(GCS_MOUNTS_ENV) or "{}").get(str(mount)):
        from miles_ext.node.adapters.gcs import mirror

        return mirror(Path(mount), uri)
    return _adapter(os.environ.get(STORAGE_ENV) or POSIX)(Path(mount))
