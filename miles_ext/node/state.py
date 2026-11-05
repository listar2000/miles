"""Run state a caller reads while a run trains: heartbeat, lease, exit record (ext-064)."""

import importlib
import json
import os
import urllib.parse
from pathlib import Path

from miles_ext.node.contract import STATE_ENV
from miles_ext.recipes import recipe as R

FILE = "file"
MODAL_DICT = "modal-dict"


class FileState:
    def __init__(self, run_id: str, root: Path = R.RESULTS):
        self.dir = Path(root) / run_id / "state"

    def _path(self, key: str) -> Path:
        return self.dir / f"{urllib.parse.quote(key, safe='')}.json"

    def get(self, key: str, default=None):
        try:
            return json.loads(self._path(key).read_text())
        except (FileNotFoundError, ValueError):
            return default

    def __setitem__(self, key: str, value) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(value, default=str))
        tmp.replace(path)

    def pop(self, key: str, default=None):
        value = self.get(key, default)
        self._path(key).unlink(missing_ok=True)
        return value


def for_run(run_id: str):
    name = os.environ.get(STATE_ENV) or FILE
    if name == FILE:
        return FileState(run_id)
    if name == MODAL_DICT:
        from miles_ext.node.adapters.modal_dict import modal_dict

        return modal_dict()
    module, _, attr = name.rpartition(".")
    if not module:
        raise ValueError(f"{STATE_ENV}={name!r}: expected {FILE}, {MODAL_DICT} or a dotted path")
    return getattr(importlib.import_module(module), attr)(run_id)
