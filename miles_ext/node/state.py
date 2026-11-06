"""Run state a caller reads while a run trains: heartbeat, lease, exit record (ext-064)."""

import importlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from miles_ext.node.contract import STATE_ENV, STATE_TOKEN_ENV, STATE_URL_ENV
from miles_ext.recipes import recipe as R

FILE = "file"
HTTP = "http"
MODAL_DICT = "modal-dict"


class HttpState:
    """Run state on a viewer wherever it runs, as W&B takes metrics: ``<url>/<run_id>/<key>``, a bearer token. A call
    that fails raises ``OSError``, and the supervisor carries on without it."""

    TIMEOUT_SECS = 10

    def __init__(self, run_id: str):
        self.base = f"{os.environ[STATE_URL_ENV].rstrip('/')}/{urllib.parse.quote(run_id, safe='')}"
        self.token = os.environ[STATE_TOKEN_ENV]

    def _call(self, method: str, key: str, body: bytes | None = None):
        req = urllib.request.Request(
            f"{self.base}/{urllib.parse.quote(key, safe='')}",
            data=body,
            method=method,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.TIMEOUT_SECS) as resp:
            return json.loads(resp.read() or b"null")

    def get(self, key: str, default=None):
        try:
            value = self._call("GET", key)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return default
            raise
        return default if value is None else value

    def __setitem__(self, key: str, value) -> None:
        self._call("PUT", key, json.dumps(value, default=str).encode())

    def pop(self, key: str, default=None):
        value = self.get(key, default)
        self._call("DELETE", key)
        return value


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
    if name == HTTP:
        return HttpState(run_id)
    if name == MODAL_DICT:
        from miles_ext.node.adapters.modal_dict import modal_dict

        return modal_dict()
    module, _, attr = name.rpartition(".")
    if not module:
        raise ValueError(f"{STATE_ENV}={name!r}: expected {FILE}, {HTTP}, {MODAL_DICT} or a dotted path")
    return getattr(importlib.import_module(module), attr)(run_id)
