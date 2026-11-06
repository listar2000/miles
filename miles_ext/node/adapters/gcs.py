"""``MILES_GCS_MOUNTS``: a node-local directory mirrored to a GCS prefix through the client, with no FUSE (ext-072)."""

import json
import os
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from miles_ext.node.contract import GCS_CREDENTIALS_ENV
from miles_ext.platform.layout import TRACKER

FILE_WORKERS = 8  # files at a time
CHUNK_BYTES = 64 * 2**20
CHUNKED_FROM_BYTES = 256 * 2**20  # a larger file moves in concurrent chunks
CHUNK_WORKERS = 16
TIMEOUT_SECS = 600
_LAST = (TRACKER,)  # a tracker on the bucket never names a save that is still uploading


@dataclass(frozen=True)
class _Known:
    """A file as this node last agreed with the bucket on it."""

    signature: tuple[int, int]  # size and mtime here
    generation: int  # the object's version there


def _client():
    from google.cloud import storage

    if info := os.environ.get(GCS_CREDENTIALS_ENV):
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_info(
            json.loads(info), scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
        )
        return storage.Client(project=credentials.project_id, credentials=credentials)
    return storage.Client(project=None)  # Application Default Credentials; objects need no project


def _signature(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return st.st_size, st.st_mtime_ns


def _skipped(rel: str) -> bool:
    """Files mid-write: every writer here renames a ``.tmp`` into place."""
    return any(part.endswith(".tmp") for part in rel.split("/"))


class GcsMirror:
    """``commit`` uploads what changed here and deletes what this node removed; ``refresh`` downloads what other nodes
    committed and removes what they deleted. An uncommitted edit to a tracked file survives a ``refresh``, while a file
    this mirror never tracked yields to the bucket's copy, as a persisted disk's stale tracker must. One commit is not
    atomic: a reader only trusts what a later signal names (a tracker, a weight sync request)."""

    live = False

    def __init__(self, mount: Path, uri: str, client=None):
        parts = urlsplit(uri)
        if parts.scheme != "gs" or not parts.netloc:
            raise ValueError(f"expected gs://bucket[/prefix], got {uri!r}")
        self.mount = Path(mount)
        self.prefix = parts.path.strip("/") + "/" if parts.path.strip("/") else ""
        self.client = client or _client()
        self.bucket = self.client.bucket(parts.netloc)
        self._known: dict[str, _Known] = {}
        self._lock = threading.Lock()

    def commit(self) -> None:
        with self._lock:
            here = self._walk()
            changed = [rel for rel, sig in here.items() if rel not in self._known or self._known[rel].signature != sig]
            last = [rel for rel in changed if rel.rsplit("/", 1)[-1] in _LAST]
            self._each(self._upload, [rel for rel in changed if rel not in last])
            self._each(self._upload, last)
            self._each(self._delete, [rel for rel in self._known if rel not in here])

    def refresh(self) -> None:
        with self._lock:
            there = {}
            for blob in self.client.list_blobs(self.bucket, prefix=self.prefix, timeout=TIMEOUT_SECS):
                rel = blob.name[len(self.prefix) :]
                if rel and not rel.endswith("/"):
                    there[rel] = blob
            fetch = []
            for rel, blob in there.items():
                known = self._known.get(rel)
                if known is None or (known.generation != blob.generation and not self._edited(rel, known)):
                    fetch.append(blob)
            self._each(self._download, fetch)
            for rel in [rel for rel in self._known if rel not in there]:
                if not self._edited(rel, self._known.pop(rel)):
                    self._unlink(rel)

    def remove(self, rel: str) -> None:
        with self._lock:
            rel = rel.strip("/")
            for blob in self.client.list_blobs(self.bucket, prefix=f"{self.prefix}{rel}", timeout=TIMEOUT_SECS):
                name = blob.name[len(self.prefix) :]
                if name == rel or name.startswith(f"{rel}/"):
                    self._delete(name)

    def _walk(self) -> dict[str, tuple[int, int]]:
        here = {}
        for root, _, names in os.walk(self.mount):
            for name in names:
                path = Path(root, name)
                rel = path.relative_to(self.mount).as_posix()
                if not _skipped(rel) and not path.is_symlink() and (sig := _signature(path)):
                    here[rel] = sig
        return here

    def _edited(self, rel: str, known: _Known) -> bool:
        return _signature(self.mount / rel) not in (None, known.signature)

    def _unlink(self, rel: str) -> None:
        path = self.mount / rel
        path.unlink(missing_ok=True)
        for parent in path.parents:
            if parent == self.mount:
                break
            try:
                parent.rmdir()
            except OSError:  # not empty
                break

    def _each(self, one: Callable[[str], None], items: Iterable) -> None:
        """Run ``one`` on every item, then raise the first failure: what succeeded stays recorded."""
        with ThreadPoolExecutor(FILE_WORKERS) as pool:
            for future in [pool.submit(one, item) for item in items]:
                future.result()

    def _upload(self, rel: str) -> None:
        from google.cloud.storage import transfer_manager
        from google.cloud.storage.retry import DEFAULT_RETRY

        path, blob = self.mount / rel, self.bucket.blob(f"{self.prefix}{rel}")
        sig = _signature(path)  # before reading: a later write leaves a mismatch, so the next commit uploads it
        if sig is None:  # gone since the walk: the next commit sees that
            return
        if sig[0] >= CHUNKED_FROM_BYTES:
            transfer_manager.upload_chunks_concurrently(
                str(path), blob, chunk_size=CHUNK_BYTES, max_workers=CHUNK_WORKERS, worker_type=transfer_manager.THREAD
            )
            blob.reload(timeout=TIMEOUT_SECS)
        else:
            # one writer per path, so a retry cannot lose another node's write
            blob.upload_from_filename(str(path), retry=DEFAULT_RETRY, timeout=TIMEOUT_SECS)
        self._known[rel] = _Known(sig, blob.generation)

    def _download(self, blob) -> None:
        from google.api_core.exceptions import NotFound
        from google.cloud.storage import transfer_manager

        rel = blob.name[len(self.prefix) :]
        path = self.mount / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            if blob.size >= CHUNKED_FROM_BYTES:
                transfer_manager.download_chunks_concurrently(
                    blob,
                    str(tmp),
                    chunk_size=CHUNK_BYTES,
                    max_workers=CHUNK_WORKERS,
                    worker_type=transfer_manager.THREAD,
                    crc32c_checksum=True,
                )
            else:
                blob.download_to_filename(str(tmp), timeout=TIMEOUT_SECS)
            sig = _signature(tmp)
            os.replace(tmp, path)
        except NotFound:  # replaced or deleted since the listing: the next refresh decides
            return
        finally:
            tmp.unlink(missing_ok=True)
        self._known[rel] = _Known(sig, blob.generation)

    def _delete(self, rel: str) -> None:
        from google.api_core.exceptions import NotFound
        from google.cloud.storage.retry import DEFAULT_RETRY

        try:
            self.bucket.blob(f"{self.prefix}{rel}").delete(retry=DEFAULT_RETRY, timeout=TIMEOUT_SECS)
        except NotFound:
            pass
        self._known.pop(rel, None)


_mirrors: dict[str, GcsMirror] = {}
_mirrors_lock = threading.Lock()


def mirror(mount: Path, uri: str) -> GcsMirror:
    """One mirror per mount in a process: what it knows about the bucket is shared by every thread that commits."""
    with _mirrors_lock:
        if str(mount) not in _mirrors:
            _mirrors[str(mount)] = GcsMirror(mount, uri)
        return _mirrors[str(mount)]
