"""The GCS mirror on an in-memory bucket, and against a real one when ``MILES_GCS_TEST_URI`` names a prefix there."""

import itertools
import os
import subprocess
import sys
import uuid
from types import SimpleNamespace

import pytest

pytest.importorskip("google.cloud.storage")

from google.api_core.exceptions import NotFound

from miles_ext.node import contract
from miles_ext.node import storage as S
from miles_ext.node.adapters import gcs

LIVE_URI = os.environ.get("MILES_GCS_TEST_URI")


class FakeBlob:
    def __init__(self, bucket, name, size=None, generation=None):
        self.store, self.name, self.size, self.generation = bucket, name, size, generation

    def upload_from_filename(self, path, retry=None, timeout=None):
        self.store.put(self.name, open(path, "rb").read())
        self.generation = self.store.objects[self.name][1]

    def download_to_filename(self, path, timeout=None):
        if self.store.objects.get(self.name, (None, None))[1] != self.generation:
            raise NotFound(self.name)
        open(path, "wb").write(self.store.objects[self.name][0])

    def delete(self, retry=None, timeout=None):
        if self.store.objects.pop(self.name, None) is None:
            raise NotFound(self.name)
        self.store.ops.append(("delete", self.name))


class FakeBucket:
    def __init__(self):
        self.objects, self.ops, self._generations = {}, [], itertools.count(1)

    def put(self, name, data):
        self.objects[name] = (data, next(self._generations))
        self.ops.append(("put", name))

    def blob(self, name):
        return FakeBlob(self, name)


class FakeClient:
    def __init__(self, store=None):
        self.store = store or FakeBucket()

    def bucket(self, name):
        return self.store

    def list_blobs(self, bucket, prefix="", timeout=None):
        return [
            FakeBlob(bucket, name, len(data), gen)
            for name, (data, gen) in sorted(bucket.objects.items())
            if name.startswith(prefix)
        ]


def _nodes(tmp_path, n=2, uri="gs://b/run1"):
    client = FakeClient()
    return client.store, [gcs.GcsMirror(tmp_path / f"node{i}", uri, client=client) for i in range(n)]


def _write(mirror, rel, text):
    path = mirror.mount / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_a_commit_reaches_another_nodes_refresh(tmp_path):
    store, (a, b) = _nodes(tmp_path)
    _write(a, "gpu_delta/v1/x", "1")
    b.refresh()
    assert not (b.mount / "gpu_delta/v1/x").exists()  # not committed
    a.commit()
    b.refresh()
    assert (b.mount / "gpu_delta/v1/x").read_text() == "1"
    assert "run1/gpu_delta/v1/x" in store.objects


def test_commit_uploads_only_what_changed(tmp_path):
    store, (a, _) = _nodes(tmp_path)
    path = _write(a, "x", "1")
    _write(a, "y", "1")
    a.commit()
    store.ops.clear()
    a.commit()
    assert store.ops == []
    path.write_text("22")
    a.commit()
    assert store.ops == [("put", "run1/x")]


def test_a_delete_committed_here_leaves_other_nodes_after_their_refresh(tmp_path):
    store, (a, b) = _nodes(tmp_path)
    path = _write(a, "d/x", "1")
    a.commit()
    b.refresh()
    path.unlink()
    a.commit()
    assert store.objects == {}
    b.refresh()
    assert not (b.mount / "d").exists()


def test_a_commit_deletes_only_what_this_node_knew(tmp_path):
    store, (a, b) = _nodes(tmp_path)
    _write(a, "shard-a", "a")
    _write(b, "shard-b", "b")
    a.commit()
    b.commit()
    a.commit()
    assert set(store.objects) == {"run1/shard-a", "run1/shard-b"}


def test_a_refreshed_copy_does_not_come_back_after_another_node_removes_it(tmp_path):
    store, (a, b) = _nodes(tmp_path)
    _write(a, "ckpt/shard", "a")
    a.commit()
    b.refresh()
    a.remove("ckpt")
    b.commit()
    assert store.objects == {}


def test_remove_takes_the_tree_not_a_sibling_with_the_same_start(tmp_path):
    store, (a, _) = _nodes(tmp_path)
    _write(a, "iter_1/x", "1")
    _write(a, "iter_10/x", "1")
    a.commit()
    a.remove("iter_1")
    a.remove("iter_1")  # already gone is fine
    assert set(store.objects) == {"run1/iter_10/x"}


def test_a_local_edit_survives_a_refresh_until_committed(tmp_path):
    store, (a, b) = _nodes(tmp_path)
    _write(a, "x", "from a")
    a.commit()
    b.refresh()
    _write(b, "x", "edited on b")
    _write(a, "x", "from a again")
    a.commit()
    b.refresh()
    assert (b.mount / "x").read_text() == "edited on b"
    b.commit()
    assert store.objects["run1/x"][0] == b"edited on b"


def test_a_file_this_node_never_tracked_yields_to_the_bucket(tmp_path):
    _, (a, b) = _nodes(tmp_path)
    _write(a, "t", "200")
    a.commit()
    _write(b, "t", "100")  # left on a persisted disk by an earlier container
    b.refresh()
    assert (b.mount / "t").read_text() == "200"


def test_a_second_refresh_downloads_nothing(tmp_path, monkeypatch):
    _, (a, b) = _nodes(tmp_path)
    _write(a, "x", "1")
    a.commit()
    b.refresh()
    monkeypatch.setattr(FakeBlob, "download_to_filename", lambda *_, **__: pytest.fail("downloaded again"))
    b.refresh()


def test_refresh_replaces_a_stale_file_of_the_same_size(tmp_path):
    _, (a, b) = _nodes(tmp_path)
    _write(a, "t", "100")
    a.commit()
    b.refresh()
    _write(a, "t", "200")
    a.commit()
    b.refresh()
    assert (b.mount / "t").read_text() == "200"


def test_a_tracker_is_uploaded_after_the_save_it_names(tmp_path):
    store, (a, _) = _nodes(tmp_path)
    _write(a, f"run/checkpoints/{gcs.TRACKER}", "1")
    for i in range(20):
        _write(a, f"run/checkpoints/iter_1/s{i}", "x")
    a.commit()
    assert store.ops[-1] == ("put", f"run1/run/checkpoints/{gcs.TRACKER}")


def test_files_mid_write_stay_local(tmp_path):
    store, (a, _) = _nodes(tmp_path)
    _write(a, ".v1.tmp/x", "1")
    _write(a, "x.123.tmp", "1")
    a.commit()
    assert store.objects == {}


def test_a_failed_upload_raises_and_the_next_commit_retries_it(tmp_path, monkeypatch):
    store, (a, _) = _nodes(tmp_path)
    _write(a, "x", "1")
    real = FakeBlob.upload_from_filename
    monkeypatch.setattr(FakeBlob, "upload_from_filename", lambda *_, **__: (_ for _ in ()).throw(OSError("503")))
    with pytest.raises(OSError):
        a.commit()
    monkeypatch.setattr(FakeBlob, "upload_from_filename", real)
    a.commit()
    assert "run1/x" in store.objects


def test_an_object_deleted_after_the_listing_is_skipped(tmp_path, monkeypatch):
    store, (a, b) = _nodes(tmp_path)
    _write(a, "x", "1")
    a.commit()
    blobs = FakeClient.list_blobs(None, store, "run1/")
    store.objects.clear()
    monkeypatch.setattr(b.client, "list_blobs", lambda *_, **__: blobs)
    b.refresh()
    assert not (b.mount / "x").exists()


def test_a_bare_bucket_uri_and_a_bad_uri(tmp_path):
    assert gcs.GcsMirror(tmp_path, "gs://b", client=FakeClient()).prefix == ""
    with pytest.raises(ValueError, match="gs://"):
        gcs.GcsMirror(tmp_path, "s3://b/x", client=FakeClient())


def test_a_mapped_mount_gets_the_mirror_and_the_rest_keep_their_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv(contract.GCS_MOUNTS_ENV, f'{{"{tmp_path}": "gs://b/p"}}')
    monkeypatch.setattr(gcs, "_client", FakeClient)
    monkeypatch.setattr(gcs, "_mirrors", {})
    assert S.for_mount(tmp_path) is S.for_mount(tmp_path)  # threads share what it knows about the bucket
    assert isinstance(S.for_mount(tmp_path), gcs.GcsMirror)
    assert isinstance(S.for_mount("/models"), S.PosixStorage)


def test_a_service_account_key_in_env_is_used(monkeypatch):
    from google.cloud import storage
    from google.oauth2 import service_account

    seen = {}
    credentials = SimpleNamespace(project_id="p")
    monkeypatch.setenv(contract.GCS_CREDENTIALS_ENV, '{"type": "service_account"}')
    monkeypatch.setattr(service_account.Credentials, "from_service_account_info", lambda info, scopes: credentials)
    monkeypatch.setattr(storage, "Client", lambda **kw: seen.update(kw))
    gcs._client()
    assert seen == {"project": "p", "credentials": credentials}


def test_posix_nodes_do_not_load_the_gcs_client():
    code = (
        "import sys; sys.modules['google.cloud.storage'] = None\n"
        "from miles_ext.node import storage\n"
        "print(type(storage.for_mount('/results')).__name__)"
    )
    env = {k: v for k, v in os.environ.items() if k != contract.GCS_MOUNTS_ENV}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout
    assert out.strip() == "PosixStorage"


@pytest.mark.skipif(not LIVE_URI, reason="MILES_GCS_TEST_URI names no bucket prefix")
def test_real_bucket_round_trip(tmp_path, monkeypatch):
    """Two nodes through a real bucket: small files, a chunked file, overwrite, delete, and nothing left behind."""
    monkeypatch.setattr(gcs, "CHUNK_BYTES", 8 * 2**20)
    monkeypatch.setattr(gcs, "CHUNKED_FROM_BYTES", 16 * 2**20)
    uri = f"{LIVE_URI.rstrip('/')}/{uuid.uuid4().hex}"
    client = _live_client()
    a = gcs.GcsMirror(tmp_path / "a", uri, client=client)
    b = gcs.GcsMirror(tmp_path / "b", uri, client=client)
    a.mount.mkdir()
    b.mount.mkdir()
    big = os.urandom(20 * 2**20)
    try:
        _write(a, f"run/checkpoints/{gcs.TRACKER}", "1")
        _write(a, "run/gpu_delta/v1/manifest.json", "{}")
        (a.mount / "run/gpu_delta/v1/big.bin").write_bytes(big)
        a.commit()
        b.refresh()
        assert (b.mount / "run/gpu_delta/v1/big.bin").read_bytes() == big
        assert (b.mount / f"run/checkpoints/{gcs.TRACKER}").read_text() == "1"

        (b.mount / "run/gpu_delta/v1/big.bin").unlink()
        (b.mount / "run/gpu_delta/v1/big.bin").write_bytes(big[::-1])
        b.commit()
        a.refresh()
        assert (a.mount / "run/gpu_delta/v1/big.bin").read_bytes() == big[::-1]

        _write(a, f"run/checkpoints/{gcs.TRACKER}", "2")
        a.commit()
        b.refresh()
        assert (b.mount / f"run/checkpoints/{gcs.TRACKER}").read_text() == "2"

        (a.mount / "run/gpu_delta/v1/manifest.json").unlink()
        a.commit()
        b.refresh()
        assert not (b.mount / "run/gpu_delta/v1/manifest.json").exists()

        a.remove("run/gpu_delta")
        b.refresh()
        assert not (b.mount / "run/gpu_delta").exists()
    finally:
        for blob in client.list_blobs(a.bucket, prefix=a.prefix):
            blob.delete()
    assert not list(client.list_blobs(a.bucket, prefix=a.prefix))


def _live_client():
    """An access token from the environment (gcloud's, on a dev box), or the adapter's own credentials."""
    token = os.environ.get("MILES_GCS_TEST_TOKEN")
    if not token:
        return gcs._client()
    from google.cloud import storage
    from google.oauth2.credentials import Credentials

    return storage.Client(project=None, credentials=Credentials(token))
