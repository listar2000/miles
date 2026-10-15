"""R8: base weights, downloaded once per model revision and shared by every run.

Layout on ``target.weights_volume`` (mounted at ``recipe.MODELS``)::

    <org>--<name>/<40-hex revision>/        the snapshot
    <org>--<name>/<40-hex revision>/.ready  JSON, written after the snapshot is committed

A run reads ``recipe.model_dir()``, which is the keyed path when the recipe sets
``model.revision`` and the preset's directory otherwise (the layout before r3,
which the existing recipes use; ``list_weights`` reports those as ``legacy``).

Concurrent prepares of one key take a lock in the Modal Dict
``<weights_volume>-locks``: the first writes the snapshot, the others wait for
its ``.ready``. A lock older than ``LOCK_STALE_SECS`` is taken over.
"""

import io
import json
import re
import time

import modal

from miles_ext.platform.target import Target
from miles_ext.recipes import recipe as R

KEY_RE = r"[A-Za-z0-9][A-Za-z0-9._-]*--[A-Za-z0-9][A-Za-z0-9._-]*/[0-9a-f]{40}"
READY = ".ready"
LOCK_STALE_SECS = 2 * 3600


def weights_key(hf_repo: str, revision: str) -> str:
    """``Qwen/Qwen3.5-9B`` at a commit sha -> ``Qwen--Qwen3.5-9B/<sha>``."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", hf_repo or ""):
        raise ValueError(f"hf_repo {hf_repo!r} must be <org>/<name>")
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError(f"revision {revision!r} must be a resolved commit sha (prepare_weights resolves branches)")
    return check_key(f"{hf_repo.replace('/', '--')}/{revision}")


def check_key(key: str) -> str:
    if not isinstance(key, str) or not re.fullmatch(KEY_RE, key):
        raise ValueError(f"weights key {key!r} is not <org>--<name>/<sha>")
    return key


def _volume(target: Target, create: bool = False) -> modal.Volume:
    return modal.Volume.from_name(target.weights_volume, environment_name=target.modal_environment,
                                  create_if_missing=create)


def _locks(target: Target) -> modal.Dict:
    return modal.Dict.from_name(f"{target.weights_volume}-locks", environment_name=target.modal_environment,
                                create_if_missing=True)


def _ready(volume: modal.Volume, key: str) -> dict | None:
    try:
        return json.loads(b"".join(volume.read_file(f"{key}/{READY}")))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None


def resolve_revision(hf_repo: str, revision: str | None, token: str | None = None) -> str:
    """A branch, tag or short sha -> the commit sha (needs ``huggingface_hub`` where it runs)."""
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import HfApi

    return HfApi(token=token).model_info(hf_repo, revision=revision or "main").sha


def prepare_weights(hf_repo: str, revision: str | None, target: Target | dict | None, *,
                    secret_name: str | None = None, hf_token: str | None = None) -> dict:
    """Make ``hf_repo`` at ``revision`` ready on the weights volume; returns ``{key, path, ready, bytes}``.

    ``revision`` may be a branch or tag; it is resolved to a commit first.
    ``secret_name`` names a Modal secret holding ``HF_TOKEN`` for gated repos
    (the run's secret); ``hf_token`` is used only to resolve the revision."""
    target = Target.of(target)
    revision = resolve_revision(hf_repo, revision, hf_token)
    key = weights_key(hf_repo, revision)
    volume = _volume(target, create=True)
    if marker := _ready(volume, key):
        return {"key": key, "path": str(R.MODELS / key), "ready": True, "bytes": marker.get("bytes"),
                "prepared": False}
    locks, owner = _locks(target), f"{time.time():.0f}-{id(object())}"
    while not locks.put(key, {"owner": owner, "ts": time.time()}, skip_if_exists=True):
        held = locks.get(key) or {}
        if time.time() - held.get("ts", 0) > LOCK_STALE_SECS:
            locks.pop(key)  # a prepare that died; take over
            continue
        time.sleep(15)
        if marker := _ready(volume, key):
            return {"key": key, "path": str(R.MODELS / key), "ready": True, "bytes": marker.get("bytes"),
                    "prepared": False}
    try:
        from miles_ext.platform import functions

        secrets = [modal.Secret.from_name(secret_name, environment_name=target.modal_environment)] if secret_name else []
        fn = functions.download.with_options(volumes={R.MODELS: volume}, secrets=secrets)
        with functions.app.run(environment_name=target.modal_environment):
            result = fn.remote(hf_repo, revision, str(R.MODELS / key))
        marker = {"hf_repo": hf_repo, "revision": revision, "bytes": result["bytes"], "prepared_at": time.time()}
        with volume.batch_upload(force=True) as batch:  # after the snapshot's commit (the function's exit)
            batch.put_file(io.BytesIO(json.dumps(marker).encode()), f"{key}/{READY}")
        return {"key": key, "path": str(R.MODELS / key), "ready": True, "bytes": result["bytes"], "prepared": True}
    finally:
        if (locks.get(key) or {}).get("owner") == owner:
            locks.pop(key)


def list_weights(target: Target | dict | None) -> list[dict]:
    """Every cached snapshot: ``{key, hf_repo, revision, ready, bytes, kind}``; kind ``legacy`` is a
    preset directory from before r3 (still used by recipes without ``model.revision``)."""
    target = Target.of(target)
    volume, out = _volume(target), []
    for top in volume.listdir("/"):
        name = top.path.strip("/")
        if "--" not in name:
            out.append({"key": name, "hf_repo": None, "revision": None, "ready": True, "bytes": None,
                        "kind": "legacy"})
            continue
        for rev in volume.listdir(name):
            key = rev.path.strip("/")
            if not re.fullmatch(KEY_RE, key):
                continue
            marker = _ready(volume, key)
            out.append({"key": key, "hf_repo": name.replace("--", "/", 1), "revision": key.split("/")[1],
                        "ready": marker is not None, "bytes": (marker or {}).get("bytes"), "kind": "snapshot"})
    return out


def delete_weights(key: str, target: Target | dict | None) -> dict:
    """Delete one keyed snapshot. Refuses legacy preset directories, anything that is not
    ``<org>--<name>/<sha>``, and a key whose prepare holds the lock."""
    target = Target.of(target)
    check_key(key)
    if _locks(target).get(key):
        raise RuntimeError(f"weights {key} are being prepared; not deleting")
    volume = _volume(target)
    try:
        volume.remove_file(key, recursive=True)
    except (FileNotFoundError, modal.exception.NotFoundError):
        return {"key": key, "deleted": False}
    return {"key": key, "deleted": True}
