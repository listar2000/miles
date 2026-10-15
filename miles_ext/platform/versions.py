"""Which trainer a run uses: ``latest`` (main's newest image) or a stable version.

A run names its trainer once, when it is created, and the platform resolves the
name here to a fixed image (``repo@sha256:...``) and the commit inside it. The
run stores that resolution and passes ``image=`` to every ``launch`` segment, so
a resume never moves to a newer build.

- ``latest``: the image tagged ``LATEST_TAG`` in ``IMAGE_REPO``, i.e. the last
  build of ``main`` (it can trail ``main`` after a doc-only landing). Its commit
  is read from the image's own config (``MILES_MOBIUS_COMMIT``), not from git.
- ``stable-YYYY-MM-DD``: an entry of the stable list (``STABLE_LIST``), each a
  commit and image digest that passed the staging smoke test and trained one
  recipe to the end. The list is read from ``main`` on GitHub, not from this
  installed copy, which may itself be an older stable version; the packaged copy
  is the fallback when GitHub cannot be reached (entries are only ever appended).
"""

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from miles_ext import stack

IMAGE_REPO, LATEST_TAG, STABLE_RE = stack.IMAGE_REPO, stack.LATEST_TAG, stack.STABLE_RE
LATEST = "latest"
STABLE_LIST = "miles_ext/stable.json"
STABLE_LIST_URL = f"{stack.FORK_REPO.replace('github.com', 'raw.githubusercontent.com')}/main/{STABLE_LIST}"
_PACKAGED = Path(__file__).resolve().parents[1] / "stable.json"
_MANIFEST_TYPES = ", ".join([
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])
TIMEOUT = 30


def resolve(trainer: str, *, registry=None, stable=None) -> dict:
    """``{"trainer", "name", "commit", "image"}`` for ``latest`` or a stable name."""
    if trainer == LATEST:
        reg = registry or Registry(IMAGE_REPO)
        digest, config = reg.manifest(LATEST_TAG)
        env = dict(e.split("=", 1) for e in config.get("config", {}).get("Env", []) if "=" in e)
        commit = env.get("MILES_MOBIUS_COMMIT")
        if not commit:
            raise RuntimeError(f"{IMAGE_REPO}:{LATEST_TAG} ({digest}) has no MILES_MOBIUS_COMMIT")
        return {"trainer": LATEST, "name": env.get("MILES_MOBIUS_VERSION") or f"main@{commit[:10]}",
                "commit": commit, "image": f"{IMAGE_REPO}@{digest}"}
    if not re.fullmatch(STABLE_RE, trainer or ""):
        raise ValueError(f"trainer {trainer!r} must be {LATEST!r} or a stable version (stable-YYYY-MM-DD)")
    versions = stable if stable is not None else stable_versions()
    for v in versions:
        if v["name"] == trainer:
            return {"trainer": trainer, "name": trainer, "commit": v["commit"], "image": v["image"]}
    raise ValueError(f"no stable version {trainer!r} (known: {[v['name'] for v in versions]})")


def stable_versions(url: str = STABLE_LIST_URL) -> list[dict]:
    """The stable list, newest last: ``[{name, commit, image, date, note}]``."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            data = json.load(r)
    except (urllib.error.URLError, OSError, ValueError):
        data = json.loads(_PACKAGED.read_text())
    check_stable_list(data)
    return data


def check_stable_list(data) -> None:
    if not isinstance(data, list):
        raise ValueError("stable list must be a JSON list")
    seen = set()
    for v in data:
        missing = {"name", "commit", "image", "date"} - set(v)
        if missing:
            raise ValueError(f"stable entry {v.get('name')!r} lacks {sorted(missing)}")
        if not re.fullmatch(STABLE_RE, v["name"]) or v["name"] in seen:
            raise ValueError(f"stable entry name {v['name']!r} is malformed or repeated")
        if not re.fullmatch(r"[0-9a-f]{40}", v["commit"]):
            raise ValueError(f"{v['name']}: commit must be a full sha")
        if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", v["image"]):
            raise ValueError(f"{v['name']}: image must be repo@sha256:<digest>")
        seen.add(v["name"])


class Registry:
    """Anonymous pulls from Docker Hub's registry API: a tag's digest and its image config."""

    def __init__(self, repo: str):
        self.repo = repo.removeprefix("docker.io/")
        self._token = None

    def _get(self, url: str, accept: str | None = None):
        if self._token is None:
            auth = ("https://auth.docker.io/token?service=registry.docker.io"
                    f"&scope=repository:{self.repo}:pull")
            with urllib.request.urlopen(auth, timeout=TIMEOUT) as r:
                self._token = json.load(r)["token"]
        headers = {"Authorization": f"Bearer {self._token}"}
        if accept:
            headers["Accept"] = accept
        return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT)

    def manifest(self, tag: str) -> tuple[str, dict]:
        """``(digest, image config)`` of ``tag`` (a single-platform image, as build.py pushes)."""
        base = f"https://registry-1.docker.io/v2/{self.repo}"
        with self._get(f"{base}/manifests/{tag}", _MANIFEST_TYPES) as r:
            digest, manifest = r.headers["Docker-Content-Digest"], json.load(r)
        with self._get(f"{base}/blobs/{manifest['config']['digest']}") as r:
            return digest, json.load(r)
