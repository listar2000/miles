"""Trainer resolution: latest from the registry, stable versions from the list."""

import json
from pathlib import Path

import pytest

from miles_ext import stack
from miles_ext.platform import versions as V

SHA = "a" * 40
DIGEST = "sha256:" + "b" * 64


class FakeRegistry:
    def __init__(self, env):
        self.env, self.asked = env, []

    def manifest(self, tag):
        self.asked.append(tag)
        return DIGEST, {"config": {"Env": self.env}}


def test_latest_reads_commit_from_image():
    reg = FakeRegistry([f"MILES_FORK_COMMIT={SHA}", f"MILES_FORK_RELEASE=main-{SHA[:12]}", "PATH=/bin"])
    got = V.resolve("latest", registry=reg)
    assert reg.asked == [stack.LATEST_TAG]
    assert got == {"trainer": "latest", "name": f"main-{SHA[:12]}", "commit": SHA,
                   "image": f"{stack.IMAGE_REPO}@{DIGEST}"}


def test_latest_without_commit_fails():
    with pytest.raises(RuntimeError, match="MILES_FORK_COMMIT"):
        V.resolve("latest", registry=FakeRegistry(["PATH=/bin"]))


def test_stable_lookup():
    entry = {"name": "stable-2026-10-15", "commit": SHA, "image": f"x/miles@{DIGEST}", "date": "2026-10-15"}
    assert V.resolve("stable-2026-10-15", stable=[entry])["image"] == f"x/miles@{DIGEST}"
    with pytest.raises(ValueError, match="no stable version"):
        V.resolve("stable-2026-10-16", stable=[entry])
    with pytest.raises(ValueError, match="must be"):
        V.resolve("fork-v0.1.2-r6", stable=[entry])


@pytest.mark.parametrize("bad", [
    {"commit": SHA, "image": f"x@{DIGEST}", "date": "d"},
    {"name": "stable-2026-10-15", "commit": SHA[:10], "image": f"x@{DIGEST}", "date": "d"},
    {"name": "stable-2026-10-15", "commit": SHA, "image": "x:stable-2026-10-15", "date": "d"},
])
def test_stable_list_checks(bad):
    with pytest.raises(ValueError):
        V.check_stable_list([bad])


def test_packaged_stable_list_is_valid():
    data = json.loads((Path(V.__file__).resolve().parents[1] / "stable.json").read_text())
    V.check_stable_list(data)
    assert data[0]["name"] == "stable-2026-10-15"


def test_stable_list_falls_back_to_packaged(monkeypatch):
    def offline(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(V.urllib.request, "urlopen", offline)
    assert V.stable_versions()[0]["name"] == "stable-2026-10-15"


def test_spawn_passes_image(monkeypatch):
    import importlib
    L = importlib.import_module("miles_ext.platform.launch")  # the package re-exports the function
    from miles_ext.platform.target import Target

    seen = {}

    class Done:
        returncode, stderr = 0, ""
        stdout = 'LAUNCHED {"app_id": "ap-1", "call_id": "fc-1"}\n'

    def run(cmd, env, **kw):
        seen.update(json.loads(env["MILES_EXT_LAUNCH"]))
        return Done()

    monkeypatch.setattr(L.subprocess, "run", run)
    image = f"{stack.IMAGE_REPO}@{DIGEST}"
    L._spawn({"name": "r"}, "run1", Target.of(None), inputs="volume", from_commit=None, image=image,
             final_retention=False, timeout=60, segment_seconds=None)
    assert seen["image"] == image and seen["from_commit"] is None


def test_fork_image_requires_digest():
    from miles_ext.platform.images import fork_image
    with pytest.raises(ValueError, match="digest"):
        fork_image(image=f"{stack.IMAGE_REPO}:main")
    with pytest.raises(ValueError, match="not both"):
        fork_image(commit=SHA, image=f"{stack.IMAGE_REPO}@{DIGEST}")
