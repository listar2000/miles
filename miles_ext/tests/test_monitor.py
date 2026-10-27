import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECIPE = {
    "name": "toy",
    "cluster": {"nodes": 2, "gpu": "B300", "timeout_hours": 6},
    "algorithm": {"num_rollout": 3, "rollout_batch_size": 4, "n_samples": 2},
    "agent": {"timeout": 600, "trial_timeout": 1200, "max_steps": 30},
    "sandbox": {"provider": "sail"},
    "tracking": {"wandb_project": "miles-toy"},
}


def _manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("MILES_MONITOR_DIR", str(tmp_path))
    monkeypatch.setattr("miles_ext.monitor.config.DIR", tmp_path)
    from miles_ext.monitor import manifest

    return manifest


def test_launcher_registration_keeps_hand_registered_fields(monkeypatch, tmp_path):
    manifest = _manifest(monkeypatch, tmp_path)
    manifest.write_manifest("ap-1", "toy-1", title="by hand", started_at="2026-10-01T00:00:00Z")
    manifest.register_launch("ap-1", "miles-fork", RECIPE, "toy-1")
    m = json.loads((tmp_path / "manifests" / "ap-1.json").read_text())
    assert (m["title"], m["started_at"]) == ("by hand", "2026-10-01T00:00:00Z")
    assert m["expected_trials"] == 24 and m["gpu"] == "2x B300:8" and m["sandbox"]["app"] == "miles-mobius-toy-1"


def test_register_launch_is_off_without_monitor_dir(monkeypatch, tmp_path):
    manifest = _manifest(monkeypatch, tmp_path)
    monkeypatch.delenv("MILES_MONITOR_DIR")
    assert manifest.register_launch("ap-1", "miles-fork", RECIPE, "toy-1") is None
    assert not (tmp_path / "manifests").exists()


@pytest.mark.skipif(sys.version_info < (3, 11), reason="the monitor reads monitor.toml with tomllib")
def test_config_rejects_unknown_keys(tmp_path):
    from miles_ext.monitor import config

    (tmp_path / "monitor.toml").write_text('wandb_entity = "me"\n')
    assert config.load(tmp_path / "monitor.toml")["wandb_entity"] == "me"
    (tmp_path / "monitor.toml").write_text('wandb_entiy = "me"\n')
    with pytest.raises(ValueError, match="wandb_entiy"):
        config.load(tmp_path / "monitor.toml")


def test_server_serves_a_registered_run(monkeypatch, tmp_path):
    """The fields a watchdog reads from /api/state, from a manifest and a saved state, with no collectors."""
    manifest = _manifest(monkeypatch, tmp_path)
    manifest.write_manifest("ap-1", "toy-1", **manifest.fields_from_recipe(RECIPE, "toy-1"))
    run = {
        "key": "miles-fork-results/toy-1",
        "run_id": "toy-1",
        "volume": "miles-fork-results",
        "registry": {},
        "attempts": [{"app_id": "ap-1"}],
        "trials": {"dirs": 5, "finished": 3, "solved": 1, "phases": {"done": 3}, "exceptions": {}},
    }
    (tmp_path / "state.json").write_text(json.dumps({"runs": {run["key"]: run}}))
    env = os.environ | {"MILES_MONITOR_DIR": str(tmp_path), "MONITOR_COLLECT": "0", "MONITOR_PORT": "0"}
    log = tmp_path / "server.log"
    with log.open("w") as out:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "miles_ext.monitor.server"],
            cwd=REPO,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    try:
        port = None
        deadline = time.monotonic() + 30
        while port is None and proc.poll() is None and time.monotonic() < deadline:
            m = re.search(r"monitor on http://127\.0\.0\.1:(\d+)", log.read_text())
            port = m and int(m.group(1))
            time.sleep(0.05)
        assert port, log.read_text()
        snap = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=30).read())
        (r,) = [r for r in snap["runs"] if r["run_id"] == "toy-1"]
        assert r["links"]["sandbox_app"] == "miles-mobius-toy-1" and r["attempts"][0]["limits"]["trial_secs"] == 1200
        assert (r["trials"]["finished"], r["trials"]["solved"]) == (3, 1)
    finally:
        proc.kill()
        proc.wait()
