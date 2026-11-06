"""The stand-in caller's launch manifests, which a run viewer on the launcher side reads."""

import json

from miles_ext.launch import manifest

RECIPE = {
    "name": "toy",
    "cluster": {"nodes": 2, "gpu": "B300", "timeout_hours": 6},
    "algorithm": {"num_rollout": 3, "rollout_batch_size": 4, "n_samples": 2},
    "agent": {"timeout": 600, "trial_timeout": 1200, "max_steps": 30},
    "sandbox": {"provider": "sail"},
    "tracking": {"wandb_project": "miles-toy"},
}


def test_launcher_registration_keeps_hand_registered_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("MILES_MONITOR_DIR", str(tmp_path))
    manifest.write_manifest("ap-1", "toy-1", title="by hand", started_at="2026-10-01T00:00:00Z")
    manifest.register_launch("ap-1", "miles-fork", RECIPE, "toy-1")
    m = json.loads((tmp_path / "manifests" / "ap-1.json").read_text())
    assert (m["title"], m["started_at"]) == ("by hand", "2026-10-01T00:00:00Z")
    assert m["expected_trials"] == 24 and m["gpu"] == "2x B300:8" and m["sandbox"]["app"] == "miles-mobius-toy-1"


def test_register_launch_is_off_without_a_viewer_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("MILES_MONITOR_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert manifest.register_launch("ap-1", "miles-fork", RECIPE, "toy-1") is None
    assert not (tmp_path / ".miles-monitor").exists()
