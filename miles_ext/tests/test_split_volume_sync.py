"""The split adapter sync's trainer side: a sync on the volume, its commit handoff, and pruning."""

import pytest


def test_sync_dir_commit_and_prune(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from miles_ext.launch.split import volume_sync as V

    monkeypatch.setattr(V, "COMMIT_DIR", tmp_path / "commit")
    commits = []
    stop = V.commit_watcher(lambda: commits.append(1))
    try:
        for version in (1, 2, 3):
            out = V.write_sync(
                tmp_path / "adapters",
                version,
                [[("default:a.lora_A.weight", torch.ones(2, 2))], [("default:b", torch.zeros(3))]],
            )
            V.request_commit(out.name, timeout=10)
            V.prune(tmp_path / "adapters", keep=2)
    finally:
        stop.set()
    assert len(commits) == 3
    assert sorted(p.name for p in (tmp_path / "adapters").iterdir()) == ["v0000002", "v0000003"]
    assert (tmp_path / "adapters" / "v0000003" / V.MANIFEST).read_text().count("default:b") == 1
