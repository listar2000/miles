"""ledger.py's Port kind: trailers, refreshes, and the marker rule for files a port adds."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("ledger", Path(__file__).resolve().parents[2] / "fork/tools/ledger.py")
ledger = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ledger)

PR = "https://github.com/sgl-project/sglang/pull/42210@9747f67b11"
EDITED = "python/sglang/srt/managers/scheduler.py"
ADDED = "python/sglang/srt/weight_sync/gpu_delta/apply.py"


def _patch(marker=True, new_file=True):
    text = f"diff --git a/{EDITED} b/{EDITED}\n--- a/{EDITED}\n+++ b/{EDITED}\n@@ -1 +1,2 @@\n x\n"
    text += "+# [fork port-001] gpu delta\n" if marker else "+y\n"
    if new_file:
        text += (
            f"diff --git a/{ADDED} b/{ADDED}\nnew file mode 100644\n--- /dev/null\n+++ b/{ADDED}\n@@ -0,0 +1 @@\n+z\n"
        )
    return text


def _commit(sha, kind, fid, files, **trailers):
    trailers = {f"Fork-{kind}": [fid], **{k.replace("_", "-"): [v] for k, v in trailers.items()}}
    return ledger.Commit(sha, f"{kind.lower()}: {fid}", trailers, files)


def _check(tmp_path, monkeypatch, series, changed=(), patch=None):
    root = tmp_path / "repo"
    (root / ledger.SGLANG_PATCHES).mkdir(parents=True)
    if patch is not None:
        (root / ledger.SGLANG_PATCHES / "port-001-gpu-delta.patch").write_text(patch)
    monkeypatch.setattr(ledger, "ROOT", root)
    monkeypatch.setattr(ledger, "commits", lambda base: series)
    monkeypatch.setattr(ledger, "git", lambda *args: "\n".join(changed))
    monkeypatch.setattr(ledger, "in_base", lambda base, path: False, raising=False)
    return ledger.cmd_check("v0")


def test_port_files_and_markers():
    assert ledger.sglang_patch_id("fork/deps/sglang/port-001-gpu-delta.patch") == "port-001"
    assert ledger.patch_files(_patch(), "port-001") == {EDITED: True, ADDED: True}
    assert ledger.patch_files(_patch(marker=False), "port-001") == {EDITED: False, ADDED: True}
    assert ledger.patch_files(_patch(), "fix-001")[ADDED] is False  # only a port's new files count as marked
    assert ledger.patch_new_files(_patch()) == {ADDED}


def test_sglang_port_commit_passes(tmp_path, monkeypatch, capsys):
    port = _commit(
        "a1",
        "Port",
        "port-001",
        ["fork/deps/sglang/port-001-gpu-delta.patch"],
        Touches=f"sglang:{EDITED}, sglang:{ADDED}",
        Upstream_PR=PR,
        Remove_When="merged",
    )
    assert _check(tmp_path, monkeypatch, [port], patch=_patch()) == 0
    assert "1 port" in capsys.readouterr().out


@pytest.mark.parametrize(
    "trailers,problem",
    [
        ({"Remove_When": "merged"}, "missing Upstream-PR"),
        ({"Upstream_PR": "https://github.com/x/y/pull/1", "Remove_When": "merged"}, "<head sha>"),
        ({"Upstream_PR": PR}, "missing Remove-When"),
    ],
)
def test_port_trailers_are_required(tmp_path, monkeypatch, capsys, trailers, problem):
    port = _commit(
        "a1",
        "Port",
        "port-001",
        ["fork/deps/sglang/port-001-gpu-delta.patch"],
        Touches=f"sglang:{EDITED}, sglang:{ADDED}",
        **trailers,
    )
    assert _check(tmp_path, monkeypatch, [port], patch=_patch()) == 1
    assert problem in capsys.readouterr().out


def test_an_edited_file_still_needs_its_marker(tmp_path, monkeypatch, capsys):
    port = _commit(
        "a1",
        "Port",
        "port-001",
        ["fork/deps/sglang/port-001-gpu-delta.patch"],
        Touches=f"sglang:{EDITED}, sglang:{ADDED}",
        Upstream_PR=PR,
        Remove_When="merged",
    )
    assert _check(tmp_path, monkeypatch, [port], patch=_patch(marker=False)) == 1
    assert f"no added [fork port-001] line in ['{EDITED}']" in capsys.readouterr().out


def test_a_refresh_may_drop_a_file_and_reuse_the_id(tmp_path, monkeypatch):
    first = _commit(
        "a1",
        "Port",
        "port-001",
        ["fork/deps/sglang/port-001-gpu-delta.patch"],
        Touches=f"sglang:{EDITED}, sglang:{ADDED}",
        Upstream_PR=PR,
        Remove_When="merged",
    )
    refresh = _commit(
        "a2",
        "Port",
        "port-001",
        ["fork/deps/sglang/port-001-gpu-delta.patch"],
        Touches=f"sglang:{EDITED}",
        Upstream_PR=PR,
        Remove_When="merged",
    )
    assert _check(tmp_path, monkeypatch, [first, refresh], patch=_patch(new_file=False)) == 0


def test_miles_port_new_files_need_no_marker_but_edited_ones_do(tmp_path, monkeypatch, capsys):
    port = _commit(
        "a1",
        "Port",
        "port-002",
        ["miles/new.py", "miles/old.py"],
        Touches="miles/new.py, miles/old.py",
        Upstream_PR=PR,
        Remove_When="merged",
    )
    root = tmp_path / "repo"
    (root / "miles").mkdir(parents=True)
    (root / "miles/new.py").write_text("x = 1\n")
    (root / "miles/old.py").write_text("y = 1  # [fork port-002]\n")
    monkeypatch.setattr(ledger, "in_base", lambda base, path: path == "miles/old.py", raising=False)
    monkeypatch.setattr(ledger, "ROOT", root)
    (root / ledger.SGLANG_PATCHES).mkdir(parents=True)
    monkeypatch.setattr(ledger, "commits", lambda base: [port])
    monkeypatch.setattr(ledger, "git", lambda *args: "miles/new.py\nmiles/old.py")
    assert ledger.cmd_check("v0") == 0
    (root / "miles/old.py").write_text("y = 1\n")
    assert ledger.cmd_check("v0") == 1
    assert "miles/old.py: no [fork <id>] marker for ['port-002']" in capsys.readouterr().out


def test_image_check_requires_a_ports_new_files_to_exist(tmp_path, monkeypatch):
    root, src = tmp_path / "repo", tmp_path / "sglang"
    (root / ledger.SGLANG_PATCHES).mkdir(parents=True)
    (root / ledger.SGLANG_PATCHES / "port-001-gpu-delta.patch").write_text(_patch())
    (src / EDITED).parent.mkdir(parents=True)
    (src / EDITED).write_text("# [fork port-001]\n")
    monkeypatch.setattr(ledger, "ROOT", root)
    monkeypatch.setattr(ledger, "_stack_value", lambda name: str(src))
    assert any("apply.py: missing" in p for p in ledger.sglang_marker_problems())
    (src / ADDED).parent.mkdir(parents=True)
    (src / ADDED).write_text("z\n")
    assert ledger.sglang_marker_problems() == []


def test_a_miles_refresh_may_restore_a_file_the_pr_no_longer_edits(tmp_path, monkeypatch):
    first = _commit(
        "a1",
        "Port",
        "port-002",
        ["miles/kept.py", "miles/dropped.py"],
        Touches="miles/kept.py, miles/dropped.py",
        Upstream_PR=PR,
        Remove_When="merged",
    )
    refresh = _commit(
        "a2",
        "Port",
        "port-002",
        ["miles/kept.py", "miles/dropped.py"],
        Touches="miles/kept.py",
        Upstream_PR=PR,
        Remove_When="merged",
    )
    root = tmp_path / "repo"
    (root / "miles").mkdir(parents=True)
    (root / "miles/kept.py").write_text("y = 1  # [fork port-002]\n")
    (root / ledger.SGLANG_PATCHES).mkdir(parents=True)
    monkeypatch.setattr(ledger, "ROOT", root)
    monkeypatch.setattr(ledger, "commits", lambda base: [first, refresh])
    monkeypatch.setattr(ledger, "git", lambda *args: "miles/kept.py")  # dropped.py is back to the base
    monkeypatch.setattr(ledger, "in_base", lambda base, path: True, raising=False)
    monkeypatch.setattr(ledger, "only_removes", lambda sha, path, base: sha == "a2" and path == "miles/dropped.py")
    assert ledger.cmd_check("v0") == 0
