"""Test staged checkpoint uploads, tracker publication, and recipe wiring."""

import os
import sys
import threading
import types
from argparse import Namespace

from miles_ext.checkpoint import staged_save as S


class FakeQueue:
    """Like modal.Queue: a non-blocking get of an empty partition returns None."""

    def __init__(self):
        self.parts = {}

    def put(self, item, partition):
        self.parts.setdefault(partition, []).append(item)

    def get(self, partition, block=True, timeout=None):
        assert not block, "the uploader never blocks on the queue"
        items = self.parts.get(partition) or []
        return items.pop(0) if items else None


def _save(stage, iteration, ranks, owner, monkeypatch):
    """What Megatron does on one node inside ``context``: shards per rank, rank 0's metadata."""
    monkeypatch.setenv(S.STAGE_ENV, str(stage))
    for rank in ranks:
        monkeypatch.setitem(sys.modules, "torch.distributed", types.SimpleNamespace(get_rank=lambda r=rank: r))
        monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(distributed=sys.modules["torch.distributed"]))
        args = Namespace(save="/vol/run/checkpoints")
        with S.context(args, iteration):
            assert args.save == str(stage)
            it = stage / f"iter_{iteration:07d}"
            (it / f"__{rank}_0.distcp").write_bytes(b"x" * 10)
            if owner and rank == ranks[0]:
                (it / "metadata.json").write_text("{}")
                (it / "common.pt").write_bytes(b"c")
                (stage / "latest_checkpointed_iteration.txt").write_text(str(iteration))
        assert args.save == "/vol/run/checkpoints"


def _settled(stage):
    for m in stage.glob(f"iter_*/{S.MARKER}*"):
        os.utime(m, (0, 0))


def _uploader(tmp_path, node, q, commits=None, train_ranks=4):
    return S.Uploader(
        tmp_path / f"stage{node}",
        tmp_path / "vol" / "run",
        "run",
        node,
        q,
        lambda: (commits if commits is not None else []).append(node),
        train_ranks=train_ranks,
        log=lambda *_: None,
    )


def test_tracker_moves_only_after_every_node_uploaded(tmp_path, monkeypatch):
    q, commits = FakeQueue(), []
    head, worker = _uploader(tmp_path, 0, q, commits), _uploader(tmp_path, 1, q, commits)
    _save(head.stage, 3, [0, 1], True, monkeypatch)
    _save(worker.stage, 3, [2, 3], False, monkeypatch)
    _settled(head.stage), _settled(worker.stage)
    tracker = tmp_path / "vol" / "run" / "checkpoints" / "latest_checkpointed_iteration.txt"

    assert head.drain(timeout=0) is False  # uploaded its half; node 1 has not reported
    assert head.busy() and not tracker.exists() and not (head.stage / "iter_0000003").exists()
    assert worker.drain(timeout=0) is True and not worker.busy()
    assert head.drain(timeout=0) is True and not head.busy()
    assert tracker.read_text() == "3"
    ckpt = tracker.parent / "iter_0000003"
    assert sorted(p.name for p in ckpt.iterdir()) == [
        "__0_0.distcp",
        "__1_0.distcp",
        "__2_0.distcp",
        "__3_0.distcp",
        "common.pt",
        "metadata.json",
    ]
    assert commits == [0, 1]  # each node committed its shards before reporting


def test_a_repeated_report_does_not_count_twice(tmp_path, monkeypatch):
    q = FakeQueue()
    head = _uploader(tmp_path, 0, q)
    _save(head.stage, 1, [0, 1], True, monkeypatch)
    _settled(head.stage)
    q.put((0, 1, 2), partition=S.report_partition("run"))  # an upload retried after its report went out
    assert head.drain(timeout=0) is False  # 2 of 4 ranks, not 4
    assert not (tmp_path / "vol" / "run" / "checkpoints" / "latest_checkpointed_iteration.txt").exists()


def test_unsettled_and_partial_saves_wait(tmp_path, monkeypatch):
    up = _uploader(tmp_path, 0, FakeQueue(), train_ranks=1)
    _save(up.stage, 1, [0], True, monkeypatch)
    assert up.drain(timeout=0) is False  # marker too fresh: another rank may still be marking
    (up.stage / "iter_0000002").mkdir()  # a save cut short: no marker, never uploaded
    _settled(up.stage)
    assert up.drain(timeout=5) is True
    assert sorted(p.name for p in up.stage.glob("iter_*")) == ["iter_0000002"]


def test_next_save_waits_for_the_upload(tmp_path, monkeypatch):
    up = _uploader(tmp_path, 0, FakeQueue(), train_ranks=1)
    _save(up.stage, 1, [0], True, monkeypatch)
    monkeypatch.setattr(S, "POLL_SECS", 0.01)
    done = threading.Event()

    def second():
        _save(up.stage, 2, [0], True, monkeypatch)
        done.set()

    threading.Thread(target=second, daemon=True).start()
    assert not done.wait(0.2)
    _settled(up.stage)
    up.drain(timeout=0)
    assert done.wait(2)


def test_exit_record_waits_for_the_last_upload(tmp_path, monkeypatch):
    import sys as _sys

    from miles_ext.tests.test_supervisor import _supervisor

    sentinel = tmp_path / "save-now"
    sup, _, commits = _supervisor(tmp_path, monkeypatch, num_rollout=3, segment_secs=None, sentinel=sentinel)
    tracker = tmp_path / "run1" / "checkpoints" / "latest_checkpointed_iteration.txt"
    tracker.parent.mkdir(parents=True)
    polls = []

    def pending():  # the last save is staged: its upload moves the tracker a few polls after Miles exits
        polls.append(1)
        if len(polls) < 3:
            return True
        tracker.write_text("3")
        return False

    sup.pending = pending
    record = sup.run([_sys.executable, "-c", "print('done')"], cwd=str(tmp_path), env={})
    assert record["reason"] == "success" and record["tracker"] == 3 and record["complete"]
    assert commits, "the tracker the upload moved was not committed"


def test_recipe_stage_dir_wires_the_hook_and_env():
    from pathlib import Path

    import yaml

    from miles_ext.recipes import recipe as R

    spec = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / "miles_ext/recipes/configs/" "dsv41-flash-tb4-overfit16.yaml"
        ).read_text()
    )
    spec["taskset"] = {k: v for k, v in spec["taskset"].items() if k not in ("tasks_dir", "prompts")}
    off = R.from_dict(spec)
    assert "--custom-megatron-save-context-path" not in R.miles_args(off, "run1", resume=False)
    assert S.STAGE_ENV not in R.trainer_env(off)
    on = R.from_dict(spec, {"checkpoint": {"stage_dir": "/tmp/miles-ckpt-stage"}})
    R.validate(on, local_inputs=False)
    argv = R.miles_args(on, "run1", resume=False)
    assert argv[argv.index("--custom-megatron-save-context-path") + 1] == "miles_ext.checkpoint.staged_save.context"
    assert R.trainer_env(on)[S.STAGE_ENV] == "/tmp/miles-ckpt-stage"
