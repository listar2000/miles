"""Test staged checkpoint uploads, tracker publication, and recipe wiring."""

import os
import shutil
import sys
import threading
import types
from argparse import Namespace
from pathlib import Path

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


def _rank(monkeypatch, rank):
    monkeypatch.setitem(sys.modules, "torch.distributed", types.SimpleNamespace(get_rank=lambda r=rank: r))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(distributed=sys.modules["torch.distributed"]))


def _save(stage, iteration, ranks, owner, monkeypatch):
    """What Megatron does on one node inside ``context``: shards per rank, rank 0's metadata."""
    monkeypatch.setenv(S.STAGE_ENV, str(stage))
    for rank in ranks:
        _rank(monkeypatch, rank)
        args = Namespace(save="/vol/run/checkpoints", no_save_optim=True, save_interval=1)
        with S.context(args, iteration):
            assert args.save == str(stage)
            it = stage / f"iter_{iteration:07d}"
            (it / f"__{rank}_0.distcp").write_bytes(b"x" * 10)
            if owner and rank == ranks[0]:
                (it / "metadata.json").write_text("{}")
                (it / "common.pt").write_bytes(b"c")
                (stage / "latest_checkpointed_iteration.txt").write_text(str(iteration))
        assert args.save == "/vol/run/checkpoints"


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
    q.put((0, 1, 2), partition=S.report_partition("run"))  # an upload retried after its report went out
    assert head.drain(timeout=0) is False  # 2 of 4 ranks, not 4
    assert not (tmp_path / "vol" / "run" / "checkpoints" / "latest_checkpointed_iteration.txt").exists()


def test_a_save_uploads_once_every_local_rank_marked_it(tmp_path, monkeypatch):
    up = _uploader(tmp_path, 0, FakeQueue(), train_ranks=2)
    (up.stage / S.RANKS).mkdir()
    (up.stage / S.RANKS / "1").touch()  # rank 1 of this node is inside the save, not done yet
    _save(up.stage, 1, [0], True, monkeypatch)
    assert up.drain(timeout=0) is False
    _save(up.stage, 1, [1], False, monkeypatch)
    (up.stage / "iter_0000002").mkdir()  # a save cut short: no marker, never uploaded
    assert up.drain(timeout=5) is True
    assert sorted(p.name for p in up.stage.glob("iter_*")) == ["iter_0000002"]


def test_rank_0_outside_container_0_is_refused(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv(S.STAGE_ENV, str(tmp_path / "stage"))
    monkeypatch.setenv(S.NODE_ENV, "1")
    _rank(monkeypatch, 0)
    with pytest.raises(RuntimeError, match="container 0"):
        with S.context(Namespace(save=str(tmp_path / "vol"), no_save_optim=True, save_interval=1), 0):
            pass


def test_busy_until_the_tracker_is_written(tmp_path, monkeypatch):
    q = FakeQueue()
    up = _uploader(tmp_path, 0, q, train_ranks=1)
    _save(up.stage, 1, [0], True, monkeypatch)
    up._upload(up._ready())  # uploaded and reported, the report not yet collected
    assert up.busy()
    tracker = tmp_path / "vol" / "run" / "checkpoints" / "latest_checkpointed_iteration.txt"
    tracker.mkdir(parents=True)  # the tracker write fails
    up._collect()
    try:
        up._track()
    except OSError:
        pass
    assert up.busy()
    tracker.rmdir()
    up._track()
    assert not up.busy() and tracker.read_text() == "1"


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


def test_adam_state_goes_in_every_kth_save_straight_to_the_volume(tmp_path, monkeypatch):
    stage, vol = tmp_path / "stage", tmp_path / "vol"
    monkeypatch.setenv(S.STAGE_ENV, str(stage))
    monkeypatch.setenv(S.OPTIM_EVERY_ENV, "3")
    _rank(monkeypatch, 0)
    seen = {}
    for it in (1, 3, 5, 11):  # save_interval 2 (saves after rollouts 1, 3, 5, ...): Adam's state in every 3rd save
        args = Namespace(save=str(vol), no_save_optim=False, save_interval=2)
        with S.context(args, it):
            seen[it] = (args.save, args.no_save_optim)
            Path(args.save, f"iter_{it:07d}").mkdir(parents=True, exist_ok=True)
        assert (args.save, args.no_save_optim) == (str(vol), False)
        shutil.rmtree(stage, ignore_errors=True)  # stands in for the upload
    assert seen == {1: (str(stage), True), 3: (str(stage), True), 5: (str(vol), False), 11: (str(vol), False)}
    assert S.has_optimizer(vol / "iter_0000005") and not S.has_optimizer(vol / "iter_0000001")
    assert not S.with_optimizer(Namespace(no_save_optim=False, save_interval=2), 6)  # a final save off the interval
    monkeypatch.setenv(S.OPTIM_EVERY_ENV, "1")
    assert S.with_optimizer(Namespace(no_save_optim=False, save_interval=10), 4)


def test_an_upload_does_not_move_the_tracker_back(tmp_path, monkeypatch):
    up = _uploader(tmp_path, 0, FakeQueue(), train_ranks=1)
    _save(up.stage, 2, [0], True, monkeypatch)
    tracker = tmp_path / "vol" / "run" / "checkpoints" / "latest_checkpointed_iteration.txt"
    tracker.parent.mkdir(parents=True)
    tracker.write_text("3")  # save 3 carried Adam's state and went to the volume directly
    assert up.drain(timeout=0) is True
    assert tracker.read_text() == "3" and (tracker.parent / "iter_0000002" / "common.pt").exists()


def test_upload_copies_large_files_in_pieces(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "COPY_PIECE", 1000)
    monkeypatch.setattr(S, "IO_CHUNK", 300)
    src = tmp_path / "src"
    src.mkdir()
    blobs = {"a.distcp": os.urandom(4321), "b.distcp": os.urandom(1000), "empty": b""}
    for name, data in blobs.items():
        (src / name).write_bytes(data)
    assert S._copy_pieces(sorted(src.iterdir()), src, tmp_path / "dst") == 5321
    assert {p.name: p.read_bytes() for p in (tmp_path / "dst").iterdir()} == blobs


def test_retention_keeps_every_kth_step_and_the_newest_adam_save(tmp_path):
    from miles_ext.platform.checkpoints import apply_retention_local, retention_plan

    saves = [{"step": s, "kind": "full"} for s in (5, 10, 15, 20, 25)]
    plan = retention_plan(saves, keep=1, keep_final_full=True, final=False, latest=25, keep_every=10)
    assert [s["step"] for s in plan] == [15, 5]
    run = tmp_path / "run"
    for s in (5, 10, 15, 20, 25):
        (run / "checkpoints" / f"iter_{s:07d}").mkdir(parents=True)
        (run / "checkpoints" / f"iter_{s:07d}" / "common.pt").write_bytes(b"c")
    (run / "checkpoints" / "iter_0000015" / S.OPTIM_MARKER).touch()
    (run / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("25")
    dropped = apply_retention_local(run, keep=1, keep_final_full=True, final=False, keep_every=10)
    assert sorted(d["step"] for d in dropped) == [5]


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
    every = R.from_dict(spec, {"checkpoint": {"optimizer_every": 4, "save_optimizer": True}})
    R.validate(every, local_inputs=False)
    assert R.trainer_env(every)[S.OPTIM_EVERY_ENV] == "4"
    assert "--no-load-optim" not in " ".join(R.miles_args(every, "run1", resume=True))
    assert "--no-load-optim" in " ".join(R.miles_args(every, "run1", resume=True, resume_optimizer=False))
