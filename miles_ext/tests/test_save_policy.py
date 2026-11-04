"""Test which saves carry Adam's state, retention of kept steps, and the recipe wiring."""

import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from miles_ext.checkpoint import save_policy as P
from miles_ext.platform.checkpoints import apply_retention_local, retention_plan
from miles_ext.recipes import recipe as R


def _spec() -> dict:
    path = Path(R.__file__).parent / "configs" / "dsv41-flash-tb4-overfit16.yaml"
    spec = yaml.safe_load(path.read_text())
    spec["taskset"] = {k: v for k, v in spec["taskset"].items() if k not in ("tasks_dir", "prompts")}
    return spec


def test_adam_state_goes_in_every_kth_save_and_marks_it(tmp_path, monkeypatch):
    monkeypatch.setenv(P.OPTIM_EVERY_ENV, "3")
    dist = types.SimpleNamespace(get_rank=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(distributed=dist))
    seen = {}
    for it in (1, 3, 5, 6):  # save_interval 2: saves after rollouts 1, 3, 5, ...; 6 is a final save off the interval
        (tmp_path / f"iter_{it:07d}").mkdir()
        args = Namespace(save=str(tmp_path), no_save_optim=False, save_interval=2)
        with P.context(args, it):
            seen[it] = args.no_save_optim
        assert args.no_save_optim is False
    assert seen == {1: True, 3: True, 5: False, 6: True}
    assert [it for it in seen if P.has_optimizer(tmp_path / f"iter_{it:07d}")] == [5]


def test_a_failed_save_restores_args_and_leaves_no_marker(tmp_path, monkeypatch):
    monkeypatch.setenv(P.OPTIM_EVERY_ENV, "1")
    args = Namespace(save=str(tmp_path), no_save_optim=False, save_interval=1)
    (tmp_path / "iter_0000000").mkdir()
    with pytest.raises(OSError), P.context(args, 0):
        raise OSError("disk full")
    assert args.no_save_optim is False and not P.has_optimizer(tmp_path / "iter_0000000")


def test_retention_keeps_kth_rollouts_and_the_newest_adam_save(tmp_path):
    saves = [{"step": s, "kind": "full"} for s in (4, 9, 14, 19, 24)]
    plan = retention_plan(saves, keep=1, keep_final_full=True, final=False, latest=24, keep_every=10)
    assert [s["step"] for s in plan] == [14, 4]  # 9 and 19 are the 10th and 20th rollouts, 24 the newest
    assert [
        s["step"] for s in retention_plan(saves, keep=1, keep_final_full=True, final=True, latest=24, keep_every=10)
    ] == [14, 4]
    run = tmp_path / "run"
    for s in (4, 9, 14, 19, 24):
        (run / "checkpoints" / f"iter_{s:07d}").mkdir(parents=True)
        (run / "checkpoints" / f"iter_{s:07d}" / "common.pt").write_bytes(b"c")
    (run / "checkpoints" / "iter_0000014" / P.OPTIM_MARKER).touch()
    (run / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("24")
    dropped = apply_retention_local(run, keep=1, keep_final_full=True, final=False, keep_every=10)
    assert [d["step"] for d in dropped] == [4]


def test_recipe_wiring_and_refusals():
    off = R.from_dict(_spec())
    assert "--custom-megatron-save-context-path" not in R.miles_args(off, "run1", resume=False)
    assert P.OPTIM_EVERY_ENV not in R.trainer_env(off)
    every = R.from_dict(_spec(), {"checkpoint": {"optimizer_every": 4, "save_optimizer": True, "keep_every": 20}})
    R.validate(every, local_inputs=False)
    argv = R.miles_args(every, "run1", resume=False)
    assert argv[argv.index("--custom-megatron-save-context-path") + 1] == "miles_ext.checkpoint.save_policy.context"
    assert R.trainer_env(every)[P.OPTIM_EVERY_ENV] == "4"
    assert "--no-load-optim" not in " ".join(R.miles_args(every, "run1", resume=True))
    assert "--no-load-optim" in " ".join(R.miles_args(every, "run1", resume=True, resume_optimizer=False))
    with pytest.raises(ValueError, match="optimizer_every needs"):
        R.validate(
            R.from_dict(_spec(), {"checkpoint": {"optimizer_every": 4, "save_optimizer": False}}), local_inputs=False
        )
    with pytest.raises(ValueError, match="keep_every must be a multiple"):  # save_interval is 10
        R.validate(R.from_dict(_spec(), {"checkpoint": {"keep_every": 25}}), local_inputs=False)
