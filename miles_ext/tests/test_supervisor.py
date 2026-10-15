"""R5/R6 without Modal or GPUs: completion from the checkpoint tracker, the deadline save
against a fake Miles process, exit reasons, and launch's early return.

    python -m pytest miles_ext/tests/test_supervisor.py
"""

import importlib
import json
import sys
import textwrap
import time

import pytest

P, SUP, L = (importlib.import_module(f"miles_ext.platform.{m}") for m in ("progress", "supervisor", "launch"))
from miles_ext.platform.target import Target
from miles_ext.recipes import recipe as R

from miles_ext.tests.test_platform import DEV, _recipe


# completion

@pytest.mark.parametrize("backend,tracker,num_rollout,complete", [
    # the two finished dev runs: FSDP 3c969ce5 (12 rollouts, tracker 12), Megatron 67d526e9 (4 rollouts, tracker 3)
    ("fsdp", 12, 12, True), ("fsdp", 11, 12, False), ("megatron", 3, 4, True), ("megatron", 2, 4, False),
    ("fsdp", None, 1, False), ("megatron", None, 1, False)])
def test_completion_per_backend(backend, tracker, num_rollout, complete):
    assert P.is_complete(backend, tracker, num_rollout) is complete


def test_next_rollout_id():
    assert P.next_rollout_id("fsdp", 12) == 12 and P.next_rollout_id("megatron", 3) == 4
    assert P.next_rollout_id("fsdp", None) == 0
    with pytest.raises(ValueError):
        P.next_rollout_id("deepspeed", 1)


def test_progress_names():
    assert P.progress_dict_name(Target.of(DEV)) == "mdev-progress"
    assert P.progress_dict_name(Target.of(None)) == "miles-progress"
    with pytest.raises(ValueError):
        P.keys("../x")


# deadline margin

def test_trigger_at():
    t0 = 1000.0
    assert SUP.trigger_at(None, None, 60, 60, t0) is None
    # measured: save 60 s + step 120 s before the deadline
    assert SUP.trigger_at(t0 + 3600, None, 120, 60, t0) == t0 + 3600 - 180
    # the lease ends first
    assert SUP.trigger_at(t0 + 3600, t0 + 1800, 120, 60, t0) == t0 + 1800 - 180
    # a short segment: the margin is capped at a quarter of it, so it still trains
    assert SUP.trigger_at(t0 + 600, None, None, None, t0) == t0 + 600 - 150


@pytest.mark.parametrize("code,tail,reason", [
    (0, [], "success"), (1, ["torch.OutOfMemoryError: CUDA out of memory."], "oom"),
    (1, ["[Rank 3] Watchdog caught collective operation timeout: WorkNCCL"], "nccl_timeout"),
    (1, ["ValueError: bad"], "error")])
def test_classify(code, tail, reason):
    assert SUP.classify(code, tail) == reason


# the supervisor around a fake Miles

FAKE_MILES = textwrap.dedent('''
    import json, os, sys, time
    run_dir, start, num_rollout, sentinel, step_secs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], float(sys.argv[5])
    ev = os.path.join(run_dir, "checkpoints", "events")
    os.makedirs(ev, exist_ok=True)
    def emit(name, e):
        e["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + ".%06dZ" % (time.time() % 1 * 1e6)
        with open(os.path.join(ev, name), "a") as f:
            f.write(json.dumps(e) + "\\n")
    for rollout_id in range(start, num_rollout):
        time.sleep(step_secs)
        emit("rollout_executor.jsonl", {"type": "metric", "metrics": {"rollout/step": rollout_id, "rollout/raw_reward_unfiltered": 0.5}})
        emit("trainer_controller_actor.jsonl", {"type": "train_group_step_end", "rollout_id": rollout_id})
        if os.path.exists(sentinel) or rollout_id == num_rollout - 1:  # FSDP: iter_{rollout_id + 1}
            os.makedirs(os.path.join(run_dir, "checkpoints", "iter_%07d" % (rollout_id + 1)), exist_ok=True)
            with open(os.path.join(run_dir, "checkpoints", "latest_checkpointed_iteration.txt"), "w") as f:
                f.write(str(rollout_id + 1))
            if os.path.exists(sentinel):
                os.remove(sentinel)
        print("step", rollout_id, flush=True)
    print("done", flush=True)
''')


def _supervisor(tmp_path, monkeypatch, num_rollout, segment_secs, sentinel):
    monkeypatch.setattr(SUP, "POLL_SECS", 0.05)
    monkeypatch.setattr(P, "HEARTBEAT_SECS", 0.1)
    monkeypatch.setattr(SUP, "SENTINEL", str(sentinel))
    recipe = _recipe()
    recipe["model"]["backend"] = "fsdp"
    recipe["algorithm"]["num_rollout"] = num_rollout
    store, commits = {}, []
    sup = SUP.Supervisor("run1", recipe, tmp_path / "run1", progress_dict=store, segment=0,
                         deadline=time.time() + segment_secs if segment_secs else None,
                         commit=lambda: commits.append(time.time()), image={"release": "test"}, backend="fsdp")
    return sup, store, commits


def _fake(tmp_path, sentinel, start, num_rollout, step_secs=0.3):
    script = tmp_path / "fake_miles.py"
    script.write_text(FAKE_MILES)
    return [sys.executable, str(script), str(tmp_path / "run1"), str(start), str(num_rollout), str(sentinel), str(step_secs)]


def test_deadline_save_stops_and_records(tmp_path, monkeypatch):
    sentinel = tmp_path / "save-now"
    sup, store, commits = _supervisor(tmp_path, monkeypatch, num_rollout=50, segment_secs=2.0, sentinel=sentinel)
    record = sup.run(_fake(tmp_path, sentinel, 0, 50), cwd=str(tmp_path), env={})
    assert record["reason"] == "segment_end" and record["retryable"] and not record["complete"]
    assert 0 < record["tracker"] < 50 and record["next_rollout_id"] == record["tracker"]
    assert commits, "the save was not committed"
    on_disk = json.loads((tmp_path / "run1" / P.EXIT_FILE).read_text())
    assert on_disk["reason"] == "segment_end" and store["exit:run1"]["reason"] == "segment_end"
    assert store["run:run1"]["phase"] == "ended" and store["run:run1"]["metrics"]["rollout/raw_reward_unfiltered"] == 0.5
    assert len((tmp_path / "run1" / P.EXITS_LOG).read_text().splitlines()) == 1


def test_run_to_the_end(tmp_path, monkeypatch):
    sentinel = tmp_path / "save-now"
    sup, store, _ = _supervisor(tmp_path, monkeypatch, num_rollout=3, segment_secs=None, sentinel=sentinel)
    record = sup.run(_fake(tmp_path, sentinel, 0, 3, step_secs=0.1), cwd=str(tmp_path), env={})
    assert record["reason"] == "success" and record["complete"] and record["tracker"] == 3


def test_lease_expiry_saves_and_stops(tmp_path, monkeypatch):
    sentinel = tmp_path / "save-now"
    sup, store, _ = _supervisor(tmp_path, monkeypatch, num_rollout=50, segment_secs=None, sentinel=sentinel)
    store["lease:run1"] = {"expires_at": time.time() + 1.0}
    record = sup.run(_fake(tmp_path, sentinel, 0, 50), cwd=str(tmp_path), env={})
    assert record["reason"] == "lease_expired" and record["retryable"]


def test_sigterm_stops(tmp_path, monkeypatch):
    sentinel = tmp_path / "save-now"
    sup, store, _ = _supervisor(tmp_path, monkeypatch, num_rollout=50, segment_secs=None, sentinel=sentinel)
    import threading
    threading.Timer(0.5, sup.sigterm.set).start()
    record = sup.run(_fake(tmp_path, sentinel, 0, 50), cwd=str(tmp_path), env={})
    assert record["reason"] == "stopped" and not record["retryable"]


def test_failure_keeps_the_tail(tmp_path, monkeypatch):
    sentinel = tmp_path / "save-now"
    sup, _, _ = _supervisor(tmp_path, monkeypatch, num_rollout=3, segment_secs=None, sentinel=sentinel)
    cmd = [sys.executable, "-c", "print('torch.OutOfMemoryError: CUDA out of memory.'); raise SystemExit(1)"]
    record = sup.run(cmd, cwd=str(tmp_path), env={})
    assert record["reason"] == "oom" and any("out of memory" in line for line in record["tail"])


# launch

def test_launch_returns_early_when_complete(monkeypatch):
    monkeypatch.setattr(P, "completion", lambda run_id, target, recipe: {
        "complete": True, "next_rollout_id": 20, "num_rollout": 20, "tracker": 20})
    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: pytest.fail("spawned a finished run"))
    out = L.launch(_recipe(), "run1", DEV, segment_seconds=3600)
    assert out["complete"] and not out["launched"] and out["call_id"] is None
    assert out["names"]["secret"] == "mdev-secret-run1"


def test_segment_timeout():
    assert L.segment_timeout(3600, 24) == 3600 + L.GRACE_SECS
    assert L.segment_timeout(None, 6) == 6 * 3600
    assert L.segment_timeout(30 * 3600, 24) == 24 * 3600


def test_trainer_short_circuits_on_every_rank():
    """The completion check sits before the rank split, so workers never wait for a Ray head."""
    src = (importlib.import_module("miles_ext.tests.test_platform").ROOT / "miles_ext/launch/modal_app.py").read_text()
    body = src[src.index("def train("):]
    assert body.index("P.is_complete(") < body.index("if rank != 0:")
    assert '"--save-trigger-sentinel", SENTINEL' in body


def test_event_tail_follows_a_replaced_file(tmp_path):
    """On resume Miles swaps in the events snapshot from the loaded save; new events still count."""
    def line(step, ts):
        return json.dumps({"type": "metric", "timestamp": ts, "metrics": {"rollout/step": step, "train/loss": step}})
    ev = tmp_path / "events"
    ev.mkdir()
    f = ev / "rollout_executor.jsonl"
    f.write_text("\n".join(line(i, "2026-10-15T04:00:00Z") for i in range(5)) + "\n")
    tail = SUP._EventTail(ev, since=time.time() - 60)
    tail.poll()
    assert tail.step is None  # earlier segments: skipped
    f.unlink()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    f.write_text(line(1, "2026-10-15T04:00:00Z") + "\n" + line(2, now) + "\n")  # shorter, new inode
    tail.poll()
    assert tail.step == 2 and tail.metrics["train/loss"] == 2


def test_event_snapshot_dir_is_not_a_save(tmp_path):
    C = importlib.import_module("miles_ext.platform.checkpoints")
    run = tmp_path / "run1"
    (run / "checkpoints" / "iter_0000000" / "debug_events").mkdir(parents=True)  # FSDP: rollout 0's snapshot
    for step in (1, 2):
        (run / "checkpoints" / f"iter_{step:07d}" / "model").mkdir(parents=True)
    (run / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("2")
    dropped = C.apply_retention_local(run, keep=1, keep_final_full=True, final=False)
    assert [(d["kind"], d["step"]) for d in dropped] == [("full", 1)]
    assert (run / "checkpoints" / "iter_0000000").exists()


# review fixes (H1-H4, M3, M4, L3)

def test_phase_of_matches_exit_to_segment():
    now = time.time()
    old_exit = {"call_id": "fc-old", "reason": "segment_end", "retryable": True}
    assert P.phase_of({"phase": "pending", "call_id": "fc-new", "heartbeat_at": now - 900}, old_exit, now) == "pending"
    assert P.phase_of({"phase": "training", "call_id": "fc-new", "heartbeat_at": now}, old_exit, now) == "training"
    assert P.phase_of({"phase": "stopping", "call_id": "fc-old", "heartbeat_at": now}, old_exit, now) == "ended"
    assert P.phase_of({"phase": "training", "call_id": "fc-new", "heartbeat_at": now - 3600}, None, now) == "no_heartbeat"
    assert P.phase_of(None, old_exit, now) == "ended"  # after teardown: only exit.json is left
    assert P.phase_of(None, None, now) == "no_heartbeat"


class _Dict(dict):
    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


def _launch_env(monkeypatch, store, running):
    monkeypatch.setattr(P, "completion", lambda run_id, target, recipe: {
        "complete": False, "next_rollout_id": 1, "num_rollout": 3, "tracker": 1})
    monkeypatch.setattr(P, "progress_dict", lambda target, create=True: store)
    monkeypatch.setattr(L, "run_secret_exists", lambda run_id, target: True)
    monkeypatch.setattr(L, "_weights_ready", lambda recipe, target: True)
    monkeypatch.setattr(L, "_call_running", lambda call_id: call_id in running)
    spawned = []
    monkeypatch.setattr(L, "_spawn", lambda *a, **k: spawned.append(1) or {"app_id": "ap-2", "call_id": "fc-2"})
    return spawned


def test_launch_marks_pending_and_refuses_a_live_segment(monkeypatch):
    store, running = _Dict(), {"fc-1"}
    spawned = _launch_env(monkeypatch, store, running)
    store["active:run1"] = {"call_id": "fc-1", "ts": time.time()}
    with pytest.raises(RuntimeError, match="still running"):
        L.launch(_recipe(), "run1", DEV)
    assert not spawned
    running.clear()  # fc-1 ended without releasing (killed): taken over
    out = L.launch(_recipe(), "run1", DEV)
    assert out["launched"] and out["call_id"] == "fc-2"
    assert store["run:run1"]["phase"] == "pending" and store["active:run1"]["call_id"] == "fc-2"


def test_launch_releases_the_claim_when_spawn_fails(monkeypatch):
    store = _Dict()
    _launch_env(monkeypatch, store, set())
    monkeypatch.setattr(L, "_spawn", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        L.launch(_recipe(), "run1", DEV)
    assert "active:run1" not in store


def test_final_step_save_keeps_heartbeating(tmp_path, monkeypatch):
    """A deadline save that lands on the last step: Miles finishes; no commit storm, heartbeats go on."""
    sentinel = tmp_path / "save-now"
    sup, store, commits = _supervisor(tmp_path, monkeypatch, num_rollout=3, segment_secs=1.6, sentinel=sentinel)
    sup.call_id = "fc-1"  # steps end at 0.5, 1.0, 1.5 s; the save is asked for at 1.2 s, so it lands on the last
    beats = []
    real = sup.heartbeat
    monkeypatch.setattr(sup, "heartbeat", lambda: beats.append(time.time()) or real())
    script = tmp_path / "slow_tail.py"
    script.write_text(FAKE_MILES.replace('print("done", flush=True)', 'time.sleep(1.5); print("done", flush=True)'))
    cmd = _fake(tmp_path, sentinel, 0, 3, step_secs=0.5)
    cmd[1] = str(script)
    record = sup.run(cmd, cwd=str(tmp_path), env={})
    assert record["reason"] == "success" and record["call_id"] == "fc-1"
    assert len(commits) <= 4, f"{len(commits)} commits"
    assert len(beats) >= 5  # still reporting through the 1.5 s tail


def test_finish_releases_only_its_own_claim(tmp_path, monkeypatch):
    sentinel = tmp_path / "save-now"
    sup, store, _ = _supervisor(tmp_path, monkeypatch, num_rollout=3, segment_secs=None, sentinel=sentinel)
    sup.call_id = "fc-1"
    store["active:run1"] = {"call_id": "fc-2"}
    sup.finish("stopped", -15)
    assert store["active:run1"] == {"call_id": "fc-2"}
    store["active:run1"] = {"call_id": "fc-1"}
    sup.finish("stopped", -15)
    assert "active:run1" not in store


def test_secret_is_replaced(monkeypatch):
    SEC = importlib.import_module("miles_ext.platform.secrets")
    store = {"mdev-secret-run1": {"WANDB_API_KEY": "old", "SAIL_API_KEY": "old"}}

    class Objects:
        @staticmethod
        def delete(name, allow_missing=False, environment_name=None):
            store.pop(name, None)

        @staticmethod
        def create(name, values, allow_existing=False, environment_name=None):
            if name in store and not allow_existing:
                raise RuntimeError("exists")
            store.setdefault(name, values)

    monkeypatch.setattr(SEC.modal.Secret, "objects", Objects)
    SEC.create_run_secret(_recipe(), "run1", DEV, {"WANDB_API_KEY": "new", "SAIL_API_KEY": "new"})
    assert store["mdev-secret-run1"]["WANDB_API_KEY"] == "new"


def test_rollouts_done_per_backend():
    assert P.rollouts_done("fsdp", 12) == 12 and P.rollouts_done("megatron", 3) == 4


def test_schema_returns_copies():
    S = importlib.import_module("miles_ext.platform.schema")
    s = S.schema()
    s["defaults"]["cluster"]["gpu"] = "B200"
    assert R.DEFAULTS["cluster"]["gpu"] != "B200"


@pytest.mark.parametrize("run_id,per_run", [("fork", False), ("k", False), ("run1", True)])
def test_gpu_app_per_run_is_exact(run_id, per_run):
    from miles_ext.platform.target import gpu_app_is_per_run
    assert gpu_app_is_per_run(run_id, Target.of({**DEV, "app_name": "miles-fork"} if not per_run else
                                                {**DEV, "app_name": f"mdev-{run_id}"})) is per_run


def test_trainer_clears_the_completion_queue_before_the_head():
    src = (importlib.import_module("miles_ext.tests.test_platform").ROOT / "miles_ext/launch/modal_app.py").read_text()
    assert src.index("completion_queue.clear(partition=run_id)") < src.index('start + ["--head"')
