"""Harbor's per-trial task hashes keep their values and are computed once per unchanged task, off the loop."""

import asyncio
import threading
from types import SimpleNamespace

from harbor.models.task.task import Task
from harbor.publisher.packager import Packager

from miles_ext.harbor import agent_function as af
from miles_ext.harbor import task_hash


def _task(tmp_path, name="t"):
    d = tmp_path / name
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "task.toml").write_text('version = "1.0"\n')
    (d / "instruction.md").write_text("do it\n")
    (d / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (d / "tests" / "test.sh").write_text("exit 0\n")
    return d


def test_values_match_harbor_and_compute_once(tmp_path, monkeypatch):
    d = _task(tmp_path)
    expected = (task_hash._upstream_content_hash(d)[0], task_hash._upstream_checksum(Task(d)))
    calls = []
    upstream = task_hash._upstream_content_hash
    monkeypatch.setattr(task_hash, "_upstream_content_hash", lambda p: calls.append(p) or upstream(p))
    task_hash.warm(d)
    assert (Packager.compute_content_hash(d)[0], Task(d).checksum) == expected
    task_hash.warm(d)
    assert len(calls) == 1


def test_changed_task_is_hashed_again(tmp_path):
    d = _task(tmp_path)
    task_hash.warm(d)
    before = Packager.compute_content_hash(d)[0]
    (d / "instruction.md").write_text("do it differently\n")
    task_hash.warm(d)
    assert Packager.compute_content_hash(d)[0] == task_hash._upstream_content_hash(d)[0] != before


def test_run_hashes_in_a_thread_before_the_trial(tmp_path, monkeypatch):
    d = _task(tmp_path)
    threads, seen = [], []
    for name in ("_upstream_content_hash", "_upstream_checksum"):
        upstream = getattr(task_hash, name)
        monkeypatch.setattr(task_hash, name, lambda x, f=upstream: threads.append(threading.current_thread()) or f(x))

    class _Trial:
        @classmethod
        async def create(cls, config):  # Harbor's two lookups, on the loop
            seen.append((Packager.compute_content_hash(config.task.path)[0], Task(config.task.path).checksum))
            raise RuntimeError("stop after create")

    monkeypatch.setattr("harbor.trial.trial.Trial", _Trial)
    monkeypatch.setattr(af, "build_trial_config", lambda *a: SimpleNamespace(task=SimpleNamespace(path=d)))
    monkeypatch.setattr(af.hf, "_ensure_provider_key", lambda: None)
    asyncio.run(af.run_trial("http://h:1/sessions/s", "p", {}, {"instance_id": "t"}))
    assert seen and len(threads) == 2 and threading.main_thread() not in threads


def test_prewarm_hashes_each_dir_once_in_order(tmp_path, monkeypatch):
    a, b = _task(tmp_path, "a"), _task(tmp_path, "b")
    calls = []
    upstream = task_hash._upstream_content_hash
    monkeypatch.setattr(task_hash, "_upstream_content_hash", lambda p: calls.append(p.name) or upstream(p))
    task_hash.prewarm([a, b, a, tmp_path / "missing"], workers=1)  # a failing dir is logged, not raised
    assert calls[:2] == ["a", "b"] and calls.count("a") == 1


def test_prepare_prewarms_the_prompt_set_in_its_own_thread(tmp_path, monkeypatch):
    _task(tmp_path, "a"), _task(tmp_path, "b")
    monkeypatch.setenv("HARBOR_TASKS_DIR", str(tmp_path))
    monkeypatch.delenv("HARBOR_ENV_IMPORT_PATH", raising=False)
    got = []
    monkeypatch.setattr(af.trial_procs, "start", lambda modules: got.append(("start", modules[1])))
    monkeypatch.setattr(
        af.task_hash, "prewarm", lambda dirs: got.append(([d.name for d in dirs], threading.current_thread().name))
    )
    samples = [SimpleNamespace(metadata={"instance_id": i}) for i in ("b", "a", "b", "nope")]
    af.prepare(SimpleNamespace(), SimpleNamespace(dataset=SimpleNamespace(samples=samples)))
    for t in threading.enumerate():
        if t.name == "harbor-prepare":
            t.join(30)
    # the trial forkserver first, then the hashes in hand-out order; the missing task is skipped
    assert got == [("start", af.__name__), (["b", "a", "b"], "harbor-prepare")]
