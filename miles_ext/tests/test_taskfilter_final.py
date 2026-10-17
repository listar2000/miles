"""Sweep runs merge into verdicts and keep lists:

    python -m pytest miles_ext/tests/test_taskfilter_final.py
"""

import csv
import json

from miles_ext.taskfilter import final


def _run(tmp_path, name, rows, verifier_cap=1800):
    run = tmp_path / f"sweep-{name}"
    run.mkdir()
    (run / "launch-1.json").write_text(json.dumps({"recipe": {"agent": {"verifier_timeout": verifier_cap}}}))
    (run / "summary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def _ok(task_id, verifier_s):
    return {"task_id": task_id, "oracle_reward": 1.0, "install_ok": True, "smoke_ok": True,
            "verifier_s": verifier_s, "verifier_timed_out": False}


def test_latest_run_wins_and_keep_lists_follow_cutoffs(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(json.dumps({"id": i, "version": "v1"}) + "\n" for i in ("a", "b", "c", "d")))
    full = _run(tmp_path, "full1", [
        _ok("a", 100.0), _ok("b", 1000.0),
        {"task_id": "c", "oracle_exception": "EnvironmentStartTimeoutError: 600 s"},
        {"task_id": None, "oracle_exception": "function: lost"},  # a row the map could not attribute
    ])
    retry = _run(tmp_path, "retry1", [_ok("c", 700.0)], verifier_cap=1800)
    out = tmp_path / "final"

    summary = final.main(["--taskset", "ts", "--tasks", str(tasks), "--out", str(out), str(full), str(retry)])

    rows = {r["task_id"]: r for r in csv.DictReader((out / "verdicts.csv").open())}
    assert rows["c"]["verdict"] == "keep" and rows["c"]["source_run"] == "retry1"
    assert rows["d"]["verdict"] == "drop" and rows["d"]["reason"] == "not run"
    assert (out / "keep_600.txt").read_text() == "a\n"
    assert (out / "keep_900.txt").read_text() == "a\nc\n"
    assert (out / "keep_1800.txt").read_text() == "a\nb\nc\n"
    assert summary["taskset"] == "ts" and summary["pass_all_filters"] == 3
    assert summary["drop_reasons"] == {"not run": 1}
    assert summary["kept_verifier_s_p50_p90_p99"] == [700.0, 1000.0, 1000.0]


def test_verifier_cap_comes_from_the_run(tmp_path):
    run = _run(tmp_path, "full1", [{**_ok("a", None), "verifier_timed_out": True}], verifier_cap=900)
    res = final.verdicts([{"id": "a", "version": "v1"}], final.load_rows([run]))
    assert res[0]["reason"] == "verifier timeout (>=900 s)"
