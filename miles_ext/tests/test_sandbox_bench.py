"""ext-040's benchmark pieces: the scripted model speaks mini-swe-agent's tool calls, traces and the report
read what Harbor and mini-swe-agent write, and the synthetic workloads run:

    python -m pytest miles_ext/tests/test_sandbox_bench.py
"""

import asyncio
import functools
import json
import subprocess
from datetime import datetime, timedelta, timezone

import pytest
from aiohttp.test_utils import TestServer

from miles_ext.obs import sandbox_profile as sp
from miles_ext.obs.sandbox_bench import report, traces
from miles_ext.obs.sandbox_bench import workloads as W
from miles_ext.obs.sandbox_bench.model_server import SUBMIT, Script, make_app


@pytest.mark.parametrize(
    "command, cls",
    [
        ("python3 -m pytest -x tests/ | tail", "test"),
        ("cd /app && ls -la", "read"),
        ("cat foo.py | grep x", "read"),
        ("nl -ba a.py | sed -n 1,9p", "read"),
        ("echo hi > a.txt", "write"),
        ("sed -i 's/a/b/' x.py", "write"),
        ("python3 run.py", "run"),
        ("grep -rn foo /", "fs-scan"),
        ("grep -rn foo /app", "read"),
        ("find / -name x 2>/dev/null", "fs-scan"),
        ("apt-get update && apt-get install -y curl", "install"),
        ("make -j8", "build"),
        ("true", "noop"),
        (SUBMIT, "submit"),
    ],
)
def test_command_classes(command, cls):
    assert W.classify(command) == cls


def test_synthetic_workloads_are_tagged_by_their_own_class():
    for workload in ("noop", "read", "test", "fs-scan", "install"):
        classes = {W.classify(c) for c in W.script(workload)}
        assert classes == {"setup", workload}, workload
    assert W.classify(W.script("write")[0]) == "setup"


def test_setup_builds_a_repo_whose_tests_pass(tmp_path):
    repo = str(tmp_path / "repo")
    subprocess.run(W.SETUP.replace(W.REPO, repo), shell=True, check=True, executable="/bin/bash")
    out = subprocess.run(
        W._test(0).replace(W.REPO, repo), shell=True, capture_output=True, text=True, executable="/bin/bash"
    )
    assert "OK" in out.stdout


def test_task_dir_is_a_harbor_task(tmp_path):
    from harbor.models.task.task import Task

    name = W.write_task(tmp_path, "read", cpus=2, memory_mb=4096)
    env = Task(tmp_path / name).config.environment
    assert env.docker_image == W.IMAGE and env.cpus == 2 and env.network_mode.value == "public"


def test_mini_swe_agent_reads_the_scripted_replies():
    """The real mini-swe-agent model class against the scripted server: each script command, then the submit."""
    from minisweagent.models.litellm_model import LitellmModel

    scripts, log = {"s1": Script(["ls -la", "cat x"])}, []

    async def go():
        async with TestServer(make_app(scripts, log)) as server:
            model = LitellmModel(
                model_name="openai/model",
                cost_tracking="ignore_errors",
                model_kwargs={
                    "api_base": str(server.make_url("/sessions/s1/v1")),
                    "api_key": "dummy",
                    "drop_params": True,
                },
            )
            query = functools.partial(model.query, [{"role": "user", "content": "go"}])
            return [(await asyncio.to_thread(query))["extra"]["actions"][0]["command"] for _ in range(3)]

    assert asyncio.run(go()) == ["ls -la", "cat x", SUBMIT]
    assert [line["step"] for line in log] == [1, 2, 3]


def _trial(d, task="t1", task_path=None, steps=((0.5, "ls", 0, 0.2), (0.4, "grep -rn x /", 1, 3.0)), wall=10.0):
    """A trial dir as Harbor and mini-swe-agent write it: (model wait s, command, rc, command s) per step."""
    (d / "agent").mkdir(parents=True)
    t0 = datetime(2026, 10, 27, tzinfo=timezone.utc)

    def at(s):
        return (t0 + timedelta(seconds=s)).isoformat()

    (d / "result.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "task_id": {"path": task_path or f"/inputs/tasks/{task.rsplit('/', 1)[-1]}"},
                "started_at": at(0),
                "finished_at": at(wall),
                "environment_setup": {"started_at": at(0), "finished_at": at(2)},
                "agent_setup": {"started_at": at(2), "finished_at": at(3)},
                "agent_execution": {"started_at": at(3), "finished_at": at(8)},
                "verifier": {"started_at": at(8), "finished_at": at(9)},
            }
        )
    )
    messages = [{"role": "user", "content": "task"}]
    for wait, cmd, rc, secs in list(steps) + [(0.1, SUBMIT, 0, 0.0)]:
        messages.append(
            {"role": "assistant", "extra": {"harbor_llm_wait_msec": wait * 1000, "actions": [{"command": cmd}]}}
        )
        messages.append({"role": "tool", "extra": {"harbor_tool_exec_msec": secs * 1000, "returncode": rc}})
    (d / traces.TRAJECTORY).write_text(json.dumps({"messages": messages}))
    sp.note_session(d, "http://127.0.0.1:30000/sessions/" + d.name)
    return d


def test_trace_keeps_commands_and_drops_the_submit(tmp_path):
    t = traces.trial_trace(_trial(tmp_path / "a"))
    assert t["instance_id"] == "t1" and t["commands"] == ["ls", "grep -rn x /"]
    assert t["think_s"] == [0.5, 0.4] and t["cmd_s"] == [0.2, 3.0] and t["rc"] == [0, 1]


def test_sample_spreads_over_tasks(tmp_path):
    ts = [{"trial": f"{task}-{i}", "instance_id": task} for task in "abc" for i in range(3)]
    picked = traces.sample(ts, 4)
    assert len({t["instance_id"] for t in picked[:3]}) == 3 and len(picked) == 4
    assert picked == traces.sample(ts, 4)  # seeded


def test_report_splits_the_wall_time_exactly(tmp_path):
    d = _trial(tmp_path / "trial-1")
    _trial(tmp_path / "trial-10")  # a session whose id extends trial-1's must not join it
    proxy = {f"/30000/sessions/{name}/v1/chat/completions": [0.1, 0.1, 0.05] for name in ("trial-1", "trial-10")}
    row = report.trial_row(d, proxy)
    pieces = sum(row[p] for p in report.PIECES)
    assert row["wall"] == 10.0 and abs(pieces - row["wall"]) < 1e-9
    assert row["calls"] == 3 and abs(row["proxy"] - 0.25) < 1e-9 and abs(row["tunnel"] - 0.75) < 1e-9
    assert [c for c, _, _ in row["commands"]] == ["read", "fs-scan", "submit"]
    s = report.summarize([row])
    assert abs(sum(s["share"].values()) - 1.0) < 1e-9 and s["tool_share_by_class"]["fs-scan"] == 0.3


def test_a_retried_call_keeps_its_wait_whole(tmp_path):
    d = _trial(tmp_path / "t")
    row = report.trial_row(d, {"/30000/sessions/t/v1/chat/completions": [0.1] * 4})
    assert not row["matched"] and row["tunnel"] == row["proxy"] == 0.0 and abs(row["model_unsplit"] - 1.0) < 1e-9
    assert abs(sum(row[p] for p in report.PIECES) - row["wall"]) < 1e-9
    s = report.summarize([row], no_result=2)
    assert abs(sum(s["share"].values()) - 1.0) < 1e-9
    assert (s["trials"], s["failed"], s["no_result"], s["unmatched_calls"]) == (3, 2, 2, 1)


def test_probes_are_not_sandbox_start(tmp_path):
    d = _trial(tmp_path / "p")
    sp.write_lines(d / sp.TRIAL_FILE, [{"op": "probe_exec", "dt": 0.25, "depth": 0}] * 2)
    row = report.trial_row(d, {})
    assert row["probes"] == 0.5 and abs(row["sandbox_start"] - 1.5) < 1e-9


def test_fidelity_counts_a_replay_cut_short(tmp_path):
    d = _trial(tmp_path / "f")
    row = report.trial_row(d, {})
    trace = {"commands": ["ls", "grep -rn x /", "make"], "rc": [0, 1, 0], "cmd_s": [0.1, 3.0, 1.0]}
    out = report.fidelity([row], [{"trial_dir": str(d), "trace": "orig"}], {"orig": trace})
    assert out["rc_match"] == 1.0 and out["compared"] == 2 and out["replays_cut_short"] == 1


def test_plan_runs_each_provider_in_its_own_sequence():
    from miles_ext.obs.sandbox_bench.app import plan_arms

    arms = plan_arms("b", ["sail", "modal"], [1, 32], 2, ["noop", "trace"], 32, 8, "src", "zero", 2, 4096, 5)
    assert [a["name"] for a in arms[:4]] == ["sail-c1-r0", "sail-c32-r0", "sail-c32-r1", "sail-c1-r1"]
    assert {a["trials"] for a in arms if a["concurrency"] == 1} == {8}
    assert {a["trace_trials"] for a in arms} == {5}  # at every concurrency
    assert {
        a["trace_trials"] for a in plan_arms("b", ["sail"], [1, 32], 1, ["trace"], 4, 2, "s", "zero", 2, 4096)
    } == {
        None  # every trace
    }


def test_trace_trials_are_set_up_from_the_tasks_volume_layout(tmp_path, monkeypatch):
    """The tb4t layout: a task's name is not always its dir; older extracts recorded the name, newer the dir."""
    from harbor.models.task.task import Task

    from miles_ext.harbor.agent_function import hf
    from miles_ext.obs.sandbox_bench import app

    src = tmp_path / "datasets" / "tb4t" / "tasks"
    names = ("software-009", "wal-1", "bp105-security-email-auth-policy-r01", "bp105-security-email-auth-policy-r02")
    for name in names:
        (src / name / "environment").mkdir(parents=True)
        (src / name / "tests").mkdir()
        (src / name / "task.toml").write_text('version = "1.0"\n')
        (src / name / "instruction.md").write_text("do it\n")
        (src / name / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        (src / name / "tests" / "test.sh").write_text("exit 0\n")
    (src / "wal-1" / "task.toml").write_text(
        'version = "1.0"\n[environment]\nnetwork_mode = "no-network"\n[agent]\nnetwork_mode = "allowlist"\n'
        'allowed_hosts = ["example.com"]\n'
    )
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    old_style = [  # (trial name, the task_name the extract kept)
        ("software-009__20261020-1__a", "mobius-labs/software-009"),
        ("wal-1__20261020-1__b", "wal-1"),
        ("bp105-security-email-auth-policy__20261020-1__c", "security-email-auth-policy-r01"),
        ("bp105-security-email-auth-policy__20261020-1__d", "bp105-security-email-auth-policy-r02"),  # a new extract
    ]
    lines = [
        {"trial": trial, "instance_id": iid, "commands": ["ls"], "think_s": [1.0], "cmd_s": [0.1], "rc": [0]}
        for trial, iid in old_style
    ]
    (traces_dir / "src.jsonl").write_text("".join(json.dumps(t) + "\n" for t in lines))
    monkeypatch.setattr(app, "TRACES_DIR", traces_dir)
    assert [r["task_dir"] for r in app.check_traces.local("src", str(src))] == [
        "software-009",
        "wal-1",
        "bp105-security-email-auth-policy-r01",
        "bp105-security-email-auth-policy-r02",
    ]
    (arm,) = app.plan_arms("b", ["modal"], [32], 1, ["noop", "trace"], 2, 1, "src", "zero", 2, 4096)
    arm["tasks_src"] = str(src)
    root = tmp_path / "root"
    root.mkdir()
    specs = app._trial_specs(arm, root)
    assert len({s["key"] for s in specs}) == len(specs) == 6
    for s in specs:  # what the agent function resolves for each trial
        env = Task(hf._task_path(root, s["instance_id"])).config.environment
        assert (env.cpus, env.memory_mb) == (2, 4096)  # sizeless tasks take the arm's size
        task = Task(hf._task_path(root, s["instance_id"])).config
        assert env.network_mode.value == "public" and task.agent.explicit_phase_policy() is None
        assert task.verifier.explicit_phase_policy() is None  # no policy switch during the trial


def test_a_trace_without_its_task_fails_before_any_trial(tmp_path, monkeypatch):
    from miles_ext.obs.sandbox_bench import app

    src = tmp_path / "tasks"
    for name in ("x-policy-r01", "y-policy-r01"):
        (src / name).mkdir(parents=True)
        (src / name / "task.toml").write_text('version = "1.0"\n')
    (tmp_path / "src.jsonl").write_text(
        json.dumps({"trial": "x-policy-r01__1__a", "instance_id": "policy-r01"})
        + "\n"
        + json.dumps({"trial": "gone__1__b", "instance_id": "gone"})
        + "\n"
    )
    monkeypatch.setattr(app, "TRACES_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="1 of 2") as e:
        app._trace_tasks("src", str(src))
    assert "gone__1__b" in str(e.value) and "x-policy-r01__1__a" not in str(e.value)


def test_extract_names_the_task_dir(tmp_path):
    d = _trial(
        tmp_path / "x",
        task="mobius-labs/security-email-auth-policy-r01",
        task_path="/datasets/tb4t-20261016/tasks/bp105-security-email-auth-policy-r01",
    )
    assert traces.trial_trace(d)["instance_id"] == "bp105-security-email-auth-policy-r01"


def _rep_traces():
    return [
        {
            "trial": "a__1__x",
            "instance_id": "a",
            "commands": ["cat a.py", "cat b.py", "cat a.py", "python -m pytest -q", "sed -i 's/x/y/' a.py"],
            "cmd_s": [0.1, 0.3, 0.2, 300.0, 0.5],
            "rc": [0, 0, 0, 124, 0],
        },
        {
            "trial": "b__1__y",
            "instance_id": "b",
            "commands": ["python -m pytest -q tests/test_x.py", "grep -rn foo .", "cat c.py"],
            "cmd_s": [12.0, 2.0, None],
            "rc": [1, 0, None],
        },
    ]


def test_a_command_that_writes_a_script_and_runs_it_is_a_run():
    assert W.classify("cat > /tmp/x.py <<'EOF'\nprint(1)\nEOF\npython /tmp/x.py") == "run"
    assert W.classify("python repro.py > out.txt 2>&1") == "run"
    assert W.classify("cat > x.py <<'EOF'\nx = 1\nEOF") == "write"
    assert W.classify("cd /app && python -m pytest -q > log.txt") == "test"


def test_command_breakdown_counts_time_by_class_and_bucket():
    b = traces.command_breakdown(_rep_traces())
    assert b["commands"] == 8 and b["untimed"] == 1
    assert b["by_class"]["test"]["timeouts"] == 1 and b["by_class"]["test"]["n"] == 2
    assert [x["s"] for x in b["by_class"]["test"]["slowest"]] == [300.0, 12.0]
    assert b["by_class"]["test"]["slowest"][0] == {"s": 300.0, "rc": 124, "cmd": "python -m pytest -q"}
    assert b["by_class"]["read"]["unique"] == 4  # cat a.py twice; grep in the working tree is a read
    assert abs(sum(c["share"] for c in b["by_class"].values()) - 1) < 1e-9
    assert b["by_bucket"]["timeout"]["n"] == 1 and b["by_bucket"]["<1 s"]["n"] == 4


def test_representative_replays_each_pick_after_its_prefix():
    rep = traces.representative(_rep_traces(), lambda t: "dir-" + t["instance_id"])
    by_trial = {r["trial"]: r for r in rep}
    picked = {r["commands"][p] for r in rep for p in r["measure"]}
    assert "python -m pytest -q" not in picked  # timed out: counted, not replayed
    assert "python -m pytest -q tests/test_x.py" in picked and "sed -i 's/x/y/' a.py" in picked
    a = by_trial["rep-a__1__x"]  # the edit replays after everything before it in its trial
    assert a["instance_id"] == "dir-a" and a["commands"][-1] == "sed -i 's/x/y/' a.py" and len(a["commands"]) == 5
    for r in rep:
        assert len(r["commands"]) == len(r["cmd_s"]) == len(r["rc"]) == len(r["think_s"])
        assert max(r["measure"]) == len(r["commands"]) - 1


def test_representative_prefers_the_shortest_prefix_among_similar_commands():
    t = {
        "trial": "t__1__x",
        "instance_id": "t",
        "commands": ["python slow.py", "python a.py", "python b.py"],
        "cmd_s": [200.0, 5.0, 5.1],
        "rc": [0, 0, 0],
    }
    early = {**t, "trial": "u__1__y", "commands": ["python c.py"], "cmd_s": [5.05], "rc": [0]}
    rep = traces.representative([t, early], lambda x: x["instance_id"], quantiles=(0.5,), max_prefix_s=60)
    assert [(r["trial"], r["commands"]) for r in rep] == [("rep-u__1__y", ["python c.py"])]


def test_fidelity_lists_the_measured_commands(tmp_path):
    d = _trial(tmp_path / "m")
    row = report.trial_row(d, {})
    trace = {"commands": ["ls", "grep -rn x /"], "rc": [0, 0], "cmd_s": [0.1, 2.5], "measure": [1]}
    out = report.fidelity([row], [{"trial_dir": str(d), "trace": "rep"}], {"rep": trace})
    assert out["measured"] == [
        {"class": "fs-scan", "recorded_s": 2.5, "replay_s": 3.0, "rc": 1, "recorded_rc": 0, "cmd": "grep -rn x /"}
    ]


def test_command_time_comes_from_timestamps_when_the_step_timing_field_is_missing(tmp_path):
    """a6's trajectories: assistant messages carry timestamp and harbor_llm_wait_msec; tool messages carry only
    raw_output, returncode, timestamp and exception_info (no harbor_tool_exec_msec)."""
    d = _trial(tmp_path / "a6", task="mobius-labs/software-009", steps=())
    t0 = 1_761_000_000.0
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for i, (wait, cmd, run, rc) in enumerate([(4.0, "ls", 0.9, 0), (30.0, "python -m pytest", 300.1, 124)]):
        t0 += wait
        messages.append(
            {
                "role": "assistant",
                "extra": {"timestamp": t0, "harbor_llm_wait_msec": wait * 1000, "actions": [{"command": cmd}]},
            }
        )
        t0 += run
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"c{i}",
                "extra": {"raw_output": "", "returncode": rc, "timestamp": t0, "exception_info": None},
            }
        )
    (d / traces.TRAJECTORY).write_text(json.dumps({"messages": messages}))
    t = traces.trial_trace(d)
    assert t["cmd_s"] == [0.9, 300.1] and t["think_s"] == [4.0, 30.0] and t["rc"] == [0, 124]
    assert traces.command_breakdown([t])["by_bucket"]["timeout"]["n"] == 1
    row = report.trial_row(d, {})
    assert [round(s, 3) for _, s, _ in row["commands"]] == [0.9, 300.1] and row["model_wait"] == 34.0
