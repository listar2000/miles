"""ext-040's benchmark pieces: the scripted model speaks mini-swe-agent's tool calls, traces and the report
read what Harbor and mini-swe-agent write, and the synthetic workloads run:

    python -m pytest miles_ext/tests/test_sandbox_bench.py
"""

import asyncio
import json
import subprocess
import threading
from datetime import datetime, timedelta, timezone

import pytest
from aiohttp import web

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
    assert env.docker_image == W.IMAGE and env.cpus == 2 and env.network_mode.value == "no-network"
    assert (
        Task(tmp_path / W.write_task(tmp_path, "install", 2, 4096)).config.environment.network_mode.value == "public"
    )


def test_mini_swe_agent_reads_the_scripted_replies(tmp_path):
    """The real mini-swe-agent model class against the scripted server: each script command, then the submit."""
    from minisweagent.models.litellm_model import LitellmModel

    scripts = {"s1": Script(["ls -la", "cat x"])}
    loop = asyncio.new_event_loop()
    started = threading.Event()
    box = {}

    async def serve():
        runner = web.AppRunner(make_app(scripts, tmp_path / "model.jsonl"))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        box["port"], box["runner"] = site._server.sockets[0].getsockname()[1], runner
        started.set()

    thread = threading.Thread(target=lambda: (loop.run_until_complete(serve()), loop.run_forever()), daemon=True)
    thread.start()
    started.wait(10)
    try:
        model = LitellmModel(
            model_name="openai/model",
            cost_tracking="ignore_errors",
            model_kwargs={
                "api_base": f"http://127.0.0.1:{box['port']}/sessions/s1/v1",
                "api_key": "dummy",
                "drop_params": True,
            },
        )
        got = [model.query([{"role": "user", "content": "go"}])["extra"]["actions"][0]["command"] for _ in range(3)]
    finally:
        asyncio.run_coroutine_threadsafe(box["runner"].cleanup(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
    assert got == ["ls -la", "cat x", SUBMIT]
    assert [line["step"] for line in map(json.loads, (tmp_path / "model.jsonl").read_text().splitlines())] == [1, 2, 3]


def _trial(d, task="t1", steps=((0.5, "ls", 0, 0.2), (0.4, "grep -rn x /", 1, 3.0)), wall=10.0):
    """A trial dir as Harbor and mini-swe-agent write it: (model wait s, command, rc, command s) per step."""
    (d / "agent").mkdir(parents=True)
    t0 = datetime(2026, 10, 27, tzinfo=timezone.utc)

    def at(s):
        return (t0 + timedelta(seconds=s)).isoformat()

    (d / "result.json").write_text(
        json.dumps(
            {
                "task_name": task,
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


def test_a_retried_call_leaves_tunnel_unknown(tmp_path):
    d = _trial(tmp_path / "t")
    row = report.trial_row(d, {"/30000/sessions/t/v1/chat/completions": [0.1] * 4})
    assert row["tunnel"] is None and row["proxy"] is None


def test_plan_runs_each_provider_in_its_own_sequence():
    from miles_ext.obs.sandbox_bench.app import plan_arms

    arms = plan_arms("b", ["sail", "modal"], [1, 32], 2, ["noop", "trace"], 32, 8, "src", "zero", 2, 4096)
    assert [a["name"] for a in arms[:4]] == ["sail-c1-r0", "sail-c32-r0", "sail-c32-r1", "sail-c1-r1"]
    assert {a["trials"] for a in arms if a["concurrency"] == 1} == {8}
    assert all(a["trace_trials"] is None for a in arms if a["concurrency"] == 32)
