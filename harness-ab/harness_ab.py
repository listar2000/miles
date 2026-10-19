"""Host-side vs in-sandbox harness A/B, CPU steps 1-3 (plan: thread "Explore · host-side vs in-sandbox harness").

Run inside the trainer image (harbor, miles_ext, mini-swe-agent present) from a miles checkout on PR #51's branch.

  timings DIR...                 step 1: install, agent and per-step times from saved Harbor trial dirs
  replay-server --traj DIR ...   step 3: OpenAI-compatible model that replays recorded mini-swe-agent actions
  trials --harness H ...         steps 2-3: run Harbor trials via miles_ext's agent function against a model URL,
                                 record per-try timing plus this process's CPU, RSS and event-loop lag
  ProbeAgent                     step 2: Harbor agent (import path ``harness_ab:ProbeAgent``) timing sandbox exec
                                 round trips and, with PROBE_ECHO_URL, in-sandbox POSTs through the tunnel

Outputs are JSON lines; summaries print p50/p90/p99.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import resource
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3) if xs else None


def summary(xs):
    return {"n": len(xs), "p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "p99": pct(xs, 0.99),
            "mean": round(statistics.fmean(xs), 3) if xs else None}


def _secs(timing):
    if not timing or not timing.get("started_at") or not timing.get("finished_at"):
        return None
    return (datetime.fromisoformat(timing["finished_at"]) - datetime.fromisoformat(timing["started_at"])).total_seconds()


# ---------------------------------------------------------------- step 1
def timings(dirs):
    setup, agent, verifier, total, llm, cmd, calls = [], [], [], [], [], [], []
    failures = {}
    for d in dirs:
        for path in Path(d).rglob("result.json"):
            try:
                r = json.loads(path.read_text())
            except ValueError:
                continue
            if "agent_setup" not in r:
                continue  # not a Harbor trial result
            for xs, key in ((setup, "agent_setup"), (agent, "agent_execution"), (verifier, "verifier")):
                if (s := _secs(r.get(key))) is not None:
                    xs.append(s)
            if (s := _secs(r)) is not None:
                total.append(s)
            if exc := r.get("exception_info"):
                failures[exc["exception_type"]] = failures.get(exc["exception_type"], 0) + 1
            meta = (r.get("agent_result") or {}).get("metadata") or {}
            llm += [x / 1000 for x in meta.get("api_request_times_msec", [])]
            cmd += [x / 1000 for x in meta.get("command_exec_times_msec", [])]
            if meta.get("api_request_times_msec"):
                calls.append(len(meta["api_request_times_msec"]))
    out = {"trials": len(total), "agent_setup_s": summary(setup), "agent_execution_s": summary(agent),
           "verifier_s": summary(verifier), "trial_s": summary(total), "llm_wait_per_call_s": summary(llm),
           "command_per_step_s": summary(cmd), "calls_per_trial": summary(calls), "failures": failures}
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- step 3 model
def _task_key(messages):
    first_user = next((m for m in messages if m.get("role") == "user"), {})
    content = first_user.get("content")
    text = content if isinstance(content, str) else json.dumps(content)
    # the task text only: the rest of the template differs by harness (uname fields)
    if m := re.search(r"Please solve this issue: (.*?)\n\nYou can execute", text, re.S):
        text = m.group(1)
    return hashlib.sha1(text.encode()).hexdigest()


def load_trajectories(root):
    """task key -> list of tries; each try is [(assistant message, llm wait s)]."""
    by_task = {}
    for path in Path(root).rglob("mini-swe-agent.trajectory.json"):
        try:
            msgs = json.loads(path.read_text()).get("messages") or []
        except ValueError:
            continue
        turns = [({k: m[k] for k in ("content", "tool_calls") if m.get(k) is not None},
                  (m.get("extra") or {}).get("harbor_llm_wait_msec", 0) / 1000)
                 for m in msgs if m.get("role") == "assistant"]
        if turns:
            by_task.setdefault(_task_key(msgs), []).append(turns)
    return by_task


def replay_app(by_task, wait_scale, default_wait):
    from aiohttp import web

    assigned, counter = {}, {}
    submit = {"content": "Done.", "tool_calls": [{"id": "submit", "type": "function", "function": {
        "name": "bash", "arguments": json.dumps({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"})}}]}

    async def chat(request):
        session = request.match_info.get("session", "")
        body = await request.json()
        msgs = body["messages"]
        key = _task_key(msgs)
        tries = by_task.get(key)
        if session not in assigned:
            i = counter.get(key, 0)
            counter[key] = i + 1
            assigned[session] = tries[i % len(tries)] if tries else []
        turns = assigned[session]
        n = sum(1 for m in msgs if m.get("role") == "assistant")
        msg, wait = turns[n] if n < len(turns) else (submit, 0.0)
        await asyncio.sleep(wait * wait_scale if tries else default_wait)
        msg = {"role": "assistant", "content": msg.get("content"), **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {})}
        return web.json_response({"id": f"r-{session}-{n}", "object": "chat.completion", "created": int(time.time()),
                                  "model": body.get("model", "model"),
                                  "choices": [{"index": 0, "message": msg,
                                               "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
                                  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    app = web.Application(client_max_size=512 * 1024**2)
    app.router.add_post("/sessions/{session}/v1/chat/completions", chat)
    return app


async def serve_replay(a):
    from aiohttp import web

    by_task = load_trajectories(a.traj)
    print(f"replay: {sum(map(len, by_task.values()))} tries over {len(by_task)} tasks", flush=True)
    runner = web.AppRunner(replay_app(by_task, a.wait_scale, a.default_wait), access_log=None)
    await runner.setup()
    first, last = (int(p) for p in a.ports.split("-"))
    for port in range(first, last + 1):
        await web.TCPSite(runner, "0.0.0.0", port).start()
    await asyncio.Event().wait()


# ---------------------------------------------------------------- steps 2-3 driver
async def run_trials(a):
    sys.path.insert(0, str(Path(__file__).parent))
    os.environ["MILES_EXT_HARNESS"] = a.harness
    os.environ["HARBOR_TRIALS_DIR"] = os.environ.get("HARBOR_TRIALS_DIR", "/tmp/harbor_trials") + f"/{a.label}"
    from miles_ext.harbor import agent_function as af

    if a.agent_import_path:  # ProbeAgent: bypass the harness binding, keep everything else
        build = af.build_trial_config

        def build_probe(*args):
            cfg = build(*args)
            cfg.agent.name, cfg.agent.import_path = None, a.agent_import_path
            return cfg

        af.build_trial_config = build_probe

    tasks = [t.strip() for t in Path(a.tasks).read_text().split() if t.strip()]
    jobs = [(tasks[i % len(tasks)], i) for i in range(a.tries)]
    first, last = (int(p) for p in a.ports.split("-"))
    sem = asyncio.Semaphore(a.concurrency)
    lags, stop = [], False

    async def ticker():
        while not stop:
            t = time.perf_counter()
            await asyncio.sleep(0.1)
            lags.append(time.perf_counter() - t - 0.1)

    async def one(task, i):
        async with sem:
            base = f"http://{a.model_host}:{first + i % (last - first + 1)}/sessions/{a.label}-{i}"
            t0 = time.time()
            out = await af.run(base, "", {"temperature": 1.0}, {"instance_id": task})
            rec = {"task": task, "try": i, "wall_s": round(time.time() - t0, 2), "exit_status": out.get("exit_status"),
                   "reward": out.get("reward"), "trial_dir": out.get("trial_dir")}
            print(json.dumps(rec), flush=True)
            return rec

    tick = asyncio.create_task(ticker())
    r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.time()
    recs = await asyncio.gather(*(one(t, i) for t, i in jobs))
    wall, r1 = time.time() - t0, resource.getrusage(resource.RUSAGE_SELF)
    stop = True
    await tick
    cpu = r1.ru_utime - r0.ru_utime + r1.ru_stime - r0.ru_stime
    print(json.dumps({"harness": a.harness, "label": a.label, "tries": len(recs), "concurrency": a.concurrency,
                      "wall_s": round(wall, 1), "try_wall_s": summary([r["wall_s"] for r in recs]),
                      "exit_status": {s: sum(r["exit_status"] == s for r in recs) for s in {r["exit_status"] for r in recs}},
                      "cpu_s": round(cpu, 1), "cores_busy": round(cpu / wall, 2), "peak_rss_mib": r1.ru_maxrss // 1024,
                      "loop_lag_s": summary(lags)}), flush=True)
    timings([os.environ["HARBOR_TRIALS_DIR"]])


# ---------------------------------------------------------------- step 2 agent
try:
    from harbor.agents.base import BaseAgent
except ImportError:  # timings / replay-server need no Harbor
    BaseAgent = object


class ProbeAgent(BaseAgent):
    """Times sandbox exec round trips (output sizes from PROBE_SIZES) and in-sandbox POSTs to PROBE_ECHO_URL."""

    SUPPORTS_ATIF = False

    def __init__(self, *args, **kwargs):
        for k in ("api_base", "api_key", "llm_call_kwargs", "step_limit", "shell", "max_seq_len", "model_info"):
            kwargs.pop(k, None)
        super().__init__(*args, **kwargs)

    @staticmethod
    def name():
        return "harness-probe"

    def version(self):
        return "1"

    async def setup(self, environment):
        return None

    async def run(self, instruction, environment, context):
        reps = int(os.getenv("PROBE_REPS", "20"))
        out = {}
        for size in [int(s) for s in os.getenv("PROBE_SIZES", "0,4096,65536").split(",")]:
            cmd = f"head -c {size} /dev/zero | tr '\\0' x" if size else "true"
            xs = []
            for _ in range(reps):
                t = time.perf_counter()
                await environment.exec(cmd, timeout_sec=60)
                xs.append(time.perf_counter() - t)
            out[f"exec_{size}B_s"] = xs
        if echo := os.getenv("PROBE_ECHO_URL"):
            for kb in [int(s) for s in os.getenv("PROBE_BODY_KB", "50,300,800").split(",")]:
                r = await environment.exec(
                    f"head -c {kb * 1024} /dev/zero | tr '\\0' x > /tmp/body && for i in $(seq {reps}); do "
                    f"curl -s -o /dev/null -w '%{{time_total}}\\n' -X POST --data-binary @/tmp/body {echo}; done",
                    timeout_sec=600)
                out[f"post_{kb}KB_s"] = [float(x) for x in (r.stdout or "").split()]
        context.metadata = {**(context.metadata or {}), "probe": out}
        print(json.dumps({k: summary(v) for k, v in out.items()}), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("timings")
    t.add_argument("dirs", nargs="+")
    s = sub.add_parser("replay-server")
    s.add_argument("--traj", required=True, help="dir with a2's trial dirs (mini-swe-agent.trajectory.json)")
    s.add_argument("--ports", required=True, help="first-last, same range the session proxy forwards")
    s.add_argument("--wait-scale", type=float, default=1.0)
    s.add_argument("--default-wait", type=float, default=12.0, help="model wait for tasks with no recording")
    r = sub.add_parser("trials")
    r.add_argument("--harness", required=True, choices=["mini-swe-agent-sandbox", "mini-swe-agent-host"])
    r.add_argument("--tasks", required=True, help="file of instance ids, one per line")
    r.add_argument("--tries", type=int, default=128)
    r.add_argument("--concurrency", type=int, default=128)
    r.add_argument("--model-host", required=True, help="IP the session URLs name (sandbox arm: what the proxy forwards to)")
    r.add_argument("--ports", required=True)
    r.add_argument("--label", default="ab")
    r.add_argument("--agent-import-path", default="", help="e.g. harness_ab:ProbeAgent for step 2")
    a = p.parse_args()
    if a.cmd == "timings":
        timings(a.dirs)
    elif a.cmd == "replay-server":
        asyncio.run(serve_replay(a))
    else:
        asyncio.run(run_trials(a))


if __name__ == "__main__":
    main()
