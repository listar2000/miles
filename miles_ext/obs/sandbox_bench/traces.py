"""Replay a past run's commands: extract each trial's command list from its mini-swe-agent trajectory.

The extract keeps only what a replay needs (commands, the recorded model wait and command time per step, exit codes),
one JSON line per trial, and is written to the results volume, never to the repo. A replay runs the commands in
order on the same task image, so a command's exit code matching the original is the fidelity check.
"""

import json
import random
from pathlib import Path
from typing import Any

from miles_ext.obs.sandbox_bench.workloads import classify

TRAJECTORY = Path("agent") / "mini-swe-agent.trajectory.json"


def trial_trace(trial_dir: Path) -> dict[str, Any] | None:
    """One trial's replay record, or None when it has no result or trajectory."""
    try:
        result = json.loads((trial_dir / "result.json").read_text())
        messages = json.loads((trial_dir / TRAJECTORY).read_text()).get("messages", [])
    except (OSError, ValueError):
        return None
    commands, think_s, cmd_s, rc = [], [], [], []
    for m in messages:
        extra = m.get("extra") if isinstance(m.get("extra"), dict) else {}
        if m.get("role") == "assistant":
            for action in extra.get("actions") or []:
                commands.append(action.get("command", ""))
                think_s.append(round((extra.get("harbor_llm_wait_msec") or 0) / 1000, 3))
                cmd_s.append(None)
                rc.append(None)
        elif m.get("role") == "tool" and commands and cmd_s[-1] is None:
            if (ms := extra.get("harbor_tool_exec_msec")) is not None:
                cmd_s[-1] = round(ms / 1000, 3)
            rc[-1] = extra.get("returncode")
    while commands and classify(commands[-1]) == "submit":  # the scripted model adds its own
        for xs in (commands, think_s, cmd_s, rc):
            xs.pop()
    if not commands:
        return None
    return {
        "trial": trial_dir.name,
        "instance_id": result.get("task_name"),
        "commands": commands,
        "think_s": think_s,
        "cmd_s": cmd_s,
        "rc": rc,
    }


def sample(traces: list[dict], n: int, seed: int = 0) -> list[dict]:
    """``n`` traces spread over tasks: one per task in a seeded order, then a second round, and so on."""
    rng = random.Random(seed)
    by_task: dict[str, list[dict]] = {}
    for t in sorted(traces, key=lambda t: t["trial"]):
        by_task.setdefault(t["instance_id"], []).append(t)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    for ts in by_task.values():
        rng.shuffle(ts)
    out, depth = [], 0
    while len(out) < n and any(len(ts) > depth for ts in by_task.values()):
        out += [by_task[k][depth] for k in tasks if len(by_task[k]) > depth][: n - len(out)]
        depth += 1
    return out


def extract(trials_dir: Path, n: int, seed: int = 0) -> list[dict]:
    traces = [t for d in sorted(trials_dir.iterdir()) if d.is_dir() and (t := trial_trace(d))]
    return sample(traces, n, seed)
