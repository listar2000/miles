"""Replay a past run's commands: extract each trial's command list from its mini-swe-agent trajectory.

The extract keeps only what a replay needs (commands, the recorded model wait and command time per step, exit codes),
one JSON line per trial, and is written to the results volume, never to the repo. A replay runs the commands in
order on the same task image, so a command's exit code matching the original is the fidelity check.
"""

import json
import random
from pathlib import Path, PurePosixPath
from typing import Any

from miles_ext.obs.sandbox_bench.workloads import classify

TRAJECTORY = Path("agent") / "mini-swe-agent.trajectory.json"


def task_dir(result: dict) -> str:
    """The trial's task dir name (the agent function's ``instance_id``), from the task path Harbor ran.

    Harbor's ``task_name`` is not the dir: it may carry an org prefix (``mobius-labs/...``) and the task's own
    name may differ from its dir (``security-email-auth-policy-r01`` in ``bp105-security-email-auth-policy-r01``)."""
    path = ((result.get("task_id") or {}).get("path")) or ""
    return PurePosixPath(path).name or (result.get("task_name") or "").rsplit("/", 1)[-1]


def resolve_task_dir(trace: dict, dirs: set[str]) -> str:
    """The dir under the tasks root that ``trace`` replays on; ``dirs``: the dir names there.

    Extracts made before 10-28 recorded ``task_name`` without its org prefix rather than the dir; for those the
    dir is the one whose name ends with that name and starts as the trial name does (Harbor names a trial after
    its dir's first 32 characters)."""
    name = trace["instance_id"].rsplit("/", 1)[-1]
    if name in dirs:
        return name
    head = trace["trial"].split("__", 1)[0]
    found = [d for d in dirs if d.endswith("-" + name) and d[:32].rstrip("_-") == head]
    if len(found) != 1:
        raise LookupError(f"trace {trace['trial']}: task {name!r} matches {sorted(found) or 'no'} task dirs")
    return found[0]


def tool_exec_s(extra: dict, since: float | None) -> float | None:
    """A step's command time from its first tool message: Harbor's step-timing field when the patch applied,
    else the tool message's timestamp minus the model reply's (``since``). mini-swe-agent stamps the reply when
    the model call returns and the tool message after the step's commands ran, so the difference is command time
    plus a few milliseconds of bookkeeping, never model wait."""
    if (ms := extra.get("harbor_tool_exec_msec")) is not None:
        return ms / 1000
    if since is not None and (ts := extra.get("timestamp")) is not None:
        return max(0.0, ts - since)
    return None


def trial_trace(trial_dir: Path) -> dict[str, Any] | None:
    """One trial's replay record, or None when it has no result or trajectory."""
    try:
        result = json.loads((trial_dir / "result.json").read_text())
        messages = json.loads((trial_dir / TRAJECTORY).read_text()).get("messages", [])
    except (OSError, ValueError):
        return None
    commands, think_s, cmd_s, rc = [], [], [], []
    since = None  # the step's model reply time, until its first tool message
    for m in messages:
        extra = m.get("extra") if isinstance(m.get("extra"), dict) else {}
        if m.get("role") == "assistant":
            since = extra.get("timestamp")
            for action in extra.get("actions") or []:
                commands.append(action.get("command", ""))
                think_s.append(round((extra.get("harbor_llm_wait_msec") or 0) / 1000, 3))
                cmd_s.append(None)
                rc.append(None)
        elif m.get("role") == "tool" and commands and cmd_s[-1] is None:
            if (secs := tool_exec_s(extra, since)) is not None:
                cmd_s[-1] = round(secs, 3)
            since = None
            rc[-1] = extra.get("returncode")
    while commands and classify(commands[-1]) == "submit":  # the scripted model adds its own
        for xs in (commands, think_s, cmd_s, rc):
            xs.pop()
    if not commands:
        return None
    return {
        "trial": trial_dir.name,
        "instance_id": task_dir(result),
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


TIMEOUT_S = 295.0  # a recorded command this long hit the harness's 300 s command timeout
_EXAMPLES, _EXAMPLE_CHARS = 5, 300  # the slowest commands kept per class, and how much of each
BUCKETS = ((1.0, "<1 s"), (10.0, "1-10 s"), (60.0, "10-60 s"), (TIMEOUT_S, "60-295 s"), (float("inf"), "timeout"))


def _bucket(s: float) -> str:
    return next(label for top, label in BUCKETS if s < top)


def _q(xs: list[float], q: float) -> float | None:
    return sorted(xs)[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def command_breakdown(traces: list[dict]) -> dict[str, Any]:
    """Where the recorded command time went: per command class and per duration bucket (no sandbox needed)."""
    timed = [
        (classify(c), s, c, rc)
        for t in traces
        for c, s, rc in zip(t["commands"], t["cmd_s"], t["rc"], strict=True)
        if s is not None
    ]
    total = sum(x[1] for x in timed) or 1.0
    unique: dict[str, set[str]] = {}
    for t in traces:
        for c in t["commands"]:
            unique.setdefault(classify(c), set()).add(" ".join(c.split()))
    by_class: dict[str, list[float]] = {}
    by_bucket: dict[str, list[float]] = {label: [] for _, label in BUCKETS}
    slowest: dict[str, list[tuple]] = {}
    for cls, s, c, rc in timed:
        by_class.setdefault(cls, []).append(s)
        by_bucket[_bucket(s)].append(s)
        slowest.setdefault(cls, []).append((s, rc, c))
    return {
        "trials": len(traces),
        "commands": sum(len(t["commands"]) for t in traces),
        "untimed": sum(len(t["commands"]) for t in traces) - len(timed),
        "total_s": total,
        "by_class": {
            cls: {
                "n": len(xs),
                "unique": len(unique.get(cls, ())),
                "total_s": sum(xs),
                "share": sum(xs) / total,
                "p50": _q(xs, 0.5),
                "p90": _q(xs, 0.9),
                "max": max(xs),
                "timeouts": sum(s >= TIMEOUT_S for s in xs),
                "slowest": [
                    {"s": s, "rc": rc, "cmd": c[:_EXAMPLE_CHARS]}
                    for s, rc, c in sorted(slowest[cls], key=lambda x: -x[0])[:_EXAMPLES]
                ],
            }
            for cls, xs in sorted(by_class.items(), key=lambda kv: -sum(kv[1]))
        },
        "by_bucket": {
            label: {"n": len(xs), "total_s": sum(xs), "share": sum(xs) / total} for label, xs in by_bucket.items()
        },
    }


def representative(traces: list[dict], dir_of, quantiles=(0.5, 0.9)) -> list[dict]:
    """A few distinct commands per class, at the given quantiles of their recorded time, as replayable traces.

    Commands that timed out are left out (the breakdown counts them; a replay would only wait out the timeout).
    Each command runs on a fresh sandbox of the task it came from, grouped one sandbox per task; ``dir_of(trace)``
    is the trace's task dir. A command replayed without the steps before it may exit differently: the report's
    fidelity check shows how often."""
    seen: dict[str, tuple[str, float, Any, dict]] = {}  # normalized command -> (class, seconds, rc, trace)
    for t in traces:
        for c, s, rc in zip(t["commands"], t["cmd_s"], t["rc"], strict=True):
            cls, key = classify(c), " ".join(c.split())
            if s is None or s >= TIMEOUT_S or cls in ("submit", "noop") or key in seen:
                continue
            seen[key] = (cls, s, rc, t)
    by_class: dict[str, list[tuple[float, str, Any, dict]]] = {}
    for key, (cls, s, rc, t) in seen.items():
        by_class.setdefault(cls, []).append((s, key, rc, t))
    by_task: dict[str, dict] = {}
    for _, xs in sorted(by_class.items()):
        xs.sort(key=lambda x: x[0])
        for i in sorted({min(len(xs) - 1, int(q * len(xs))) for q in quantiles}):
            s, command, rc, t = xs[i]
            task = dir_of(t)
            rep = by_task.setdefault(
                task,
                {"trial": f"rep-{task}", "instance_id": task, "commands": [], "think_s": [], "cmd_s": [], "rc": []},
            )
            rep["commands"].append(command)
            rep["think_s"].append(0.0)
            rep["cmd_s"].append(s)
            rep["rc"].append(rc)
    return list(by_task.values())
