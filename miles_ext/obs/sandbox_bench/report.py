"""The breakdown: join each trial's three timing sources and aggregate per provider, concurrency and workload.

Per trial, every piece is a duration measured on one clock (the trial's wall time splits exactly into Harbor's
phases plus teardown; the agent phase splits into model round trips, commands and the rest):

    sandbox_start     Harbor's environment setup (create, image, boot) minus the calibration probes
    probes            the no-op exec probes the benchmark runs right after start (calibration, not trial work)
    harness_install   Harbor's agent setup (mini-swe-agent install and patches)
    proxy             the session proxy's time for the trial's model calls, receipt to response (host clock);
                      with a scripted model that is the proxy, in training it also holds the model's time
    tunnel            model wait seen in the sandbox minus proxy time: tunnel, network and client overhead
    model_unsplit     the model wait of a trial whose calls do not match its proxy lines one to one (a retry):
                      proxy and tunnel stay unknown there, so the wait is kept whole rather than dropped
    tools             command time in the sandbox (by command class in ``tool_share_by_class``)
    agent_other       the rest of the agent phase: mini-swe-agent start-up, Harbor's exec and polling
    verifier          Harbor's verifier phase
    teardown          what is left of the trial's wall time (stop, downloads, result write)

Calibration: ``probe_exec`` spans are no-op host-to-sandbox exec calls right after start.
"""

import json
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from miles_ext.obs.sandbox_bench.traces import TRAJECTORY, tool_exec_s
from miles_ext.obs.sandbox_bench.workloads import classify
from miles_ext.obs.sandbox_profile import TRIAL_FILE

MANIFEST = "manifest.json"
PROXY_LOG = "proxy.jsonl"
PIECES = (
    "sandbox_start",
    "probes",
    "harness_install",
    "proxy",
    "tunnel",
    "model_unsplit",
    "tools",
    "agent_other",
    "verifier",
    "teardown",
)


def _lines(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except OSError:
        return []


def _span(timing: dict | None) -> float:
    if not timing or not timing.get("started_at") or not timing.get("finished_at"):
        return 0.0
    return (
        datetime.fromisoformat(timing["finished_at"]) - datetime.fromisoformat(timing["started_at"])
    ).total_seconds()


def trial_row(trial_dir: Path, proxy_by_session: dict[str, list[float]]) -> dict[str, Any] | None:
    """One trial's breakdown, or None when Harbor wrote no result."""
    try:
        result = json.loads((trial_dir / "result.json").read_text())
    except (OSError, ValueError):
        return None
    wall = _span({"started_at": result.get("started_at"), "finished_at": result.get("finished_at")})
    phases = {k: _span(result.get(k)) for k in ("environment_setup", "agent_setup", "agent_execution", "verifier")}
    spans = _lines(trial_dir / TRIAL_FILE)
    session = next((s["path"] for s in spans if s.get("op") == "session"), None)
    try:
        messages = json.loads((trial_dir / TRAJECTORY).read_text()).get("messages", [])
    except (OSError, ValueError):
        messages = []
    waits, commands = [], []  # commands: (class, seconds, return code)
    pending, since = [], None
    for m in messages:
        extra = m.get("extra") if isinstance(m.get("extra"), dict) else {}
        if m.get("role") == "assistant" and extra.get("harbor_llm_wait_msec") is not None:
            waits.append(extra["harbor_llm_wait_msec"] / 1000)
            pending, since = [a.get("command", "") for a in extra.get("actions") or []], extra.get("timestamp")
        elif m.get("role") == "tool" and pending and (secs := tool_exec_s(extra, since)) is not None:
            commands.append((classify(pending[0]), secs, extra.get("returncode")))
            pending, since = [], None
    proxied = [
        dt for path, dts in proxy_by_session.items() if session and path.startswith(session + "/") for dt in dts
    ]
    # a call is matched to its proxy line by order; a retry (more proxy lines than waits) leaves tunnel unknown
    matched = len(proxied) == len(waits)
    tools = sum(s for _, s, _ in commands)
    probes = [s["dt"] for s in spans if s.get("op") == "probe_exec"]
    row = {
        "trial": trial_dir.name,
        "wall": wall,
        "sandbox_start": phases["environment_setup"] - sum(probes),
        "probes": sum(probes),
        "harness_install": phases["agent_setup"],
        "proxy": sum(proxied) if matched else 0.0,
        "tunnel": sum(waits) - sum(proxied) if matched else 0.0,
        "model_unsplit": 0.0 if matched else sum(waits),
        "matched": matched,
        "model_wait": sum(waits),
        "tools": tools,
        "agent_other": phases["agent_execution"] - sum(waits) - tools,
        "verifier": phases["verifier"],
        "teardown": wall - sum(phases.values()),
        "calls": len(waits),
        "tunnel_per_call": [w - p for w, p in zip(waits, proxied, strict=True)] if matched else [],
        "commands": commands,
        "probe_exec": probes,
        "host_exec": [s["dt"] for s in spans if s.get("op") == "exec" and s.get("depth") == 0],
        "start_span": next((s["dt"] for s in spans if s.get("op") == "start"), None),
        "exception": (result.get("exception_info") or {}).get("exception_type"),
    }
    return row


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def _median_ci(xs: list[float], n: int = 1000, seed: int = 0) -> tuple[float | None, float | None]:
    """95% bootstrap interval of the median."""
    if len(xs) < 2:
        return None, None
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n) - 1]


def _stats(xs: list[float]) -> dict[str, Any]:
    lo, hi = _median_ci(xs)
    return {"n": len(xs), "p50": _q(xs, 0.5), "p90": _q(xs, 0.9), "ci95": [lo, hi]}


def summarize(rows: list[dict], no_result: int = 0) -> dict[str, Any]:
    """Shares of total trial wall time per piece, and per-trial / per-call / per-command distributions.

    ``no_result``: trials that ended without a Harbor result (no row); they count as failed."""
    total = sum(r["wall"] for r in rows) or 1.0
    share = {p: sum(r[p] for r in rows) / total for p in PIECES}
    by_class: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for cls, s, _ in r["commands"]:
            by_class[cls].append(s)
    return {
        "trials": len(rows) + no_result,
        "failed": sum(1 for r in rows if r["exception"]) + no_result,
        "no_result": no_result,
        "unmatched_calls": sum(1 for r in rows if not r["matched"]),
        "share": share,
        "tool_share_by_class": {c: sum(xs) / total for c, xs in sorted(by_class.items())},
        "wall": _stats([r["wall"] for r in rows]),
        "per_trial": {p: _stats([r[p] for r in rows]) for p in PIECES},
        "tunnel_per_call": _stats([x for r in rows for x in r["tunnel_per_call"]]),
        "command_by_class": {c: _stats(xs) for c, xs in sorted(by_class.items())},
        "probe_exec": _stats([x for r in rows for x in r["probe_exec"]]),
        "host_exec_calls": _stats([float(len(r["host_exec"])) for r in rows]),
    }


def fidelity(rows: list[dict], trials: list[dict], traces: dict[str, dict]) -> dict[str, Any]:
    """Replays only: how often a command's exit code matches the original, and replay / original command time."""
    by_trial = {r["trial"]: r for r in rows}
    same, compared, ratios = 0, 0, defaultdict(list)
    replays, short = 0, 0
    for t in trials:
        row, trace = by_trial.get(Path(t.get("trial_dir") or "").name), traces.get(t.get("trace") or "")
        if not row or not trace:
            continue
        replayed = [c for c in row["commands"] if c[0] != "submit"]
        replays += 1
        short += len(replayed) < len(trace["commands"])  # the replay stopped before its last command
        for (cls, s, rc), orig_rc, orig_s in zip(replayed, trace["rc"], trace["cmd_s"], strict=False):
            if orig_rc is not None and rc is not None:
                compared += 1
                same += rc == orig_rc
            if orig_s:
                ratios[cls].append(s / orig_s)
    return {
        "rc_match": same / compared if compared else None,
        "compared": compared,
        "replays": replays,
        "replays_cut_short": short,
        "time_ratio_by_class": {c: _stats(xs) for c, xs in sorted(ratios.items())},
    }


def arm_report(arm_dir: Path, traces: dict[str, dict] | None = None) -> dict[str, Any]:
    manifest = json.loads((arm_dir / MANIFEST).read_text())
    proxy_by_session: dict[str, list[float]] = defaultdict(list)
    for line in sorted(_lines(arm_dir / PROXY_LOG), key=lambda r: r["t0"]):
        if line.get("path", "").endswith("/chat/completions"):
            proxy_by_session[line["path"]].append(line["dt"])
    by_workload: dict[str, list[dict]] = defaultdict(list)
    trials_by_workload: dict[str, list[dict]] = defaultdict(list)
    for t in manifest["trials"]:
        trials_by_workload[t["workload"]].append(t)
        if t.get("trial_dir") and (row := trial_row(Path(t["trial_dir"]), proxy_by_session)):
            by_workload[t["workload"]].append(row)
    out = {"arm": manifest["arm"], "workloads": {}}
    for workload in sorted(trials_by_workload):
        rows = by_workload.get(workload, [])
        out["workloads"][workload] = summarize(rows, no_result=len(trials_by_workload[workload]) - len(rows))
        if traces is not None and any(t.get("trace") for t in trials_by_workload[workload]):
            out["workloads"][workload]["fidelity"] = fidelity(rows, trials_by_workload[workload], traces)
    return out


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _sec(x: float | None) -> str:
    return "-" if x is None else (f"{x:.3f}" if x < 1 else f"{x:.1f}")


def markdown(arms: list[dict]) -> str:
    """One table of wall-time shares per (provider, concurrency, workload), then per-call and per-command times."""
    head = ["provider", "conc", "rep", "workload", "trials", "wall p50 s"] + list(PIECES)
    lines = ["## Share of trial wall time", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rows2 = [
        "## Per call and per command (seconds, p50 / p90)",
        "",
        "| provider | conc | rep | workload | tunnel per call | no-op exec | commands by class |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        a = arm["arm"]
        for workload, s in arm["workloads"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        a["provider"],
                        str(a["concurrency"]),
                        str(a["repeat"]),
                        workload,
                        f"{s['trials']} ({s['failed']} failed)",
                        _sec(s["wall"]["p50"]),
                    ]
                    + [_pct(s["share"][p]) for p in PIECES]
                )
                + " |"
            )
            cmds = "; ".join(
                f"{c} {_sec(v['p50'])}/{_sec(v['p90'])} (n={v['n']})" for c, v in s["command_by_class"].items()
            )
            rows2.append(
                f"| {a['provider']} | {a['concurrency']} | {a['repeat']} | {workload} | "
                f"{_sec(s['tunnel_per_call']['p50'])} / {_sec(s['tunnel_per_call']['p90'])} | "
                f"{_sec(s['probe_exec']['p50'])} / {_sec(s['probe_exec']['p90'])} | {cmds} |"
            )
    return "\n".join(lines + [""] + rows2) + "\n"
