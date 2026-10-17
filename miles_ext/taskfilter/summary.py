"""One summary row per trial, from Harbor's ``result.json`` and the install probe record."""

from datetime import datetime
from typing import Any

ROW_KEYS = ("task_id", "version", "env_build_s", "install_ok", "smoke_ok",
            "install_s", "install_error", "oracle_reward", "oracle_exception", "verifier_s", "verifier_timed_out", "total_s")


def seconds(timing: dict | None) -> float | None:
    """Seconds between ``started_at`` and ``finished_at`` of a Harbor TimingInfo, or None."""
    if not timing or not timing.get("started_at") or not timing.get("finished_at"):
        return None
    start, end = (datetime.fromisoformat(str(timing[k]).replace("Z", "+00:00")) for k in ("started_at", "finished_at"))
    return round((end - start).total_seconds(), 1)


def summarize(task_id: str, version: str, result: dict | None, probe: dict | None,
              verifier_cap_s: float, error: str | None = None) -> dict[str, Any]:
    """The filter row for one trial. ``error`` is a runner-side failure (no result.json)."""
    result, probe = result or {}, probe or {}
    exc = result.get("exception_info") or {}
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    verifier_s = seconds(result.get("verifier"))
    exc_type = exc.get("exception_type") or ""
    timed_out = (verifier_s is not None and verifier_s >= verifier_cap_s - 5) or (
        "Timeout" in exc_type and (result.get("verifier") or {}).get("started_at") is not None
        and not rewards)
    oracle_exception = f"{exc_type}: {str(exc.get('exception_message') or '')[:300]}" if exc_type else error
    return {
        "task_id": task_id,
        "version": version,
        "env_build_s": seconds(result.get("environment_setup")),
        "install_ok": probe.get("install_ok"),
        "smoke_ok": probe.get("smoke_ok"),
        "install_s": probe.get("install_s"),
        "install_error": probe.get("error"),
        "oracle_reward": rewards.get("reward"),
        "oracle_exception": oracle_exception,
        "verifier_s": verifier_s,
        "verifier_timed_out": bool(timed_out),
        "total_s": seconds(result),
    }


def reasons(row: dict | None, verifier_cap_s: float) -> list[str]:
    """Why a task is dropped, from its row; ``[]`` keeps it. The one owner of the filter rules
    (filter 1, GPU, is decided from task.toml, not here). The first reason is the primary one."""
    if not row:
        return ["not run"]
    exc = row.get("oracle_exception") or ""
    kind = exc.split(":")[0]
    # the sandbox never ran the task: one reason says why
    if kind == "SailboxExecutionError":
        return ["image spec rejected by Sail (registry or Dockerfile syntax)"]
    if kind == "ValueError" and "largest Sailbox" in exc:
        return ["resources exceed largest Sailbox"]
    if kind == "InvalidArgumentError" and "10000 entries" in exc:
        return ["build context >10000 files"]
    if kind in ("ImageBuildError", "InvalidArgumentError", "ValueError"):
        return [f"image build failed ({kind})"]
    if kind == "EnvironmentStartTimeoutError":
        return ["image build timeout"]
    out = []
    if kind == "AgentTimeoutError":
        out.append("oracle over its time limit")
    if row.get("verifier_timed_out") or kind == "VerifierTimeoutError":
        out.append(f"verifier timeout (>={verifier_cap_s:g} s)")
    if kind == "DownloadVerifierDirError":
        out.append("verifier output download error")
    if not out and (row.get("oracle_reward") or 0) < 1:
        out.append("oracle fail" + (f" ({kind})" if kind else ""))
    if not (row.get("install_ok") and row.get("smoke_ok")):
        out.append("mini-swe-agent install fail")
    return out
