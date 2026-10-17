"""Merge sweep runs into per-task verdicts and keep lists.

    python -m miles_ext.taskfilter.final --taskset <name> --tasks <tasks.jsonl> --out <dir> <sweep dir>...

``--tasks`` is the JSONL the sweep ran on (``tasks.py``). Each sweep dir is a
``taskfilter/sweep-<run>/`` of the Volume; later runs override earlier ones
(full sweep first, then retries). The drop rules are ``summary.reasons``, with
the verifier cap each run recorded in its ``launch-*.json``. Writes
``verdicts.csv``, ``keep_<cutoff>.txt`` (kept tasks whose verifier took at most
the cutoff) and ``summary.json`` to ``--out``.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from miles_ext.taskfilter.summary import reasons

CUTOFFS_S = (300, 600, 900, 1200, 1800)
LIST_CUTOFFS_S = (600, 900, 1800)


def verifier_cap_s(run_dir: Path) -> float:
    """The verifier cap the run's newest launch record says it ran with."""
    launches = sorted(run_dir.glob("launch-*.json"))
    if not launches:
        raise SystemExit(f"{run_dir}: no launch-*.json, so its verifier cap is unknown")
    return json.loads(launches[-1].read_text())["recipe"]["agent"]["verifier_timeout"]


def load_rows(run_dirs: list[Path]) -> dict[str, dict]:
    """task id -> its latest row, tagged with the run it came from and that run's verifier cap."""
    rows = {}
    for run_dir in run_dirs:
        cap = verifier_cap_s(run_dir)
        for line in (run_dir / "summary.jsonl").read_text().splitlines():
            row = json.loads(line) if line.strip() else {}
            if row.get("task_id"):
                rows[row["task_id"]] = {**row, "source_run": run_dir.name.removeprefix("sweep-"), "cap_s": cap}
    return rows


def verdicts(tasks: list[dict], rows: dict[str, dict]) -> list[dict]:
    out = []
    for task in sorted(tasks, key=lambda t: t["id"]):
        row = rows.get(task["id"])
        why = reasons(row, row["cap_s"] if row else 0)
        row = row or {}
        out.append({
            "task_id": task["id"], "version": task["version"], "verdict": "drop" if why else "keep",
            "reason": why[0] if why else "", "all_reasons": "; ".join(why),
            "verifier_s": row.get("verifier_s"), "oracle_reward": row.get("oracle_reward"),
            "env_build_s": row.get("env_build_s"), "install_s": row.get("install_s"), "total_s": row.get("total_s"),
            "source_run": row.get("source_run", ""),
            "error": (row.get("oracle_exception") or row.get("install_error") or "").replace("\n", " ")[:200],
        })
    return out


def summarize_verdicts(taskset: str, res: list[dict], run_dirs: list[Path]) -> tuple[dict, dict[int, list[str]]]:
    kept = [r for r in res if r["verdict"] == "keep"]
    timed = [r for r in kept if r["verifier_s"] is not None]  # a kept row always has one; guard anyway
    keep = {c: sorted(r["task_id"] for r in timed if r["verifier_s"] <= c) for c in CUTOFFS_S}
    vs = sorted(r["verifier_s"] for r in timed)
    summary = {
        "taskset": taskset, "tasks": len(res), "runs": [d.name for d in run_dirs],
        "rows_by_source": dict(Counter(r["source_run"] or "none" for r in res)),
        "pass_all_filters": len(kept),
        "drop_reasons": dict(Counter(r["reason"] for r in res if r["verdict"] == "drop").most_common()),
        "keep_by_verifier_cutoff_s": {str(c): len(ids) for c, ids in keep.items()},
        "kept_verifier_s_p50_p90_p99": [vs[min(int(len(vs) * q), len(vs) - 1)] for q in (0.5, 0.9, 0.99)] if vs else None,
    }
    return summary, keep


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--taskset", required=True, help="the platform taskset the tasks came from (recorded only)")
    ap.add_argument("--tasks", required=True, type=Path, help="the JSONL the sweep ran on")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("runs", nargs="+", type=Path, help="sweep dirs, oldest first")
    args = ap.parse_args(argv)

    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    res = verdicts(tasks, load_rows(args.runs))
    summary, keep = summarize_verdicts(args.taskset, res, args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "verdicts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(res[0]))
        writer.writeheader()
        writer.writerows(res)
    for c in LIST_CUTOFFS_S:
        (args.out / f"keep_{c}.txt").write_text("".join(f"{i}\n" for i in keep[c]))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
