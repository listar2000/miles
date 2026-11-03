"""SFT records from the raw GLM-5.3-Flash Pi eval exports, one eval at a time (idempotent, manifest-tracked).

    PYTHONPATH=<miles checkout> python experiments/glm53_pi_sft/build_records.py [--root ~/data/sft-glm53-pi] [--eval <uuid> ...]

For each raw/<eval_id>/ with a .done marker and no line in records/_manifest.jsonl yet:
  F1  keep trials with outcome "passed" and reward == 1.0 (and an available trace);
  F3  drop trials with a malformed tool call (arguments not an object, unknown tool). Missing "required" keys are
      not checked: Pi's strict schemas mark nullable keys (read's offset/limit) required and Pi accepts them omitted;
  F10 drop trials whose last agent step is a tool call (cut off by the agent timeout);
  F9  drop trials whose task prompt shares >= 20% of its word 8-grams with a TB 2.1 instruction (the eval benchmark);
  F8  tasks whose sha1(task) falls in the first 3% go to the held-out split (all their trials);
then miles_ext.sft.connectors.atif converts the rest (F2: its own refusals). Writes records/<eval_id>.jsonl,
records/<eval_id>.val.jsonl, their .report.json files (connector drops plus the filter drops above) and appends
a manifest line. Kept as is (user decisions 2026-11-03): the TB canary, steps without reasoning (trained),
both passing trials of a task.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from miles_ext.sft.connectors.atif import AtifOptions, convert_trial
from miles_ext.sft.connectors.base import report_path, write_records

VAL_FRACTION = 0.03
CONTAM_THRESHOLD = 0.2
TB21_TASKS = Path.home() / "data/tmax/tb21/tasks"
OPTIONS = AtifOptions(chat_template_kwargs={"enable_thinking": True})


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def tb21_ngrams() -> set[tuple[str, ...]]:
    grams: set[tuple[str, ...]] = set()
    for path in TB21_TASKS.glob("*/instruction.md"):
        grams |= ngrams(path.read_text())
    return grams


def held_out(task: str) -> bool:
    return int(hashlib.sha1(task.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < VAL_FRACTION


def check_trial(doc: dict, bench: set) -> str | None:
    """The filter reason that drops this trajectory, or None."""
    tools = {
        t["function"]["name"]: t["function"].get("parameters") or {}
        for t in doc["agent"].get("tool_definitions") or []
    }
    agent = [s for s in doc["steps"] if s.get("source") == "agent"]
    if not agent or agent[-1].get("tool_calls"):
        return "timed_out_on_tool_call"
    for step in agent:
        for call in step.get("tool_calls") or []:
            args, schema = call.get("arguments"), tools.get(call.get("function_name"))
            if schema is None:
                return "unknown_tool"
            if not isinstance(args, dict):
                return "arguments_not_object"
    prompt = next((s.get("message") for s in doc["steps"] if s.get("source") == "user"), "") or ""
    grams = ngrams(prompt if isinstance(prompt, str) else json.dumps(prompt))
    if grams and len(grams & bench) / len(grams) >= CONTAM_THRESHOLD:
        return "tb21_overlap"
    return None


def build(raw: Path, out: Path, bench: set) -> dict:
    manifest = json.loads((raw / "manifest.json").read_text())
    filters = {"train": collections.Counter(), "val": collections.Counter()}
    trials = {"train": [], "val": []}
    for t in manifest["trials"]:
        split = "val" if held_out(t["task"]) else "train"
        if t["outcome"] != "passed" or t.get("reward") != 1.0:
            filters[split]["not_passed_reward_1"] += 1
            continue
        if t["trace"]["status"] != "available":
            filters[split]["no_trace"] += 1
            continue
        trial = raw / t["trace"]["folder"]
        reason = check_trial(json.loads((trial / "agent/trajectory.json").read_text()), bench)
        if reason:
            filters[split][reason] += 1
            continue
        trials[split].append(trial)
    line = {"eval_id": manifest["evaluation_id"], "trials": len(manifest["trials"]), "splits": {}}
    for split, suffix in (("train", ""), ("val", ".val")):
        path = out / f"{manifest['evaluation_id']}{suffix}.jsonl"
        items = (item for trial in trials[split] for item in convert_trial(trial, OPTIONS))
        report = write_records(items, path)
        rep = json.loads(report_path(path).read_text())
        rep["filtered"] = dict(filters[split])
        report_path(path).write_text(json.dumps(rep, indent=2) + "\n")
        line["splits"][split] = {
            "records": report.records,
            "filtered": dict(filters[split]),
            "connector_dropped": dict(report.dropped),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path.home() / "data/sft-glm53-pi")
    parser.add_argument("--eval", action="append", default=None, help="only these eval ids")
    args = parser.parse_args()
    out = args.root / "records"
    out.mkdir(exist_ok=True)
    manifest = out / "_manifest.jsonl"
    done = {json.loads(line)["eval_id"] for line in manifest.read_text().splitlines()} if manifest.exists() else set()
    bench = tb21_ngrams()
    for raw in sorted((args.root / "raw").iterdir()):
        if not (raw / ".done").exists() or raw.name in done or (args.eval and raw.name not in args.eval):
            continue
        line = build(raw, out, bench)
        with open(manifest, "a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line))


if __name__ == "__main__":
    main()
