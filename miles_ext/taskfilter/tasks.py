"""Build the sweep's ``--tasks`` JSONL from a platform taskset, applying filter 1 (GPU) on the way.

    python -m miles_ext.taskfilter.tasks --taskset <slug> --project <id> --out tasks.jsonl

Each line is ``{"id", "version", "gpus"}`` for one task of the taskset's current
revision. A task whose ``task.toml`` asks for GPUs (``[environment].gpus``) needs
no sandbox to be dropped: it goes to ``<out>.gpu.jsonl`` instead. Only
``task.toml`` is read, through the platform API, into memory.
"""

import argparse
import json
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from miles_ext.taskfilter.mobius import credentials, fetch, taskset_tasks


def gpus(task_toml: dict) -> int:
    return int((task_toml.get("environment") or {}).get("gpus") or 0)


def task_line(task: dict, token: str, api: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "task.toml"
        fetch(f"tasks/{task['id']}/versions/{task['version']}/task.toml", dest, token, api)
        toml = tomllib.loads(dest.read_text())
    return {"id": task["id"], "version": task["version"], "gpus": gpus(toml)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--taskset", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args(argv)

    token, api = credentials()
    entries = {t["id"]: t for t in taskset_tasks(args.taskset, args.project)}
    with ThreadPoolExecutor(args.threads) as pool:
        lines = list(pool.map(lambda t: task_line(t, token, api), entries.values()))
    keep = [x for x in lines if not x["gpus"]]
    gpu = [x for x in lines if x["gpus"]]
    args.out.write_text("".join(json.dumps(x) + "\n" for x in keep))
    args.out.with_suffix(".gpu.jsonl").write_text("".join(json.dumps(x) + "\n" for x in gpu))
    print(f"{args.taskset}: {len(lines)} tasks, {len(keep)} to sweep, {len(gpu)} need GPUs -> {args.out}")


if __name__ == "__main__":
    main()
