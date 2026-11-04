"""Copy the training records whose teacher run took at most --max-seconds of agent time; held-out records as they are."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def agent_seconds(trajectory: Path) -> float:
    stamps = [s["timestamp"] for s in json.loads(trajectory.read_text())["steps"] if s.get("timestamp")]
    parse = lambda t: datetime.fromisoformat(t.replace("Z", "+00:00"))  # noqa: E731
    return (parse(stamps[-1]) - parse(stamps[0])).total_seconds()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="a records directory (build_records.py)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.input.glob("*.jsonl")):
        if path.name.startswith("_"):
            continue
        if path.name.endswith(".val.jsonl"):
            shutil.copyfile(path, args.output / path.name)
            continue
        kept = dropped = 0
        with open(path) as f, open(args.output / path.name, "w") as out:
            for line in f:
                meta = json.loads(line)["metadata"]
                if agent_seconds(Path(meta["trial"]) / meta["trajectory"]) <= args.max_seconds:
                    out.write(line)
                    kept += 1
                else:
                    dropped += 1
        print(f"{path.name}: kept {kept}, dropped {dropped}")


if __name__ == "__main__":
    main()
