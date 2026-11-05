"""``python -m miles_ext.node <role> --spec <file>``: RFC 0005's entry commands."""

import argparse
import json
import sys
from pathlib import Path

from miles_ext.node import contract
from miles_ext.recipes import recipe as R


def load_spec(path: str) -> tuple[dict, str]:
    """A spec is the resolved recipe plus the run id: ``{"run_id": ..., "recipe": {...}}``."""
    spec = json.loads(Path(path).read_text())
    return R.from_dict(spec["recipe"]), spec["run_id"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m miles_ext.node")
    ap.add_argument("role", choices=["manifest"])
    ap.add_argument("--spec", required=True, help='JSON file: {"run_id": ..., "recipe": {...}}')
    a = ap.parse_args(argv)
    recipe, run_id = load_spec(a.spec)
    print(json.dumps(contract.manifest(recipe, run_id), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
