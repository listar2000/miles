"""Register a launched run with the monitor by hand: a launch made without MILES_MONITOR_DIR set, a backfill, or
a title and notes for one the launcher registered. Run it once the launch log names the app:

    python -m miles_ext.monitor.register <run_id> <launch log> <recipe yaml> [--title "..."] [--notes "..."]

It reads the app id from the log ("https://modal.com/apps/<workspace>/<env>/ap-..."), the start time from Modal,
and the run's shape and limits from the recipe; then writes manifests/<app_id>.json. Running it again for a
relaunch (a new app id in a new log) adds that attempt.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import config
from .manifest import fields_from_recipe, write_manifest

APP_URL = re.compile(r"modal\.com/apps/[\w-]+/[\w-]+/(ap-[A-Za-z0-9]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("log")
    ap.add_argument("recipe")
    ap.add_argument("--title")
    ap.add_argument("--notes")
    ap.add_argument("--wait", type=int, default=300, help="seconds to wait for the app id to show up in the log")
    a = ap.parse_args()
    log = Path(a.log).resolve()
    deadline = time.time() + a.wait
    app_id = None
    while not app_id:
        m = APP_URL.findall(log.read_text(errors="replace")) if log.exists() else []
        app_id = m[-1] if m else None
        if not app_id:
            if time.time() > deadline:
                sys.exit(f"no Modal app URL in {log} after {a.wait}s")
            time.sleep(5)
    apps = json.loads(
        subprocess.run(
            [config.load()["modal_cli"], "app", "list", "--json"], capture_output=True, text=True, timeout=60
        ).stdout
        or "[]"
    )
    app = next((x for x in apps if x.get("app_id") == app_id), {})
    created = app.get("created_at")  # "2026-10-16 15:21:56+00:00"
    started = (
        datetime.fromisoformat(created).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if created else None
    )

    recipe = yaml.safe_load(Path(a.recipe).read_text())
    fields = fields_from_recipe(recipe, a.run_id) | {
        "gpu_app_name": app.get("description"),
        "recipe": str(Path(a.recipe)),
        "log_path": str(log),
    }
    fields |= {
        k: v for k, v in {"title": a.title, "notes": a.notes, "started_at": started}.items() if v
    }  # the app's own start
    path = write_manifest(app_id, a.run_id, **fields)
    print(f"registered {a.run_id} attempt {app_id} -> {path}")


if __name__ == "__main__":
    main()
