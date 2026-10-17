"""Reading tasks from the Mobius platform: credentials, task files, taskset listings.

Task versions live under the platform's ``frontierbench`` storage connection
(staging: ``CONNECTION``), at ``tasks/<id>/versions/<version>/{task.toml,
instruction.md, files.zip}``; files are read through the API, never from the
bucket directly.
"""

import json
import os
import subprocess
import time
import tomllib
from pathlib import Path

CONNECTION = "71db29e3-b018-4116-a5f9-008af0c39c17"  # frontierbench storage connection on staging
PREFIX = "frontierbench"
_ATTEMPTS = 4


def credentials() -> tuple[str, str]:
    """(token, API URL) from ``MOBIUS_TOKEN``/``MOBIUS_API_URL``, else the mobius CLI's saved login."""
    token, api = os.environ.get("MOBIUS_TOKEN"), os.environ.get("MOBIUS_API_URL")
    root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "mobius"
    if not token and (root / "credentials").is_file():
        token = tomllib.loads((root / "credentials").read_text()).get("token")
    if not api and (root / "config.toml").is_file():
        api = tomllib.loads((root / "config.toml").read_text()).get("api_url")
    if not token or not api:
        raise SystemExit("no Mobius token/API URL: run `mobius login` or set MOBIUS_TOKEN and MOBIUS_API_URL")
    return token, api


def fetch(path: str, dest: Path, token: str, api: str) -> None:
    """One file of the storage connection -> ``dest``. Retries 5xx, 429 and connection errors."""
    import httpx

    url = f"{api}/api/v1/storage-connections/{CONNECTION}/files/{PREFIX}/{path}"
    with httpx.Client(headers={"authorization": f"Bearer {token}"}, follow_redirects=False, timeout=600) as client:
        for attempt in range(_ATTEMPTS):
            try:
                r = client.get(url, params={"download": "true"})
                if r.status_code < 500 and r.status_code != 429:
                    break
            except httpx.TransportError:
                if attempt == _ATTEMPTS - 1:
                    raise
            if attempt < _ATTEMPTS - 1:
                time.sleep(5 * 2**attempt)
        if r.status_code in (301, 302, 303, 307, 308):  # large files: a signed bucket URL, no auth header
            with httpx.stream("GET", r.headers["location"], timeout=600) as s, dest.open("wb") as f:
                s.raise_for_status()
                for chunk in s.iter_bytes(8 << 20):
                    f.write(chunk)
            return
        r.raise_for_status()
        dest.write_bytes(r.content)


def fetch_task(task_id: str, version: str, tasks_dir: Path, token: str, api: str) -> Path:
    """Task version -> ``tasks_dir/<task_id>`` (files.zip unpacked, then the canonical task.toml/instruction.md)."""
    import zipfile

    base = f"tasks/{task_id}/versions/{version}"
    task_dir = tasks_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    archive = tasks_dir / f"{task_id}.zip"
    fetch(f"{base}/files.zip", archive, token, api)
    with zipfile.ZipFile(archive) as z:
        root = task_dir.resolve()
        for member in z.infolist():
            if not (task_dir / member.filename).resolve().is_relative_to(root):
                raise ValueError(f"unsafe path in {task_id} archive: {member.filename}")
        z.extractall(task_dir)
    archive.unlink()
    for name in ("task.toml", "instruction.md"):
        fetch(f"{base}/{name}", task_dir / name, token, api)
    return task_dir


def taskset_tasks(slug: str, project: str, page: int = 100) -> list[dict]:
    """The current revision's task entries of a taskset (``id``, ``version``, ...), via the mobius CLI."""
    tasks, offset = [], 0
    while True:
        out = subprocess.run(
            ["mobius", "taskset", "show", slug, "--project", project, "--json", "--limit", str(page),
             "--offset", str(offset)], check=True, capture_output=True, text=True).stdout
        batch = json.loads(out)["current"]["tasks"]
        tasks += batch
        if len(batch) < page:
            return tasks
        offset += page
