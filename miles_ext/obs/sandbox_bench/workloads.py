"""Synthetic workloads, the Harbor task they run in, and the command classes every replayed command is tagged with.

Each synthetic workload is a fixed list of shell commands on one public image pinned by digest, so a difference
between providers cannot come from the task. The first command builds a small Python repo in the sandbox (class
``setup``), so read, write and test need no image build and no network. Nothing here is stored data: the scripts are
generated, and only their names and this file's code define them.
"""

import re
from pathlib import Path


IMAGE = "docker.io/library/python:3.12-slim@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f"
REPO = "/tmp/sbx/repo"
N_MODULES = 40

# the repo: N_MODULES modules of plain functions and one unittest file each, some CPU work per test
_SETUP_PY = f'''
import pathlib
root = pathlib.Path("{REPO}")
(root / "pkg").mkdir(parents=True, exist_ok=True)
(root / "tests").mkdir(exist_ok=True)
(root / "pkg" / "__init__.py").write_text("")
(root / "tests" / "__init__.py").write_text("")
for m in range({N_MODULES}):
    funcs = "".join(
        f"def f_{{m}}_{{k}}(n):\\n    total = 0\\n    for i in range(n):\\n        total += (i * {{k}} + {{m}}) % 7\\n    return total\\n\\n\\n"
        for k in range(30))
    (root / "pkg" / f"mod_{{m}}.py").write_text(f'"""module {{m}}"""\\nimport math\\n\\nx_{{m}} = {{m}}\\n\\n\\n' + funcs)
    tests = "".join(f"    def test_{{k}}(self):\\n        self.assertGreaterEqual(mod.f_{{m}}_{{k}}(200000), 0)\\n\\n" for k in range(30))
    (root / "tests" / f"test_mod_{{m}}.py").write_text(
        f"import unittest\\n\\nfrom pkg import mod_{{m}} as mod\\n\\n\\nclass T(unittest.TestCase):\\n" + tests)
'''
SETUP = f"python3 - <<'SBX_EOF'\n{_SETUP_PY}SBX_EOF"


def _read(i: int) -> str:
    m = i % N_MODULES
    return [
        f"cat {REPO}/pkg/mod_{m}.py",
        f"grep -rn 'def f_{m}_1' {REPO}",
        f"ls -la {REPO}/pkg | head -50",
        "find /usr/local/lib/python3.12 -name '*.py' | wc -l",
        f"sed -n '1,80p' {REPO}/tests/test_mod_{m}.py",
        "grep -rn 'def loads' /usr/local/lib/python3.12/json",
        f"wc -l {REPO}/pkg/*.py | tail -1",
        "head -c 4000 /usr/local/lib/python3.12/argparse.py",
    ][i % 8]


def _write(i: int) -> str:
    m = i % N_MODULES
    body = "\n".join(f"line_{j} = {j} * {i}" for j in range(60))
    return [
        f"cat <<'SBX_EOF' > {REPO}/pkg/new_{i}.py\n{body}\nSBX_EOF",
        f"sed -i 's/x_{m} = /x_{m}_renamed = /' {REPO}/pkg/mod_{m}.py",
        f"cp {REPO}/pkg/mod_{m}.py /tmp/sbx/copy_{i}.py && echo copied",
        f"printf '\\n# edited {i}\\n' >> {REPO}/pkg/mod_{m}.py",
        f"rm -f /tmp/sbx/copy_{i - 2}.py && mkdir -p /tmp/sbx/out_{i} && touch /tmp/sbx/out_{i}/done",
    ][i % 5]


def _test(i: int) -> str:
    if i % 5 == 4:
        return f"cd {REPO} && python3 -m unittest discover -s tests -t . 2>&1 | tail -3"
    return f"cd {REPO} && python3 -m unittest tests.test_mod_{i % N_MODULES} 2>&1 | tail -3"


def _fs_scan(i: int) -> str:
    return [
        "grep -R 'SBX_NEEDLE' / 2>/dev/null | head -5",
        "find / -name '*.so' 2>/dev/null | wc -l",
        "grep -rl 'import os' / 2>/dev/null | wc -l",
        "find / -type f -newer /etc/hostname 2>/dev/null | wc -l",
    ][i % 4]


_INSTALL_PACKAGES = (
    "requests==2.32.3",
    "pyyaml==6.0.2",
    "numpy==2.1.3",
    "rich==13.9.4",
    "pydantic==2.10.3",
    "httpx==0.28.1",
)


def _install(i: int) -> str:
    package = _INSTALL_PACKAGES[i % len(_INSTALL_PACKAGES)]
    return f"pip install --no-cache-dir --disable-pip-version-check -q --target /tmp/sbx/pkgs_{i} {package} && echo ok"


# name -> (steps after setup, command for step i)
SYNTHETIC = {
    "noop": (40, lambda i: "true"),
    "read": (40, _read),
    "write": (40, _write),
    "test": (20, _test),
    "fs-scan": (8, _fs_scan),
    "install": (6, _install),
}
TRACE = "trace"  # the workload name of replayed traces (traces.py)


def script(workload: str) -> list[str]:
    """The commands of one synthetic workload, setup first; the scripted model adds the submit."""
    steps, command = SYNTHETIC[workload]
    return [SETUP] + [command(i) for i in range(steps)]


TASK_PREFIX = "sbx-"


def write_task(root: Path, workload: str, cpus: int, memory_mb: int) -> str:
    """Write the Harbor task dir of a synthetic workload under ``root``; return its instance id."""
    name = TASK_PREFIX + workload
    d = root / name
    (d / "tests").mkdir(parents=True, exist_ok=True)
    (d / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 120.0\n\n[agent]\ntimeout_sec = 3600.0\n\n'
        f'[environment]\ndocker_image = "{IMAGE}"\ncpus = {cpus}\nmemory_mb = {memory_mb}\n'
        # public, as replayed tasks are (app._prepare_task): no network policy switch during the trial
        'network_mode = "public"\n'
    )
    (d / "instruction.md").write_text(f"Sandbox benchmark workload {workload}: run the commands you are given.\n")
    (d / "tests" / "test.sh").write_text(
        "#!/bin/bash\nmkdir -p /logs/verifier && echo 1 > /logs/verifier/reward.txt\n"
    )
    (d / "tests" / "test.sh").chmod(0o755)
    return name


_AT_COMMAND = r"(?:^|[\n;&|(`]|\$\(|\b(?:then|do|else|xargs|time|sudo|timeout\s+\S+|cd\s+\S+\s*&&))\s*"
# Command classes, first match wins. The order puts the more specific pattern first
# (a pytest run that pipes to grep is a test; grep -R over / is a scan, not a read; a command that writes a
# script and runs it, or runs a script that redirects its output, is a run: the time is the run's).
CLASSES = (
    ("submit", r"^\s*echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\s*$"),
    ("noop", r"^\s*(true|:)\s*$"),
    ("setup", re.escape(f'pathlib.Path("{REPO}")')),
    (
        "test",
        r"\b(pytest|unittest|tox|nosetests|go test|cargo test|npm (run )?test|yarn test|make (check|test)|ctest|"
        r"mvn test|gradle test|rspec|phpunit|jest)\b",
    ),
    (
        "install",
        r"\b(pip3? install|uv (pip )?(install|add|sync)|apt(-get)? (install|update)|npm (i|install|ci)\b|"
        r"yarn (add|install)|conda install|poetry (add|install)|cargo (install|fetch)|go (get|mod download)|"
        r"gem install|apk add)",
    ),
    ("build", r"\b(make|cmake|gcc|g\+\+|clang|cargo build|go build|npm run build|mvn|gradle|javac|rustc|tsc)\b"),
    (
        "fs-scan",
        r"(\bgrep\b[^|;&]*\s-[A-Za-z]*[rR][A-Za-z]*\b[^|;&]*\s/(\s|$|\*)|\bfind\s+/(\s|$)|"
        r"\b(du|ls)\s+-[A-Za-z]*R?[A-Za-z]*\s+/(\s|$))",
    ),
    ("run", _AT_COMMAND + r"(python3?|node|bash|sh|ruby|perl|java|go run|cargo run|\./\S+)\b"),
    (
        "write",
        r"(\bsed\s+-i|\bcat\s*>|<<\s*'?\w+'?\s*>|\btee\b|\bcp\b|\bmv\b|\brm\b|\bmkdir\b|\btouch\b|\bpatch\b|"
        r"git (apply|checkout|commit|add|stash|reset)|\bchmod\b|[^2&]>\s*[^&\s])",
    ),
    (
        "read",
        _AT_COMMAND
        + r"(cat|head|tail|less|more|grep|rg|ag|find|ls|wc|sed|awk|tree|stat|file|diff|git (log|show|diff|status|"
        r"blame|grep)|pwd|echo|which|env|printenv|uname|ps|df|du|nl|sort|uniq|cut|xxd|od|strings)\b",
    ),
)
_COMPILED = tuple((name, re.compile(pattern)) for name, pattern in CLASSES)


def classify(command: str) -> str:
    for name, pattern in _COMPILED:
        if pattern.search(command):
            return name
    return "other"
