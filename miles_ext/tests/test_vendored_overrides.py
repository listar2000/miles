"""Overrides of vendored methods stay tied to the method they replace.

``miles_ext.harbor.sail.SailEnvironment._create_sailbox`` copies the vendored
``harbor_sail.SailEnvironment._create_sailbox`` with one change. ``ledger.py``
checks the vendored file's blob; this checks the method itself, so a re-vendor
that changes it fails here until the override is re-checked and the hash below
updated. Reads source only (no harbor or sail import):

    python -m pytest miles_ext/tests/test_vendored_overrides.py
"""

import ast
import hashlib
import inspect
from pathlib import Path

from miles_ext.recipes import recipe as R

ROOT = Path(inspect.getfile(R)).parents[2]

# (vendored file, class, method) -> sha256 of the method's source the override was written against
OVERRIDDEN = {
    ("miles_ext/harbor/sail/harbor_sail.py", "SailEnvironment", "_create_sailbox"):
        "3b68d48d13f8cac479b31908da95460d670cb6762f479a7ba31121176af58f84",
}


def _method_source(path: str, cls: str, name: str) -> str:
    src = (ROOT / path).read_text()
    klass = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in klass.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    return ast.get_source_segment(src, fn)


def test_overridden_methods_unchanged():
    for (path, cls, name), digest in OVERRIDDEN.items():
        got = hashlib.sha256(_method_source(path, cls, name).encode()).hexdigest()
        assert got == digest, (f"{path}:{cls}.{name} changed; re-check the override in miles_ext and "
                               f"update its hash to {got}")
