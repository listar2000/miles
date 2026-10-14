"""Upstream modules that live outside Miles' installed packages.

Miles ships its in-process Harbor agent function under ``examples/`` (not part
of the ``miles`` package). The fork has that tree checked out next to
``miles/``, so we load the file from there rather than keeping a copy: what
runs is always the version of the branch's upstream base.
"""

import importlib.util
import sys
from functools import cache
from pathlib import Path

import miles

REPO_ROOT = Path(miles.__file__).resolve().parents[1]
HARBOR_AGENT_FUNCTION = REPO_ROOT / "examples/experimental/harbor/harbor_agent_function.py"
# what miles_ext.harbor.agent_function uses from it; checked at load, so an
# upstream rename fails the image build instead of a rollout
HARBOR_AGENT_FUNCTION_API = (
    "HARNESS_BINDINGS", "HarnessBinding", "_mini_swe_agent_env", "build_trial_config", "_trial_timeout_s",
    "_ensure_provider_key", "_failed", "trial_result_to_metadata", "_environment_config",
)


@cache
def harbor_agent_function():
    """``examples/experimental/harbor/harbor_agent_function.py`` as a module (one instance per process)."""
    name = "miles_examples_harbor_agent_function"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARBOR_AGENT_FUNCTION)
    if spec is None or not HARBOR_AGENT_FUNCTION.exists():
        raise ImportError(f"Miles' Harbor agent function not found at {HARBOR_AGENT_FUNCTION}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    missing = [n for n in HARBOR_AGENT_FUNCTION_API if not hasattr(module, n)]
    if missing:
        raise ImportError(f"{HARBOR_AGENT_FUNCTION} no longer defines {missing}; adapt miles_ext.harbor.agent_function")
    return module
