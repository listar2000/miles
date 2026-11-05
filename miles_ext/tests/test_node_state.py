"""Run state behind MILES_STATE, and the trainer role's env."""

import subprocess
import sys

import pytest

from miles_ext.node import contract as C
from miles_ext.node import state as S


def test_file_state_round_trip(tmp_path):
    state = S.FileState("run1", tmp_path)
    assert state.get("run:run1") is None
    state["run:run1"] = {"phase": "training"}
    assert state.get("run:run1") == {"phase": "training"}
    assert state.pop("run:run1") == {"phase": "training"}
    assert state.pop("run:run1", "gone") == "gone"


def test_an_unknown_state_store_is_refused(monkeypatch):
    monkeypatch.setenv(C.STATE_ENV, "redis")
    with pytest.raises(ValueError, match="redis"):
        S.for_run("run1")


def test_the_trainer_role_imports_without_modal():
    code = (
        "import sys, types; sys.modules['modal'] = None; sys.modules.setdefault('ray', types.ModuleType('ray'))\n"
        "import miles_ext.node.trainer"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_gang_reads_the_contracts_env(monkeypatch):
    pytest.importorskip("aiohttp")
    from miles_ext.node.trainer import Gang

    for name, value in (
        (C.NODE_RANK_ENV, "1"),
        (C.NNODES_ENV, "2"),
        (C.HEAD_ADDR_ENV, "10.0.0.1"),
        (C.NODE_IP_ENV, "10.0.0.2"),
    ):
        monkeypatch.setenv(name, value)
    assert Gang.from_env() == Gang(1, 2, "10.0.0.1", "10.0.0.2")
