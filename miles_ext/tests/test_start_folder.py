"""The in-sandbox agent runs commands where Harbor starts it (the task's workdir), not in a
folder we pick; MSWEA_CWD still overrides:

    python -m pytest miles_ext/tests/test_start_folder.py
"""

from miles_ext.harbor import agent_function as af


def test_no_folder_by_default(monkeypatch):
    monkeypatch.delenv("MSWEA_CWD", raising=False)
    assert af._shell_config()["cwd"] == ""


def test_mswea_cwd_overrides(monkeypatch):
    monkeypatch.setenv("MSWEA_CWD", "/testbed")
    assert af._shell_config()["cwd"] == "/testbed"
