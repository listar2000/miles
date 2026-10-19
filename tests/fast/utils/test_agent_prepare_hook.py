# [fork hook-005]
import sys
from types import SimpleNamespace

from miles.utils.misc import call_agent_prepare_hook


def test_prepare_hook_calls_the_agent_modules_prepare_once(tmp_path, monkeypatch):
    (tmp_path / "with_prepare.py").write_text("seen = []\ndef run(): pass\ndef prepare(args, ds): seen.append(ds)\n")
    (tmp_path / "without_prepare.py").write_text("def run(): pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for path in ("with_prepare.run", "without_prepare.run", None):
        call_agent_prepare_hook(SimpleNamespace(custom_agent_function_path=path), "ds")
    assert sys.modules["with_prepare"].seen == ["ds"]
