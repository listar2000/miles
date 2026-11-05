"""Node-side storage: the adapter MILES_STORAGE names, with no provider SDK unless it is selected."""

import subprocess
import sys
from unittest.mock import Mock

import pytest
from miles_ext.node import contract
from miles_ext.node import storage as S


class Custom:
    def __init__(self, mount):
        self.mount = mount


def test_posix_is_the_default_and_needs_no_modal():
    code = (
        "import sys; sys.modules['modal'] = None\n"
        "from miles_ext.node import storage\n"
        "s = storage.for_mount('/results'); s.commit(); s.refresh(); s.remove('x'); print(type(s).__name__)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "PosixStorage"


def test_a_dotted_path_loads_a_callers_adapter(monkeypatch):
    monkeypatch.setenv(contract.STORAGE_ENV, f"{__name__}.Custom")
    assert isinstance(S.for_mount("/results"), Custom)


def test_an_unknown_name_is_refused(monkeypatch):
    monkeypatch.setenv(contract.STORAGE_ENV, "s3")
    with pytest.raises(ValueError, match="s3"):
        S.for_mount("/results")


def test_modal_adapter_commits_the_volume_mounted_there(monkeypatch):
    modal = pytest.importorskip("modal")
    from miles_ext.node.adapters import modal_volume as M

    volume = Mock()
    volume.remove_file.side_effect = modal.exception.NotFoundError("gone")
    from_name = Mock(return_value=volume)
    monkeypatch.setattr(M.modal.Volume, "from_name", from_name)
    monkeypatch.setenv(contract.STORAGE_ENV, S.MODAL_VOLUME)
    monkeypatch.setenv(M.VOLUMES_ENV, '{"/results": "vol-a"}')

    store = S.for_mount("/results")
    store.commit()
    store.refresh()
    store.remove("run1/checkpoints/iter_0000001")  # already gone is fine

    from_name.assert_called_once_with("vol-a")
    volume.commit.assert_called_once()
    volume.reload.assert_called_once()
