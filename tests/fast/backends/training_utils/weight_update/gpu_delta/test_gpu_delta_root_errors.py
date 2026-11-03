"""[fork fix-080] A rank-0 GPU-delta action's error reaches every rank with its notes (the engine's reply)."""

import pytest

from miles.backends.training_utils.weight_update.protocols.gpu_delta import protocol


def test_root_errors_keep_the_engine_reply(monkeypatch):
    monkeypatch.setattr(protocol.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(protocol.dist, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(protocol, "get_gloo_group", lambda: None)

    def refuse():
        error = RuntimeError("Client error '409 Conflict'")
        error.add_note("response.text='unsupported GPU delta model mapping: Glm5NextForConditionalGeneration'")
        raise error

    with pytest.raises(RuntimeError, match="409 Conflict.*unsupported GPU delta model mapping"):
        protocol._on_root(refuse)
