"""fix-076: a recovery publication passes the post-write visibility hook before engines read it."""  # [fork fix-076]

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.backends.training_utils.weight_update.protocols.gpu_delta import protocol as gpu_delta


def _protocol(events, hook_fails=False):
    protocol = gpu_delta.UpdateWeightFromGpuDelta(
        Namespace(custom_update_weight_post_write_path=None, hf_checkpoint="hf")
    )

    def hook(args, directory, engines):
        events.append(("hook", directory))
        if hook_fails:
            raise OSError("commit failed")

    protocol._post_write_hook = hook
    protocol.rollout_engines = []
    protocol._stream_id, protocol._cohort, protocol._plan = "stream", SimpleNamespace(plan_digest="plan"), {}
    protocol._hf_snapshot, protocol._base_sha256 = {}, "base"
    protocol._recovery_payload = SimpleNamespace(write=lambda *args: {"metadata": {}})
    return protocol


def _gather(shard, shards, dst, group):
    shards[0] = shard


@pytest.fixture
def one_rank():
    with (
        patch.object(gpu_delta, "_gather_all", side_effect=lambda value: [value]),
        patch.object(gpu_delta, "get_gloo_group", return_value=None),
        patch.object(gpu_delta.dist, "get_rank", return_value=0),
        patch.object(gpu_delta.dist, "get_world_size", return_value=1),
        patch.object(gpu_delta.dist, "gather_object", side_effect=_gather),
        patch.object(gpu_delta.dist, "broadcast_object_list"),
    ):
        yield


def test_recovery_publication_is_made_visible_after_sealing(tmp_path, one_rank):
    events = []
    seal = lambda directory, shards: events.append(("seal", str(directory))) or {"target_version": 3}  # noqa: E731
    with patch.object(gpu_delta.gpu_delta_publication, "seal_publication", side_effect=seal):
        descriptor = _protocol(events)._publish_recovery(tmp_path / "recovery_v000003", 3)
    assert descriptor == {"target_version": 3}
    assert events == [("seal", str(tmp_path / "recovery_v000003")), ("hook", str(tmp_path / "recovery_v000003"))]


def test_a_failed_visibility_hook_fails_the_recovery_publication(tmp_path, one_rank):
    events = []
    with patch.object(gpu_delta.gpu_delta_publication, "seal_publication", return_value={}):
        with pytest.raises(RuntimeError, match="recovery publication visibility failed"):
            _protocol(events, hook_fails=True)._publish_recovery(tmp_path / "recovery_v000003", 3)
