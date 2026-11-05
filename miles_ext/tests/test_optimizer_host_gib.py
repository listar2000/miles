"""The pinned optimizer tier is sized from the trainer node's memory, after the other host users (fix-073)."""

import pytest

from miles_ext.recipes import recipe as R


def _recipe(**overrides) -> dict:
    spec = {
        "model": {"preset": "glm-5.3-flash"},
        "cluster": {"gpu": "B300", "nodes": 2, "train_gpus": 8, "memory_gib": 2304},
        "optimizer": {"offload": "disk"},
    }
    for section, values in overrides.items():
        spec[section] = {**spec.get(section, {}), **values}
    return R.from_dict(spec)


def test_together_mode_splits_what_the_reserve_leaves():
    assert R.optimizer_host_gib(_recipe()) == (2304 - R.TRAINER_HOST_RESERVE_GIB) // 8


def test_gpu_delta_also_leaves_its_model_copies():
    together = R.optimizer_host_gib(_recipe())
    split = R.optimizer_host_gib(_recipe(cluster={"placement": "split"}, rollout={"weight_sync": "gpu-delta"}))
    copies = R.GPU_DELTA_HOST_COPIES * R.PRESETS["glm-5.3-flash"].served_gib
    assert split == (2304 - R.TRAINER_HOST_RESERVE_GIB - copies) // 8 < together


def test_gpu_delta_copies_spread_over_the_trainer_nodes():
    recipe = _recipe(
        cluster={"placement": "split", "nodes": 2, "train_gpus": 16, "memory_gib": 2560},
        rollout={"weight_sync": "gpu-delta"},
    )
    copies = R.GPU_DELTA_HOST_COPIES * R.PRESETS["glm-5.3-flash"].served_gib
    assert R.optimizer_host_gib(recipe) == (2560 - R.TRAINER_HOST_RESERVE_GIB - copies / 2) // 8


def test_a_partly_filled_trainer_node_is_sized_by_the_fullest_node():
    """12 trainer GPUs sit 8 + 4 on two nodes: each GPU gets the share of a full node."""
    full = R.optimizer_host_gib(_recipe())
    assert R.optimizer_host_gib(_recipe(cluster={"nodes": 3, "train_gpus": 12})) == full


@pytest.mark.parametrize(
    "overrides", [{"optimizer": {"offload": None}}, {"cluster": {"memory_gib": 1024}}], ids=["no-disk", "small-node"]
)
def test_no_budget_without_disk_streaming_or_room(overrides):
    assert R.optimizer_host_gib(_recipe(**overrides)) == 0


def test_every_gpu_delta_preset_knows_its_served_size():
    assert all(R.PRESETS[name].served_gib for name in R.GPU_DELTA_PRESETS)
