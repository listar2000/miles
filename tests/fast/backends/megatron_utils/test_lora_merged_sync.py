"""--lora-merged-sync [fork hook-006]: engines get the full weights, with full-parameter weight-version checks."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.utils import restore_lora_adapter, save_lora_checkpoint
from miles.backends.megatron_utils.megatron_config import resolve_args_checkpoint_load
from miles.backends.training_utils.model_companion import ModelCompanion, ModelCompanionWeightVersionUtils
from miles.rollout.session.config import compute_session_server_config
from miles.utils.arguments import _resolve_sample_ownership_check
from miles.utils.lora.arguments import _validate_lora_merged_sync, _validate_lora_merged_sync_placement
from miles.utils.lora.utils import lora_rollout_enabled, lora_without_weight_versions
from miles.utils.types import SampleLineage
from miles.utils.weight_version import assert_weight_version_is_published


def _args(**overrides):
    base = dict(
        lora_rank=8,
        lora_adapter_path=None,
        lora_merged_sync=True,
        lora_train_only=False,
        multi_lora=False,
        colocate=False,
        offload_train=False,
        megatron_to_hf_mode="raw",
        debug_rollout_only=False,
        debug_train_only=False,
        debug_skip_weight_update=False,
        update_weights_interval=1,
    )
    return Namespace(**{**base, **overrides})


@pytest.mark.parametrize("merged", [True, False])
def test_merged_sync_serves_full_weights_and_checks_versions(merged):
    args = _args(lora_merged_sync=merged)
    assert lora_rollout_enabled(args) is not merged
    assert lora_without_weight_versions(args) is not merged  # gates the actor's verify_engine_version
    if merged:
        with pytest.raises(AssertionError, match="staleness accounting"):
            assert_weight_version_is_published(args, rollouts_since_publish=10)
    else:
        assert_weight_version_is_published(args, rollouts_since_publish=10)


@pytest.mark.parametrize("merged", [True, False])
def test_merged_sync_allows_the_sample_ownership_checker(merged):
    args = _args(
        lora_merged_sync=merged,
        enable_sample_ownership_checker=True,
        custom_convert_samples_to_train_data_path=None,
        sample_ownership_grace_steps=None,
        ci_test=False,
        train_backend="megatron",
        megatron_config=None,
        debug_disable_optimizer=False,
        num_critic_only_steps=0,
        save_debug_event_data="/events",
    )
    if merged:
        _resolve_sample_ownership_check(args)
    else:
        with pytest.raises(ValueError, match="LoRA training has no model companion info"):
            _resolve_sample_ownership_check(args)


def test_session_server_requests_no_adapter_under_merged_sync():
    args = _args(
        miles_router_timeout=None,
        hf_checkpoint=None,
        chat_template_path=None,
        tito_model="qwen3_5",
        apply_chat_template_kwargs=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        sglang_speculative_algorithm=None,
        save_debug_trajectory_data=None,
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="raw",
    )
    config = compute_session_server_config(args, host="h", port=1, instance_id=None, backend_url="http://b")
    assert not lora_rollout_enabled(config)


@pytest.mark.parametrize(
    "override",
    [
        {"megatron_to_hf_mode": "bridge"},
        {"lora_train_only": True},
        {"multi_lora": True},
        {"colocate": True},
        {"offload_train": True},
    ],
)
def test_merged_sync_refuses_incompatible_modes(override):
    with pytest.raises(AssertionError, match="--lora-merged-sync"):
        _validate_lora_merged_sync(_args(**override))
        _validate_lora_merged_sync_placement(_args(**override))


@pytest.mark.parametrize("resume_dir", [None, "/run/ckpt"])
def test_lora_restore_keeps_sample_counts_only_on_a_resume(tmp_path, resume_dir):
    def model():
        chunk = nn.Module()
        chunk.model_companion = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
        return [chunk]

    optimizer = SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda state: None)
    args = Namespace(
        megatron_to_hf_mode="raw",
        lora_adapter_targets=[],
        experts_shared_outer_loras=False,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        no_save_optim=False,
        no_load_optim=False,
        lora_adapter_path=str(tmp_path / "adapter"),
        lora_resume_dir=resume_dir,
    )
    saved = model()
    for _ in range(3):
        ModelCompanionWeightVersionUtils.bump_weight_version(saved)
    saved[0].model_companion.record_sample_consumptions(
        [SampleLineage(source_sample_index=0, output_index=0, output_count=1)]
    )
    save_lora_checkpoint(
        saved, args, args.lora_adapter_path, publisher=None, optimizer=optimizer, opt_param_scheduler=None, iteration=3
    )

    resumed = model()
    assert restore_lora_adapter(resumed, args, optimizer=optimizer, opt_param_scheduler=None) == 3
    assert ModelCompanionWeightVersionUtils.weight_version(resumed) == 3
    expected = saved if resume_dir else model()
    assert torch.equal(resumed[0].model_companion.sample_consumptions, expected[0].model_companion.sample_consumptions)


def test_lora_load_dir_resumes_the_latest_fully_saved_iteration(tmp_path):
    run = tmp_path / "ckpt"
    for i in (0, 1):
        (run / f"iter_{i:07d}" / "adapter").mkdir(parents=True)
    (run / "rollout" / "0").mkdir(parents=True)  # iteration 1's rollout state was never written
    args = Namespace(
        load=str(run),
        ref_load="/base",
        ref_ckpt_step=None,
        lora_rank=8,
        lora_adapter_path="/seed-adapter",
        megatron_to_hf_mode="raw",
        no_load_optim=False,
        start_rollout_id=None,
    )
    resolve_args_checkpoint_load(args)
    assert args.lora_adapter_path == str(run / "iter_0000000" / "adapter")
    assert (args.load, args.lora_resume_dir, args.no_load_optim, args.start_rollout_id) == (
        "/base",
        str(run),
        False,
        None,
    )
