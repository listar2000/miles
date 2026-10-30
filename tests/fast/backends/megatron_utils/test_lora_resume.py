"""A LoRA run resumes from its own --load: adapter, optimizer and model companions [fork hook-006] [fork hook-015]."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.utils import restore_lora_adapter, save_lora_checkpoint
from miles.backends.megatron_utils.megatron_config import resolve_args_checkpoint_load
from miles.backends.training_utils.model_companion import ModelCompanion, ModelCompanionWeightVersionUtils
from miles.utils.types import SampleLineage


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
        ckpt_step=None,
        exit_on_missing_checkpoint=False,
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
