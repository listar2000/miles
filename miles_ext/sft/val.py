"""``SftTrainRayActor`` (``--custom-trainer-actor-class``): validation rollout ids run forward-only and log
``val/nll``, the mean NLL over trained tokens, with ``val/tokens`` and ``val/samples`` at ``train/step``."""

import logging

from miles_ext.sft.rollout import VAL_BASE, is_val

from miles.backends.megatron_utils.actor import MegatronTrainRayActor

logger = logging.getLogger(__name__)


class SftTrainRayActor(MegatronTrainRayActor):
    def train(self, rollout_id, rollout_data_ref, witness_info=None, attempt=0, external_data=None):
        if is_val(rollout_id):
            return self._val_nll(rollout_id, rollout_data_ref, witness_info)
        return super().train(
            rollout_id, rollout_data_ref, witness_info=witness_info, attempt=attempt, external_data=external_data
        )

    def _val_nll(self, rollout_id: int, rollout_data_ref, witness_info=None):
        import torch
        import torch.distributed as dist

        from miles.backends.megatron_utils.model import forward_only
        from miles.backends.training_utils.data.context_parallel import _slice_loss_mask_for_local_cp
        from miles.backends.training_utils.data.rollout import get_data_iterator, get_rollout_data
        from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
        from miles.backends.training_utils.parallel import get_parallel_state
        from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
        from miles.utils.tracking_utils import tracking

        args = self.args
        torch.cuda.empty_cache()  # the full-vocab logits of the forward need memory training leaves fragmented
        rollout_data, store_get_result = get_rollout_data(
            args=args,
            rollout_data_ref=rollout_data_ref,
            witness_info=witness_info,
            train_parallel_config=get_parallel_state().train_parallel_config(supports_precomputed_schedule=True),
        )
        with store_get_result:
            data_iterator, num_microbatches = get_data_iterator(args, self.model, rollout_data)
            out = forward_only(
                get_log_probs_and_entropy,
                args,
                self.model,
                data_iterator,
                num_microbatches,
                rollout_id=rollout_id,
                store_prefix="val_",
                fp32_output=False,
            )
            totals = torch.zeros(3, dtype=torch.float64, device=torch.cuda.current_device())
            if get_parallel_state().is_pp_last_stage:
                max_seq_lens = rollout_data.get("max_seq_lens")
                for i, log_probs in enumerate(out["val_log_probs"]):
                    if get_parallel_state().cp.size > 1:
                        mask = _slice_loss_mask_for_local_cp(
                            rollout_data["total_lengths"][i],
                            rollout_data["response_lengths"][i],
                            rollout_data["loss_masks"][i],
                            args.qkv_format,
                            max_seq_lens[i] if max_seq_lens else None,
                        )
                    else:  # without CP every rank holds the whole response
                        mask = torch.as_tensor(rollout_data["loss_masks"][i])
                    assert mask.numel() == log_probs.numel(), (i, mask.numel(), log_probs.numel())
                    mask = mask.to(log_probs.device, torch.float64)
                    totals[0] -= (log_probs.double() * mask).sum()
                    totals[1] += mask.sum()
                    if get_parallel_state().cp.rank == 0 and sum(rollout_data["loss_masks"][i]) > 0:  # not padding
                        totals[2] += 1
            dist.all_reduce(totals)  # tp ranks repeat the same sums; divided out below
            tp = get_parallel_state().tp.size
            nll, tokens, samples = (totals / tp).tolist()
            step = rollout_id - VAL_BASE
            if dist.get_rank() == 0:
                _define_val_metric()
                metrics = {
                    "val/nll": nll / max(tokens, 1),
                    "val/tokens": tokens,
                    "val/samples": samples,
                    "train/step": step,
                }
                tracking.log(args, metrics, step_key="train/step")
                logger.info(f"validation at step {step}: {metrics}")
        return TrainStepOutput(outcome=TrainStepOutcome.NORMAL)


def _define_val_metric():
    """Miles defines step metrics for train/, rollout/, eval/ and perf/ only (wandb_utils)."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.define_metric("val/*", step_metric="train/step")
    except ImportError:
        pass
