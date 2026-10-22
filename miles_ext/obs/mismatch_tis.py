"""Miles' vanilla TIS (same weights, same loss) plus train/rollout mismatch metrics, logged as `train/<name>`.

Use with `--custom-tis-function-path miles_ext.obs.mismatch_tis.vanilla_tis_with_mismatch_metrics`. Per token,
with r = pi / mu (trainer over engine):

- `mismatch_k1`: log mu - log pi, the k1 estimate of KL(mu || pi).
- `mismatch_k3`: r - 1 - log r, the k3 estimate of KL(mu || pi) (non-negative, lower variance).
- `tis_clip_high_frac` / `tis_clip_low_frac`: r above `--tis-clip` / below `--tis-clip-low`; they sum to `tis_clipfrac`.
- `logprob_absdiff_gt_{0.5,1,2}`: |log pi - log mu| above each threshold.

`mis_with_mask_health` runs Miles' MIS (examples/.../mis.py) unchanged and adds metrics that show whether its ratio
band masks the tokens whose probability actually moved (prime-rl docs/ipo.md): a ratio band masks a rare token going
0.01 -> 0.06 (ratio 6) and keeps a likely one going 0.9 -> 0.2 (ratio 0.22). "Masked" is a loss token that MIS's
returned mask drops (TIS mask, RS and veto alike). With dp = |pi - mu|, each value is reduced like every other loss
metric (per-sample mean by default, per-token mean with --calculate-per-token-loss):

- `mis_prob_absdiff`: dp. `mis_masked_prob_absdiff` / `mis_masked_logprob_absdiff`: dp and |log pi - log mu| on
  masked tokens, 0 elsewhere.
- `mis_prob_absdiff_gt_0.3`: what IPO at its default eps 0.3 would mask.
- `mis_kept_prob_absdiff_gt_0.3`: kept by MIS although dp > 0.3 (drift the band misses).
- `mis_masked_prob_absdiff_lt_0.05`: masked by MIS although dp < 0.05 (rare tokens that barely moved).
"""

from argparse import Namespace
from typing import Any

import torch

from miles.backends.training_utils.loss.hub.corrections import vanilla_tis_function

ABSDIFF_THRESHOLDS = (0.5, 1.0, 2.0)
PROB_MOVED, PROB_STILL = 0.3, 0.05


def mismatch_metrics(args: Namespace, train_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor) -> dict:
    log_ratio = (train_log_probs - rollout_log_probs).detach()  # log pi - log mu
    ratio = torch.exp(log_ratio)
    metrics = {
        "mismatch_k1": -log_ratio,
        "mismatch_k3": ratio - 1 - log_ratio,
        "tis_clip_high_frac": (ratio > args.tis_clip).float(),
        "tis_clip_low_frac": (ratio < args.tis_clip_low).float(),
    }
    absdiff = log_ratio.abs()
    for threshold in ABSDIFF_THRESHOLDS:
        metrics[f"logprob_absdiff_gt_{threshold:g}"] = (absdiff > threshold).float()
    return metrics


def vanilla_tis_with_mismatch_metrics(
    args: Namespace,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    pg_loss, loss_masks, metrics = vanilla_tis_function(
        args,
        pg_loss=pg_loss,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
        **kwargs,
    )
    metrics |= mismatch_metrics(args, torch.cat(train_log_probs, dim=0), torch.cat(rollout_log_probs, dim=0))
    return pg_loss, loss_masks, metrics


def mis_mask_health(train_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor, masked: torch.Tensor) -> dict:
    log_ratio = (train_log_probs - rollout_log_probs).detach()
    prob_absdiff = (train_log_probs.exp() - rollout_log_probs.exp()).abs().detach()
    moved = (prob_absdiff > PROB_MOVED).float()
    return {
        "mis_prob_absdiff": prob_absdiff,
        "mis_masked_prob_absdiff": prob_absdiff * masked,
        "mis_masked_logprob_absdiff": log_ratio.abs() * masked,
        f"mis_prob_absdiff_gt_{PROB_MOVED:g}": moved,
        f"mis_kept_prob_absdiff_gt_{PROB_MOVED:g}": moved * (1 - masked),
        f"mis_masked_prob_absdiff_lt_{PROB_STILL:g}": (prob_absdiff < PROB_STILL).float() * masked,
    }


def mis_with_mask_health(
    args: Namespace,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    from examples.infra_features.train_infer_mismatch_helper.mis import compute_mis_weights_with_cp

    from miles.backends.training_utils.data.context_parallel import slice_log_prob_with_cp

    pg_loss, modified_masks, metrics = compute_mis_weights_with_cp(
        args,
        pg_loss=pg_loss,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        **kwargs,
    )
    max_seq_lens = kwargs.get("max_seq_lens")
    masked = torch.cat(
        [
            slice_log_prob_with_cp(
                loss_mask.float() * (1 - modified_mask.float()),
                total_lengths[i],
                response_lengths[i],
                args.qkv_format,
                max_seq_lens[i] if max_seq_lens is not None else None,
            )
            for i, (loss_mask, modified_mask) in enumerate(zip(loss_masks, modified_masks, strict=True))
        ]
    )
    metrics |= mis_mask_health(torch.cat(train_log_probs, dim=0), torch.cat(rollout_log_probs, dim=0), masked)
    return pg_loss, modified_masks, metrics
