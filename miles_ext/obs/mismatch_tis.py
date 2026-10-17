"""Miles' vanilla TIS (same weights, same loss) plus train/rollout mismatch metrics, logged as `train/<name>`.

Use with `--custom-tis-function-path miles_ext.obs.mismatch_tis.vanilla_tis_with_mismatch_metrics`. Per token,
with r = pi / mu (trainer over engine):

- `mismatch_k1`: log mu - log pi, the k1 estimate of KL(mu || pi).
- `mismatch_k3`: r - 1 - log r, the k3 estimate of KL(mu || pi) (non-negative, lower variance).
- `tis_clip_high_frac` / `tis_clip_low_frac`: r above `--tis-clip` / below `--tis-clip-low`; they sum to `tis_clipfrac`.
- `logprob_absdiff_gt_{0.5,1,2}`: |log pi - log mu| above each threshold.
"""

from argparse import Namespace
from typing import Any

import torch

from miles.backends.training_utils.loss.hub.corrections import vanilla_tis_function

ABSDIFF_THRESHOLDS = (0.5, 1.0, 2.0)


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
