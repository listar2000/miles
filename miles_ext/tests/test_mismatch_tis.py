from argparse import Namespace

import torch
from miles_ext.obs.mismatch_tis import vanilla_tis_with_mismatch_metrics

from miles.backends.training_utils.loss.hub.corrections import vanilla_tis_function

ARGS = Namespace(tis_clip=2.0, tis_clip_low=0.5)


def _inputs():
    # log pi - log mu per token: 0, +1 (r=e>2), -1 (r=1/e<0.5), +0.25
    train = [torch.tensor([-1.0, -0.5]), torch.tensor([-2.0, -0.75])]
    rollout = [torch.tensor([-1.0, -1.5]), torch.tensor([-1.0, -1.0])]
    masks = [torch.ones(2), torch.ones(2)]
    return dict(pg_loss=torch.ones(4), train_log_probs=train, rollout_log_probs=rollout, loss_masks=masks)


def test_loss_identical_to_vanilla():
    base_loss, base_masks, base_metrics = vanilla_tis_function(ARGS, **_inputs())
    loss, masks, metrics = vanilla_tis_with_mismatch_metrics(ARGS, **_inputs())
    assert torch.equal(loss, base_loss) and all(torch.equal(a, b) for a, b in zip(masks, base_masks, strict=True))
    for key, value in base_metrics.items():
        assert torch.equal(metrics[key], value)


def test_mismatch_metrics_values():
    _, _, m = vanilla_tis_with_mismatch_metrics(ARGS, **_inputs())
    log_ratio = torch.tensor([0.0, 1.0, -1.0, 0.25])
    assert torch.allclose(m["mismatch_k1"], -log_ratio)
    assert torch.allclose(m["mismatch_k3"], log_ratio.exp() - 1 - log_ratio)
    assert m["tis_clip_high_frac"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert m["tis_clip_low_frac"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert torch.equal(m["tis_clip_high_frac"] + m["tis_clip_low_frac"], m["tis_clipfrac"])
    assert m["logprob_absdiff_gt_0.5"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert m["logprob_absdiff_gt_2"].tolist() == [0.0, 0.0, 0.0, 0.0]
