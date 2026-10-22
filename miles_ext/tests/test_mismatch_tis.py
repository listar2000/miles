from argparse import Namespace

import pytest
import torch
from miles_ext.obs.mismatch_tis import mis_mask_health, mis_with_mask_health, vanilla_tis_with_mismatch_metrics

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


def test_mis_mask_health_ipo_cases():
    # trainer vs engine probability: rare 0.005 -> 0.04 (ratio 8), likely 0.2 -> 0.9 (ratio 0.22), unchanged
    train = torch.tensor([0.04, 0.2, 0.5]).log()
    rollout = torch.tensor([0.005, 0.9, 0.5]).log()
    m = mis_mask_health(train, rollout, masked=torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(m["mis_prob_absdiff"], torch.tensor([0.035, 0.7, 0.0]))
    assert torch.allclose(m["mis_masked_prob_absdiff"], torch.tensor([0.035, 0.0, 0.0]))
    assert torch.allclose(
        m["mis_masked_logprob_absdiff"], torch.tensor([8.0, 1.0, 1.0]).log() * torch.tensor([1.0, 0, 0])
    )
    assert m["mis_prob_absdiff_gt_0.3"].tolist() == [0.0, 1.0, 0.0]
    assert m["mis_kept_prob_absdiff_gt_0.3"].tolist() == [0.0, 1.0, 0.0]
    assert m["mis_masked_prob_absdiff_lt_0.05"].tolist() == [1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "config",
    [
        dict(use_tis=True, tis_mode="mask", use_rs=False, tis_lower_bound=0.5, rs_veto_threshold=None),
        dict(use_tis=True, tis_mode="truncate", use_rs=True, tis_lower_bound=0.0, rs_veto_threshold=1e-4),
        dict(use_tis=False, tis_mode="mask", use_rs=True, tis_lower_bound=0.5, rs_veto_threshold=None),
    ],
)
def test_masked_matches_upstream_mis(monkeypatch, config):
    from examples.infra_features.train_infer_mismatch_helper.mis import compute_mis_weights

    import miles.backends.training_utils.data.context_parallel as cp

    monkeypatch.setattr(cp, "get_parallel_state", lambda: Namespace(cp=Namespace(size=1, rank=0, group=None)))
    args = Namespace(
        tis_upper_bound=2.0,
        tis_level="token",
        rs_level="token",
        rs_lower_bound=None,
        rs_upper_bound=None,
        tis_batch_normalize=False,
        qkv_format="thd",
        **config,
    )
    gen = torch.Generator().manual_seed(0)
    lengths = [64, 32]
    train = [-torch.rand(n, generator=gen) * 3 for n in lengths]
    rollout = [t + torch.randn(n, generator=gen) * 0.8 for t, n in zip(train, lengths, strict=True)]
    masks = [torch.ones(64), (torch.arange(32) % 5 != 0).float()]
    _, upstream_masks, _ = compute_mis_weights(
        args, train_log_probs=train, rollout_log_probs=rollout, loss_masks=masks
    )
    _, _, metrics = mis_with_mask_health(
        args,
        pg_loss=torch.ones(96),
        train_log_probs=train,
        rollout_log_probs=rollout,
        loss_masks=masks,
        total_lengths=[n + 8 for n in lengths],
        response_lengths=lengths,
    )
    expected = torch.cat([m * (1 - u) for m, u in zip(masks, upstream_masks, strict=True)])
    assert torch.equal((metrics["mis_masked_logprob_absdiff"] > 0).float(), expected)
