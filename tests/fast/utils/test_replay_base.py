from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["replay"])

import torch

from miles.utils.replay_base import BaseReplayManager


class _FakeReplay:
    def __init__(self, *top_indices):
        self.top_indices = list(top_indices)

    def pop_forward(self):
        return self.top_indices.pop(0)

    def pop_backward(self):
        return self.pop_forward()


def _topk(scores, topk):
    return torch.topk(scores, topk, dim=1).indices.to(torch.int32)


def _make_replay_manager(top_indices):
    manager = BaseReplayManager()
    manager.enable_check_replay_result = False
    manager.enabled = True
    manager.stage = "replay_forward"
    manager.set_current(_FakeReplay(top_indices))
    return manager


def test_get_topk_fn_fills_all_invalid_rows_with_arange():
    # an all-(-1) row is a masked/pad token; fill it with arange to avoid reading
    # invalid (-1) positions downstream
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    manager = _make_replay_manager(torch.tensor([[-1, -1, -1]], dtype=torch.int32))

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), torch.tensor([[0, 1, 2]], dtype=torch.int32))


def test_get_topk_fn_preserves_partial_padding():
    # a row with some valid picks keeps its -1 padding (only all-(-1) rows are filled)
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    replayed_top_indices = torch.tensor([[2, -1, -1]], dtype=torch.int32)
    manager = _make_replay_manager(replayed_top_indices)

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), replayed_top_indices)


def test_get_topk_fn_spreads_pad_rows_over_all_experts():
    # [fork fix-014] pad rows go round-robin over the experts, so no expert-parallel rank gets all of them
    num_experts, topk, num_ranks = 384, 6, 8
    scores = torch.zeros(4096, num_experts)
    real = torch.stack([torch.randperm(num_experts)[:topk] for _ in range(1024)]).to(torch.int32)
    replayed = torch.full((4096, topk), -1, dtype=torch.int32)
    replayed[:1024] = real
    manager = _make_replay_manager(replayed)

    top_indices = manager.get_topk_fn(_topk, return_probs=False)(scores, topk)

    torch.testing.assert_close(top_indices[:1024], real)
    pads = top_indices[1024:]
    assert all(len(set(row.tolist())) == topk for row in pads)
    per_rank = torch.bincount(pads.flatten().long() // (num_experts // num_ranks), minlength=num_ranks)
    assert per_rank.max() - per_rank.min() <= topk, per_rank
