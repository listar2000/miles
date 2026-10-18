import logging
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from miles.backends.megatron_utils import initialize
from miles.backends.training_utils.parallel import GroupInfo, ParallelState


def _parallel_state(
    *,
    tp_rank: int = 0,
    pp_rank: int = 0,
    pp_size: int = 1,
    intra_dp_rank: int = 0,
    intra_dp_size: int = 1,
    intra_dp_cp_rank: int = 0,
    intra_dp_cp_size: int = 1,
    indep_dp_rank: int = 0,
    indep_dp_size: int = 1,
) -> ParallelState:
    intra_dp = GroupInfo(rank=intra_dp_rank, size=intra_dp_size, group=None)
    indep_dp = GroupInfo(rank=indep_dp_rank, size=indep_dp_size, group=None)
    return ParallelState(
        intra_dp=intra_dp,
        intra_dp_cp=GroupInfo(rank=intra_dp_cp_rank, size=intra_dp_cp_size, group=None),
        cp=GroupInfo(rank=0, size=1, group=None),
        tp=GroupInfo(rank=tp_rank, size=max(tp_rank + 1, 1), group=None),
        pp=GroupInfo(rank=pp_rank, size=pp_size, group=None),
        ep=GroupInfo(rank=0, size=1, group=None),
        etp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=indep_dp,
    )


def _patch_state(monkeypatch, state: ParallelState) -> None:
    monkeypatch.setattr(initialize, "get_parallel_state", lambda: state)


class TestLocalReplicaMainRank:
    def test_each_ft_cell_has_one_local_representative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Independent-DP alive rank does not suppress another cell's local representative."""
        _patch_state(monkeypatch, _parallel_state(indep_dp_rank=2, indep_dp_size=3))

        assert initialize.is_local_replica_megatron_main_rank()

    def test_nonzero_intra_dp_rank_is_not_a_local_representative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ordinary data-parallel copies emit only one snapshot per cell."""
        _patch_state(monkeypatch, _parallel_state(intra_dp_rank=1, intra_dp_size=2))

        assert not initialize.is_local_replica_megatron_main_rank()


class TestSetRandomSeedFromArgs:
    def test_it_reseeds_rngs_and_only_rank_zero_logs(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Repeated setup reproduces CPU RNG streams while only rank zero announces the seed."""
        _patch_state(monkeypatch, _parallel_state())
        monkeypatch.setattr(initialize.tensor_parallel, "model_parallel_cuda_manual_seed", lambda *args: None)
        args = SimpleNamespace(
            rank=0,
            seed=1729,
            data_parallel_random_init=True,
            te_rng_tracker=False,
            inference_rng_tracker=True,
        )

        with caplog.at_level(logging.INFO, logger=initialize.__name__):
            initialize.set_random_seed_from_args(args)
            first_values = (random.random(), np.random.random(), torch.rand(1).item())
            args.rank = 1
            initialize.set_random_seed_from_args(args)
            second_values = (random.random(), np.random.random(), torch.rand(1).item())

        assert second_values == first_values
        assert [record.message for record in caplog.records] == ["> setting random seeds to 1729 ..."]


def test_intra_mode_first_inner_rank_is_main_rank(monkeypatch):
    """INTRA mode: last pp stage, tp rank 0, intra_dp_cp rank 0 is the main rank."""
    _patch_state(monkeypatch, _parallel_state(pp_rank=1, pp_size=2, intra_dp_cp_rank=0, intra_dp_cp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is True


def test_intra_mode_non_first_inner_rank_is_not_main_rank(monkeypatch):
    """INTRA mode: a non-zero intra_dp_cp rank is not the main rank."""
    _patch_state(monkeypatch, _parallel_state(pp_rank=1, pp_size=2, intra_dp_cp_rank=1, intra_dp_cp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is False


def test_indep_mode_first_replica_first_inner_is_main_rank(monkeypatch):
    """INDEP mode: first replica plus first inner rank yields the main rank."""
    _patch_state(
        monkeypatch,
        _parallel_state(
            pp_rank=1, pp_size=2, intra_dp_cp_rank=0, intra_dp_cp_size=2, indep_dp_rank=0, indep_dp_size=2
        ),
    )
    assert initialize.is_first_replica_megatron_main_rank() is True


def test_indep_mode_second_replica_main_rank_is_not_first_replica(monkeypatch):
    """INDEP regression: second replica's main rank is not the first-replica main rank."""
    _patch_state(
        monkeypatch,
        _parallel_state(
            pp_rank=1, pp_size=2, intra_dp_cp_rank=0, intra_dp_cp_size=2, indep_dp_rank=1, indep_dp_size=2
        ),
    )
    assert initialize.is_first_replica_megatron_main_rank() is False


def test_non_zero_tp_rank_is_not_main_rank(monkeypatch):
    """A non-zero tp rank is never the main rank even on the first replica's first inner rank."""
    _patch_state(monkeypatch, _parallel_state(tp_rank=1, pp_rank=1, pp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is False


def test_non_last_pp_stage_is_not_main_rank(monkeypatch):
    """A pp rank that is not the last pipeline stage is never the main rank."""
    _patch_state(monkeypatch, _parallel_state(tp_rank=0, pp_rank=0, pp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is False


# [fork fix-011] begin
class _RoutedLayer(torch.nn.Module):
    """Keeps a tensor from its forward on the module, like the alltoall dispatcher's self.probs."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.proj = torch.nn.Linear(width, width, bias=False)
        self.router = torch.nn.Linear(width, 4, bias=False)
        self.probs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x[..., : self.width].contiguous()
        self.probs = torch.softmax(self.router(hidden), dim=-1)
        out = self.proj(hidden) * self.probs[..., :1]
        return x + torch.nn.functional.pad(out, (0, x.shape[-1] - self.width))


class TestReleaseRecomputeInputs:
    def _train(self, monkeypatch: pytest.MonkeyPatch, release: bool) -> tuple[torch.Tensor, int]:
        from megatron.core.tensor_parallel import random as tp_random

        monkeypatch.setattr(tp_random, "_get_all_rng_states", lambda: (None, None, None))
        monkeypatch.setattr(tp_random, "_set_all_rng_states", lambda *args, **kwargs: None)
        original = tp_random.CheckpointFunction.__dict__["backward"]
        width, rows = 64, 4096
        try:
            if release:
                initialize._release_recompute_inputs()
            torch.manual_seed(0)
            layers = torch.nn.ModuleList(_RoutedLayer(width) for _ in range(4))
            for _ in range(2):
                hidden = torch.randn(rows, 1, width + 3, requires_grad=True)
                for layer in layers:
                    hidden = tp_random.checkpoint(layer, False, hidden)
                hidden.sum().backward()
                del hidden
        finally:
            tp_random.CheckpointFunction.backward = original
        alive = {
            t.untyped_storage().data_ptr()
            for layer in layers
            for t in self._graph_leaves(layer.probs)
            if t.numel() >= rows * width
        }
        grads = torch.cat([p.grad.flatten() for p in layers.parameters()])
        return grads, len(alive)

    @staticmethod
    def _graph_leaves(tensor: torch.Tensor) -> list[torch.Tensor]:
        leaves, stack, seen = [], [tensor.grad_fn], set()
        while stack:
            node = stack.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            if hasattr(node, "variable"):
                leaves.extend([node.variable] + ([node.variable.grad] if node.variable.grad is not None else []))
            stack.extend(next_node for next_node, _ in node.next_functions)
        return leaves

    def test_release_frees_recompute_inputs_and_keeps_grads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Module-held recompute outputs no longer pin each layer's input and its grad; grads are unchanged."""
        base_grads, base_alive = self._train(monkeypatch, release=False)
        grads, alive = self._train(monkeypatch, release=True)

        assert base_alive == 8  # four layers, each keeping its input and the input's grad
        assert alive == 0
        assert torch.equal(grads, base_grads)

    def test_env_switch_leaves_megatron_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MILES_RECOMPUTE_RELEASE=0 keeps Megatron's own backward."""
        from megatron.core.tensor_parallel import random as tp_random

        monkeypatch.setenv("MILES_RECOMPUTE_RELEASE", "0")
        original = tp_random.CheckpointFunction.__dict__["backward"]
        initialize._release_recompute_inputs()

        assert tp_random.CheckpointFunction.__dict__["backward"] is original


# [fork fix-011] end
