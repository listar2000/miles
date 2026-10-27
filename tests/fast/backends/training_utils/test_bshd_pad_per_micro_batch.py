"""[fork fix-015] bshd pads each micro-batch to its own longest sample, not the whole batch's longest."""

from argparse import Namespace
from types import SimpleNamespace

import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.metrics.log_utils import aggregate_forward_results
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles_plugins.models.deepseek_v4.ops import rope


def _parallel_state():
    trivial = GroupInfo(rank=0, size=1, group=None)
    return ParallelState(
        intra_dp=trivial,
        intra_dp_cp=trivial,
        cp=trivial,
        tp=trivial,
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=trivial,
    )


def _load_max_seq_lens(monkeypatch, total_lengths, *, micro_batch_size=1, micro_batch_indices=None):
    parallel_state = _parallel_state()
    rollout_data = {
        "tokens": [list(range(n)) for n in total_lengths],
        "loss_masks": [[1] for _ in total_lengths],
        "total_lengths": list(total_lengths),
        "response_lengths": [1 for _ in total_lengths],
    }
    if micro_batch_indices is not None:
        rollout_data["micro_batch_indices"] = micro_batch_indices
    args = Namespace(
        enable_witness=False,
        qkv_format="bshd",
        data_pad_size_multiplier=16,
        compress_ratios=[],
        micro_batch_size=micro_batch_size,
    )
    monkeypatch.setattr(data_utils, "process_rollout_data", lambda *a, **k: (rollout_data, object()))
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    return data_utils.get_rollout_data(args, object())[0]["max_seq_lens"]


def test_each_sample_is_padded_to_its_own_length(monkeypatch):
    assert _load_max_seq_lens(monkeypatch, [5, 30, 12, 17]) == [16, 32, 16, 32]


def test_samples_of_a_micro_batch_share_its_longest(monkeypatch):
    assert _load_max_seq_lens(monkeypatch, [5, 30, 12, 7, 3], micro_batch_size=2) == [32, 32, 16, 16, 16]


def test_explicit_micro_batch_indices_set_the_groups(monkeypatch):
    lens = _load_max_seq_lens(monkeypatch, [5, 30, 12, 40], micro_batch_indices=[[0, 2], [1, 3]])
    assert lens == [16, 48, 16, 48]


def _visit_order(monkeypatch, total_lengths, *, global_batch_size, micro_batch_size=1, **extra):
    monkeypatch.setattr(data_utils, "get_parallel_state", _parallel_state)
    rollout_data = {"tokens": [[0] * n for n in total_lengths], "total_lengths": list(total_lengths), **extra}
    args = Namespace(
        qkv_format="bshd",
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        use_dynamic_global_batch_size=False,
        use_dynamic_batch_size=False,
    )
    iterators, num_microbatches = data_utils.get_data_iterator(args, None, rollout_data)
    order = [iterators[0].get_next(["total_lengths"])["total_lengths"] for _ in range(sum(num_microbatches))]
    return order, iterators[0]


def test_each_step_visits_its_longest_micro_batch_first(monkeypatch):
    # two steps of four samples; samples never move between steps
    order, _ = _visit_order(monkeypatch, [5, 30, 12, 17, 40, 1, 8, 9], global_batch_size=4)
    assert order == [[30], [17], [12], [5], [40], [9], [8], [1]]


def test_micro_batches_are_ordered_by_their_longest_sample(monkeypatch):
    order, _ = _visit_order(monkeypatch, [5, 30, 12, 7, 31, 3], global_batch_size=6, micro_batch_size=2)
    assert order == [[31, 3], [5, 30], [12, 7]]


def test_rollout_side_micro_batches_are_reordered_within_each_step(monkeypatch):
    order, _ = _visit_order(
        monkeypatch,
        [5, 30, 12, 40],
        global_batch_size=4,
        micro_batch_indices=[[0, 2], [1], [3]],
        num_microbatches=[2, 1],
    )
    assert order == [[30], [5, 12], [40]]


def test_forward_results_come_back_in_sample_order(monkeypatch):
    lengths = [5, 30, 12, 17]
    order, iterator = _visit_order(monkeypatch, lengths, global_batch_size=4)
    store = [{"log_probs": [f"logp-of-{n}" for n in batch]} for batch in order]
    args = Namespace(use_dynamic_batch_size=False)
    assert aggregate_forward_results(store, iterator, args)["log_probs"] == [f"logp-of-{n}" for n in lengths]


def test_rope_table_does_not_depend_on_the_requested_length():
    config = SimpleNamespace(
        original_max_position_embeddings=65536, rotary_scaling_factor=16, beta_fast=32, beta_slow=1
    )
    rope.precompute_freqs_cis.cache_clear()
    longest = rope.wrapped_precompute_freqs_cis(config, 64, 10000.0, False, 3000, None)
    for n in (1, 1000, 1024, 1025, 2999):
        table = rope.wrapped_precompute_freqs_cis(config, 64, 10000.0, False, n, None)
        assert table.shape[0] == n
        assert torch.equal(table, longest[:n])
        assert torch.equal(table, rope.precompute_freqs_cis.__wrapped__(64, n, 65536, 10000.0, 16, 32, 1))
    # one table per power of two: 4096 (3000, 2999), 2048 (1025), 1024 (1000, 1024), 1
    assert rope.precompute_freqs_cis.cache_info().currsize == 4
