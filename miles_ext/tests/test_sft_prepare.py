"""Offline SFT prepare, the shard-reading rollout function and the parity report (ext-048):

    python -m pytest miles_ext/tests/test_sft_prepare.py
"""

import json
from types import SimpleNamespace

from miles_ext.sft import prepare as prepare_module
from miles_ext.sft.parity import compare, full_loss_mask
from miles_ext.sft.postprocess import load_processors
from miles_ext.sft.render import SftRenderer
from miles_ext.sft.rollout import generate_rollout
from miles_ext.tests.test_sft_render import AGENT, CHAT, TOOLS, _qwen_tito

from miles.utils.types import Sample


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class _Buffer:
    def __init__(self, ids):
        self.ids = ids

    def get_samples(self, n):
        return [[Sample(prompt=str(i))] for i in self.ids[:n]]


def _prepare(tmp_path, rows, **kwargs):
    _write_jsonl(tmp_path / "in.jsonl", rows)
    out = tmp_path / "out"
    out.mkdir()
    counts = prepare_module.prepare(
        inputs=[str(tmp_path / "in.jsonl")],
        output=out,
        renderer=SftRenderer(_qwen_tito()),
        processors=load_processors(["miles_ext.sft.postprocess.normalize_openai_messages"]),
        max_tokens=kwargs.get("max_tokens"),
    )
    return out, counts


def test_prepared_samples_come_back_from_the_rollout_function_unchanged(tmp_path):
    split = {"messages": CHAT, "steps": [{"context": [0, 1], "completion": 2}, {"context": [0, 3], "completion": 4}]}
    rows = [{"messages": AGENT, "tools": TOOLS}, {"messages": [{"role": "user", "content": "x"}]}, split]
    out, counts = _prepare(tmp_path, rows)
    assert (counts.records, counts.records_skipped, counts.records_split, counts.samples) == (3, 1, 1, 3)
    assert counts.errors.most_common(1)[0][0].startswith("RecordError: no assistant")

    index = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    prompts = [json.loads(line)["text"] for line in (out / "prompts.jsonl").read_text().splitlines()]
    assert prompts == ["0", "1", "2"] and [row["source"].rsplit(":", 1)[1] for row in index] == ["1", "3", "3"]

    expected = SftRenderer(_qwen_tito()).render(prepare_module.parse_record({"messages": AGENT, "tools": TOOLS}))[0]
    args = SimpleNamespace(prompt_data=str(out / "prompts.jsonl"), rollout_batch_size=3)
    groups = generate_rollout(args, 0, _Buffer([0, 1, 2]))
    first = groups[0][0]
    assert first.tokens == expected.token_ids
    assert full_loss_mask(first.tokens, first.response_length, first.loss_mask) == expected.loss_mask
    assert first.reward == 0 and first.loss_mask[0] == 1
    assert sum(sum(g[0].loss_mask) for g in groups) == counts.trained_tokens


def test_samples_over_max_tokens_are_dropped_not_cut(tmp_path):
    _, counts = _prepare(tmp_path, [{"messages": AGENT, "tools": TOOLS}, {"messages": CHAT}], max_tokens=200)
    assert counts.samples_too_long == 1 and counts.samples == 1


def test_shards_roll_over(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_module, "SHARD_TOKENS", 150)
    out, counts = _prepare(tmp_path, [{"messages": CHAT}] * 3)
    assert counts.samples == 3 and len(list((out / "shards").glob("*.tokens.npy"))) == 3
    groups = generate_rollout(
        SimpleNamespace(prompt_data=str(out / "prompts.jsonl"), rollout_batch_size=3), 0, _Buffer([2])
    )
    assert len(groups[0][0].tokens) == json.loads((out / "samples.jsonl").read_text().splitlines()[2])["length"]


def test_parity_report_points_at_the_first_difference():
    tito = _qwen_tito()
    (rendered,) = SftRenderer(tito).render(prepare_module.parse_record({"messages": CHAT}))
    same = compare(rendered, served_tokens=rendered.token_ids, served_loss_mask=rendered.loss_mask)
    assert same.tokens_equal and same.mask_equal and same.first_token_diff is None
    served = list(rendered.token_ids)
    served[10] += 1
    diff = compare(
        rendered, served_tokens=served[:-1], served_loss_mask=rendered.loss_mask[:-1], tokenizer=tito.tokenizer
    )
    assert not diff.tokens_equal and diff.first_token_diff == 10 and diff.mask_diff_count == 1
    assert diff.rendered_window and diff.served_window
