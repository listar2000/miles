# ext-049: SFT record connectors, Harbor ATIF first

## Why
SFT data arrives as Harbor ATIF trajectories, HF datasets and rollout dumps, while
`miles_ext.sft` (ext-048) reads only its own record format. Harbor's own export
(`harbor traces export`) flattens ATIF into role/content text: reasoning becomes inline
`<think>`, tool calls become JSON text and system steps become user messages, so it loses
what the tokenizer needs. Design: [RFC 0003](../rfcs/0003-sft-record-connectors.md).

## Decision
- A connector yields record dicts or `Dropped` items; `write_records` validates every record
  with `parse_record` (still the one owner of the format), writes JSONL and a report of drops
  by reason. Connectors run as a separate step before `prepare`.
- The ATIF connector implements the spec only and takes trajectories as given. The unit is a
  Harbor trial directory. `is_copied_context` and `llm_call_count: 0` steps are context only;
  steps by another model are `train: false`; `llm_call_count > 1`, `boundary: truncate` or an
  unknown boundary, multimodal content and unknown `source_call_id`s refuse the record.
  `replace` boundaries follow the spec literally. Each subagent (embedded or file reference)
  is its own record.
- `completion_token_ids` are copied only with `--token-ids`: they are tokens of whatever model
  produced the trajectory.
- Rejected: reading `harbor traces export` output (lossy, above); per-harness context rules
  (they would make the connector depend on how each harness writes ATIF, which is Harbor's job);
  filtering by reward in the connector (selection is a separate concern).

## How it is applied
New files: `miles_ext/sft/connectors/` (`base.py`, `atif.py`),
`miles_ext/tests/test_sft_connector_atif.py`, and Harbor's own trajectories copied into
`miles_ext/tests/fixtures/atif/` with their source commit. Run
`python -m pytest miles_ext/tests/test_sft_connector_atif.py`; its last test runs `prepare` on the
records and needs the RFC 0002 test dependencies. On a Harbor bump, copy newer goldens from
Harbor's `tests/golden/terminus_2/` and rerun.

## Remove when
Upstream Miles reads ATIF into SFT data laid out like session-server rollouts.
