# RFC 0003: Connectors from other data formats to SFT records, with Harbor ATIF first

Status: accepted 2026-11-01; implemented in `miles_ext/sft/connectors/` (ext-049).

## What is decided

From the user (2026-11-01): SFT data arrives in many formats. A **connector** turns one unit
of a source format into SFT records ([RFC 0002](0002-tito-aware-sft-tokenization.md) "The
record"), so the tokenizer stays the only reader of the record. Harbor's Agent Trajectory
Interchange Format (ATIF) is the first connector and the reference for later ones, because it
will be a common source for this fork and its spec already covers context management and
subagents. This RFC is CPU only. Launching SFT on Modal is separate work.

Decisions:

1. **The ATIF spec is the contract.** A trajectory is taken as given. How a harness (Claude
   Code, Codex, Terminus-2) becomes ATIF is Harbor's job, and agent text is used as written.
   Producer-specific `extra` fields are ignored; the spec's own `extra.context_management`
   convention is the one exception.
2. **The unit of a Harbor source is a trial directory**, because continuations and subagents
   live in their own files beside `agent/trajectory.json`.
3. **Connectors write records to JSONL** in a separate step. `miles_ext.sft.prepare` reads
   that file as before, so records stay inspectable and diffable.
4. **No filtering in connectors.** A trajectory given to a connector is SFT data as it is.
   Choosing which trials to train on (by reward, task, length) is a separate concern.
5. **Contexts the spec does not determine are refused**, never guessed, and counted by reason.
6. **Subagents become their own records**, included by default.
7. **Render settings stay per record** (`chat_template_kwargs` is a connector option).
   Per-step values such as ATIF `reasoning_effort` are not mapped yet.

**Why this lives in the fork.** The fork keeps data preparation out (`fork/guide/help.md`), and
connectors are an exception the owner approved on 2026-11-01, with this scope: a connector
only translates a source format's sessions into the record `miles_ext.sft` trains on, deciding
each model call's context and what is trained. That is part of the training data contract
(RFC 0002's layer 2), so it is owned next to the record format and tested against it.
Choosing, cleaning or generating data (selection by reward, dataset builds, rollouts to make
data) stays outside the fork.

## Connector interface

A connector is a function from one source unit to an iterator of items. An item is either a
record dict or a `Dropped(source, reason, detail)`. `miles_ext.sft.connectors.base.write_records`
takes those items, validates every record with `parse_record`, and writes:

- `<output>.jsonl`: one record per line;
- `<output>.report.json`: records written, and drops counted by reason with a few examples.

A record that fails `parse_record` is a connector bug; it is reported as
`invalid_record` and never written. A new connector is one module with its own `main()` that
calls `write_records`, plus tests over real source files.

## ATIF to records

Run `python -m miles_ext.sft.connectors.atif --input <trial dir or job dir> --output records.jsonl`.
A trial directory is one with `agent/trajectory.json`; a job directory is searched for them.
Multi-stage tasks also have `steps/<name>/agent/trajectory.json`, and each is a separate main
trajectory of the same trial. Other files are reached only through references, so a
continuation or summarization file is never read as a main trajectory.

### Steps to messages

ATIF steps are walked in order, keeping the message store and the current context (the
messages the model would receive at its next call).

| ATIF | Record |
| --- | --- |
| `source: user` | a `user` message, added to the context |
| `source: system` without `context_management` | a `system` message (when the text is not empty), added to the context. Its observation records an event, not model input, and is not added |
| `source: system` with `context_management` | changes the context (below); its own `message` is a log line and is not added |
| `source: agent` | an `assistant` message: `message` to `content`, `reasoning_content`, `tool_calls` (`tool_call_id` to `id`, the `arguments` object to a JSON string). Added to the context after its step is recorded |
| agent `observation.results[]` with `source_call_id` | a `tool` message with that `tool_call_id`, in the order of `results` |
| agent `observation.results[]` without `source_call_id` | a `user` message (harnesses that talk to the model in text) |
| `agent.tool_definitions` | `tools` |
| `metrics.completion_token_ids` | `completion_token_ids` on the assistant message, only with `--token-ids` |
| content as `ContentPart`s | text parts concatenated as they are; an image or audio part refuses the record (`multimodal`) |

Tool calls without a result are kept as they are; the spec does not require a result per call.
A result whose `source_call_id` names no tool call of its step refuses the record.

### Which agent steps are trained

An agent step becomes a record step with its current context and its reply.

- `is_copied_context: true`: the message is context only and is never a completion. ATIF says
  such steps MUST NOT be SFT data; they were trained where they were produced. No record step
  is made for it.
- `llm_call_count: 0` (a deterministic dispatch, which the spec says SFT MUST filter): context
  only, no record step.
- `llm_call_count > 1`: one step aggregates several calls whose contexts cannot be told apart,
  so the record is refused (`aggregated_step`).
- A step by another model: a record step with `train: false`, so it is context the model saw
  but is not trained. The model trained on is `--model`, by default the main trajectory's
  `agent.model_name`; a step's model is its `model_name`, else its trajectory's
  `agent.model_name`. Names are compared by their last `/`-separated part, so `gpt-4o` and
  `openai/gpt-4o` match. Without any model name, every step is trained.

When every assistant message is a trained step whose context is everything before it, the
record has no `steps` and uses RFC 0002's append-only shorthand.

`completion_token_ids` are tokens of the model that produced the trajectory. They are copied
only with `--token-ids`, which asserts that this model has the same tokenizer as the model being
trained; wrong ids would train silently wrong tokens.

### Context management

The spec's normative rule, applied literally:

- `boundary: replace`: the context becomes the boundary step's `observation.results[].content`
  (each non-empty one as a `user` message) followed by every later step. Earlier messages stay
  in the store, and replies before the boundary are trained in their own samples. When the
  boundary step has no content, the context restarts empty. ATIF does not say which role the
  summary has; `user` is what compacting agents commonly send.
- `boundary: append`: the content is added to the current context as `user` messages.
- `boundary: truncate`, a missing boundary, or an unknown one: the spec does not say what the
  context became, so the record is refused (`context_undetermined`).
- `continued_trajectory_ref`: the continuation file (relative to the trajectory's directory) is
  read and its steps follow in the same record. A continuation starts with the context it
  carried over, marked `is_copied_context`. A missing continuation refuses the record
  (`missing_continuation`): the rest of the session is unknown.

### Subagents

A `subagent_trajectory_ref` in any observation result is resolved as the spec says: by
`trajectory_id` against `subagent_trajectories` of the trajectory that holds the reference
(then its parents'), or by `trajectory_path` relative to that trajectory's directory. Each
subagent becomes its own record, with its own tools and continuations, and its own subagents in
turn. The parent record is unchanged: it holds only the tool result the parent saw. A reference
that cannot be resolved drops only that subagent (`missing_subagent`, or `unresolvable_subagent`
for a reference with only `session_id`). A file referenced twice is converted once.

### Metadata

`metadata` holds `source: "atif"`, the trial directory, the trajectory (its file relative to the
trial, or `#<trajectory_id>` when embedded), `agent`, `agent_version`, `model`, and for a
subagent `parent` (the parent trajectory). It never affects tokens.

## Known gaps on the Harbor side

Taking ATIF as given means some harness behavior comes through as written. When we produce
SFT data ourselves with `harbor run`, these need a Harbor change, not a connector change:

- Terminus-2's compaction boundary step carries no summary content (the summary lives in its
  three summarization subagents), so the spec rule restarts the context empty plus the handoff.
- Terminus-2 without `trajectory_config.raw_content` writes "Analysis/Plan" text rebuilt from
  its parsed reply, not the reply itself.
- The Claude Code converter puts subagent steps in the main trajectory, marked only by
  `extra.is_sidechain` and `extra.agent_id`; by the spec they are main-trajectory steps.

## Tests

`miles_ext/tests/test_sft_connector_atif.py`, CPU only, over real trial-directory layouts:

- Harbor's own trajectories, copied with their source commit into
  `miles_ext/tests/fixtures/atif/` (Apache-2.0): Terminus-2 goldens for context
  summarization (a `replace` boundary and three file-ref subagents), linear history (a
  continuation with copied context), an invalid-JSON reply with reasoning, a timeout, and a
  Claude Code example with a structured tool call.
- Small trial directories written by the tests for each rule above: parallel tool calls,
  text observations, `append` and `truncate` boundaries, embedded and nested subagents,
  missing references, other-model steps, `llm_call_count`, multimodal content, and unknown
  `source_call_id`.
- Every written record passes `parse_record`, and the records from the Harbor fixtures go
  through `prepare` with the offline Qwen3 tokenizer of the RFC 0002 tests.

## What it conflicts with

Nothing in upstream Miles. It adds files under `miles_ext/` only and reads records with
RFC 0002's `parse_record`, which stays the owner of the record format.

## Order

1. ext-049: connector base, the ATIF connector, fixtures and tests (this RFC).
2. Later, each its own change: more connectors (Nemotron-style datasets, Miles rollout dumps
   with `completion_token_ids`), selecting trials, per-step render settings if real data
   needs them, and the Harbor-side fixes above for data we produce.
