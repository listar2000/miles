# RFC 0002: SFT data tokenized the way the model sees it when served

Status: accepted 2026-11-01; implemented in `miles_ext/sft/` (ext-048) and
`TITOTokenizer.tokenize_completion` (hook-017). A later RFC covers the adapters
that build SFT records (Harbor ATIF, rejection sampling of rollouts) and
launching SFT on Modal.

## What is decided

From the user (2026-10-31 to 2026-11-01): SFT input is a standard **text** format
based on OpenAI chat messages. Turning that format into training tokens must
reproduce what the model receives and samples when it is served by the Miles session
server, so SFT and RL train on the same token layout. The tokenizer is shared with the
RL path (`TITOTokenizer`): when a model family is supported for RL, SFT gets it with at
most one small override. The format and the tokenizer are harness-agnostic. Picking SFT
data from the harness the model will later run in is the job of whoever runs the SFT.
Building records from other sources (Harbor ATIF, Nemotron-style datasets, rejection
sampling) is upstream of this RFC.

## Why

Miles already has an SFT trainer path (`--debug-train-only` with a rollout function
that returns stored tokens). What it lacks is a tokenizer that matches serving:

- The stock `miles/rollout/sft_rollout.py` uses `MultiTurnLossMaskGenerator`
  (`miles/utils/mask_utils.py`). It renders each message on its own with the model's
  **native** chat template and stitches the pieces together, with hard-coded types
  `qwen`, `qwen3` and `distill_qwen`. The session server renders with each family's
  **fixed** template, keeps past reasoning, and keeps each turn's sampled tokens
  verbatim. The two layouts differ, and nothing covers DeepSeek, GLM, Kimi or the
  other TITO families.
- A conversation log is not what the model saw. A harness may drop turns, summarize,
  or edit history between calls. One message list cannot say what the context of each
  call was, so it cannot be tokenized correctly on its own.
- Our Qwen3-30B SWE replication (Nemotron-SFT-SWE data, 2026-10-30) replayed the
  session server loop by hand for Qwen only. It surfaced two silent mismatches: a missing
  tool schema in the data, and `text\n\n<tool_call>` re-rendered as `text\n<tool_call>`.
  Both only show up when you compare against real rollouts.

## Three layers of "messages"

1. **Session**: everything every role produced, kept for auditing. ATIF trajectories
   and most SFT datasets are sessions.
2. **Context**: what the harness sent to the model for one call. It is derived from the
   session (truncation, summaries, edited copies).
3. **Tokens**: what the chat template and the serving stack make of a context. For
   example, a template may drop old `<think>` blocks.

Going from 1 to 2 belongs upstream: only the person running SFT knows the harness, and
ATIF already records the rules for it (`extra.context_management`,
`is_copied_context`). Going from 2 to 3 belongs to the tokenizer alone, and the data
never encodes it.

## The record

One JSON object per line:

```json
{
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
               {"role": "assistant", "content": "...", "reasoning_content": "...",
                "tool_calls": [{"id": "call_0", "type": "function",
                                "function": {"name": "bash", "arguments": "{\"cmd\": \"ls\"}"}}]},
               {"role": "tool", "tool_call_id": "call_0", "content": "..."}],
  "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
  "chat_template_kwargs": {"enable_thinking": true},
  "steps": [{"context": [0, 1], "completion": 2}],
  "metadata": {"source": "...", "harness": "..."}
}
```

- `messages` holds OpenAI chat messages (roles `system`, `user`, `assistant`, `tool`;
  string `content`). It also accepts `reasoning_content` and `tool_calls` on assistant
  messages; tool-call `arguments` must be a JSON string. An assistant message may carry
  `completion_token_ids`: the exact tokens the engine sampled, for example from Miles RL
  rollouts or ATIF `completion_token_ids`. When present, they are used instead of a
  re-render.
- `tools` (OpenAI function tools) and `chat_template_kwargs` (thinking mode, reasoning
  effort and other render settings of the harness) are set per record, not per step.
  Thinking mode is a template setting rather than a message edit because it changes
  the generation prompt itself, for example an empty `<think></think>`.
- `steps` makes layer 2 explicit. A step is one model call: `context` is an ordered
  list of indices into `messages`, and `completion` is the index of the assistant reply.
  `train` defaults to true; with `false`, the step is context only. With `steps`,
  `messages` is a message store: summaries and harness-edited copies are stored as
  their own messages and pointed to. There is one way to express a context.
- Without `steps`, every assistant message *i* is a trained step with context
  `messages[:i]`. That is an append-only harness, and most datasets are one.
- `metadata` is carried through and never affects tokens.

## Tokenizing steps

The unit of SFT is a step, not a conversation. Steps are tokenized with the session
server's own rule (`miles/rollout/session/v2/`), whose code the tokenizer calls
instead of copying:

1. Each record resolves its render settings once with the family's request rules
   (`TITOTokenizer.resolve_request_args`), exactly as for a served request.
2. **Attach** (`SessionTree.find_attach_point`). If an earlier step's context plus
   its reply is a message-level prefix of this context, the prompt is that step's
   tokens plus `merge_tokens` for the new messages
   (`session_state._render_token_ids`). Otherwise, the context is rendered whole, in one
   pass.
3. The reply's tokens are `completion_token_ids` if given, else
   `TITOTokenizer.tokenize_completion` (below). They are trained if the step trains.
   Prompt tokens, including the assistant opener, never are.
4. The steps form a tree, and each leaf is one training sample. A reply shared by
   several leaves is trained in the earliest one only, using the session server's
   ownership rule (`postprocessor_hub.default_postprocess`).

Example: a harness keeps one past exchange. The session is (S1, U1, A1, U2, A2, U3, A3),
and the contexts are C1 = (S1, U1), C2 = (S1, U1, A1, U2) and C3 = (S1, U2, A2, U3).
C2 extends C1 + A1, so A1 and A2 share one sample. C3 does not extend C2 + A2, so C3 is
rendered whole: A2 sits in its prompt untrained, and only A3 is trained. At no point is
there a prompt (S1, U2) with A2 as the response, because inference never had one.

## Thinking retention

Where the model will be served is declared by the person running SFT, not guessed.
One setting, `thinking_retention`, takes prime-rl's names:

| Value | Serving it matches | Prompt of each step | Shares a sample with the previous step when |
| --- | --- | --- | --- |
| `all` (default) | the Miles session server | TITO fixed template, which keeps past reasoning, plus `merge_tokens` | the context extends it at message level (the rule above) |
| `template` | a plain server that re-renders every request (for example vLLM with the model's own template) | the model's native template with the record's `chat_template_kwargs`, rendered whole | the render really starts with the previous step's tokens |

In `template` mode the native template decides what it drops, and the token check then
splits samples by itself. prime-rl's third value, `tool_cycle`, keeps reasoning within a
tool loop and drops it at a new user message. Here that is what `template` already does
with a template that behaves that way (Qwen3, DeepSeek), so it is not a separate value.
In agentic data with a single user message, all modes give the same tokens.

## The model-family hook

The one family-specific piece that RL lacks is the reverse of the response parser:
which tokens the engine sampled for a given assistant message. hook-017 adds
`TITOTokenizer.tokenize_completion(message, *, template_args)`. Its default renders a
probe prompt with and without the message, encodes the difference after the
generation prompt, and drops one trailing token listed in the family's
`trailing_token_ids` (template text the engine never samples, such as Qwen's `\n` after
`<|im_end|>`). A family whose stop token is emitted elsewhere (GLM closes a turn with
the next role's opener) overrides it. A family is SFT-supported once its tests compare
`tokenize_completion` and full trajectories against the session server's renders.
At first, these are the Qwen3 family and DeepSeek V4.1.

prime-rl solves the same problem with a hand-written renderer per family that marks
every token as sampled or not (`renderers` package). Its SFT path renders a whole
conversation once and does not split at rewritten history, so it does not handle
layer 2 at all. Hand-written renderers would duplicate the TITO families, so we keep
templates as the owner and add one method.

## Post-processors

A post-processor is `fn(record) -> record`, named by dotted path and run as an ordered
list before tokenization. It may edit messages, set steps, or both. A few message edits
ship with the tokenizer (moving inline `<think>` into `reasoning_content`, trimming
trailing whitespace of assistant text). A processor that builds steps from a session
(emulating a harness) uses the same interface. The record is validated after the last
processor.

## Output and checks

`python -m miles_ext.sft.prepare` renders records offline (CPU) into token shards, a
prompt file and a manifest, with counts of samples, tokens and trained tokens.
`miles_ext.sft.rollout.generate_rollout` (`--rollout-function-path`) serves those
samples to `--debug-train-only` training without re-rendering anything.
`miles_ext.sft.parity` compares a sample the session server recorded in a real
rollout (tokens and loss mask) with the render of the same record. It reports
whether tokens and masks are equal, and where they first differ. It is the check that
the SFT layout matches serving for a given harness. Pairing rollouts with records
depends on the harness, so that part belongs to the adapters.

## What it conflicts with

- `miles/rollout/sft_rollout.py` and `--loss-mask-type` stay as they are; this path
  does not use them.
- hook-017 adds one method to an upstream class (`TITOTokenizer`); it is a candidate
  for upstream.
- The session server code is imported, not copied. A sync that changes
  `SessionTree`, `_render_token_ids` or `default_postprocess` must keep the SFT tests
  passing.

## Order

1. hook-017: `tokenize_completion` with tests on Qwen3 and DeepSeek V4.1 tokenizers.
2. ext-048: record, post-processors, step tokenizer, offline prepare, rollout function,
   parity, CPU tests.
3. Later RFC: adapters (ATIF, rejection sampling, `completion_token_ids` collection),
   per-step tools if ATIF needs them, and launching SFT through the fork's recipes on
   Modal. A short GPU smoke run on prepared shards comes first there.
