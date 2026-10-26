# ext-038: single-turn tasks through the session server (`taskset.harness: single-turn`)

## Why
Short verifiable tasks (one prompt, one reply, a deterministic grade) need no task directories or
sandboxes. They let an overfit separate algorithm questions from long agent rollouts.

## Decision
- One recipe value, `taskset.harness: single-turn`. The prompt rows carry their own answers: `metadata.task`
  names a verifier and `metadata.label` is the expected answer as a string. The recipe refuses task dirs or a
  task volume and requires `taskset.prompts`.
- Each try goes through Miles' agentic generate (`--custom-agent-function-path miles_ext.tasks.single_turn.run`):
  one chat completion through the session server, graded in process; the returned `reward` lands in
  metadata for the existing `miles_ext.harbor.reward.reward_func`. Tracing, R3 routing replay, rollout
  log-probs, DSpark, LoRA adapter serving and score centering are the agent-run path unchanged.
- Rejected: Miles' plain single-turn generate. It renders prompts with the tokenizer's Jinja chat template, and
  DeepSeek-V4.1 has none; supporting it there means maintaining a second template. The session server already
  renders V4.1 with SGLang's encoder.
- Only graders live in the fork. Task generators and interpreters stay with the task data (miles-ops
  `notes/oodtask/`), since training imports only the grade.
- STAK label format: a nonempty list of non-negative decimal integers in canonical form (single spaces, no
  leading zeros except `0`). The grader reads the text after the last `</think>`, requires exactly one
  `<answer>...</answer>`, normalizes the reply's integers and compares; missing, repeated or malformed answers
  score 0.

## How it is applied
`miles_ext/tasks/single_turn.py` (agent function, `VERIFIERS`, `check_row`), `miles_ext/tasks/stackvm.py`
(STAK grader), `miles_ext/recipes/recipe.py` (harness value, `AGENT_FUNCTIONS`, no Harbor env or task inputs),
`miles_ext/platform/validate.py` (checks prompt rows instead of building a Harbor trial),
`miles_ext/launch/modal_app.py` (no task-dir inputs). Verify: `python -m pytest miles_ext/tests/test_single_turn.py -q`;
end to end, two STAK training runs with score centering enabled and disabled (DeepSeek-V4.1-Flash LoRA r32, DSpark, R3,
top-p 0.95, 13 and 14 steps on 2026-10-26): every request returned 200 at
`/sessions/<id>/v1/chat/completions` and reward rose 0.58 to 0.93 / 0.97. On an upstream sync, recheck the
agent-function signature and the session server's base URL.

## Remove when
Miles ships a prompt-only harness that renders through the session server and grades per row.
