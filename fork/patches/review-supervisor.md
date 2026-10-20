# review-supervisor: independent PR reviews run by a supervised agent

## Why

Independent reviews were done by a Claude thread whose instructions changed from
request to request, so two reviews of the same PR could check different things, and
the reviewer sometimes skipped the required checks or test proofs. Moving reviews to
Codex (to spend Codex credits instead of Claude's) was the moment to pin the protocol.

## Decision

- **One skill, any agent.** `.claude/skills/miles-review/SKILL.md` is the reviewer's
  whole guidance; `review.py` starts the agent through an adapter table (`AGENTS`), so
  Claude, Gemini or another CLI is one more entry. Rejected: a Codex-specific prompt
  folder, which would fork the rules.
- **The script runs, the agent judges.** `review.py` runs the ledger, pre-commit and
  `miles_ext/tests` (not `tests/fast`, which needs torch) on head and base, and proves each new test (head, then the base's
  production files). The agent gets everything in one read-only prompt and runs
  nothing. Chosen because Codex's sandbox (bubblewrap) cannot create mounts inside the
  launch host's container, and because scripted checks are identical on every run.
  Rejected: running the agent unsandboxed with the user's GitHub login.
- **Checked output.** The answer must match `output.schema.json`; the script rejects a
  verdict that does not follow from the severities, findings off the changed files or
  without an exact fix, a failing ledger without a ledger finding, and a new test that
  passes on the base without a test finding, then reruns the agent once.

Pilot (Codex GPT-6-Astra, effort high): 12 of the 13 problems the Claude reviewer found
on #48 and #49; on held-out #45 it was slightly stricter than Claude.

## How it is applied

`python .claude/skills/miles-review/review.py <N> --python <venv>/bin/python
[--board coord/BOARD.md] [--prior <review.md>]` on the launch host. "Stacked on #N." PRs
are diffed against #N's head; prompts over Codex's 1M-character limit fall back to the
diff. Fork ids used on any branch (merged or not) and the board's reservations are passed
in, so an id taken on an abandoned branch counts as taken. Verify a change to the protocol with `--result <saved answer>`: the saved answer replaces
the first agent call (a protocol break still triggers the one real retry).
Upstream sync PRs (merge diffs) are too large for one prompt and stay with a Claude
reviewer.

## Remove when

Never as such; replace the adapter when the reviewing agent changes.
