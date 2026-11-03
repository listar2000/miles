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
  `miles_ext/tests` (not `tests/fast`, which needs torch) on head and base, and proves the new tests (head, then the base's
  production files; one pytest process per side, as one per test took an hour on #144). The agent gets everything in one read-only prompt and runs
  nothing. Chosen because Codex's sandbox (bubblewrap) cannot create mounts inside the
  launch host's container, and because scripted checks are identical on every run.
  Rejected: running the agent unsandboxed with the user's GitHub login.
- **Checked output.** The answer must match `output.schema.json`; the script rejects a
  verdict that does not follow from the severities, findings off the changed files or
  without evidence and a suggested fix, a failing ledger without a ledger finding, and a new test that
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
reviewer. A PR that claims a measured improvement gets `--deep`: the agent also checks
whether the numbers hold (like-for-like baseline, same work, noise, what the timers span)
and what the change can break (RL correctness, numerics, memory, R3, LoRA sync), reading
the code passed with `--context` and the logs passed with `--evidence`.

**Automatic reviews.** `review_daemon.py` runs on a private, always-on Sail box
(`mdev-review`) with its own Codex login, and starts `review.py --checks-on modal --post`
itself when a ready PR gets a new head or its owner comments `/review [deep]` (trigger
table in SKILL.md). Polling GitHub, rather than a webhook, needs no public endpoint and
no secret in the repository; the review comments' markers are the only state that
matters. The PR's own code runs in a CPU Modal sandbox on the trainer image, never on the
box that holds the Codex and GitHub logins, which also tests it in the image.
`/merge` lands a single ready PR the merge desk's way (fold, same-code check, ledger,
push of the branch, then `main`), so landing needs no Claude thread either; stacks
and tags stay with the desk.
Up to `--parallel` (default 5) reviews run at once, each `review.py` in its own process
group with its log in `logs/`; a new head or a closed PR stops only that PR's review, and
a daemon stop ends them all. `/merge` still runs one at a time. Git writes to the shared
checkout (fetches, the review clone, the merge worktree) take one file lock, and the
checkout moves to a newer `main` only when no review runs. The daemon normally fills free
slots while reviews run; once `main` has moved it starts none until the running reviews
finish, then updates, so every review reads one protocol version. The heartbeat's `running` is a list.
Each round edits one summary comment in place and posts a "Comment" review whose line
comments carry the findings on changed lines (exact ones as GitHub suggestions); fixes are
recommendations the author decides on. The summary is the newest-created review comment
(collapsing older ones as outdated bumps their edit time, so edit time cannot pick it); a
`/review` counts as answered once the summary was edited after it. Later rounds continue the PR's Codex session with
the diff since the reviewed head (a fresh full round on a rebase, a protocol change or
`/review full`), so one judge carries its reasoning across rounds; check results are cached by
the head and base trees. A PR is reviewed deep when it changes runtime code (the trainer, an
SGLang patch, a fix/hook/port patch page), carries the `deep-review` label, or had a deep round
before, so its session stays deep; a deep follow-up sends only the context and evidence the
session has not read. Rejected: the trainer-paths-only rule, which reviewed SGLang patches
(#143, #145) as normal. Each round sets a `codex-review` commit status, which a branch rule
on `main` can require. Rejected: always-fresh sessions with the prior summary only, which
repeated the whole read each round and lost the reasoning behind earlier points. Approve and Request changes are not used: the box
posts as the PR author, and GitHub refuses both on one's own PR (a reviewer GitHub App
would allow them).
Rejected: Codex cloud review (no choice of model or effort) and GitHub Actions (ChatGPT
logins are not for public-repository CI; an API key costs about $1.20 per review).

## Remove when

Never as such; replace the adapter when the reviewing agent changes.
