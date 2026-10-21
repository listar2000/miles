---
name: miles-review
description: Review or simplify a listar2000/miles PR for fork conventions, sloppy comments, verbose tests and PR-body rules, with evidence for every finding. Use for every independent reviewer pass and before marking your own PR ready.
---

# Reviewing a miles-mobius PR

One review procedure, so every reviewer checks the same things and reports them
the same way. It removes clutter without changing behavior: unneeded comments,
tests that cannot catch a regression, PR-body noise, and convention breaks that
would make the next upstream sync harder.

The rules themselves live elsewhere; this skill only says how to check them:

- `fork/README.md` (rulebook) and `.claude/skills/miles-fork/SKILL.md` (procedure)
  for kinds, trailers, markers, patch pages and landing;
  `fork/rfcs/0001-main-and-stable-versions.md` wins where they disagree.
- `.github/pull_request_template.md` for the PR body.
- `.claude/rules/*.md` (upstream's) for any code under `miles/`, notably
  `general-code-style.md` and `single-source-of-truth.md`.

Fork PRs run no CI (GitHub Actions only builds the image after a merge), so
the checks below are the only gate before a merge.

It has two users:

- **Authors** apply it while they work and run the full pass before asking for
  a merge: fix what it finds, keep one commit per fork id, re-run every
  check, and push their own topic branch.
- **The independent reviewer** enforces it on every open PR. The reviewer
  reports only: never push to the PR branch and never comment on GitHub unless
  asked; the PR's owner applies fixes.

The independent reviewer is a fresh agent (Claude, Codex or another) started by
`review.py`, which runs Setup and the Ledger, Lint and Extension-tests checks
for it (not `tests/fast`); see "Supervised runs" at the end. This file is its whole guidance.

The reviewer's job is workflow and best practice: comments, tests, PR hygiene
and fork conventions. It is never technical depth (whether the algorithm,
memory math or recipe is right); that belongs to the PR's owner and its
technical reviewers. A technical doubt noticed in passing goes under Handoffs,
not into the verdict. The exception is a deep review (`review.py --deep`, asked
for a PR that claims a measured improvement): it adds section 5, and its
technical findings count toward the verdict.

## Setup

Work in a separate worktree so the PR head never replaces your own checkout.

```bash
git fetch --unshallow --tags origin 2>/dev/null || git fetch --tags origin
git fetch origin main "pull/$N/head:pr$N"
git worktree add "$SCRATCH/pr$N" "pr$N"
cd "$SCRATCH/pr$N"
BASE=$(git merge-base origin/main HEAD)
git log --format='%h %s%n%(trailers:only)' $BASE..HEAD
git diff --stat $BASE HEAD
```

`ledger.py` needs the full history and the upstream tags: in a shallow clone it
fails with a git error, which is a setup problem, not a finding.

## Required checks

Run all of them on the PR head; run the tests on `$BASE` too.

| Check | Command | Pass |
| --- | --- | --- |
| Ledger | `python fork/tools/ledger.py check` | `ledger check: ok` |
| Lint and format | `pre-commit run --files $(git diff --name-only $BASE HEAD)` | no hook fails on a line the PR adds |
| Extension tests | `python -m pytest miles_ext/tests -q -p no:cacheprovider` | same or better than on `$BASE` |
| Upstream tests | `python -m pytest tests/fast/<area touched by a fix or hook> -q` | passes (needs torch; add `--noconftest` if conftest pulls GPU deps) |

- `pre-commit` runs ruff (E, F, B, UP), autoflake, isort and black at 119
  columns. It also rewrites untouched lines of a touched file: a failure that
  `$BASE` already has is pre-existing, so note it and don't count it against
  the PR. Revert the hooks' edits afterwards (`git checkout -- .`).
- `ruff check --extend-select RUF100 <files>` lists `# noqa` markers that
  suppress nothing.
- A cloud container lacks harbor, mini-swe-agent and torch, so a few
  `miles_ext/tests` files fail to collect there (install `pytest aiohttp modal
  pyyaml` for the rest). Only a difference between `$BASE` and the head counts.
- Never launch GPUs or Modal apps for a review. For a red GPU check on an
  upstream-style PR, follow `.claude/rules/ci-failure-triage.md`.

Out of scope: files listed in `fork/vendored.json` (byte-for-byte copies),
prompt and template literals a model sees, and upstream code the PR does not
touch.

## 1. Fork conventions

Each break is **blocking** unless marked otherwise.

- **Fits the fork.** Ask "does training import or need this?" Data prep,
  taskset QA and one-off campaign scripts belong elsewhere, however well written.
- **One topic, one commit per fork id.** Squash work-in-progress commits: the
  PR is one commit on top of `main` (it lands by fast-forward), whose subject
  is the PR title, `type(scope): what`, with `[fork fix-NNN]` / `[fork
  hook-NNN]` at the end for a fix or hook. The ledger alone may force more: a
  fix or hook plus the infra commit that adds its patch page, or two fix ids of
  one topic. Then each commit carries one id, and the PR body's first line says
  "Merge with rebase, not squash".
- **Ledger shape.** One `Fork-*` trailer per commit. A fix or hook commit
  touches only upstream files, and its id is used by that one commit. An ext or
  infra commit touches only the fork's own paths (`OWN_PATHS` in
  `fork/tools/ledger.py`). Two topics (say a fix and an unrelated extension)
  are two PRs: split them.
- **Ids.** New ids are the next free number from `python fork/tools/ledger.py
  list`; numbers are never reused (retired ones included). An ext id may span
  several commits of the same extension.
- **Upstream edits.** The smallest possible insertion, a `[fork fix-NNN]`
  marker at every touched spot, and the `Touches:`, `Evidence:`,
  `Upstream-Status:` and `Remove-When:` trailers.
- **Extensions.** New files under `miles_ext/`, plugged in through Miles' flags
  or Harbor's `import_path`. A vendored file is never edited; adaptations go in
  a subclass next to it.
- **Patch page.** Required for every hook, vendored file, and change with a
  design choice someone may revisit. The commit body and the PR body both carry
  `Patch page: fork/patches/<file>`, or `Patch page: none (small, obvious fix)`.
  A change to a patched area updates its page in the same commit. A missing line
  is should-fix.
- **Evidence a reader can open.** Commit and PR evidence points at GitHub,
  W&B, Linear or files in the repo. Host-only paths (`investigations/...`) and
  scripts that are not in the repo are should-fix: quote the numbers instead.
- **Caller-facing changes** (the `miles_ext.platform` API, recipe or run-spec
  fields and defaults, image refs, Modal names, teardown) are listed under
  Contract.

## 2. Comments and docstrings

Find every full-line comment block of three or more lines in the changed
Python files, then read the diff for docstrings:

```bash
awk 'function f(){if(n>=3)print p":"s": "n"-line comment block";n=0}
     FNR==1{f();p=FILENAME} /^[ \t]*#/{if(!n)s=FNR;n++;next} {f()} END{f()}' \
    $(git diff --name-only $BASE HEAD -- '*.py')
```

**Cut** (should-fix): history ("used to", run ids, "this fixed X"),
counterfactuals, claims about tests, narration of the next line, a rule stated
again where it is already stated, statistics from one run, design essays and
deferred plans, stale cross-references, commented-out code.

**Keep**: at most one intent comment per logical block, two lines or fewer,
stating a constraint the code cannot express and whose omission would cause
misuse. It sits at the declaration. History belongs in the commit message,
rationale in the patch page or PR body. A module docstring is one purpose
sentence, plus a contract table only when the module owns that contract.
Function docstrings are short imperative prose; trivial helpers need none.

Give the replacement text for every comment you cut or shorten.

## 3. Tests

**Is it worth having?** A test earns its place when it pins behavior a caller
relies on and would go red on a realistic regression. Flag (should-fix):

- *declaration pinning*: asserting a constant, default or kwarg equals what the
  code literally says;
- *fake-only*: only a mock or stub is exercised, never the code under review;
- *duplicate*: an existing test already goes red for the same bug;
- *implementation coupling*: private attributes, call order, or a library's
  internals (for example `site._server.sockets`) instead of its public API;
- *bookkeeping*: counters or assertions that the main assertion already implies.

**Right place.** Extension code is tested in `miles_ext/tests/` on CPU in
seconds; upstream code touched by a fix or hook in `tests/fast/`, in upstream's
style. No GPU tests in a review.

**Right shape.** The smallest test that goes red on the bug. Prefer the
library's test helpers (`aiohttp.test_utils.TestServer`, pytest's
`monkeypatch` and `tmp_path`) over hand-built servers and fixtures. Red must
come fast: a test that only fails by hanging until a timeout or a slow cleanup
is should-fix.

**Ratios** are flags to look closer, not findings: added test lines over 1.5x
added production lines, or comments over 15% of added non-blank lines.

**Prove it.** Run each new test on the head (green), then copy the `$BASE`
version of the production files it covers into the worktree and run it again:
it must go red, quickly. Record both results and times, then restore
(`git checkout -- .`). When you propose a smaller test, write it, and prove it
green on the head and red on `$BASE` the same way before you report it.

## 4. PR body and commit message

- Sections follow `.github/pull_request_template.md`: Outcome (before and
  after), Contract (or "No caller-facing changes."), Side-effects (upstream
  files touched with their markers; delete the section if it would say "none"),
  Done, Verification with the exact commands and counts. "Stacked on #N." is
  the first line when it applies.
- Outcome opens with a plain-language result a non-specialist gets at a
  glance (bold line or small before/after table), technical detail below. A
  term a newcomer would not know gets a short "Terms:" line. No run ids
  (`overfit-1019-a4`) or internal jargon anywhere a reader lands: body, commit
  messages, patch pages, comments; describe the run and keep its numbers.
- A patch page added in its own commit uses `Fork-Infra: patch-<id>` and the
  subject `infra(patches): patch page for <id>, ...`.
- No "Requested by" header or attribution marker, no generated-by footer, and
  no claude.ai links (sessions, threads, artifacts). These are should-fix. A
  `Co-Authored-By:` commit trailer is fine.
- Every claim matches the diff and the checks you ran (for example "fails
  without the fix": did it fail, and how?).
- Authors update a body from a file. `gh pr edit N --body-file <file>` fails
  on this repo with a GraphQL "Projects (classic)" deprecation error; use
  `gh api -X PATCH repos/listar2000/miles/pulls/N -F body=@<file>`, or the
  GitHub MCP `update_pull_request` tool.

## 5. Technical depth (deep reviews only)

Read the change itself and the evidence behind its claims (`evidence/` files
and the code the PR touches or calls, given with `--context`). Two questions:

**Does the claimed improvement hold?** Check each number in the Outcome:
- Like for like: baseline and candidate differ only by this change (same base
  commit, recipe, hardware, node count, parallelism, seeds), or the PR says
  what else differs.
- Same work: the same samples, tokens and sequence lengths in both arms. A
  speedup from doing less work (fewer tokens, a skipped pass, a smaller batch)
  is reported as such, not as speed.
- Noise: repeated runs or a stated spread. A gain within run-to-run variation,
  or from one warm-up-affected step, is not a result.
- Timers: the timer spans what the claim names (CUDA work synchronized before
  the clock stops; overlapped or moved work counted once; first step and
  compile time treated consistently). A metric's formula matches its name.
- Scope: a result from a small model, one pipeline stage or one node is not
  extrapolated to the full run without saying so.

**Is it safe?** Trace what the change alters on every path it can reach:
- RL correctness: old/new logprobs, importance-sampling ratios and off-policy
  corrections when a logprob pass is skipped, cached or reused; staleness
  above 0; several minibatches per rollout; advantage and loss masks.
- Numerics: dtypes, casts, reductions, determinism, NaN/inf handling.
- Memory: peak GPU and host memory, pinned buffers and their lifetime.
- State: routing replay (R3), LoRA adapter and weight sync to the engines,
  checkpoint save and resume, process restarts.
- Switches: the change is off by default or behind a flag, and off restores the
  old behavior exactly; an invalid combination is refused at startup.

Rules `measurement` (a claim the evidence does not support as stated) and
`technical` (a correctness, numerics, memory or state risk). Evidence quotes the
code lines or the evidence file; `fix` is the exact code change, or the exact
command and comparison that would settle it (marked unproven). A finding may
anchor in a `--context` file the change affects.

## Severity

- **Blocking**: a broken hard rule (ledger fails, wrong kind or paths, unmarked
  upstream edit, edited vendored file, a secret, lint failing on added lines),
  or a test that cannot go red. In a deep review also: a headline claim the
  evidence does not support, or a correctness risk on a path the PR enables by
  default or in its recipe.
- **Should-fix**: a convention or clarity problem the author fixes before merge.
- **Nit**: optional.

Problems outside the PR's topic are handoffs to their owner, not fixes for
this PR.

## Report

In a supervised run, return the JSON in "Supervised runs" instead. Otherwise
write the review to `/mnt/project-files/reviews/pr<N>-<slug>-review.md` when
that folder exists (otherwise put it in your reply):

1. Verdict on the first line: ready / ready after the should-fixes / blocked.
2. Findings by severity, each with `file:line`, the evidence (command and
   output), and the exact fix (replacement text or a diff).
3. Any proposed rewrite (comment or test) in full, with its green and red proof.
4. The checks you ran, with counts and times, head against `$BASE`.
5. Handoffs, each naming its owner.

Then remove the worktree: `git worktree remove "$SCRATCH/pr$N"`.

## Done means

Every required check ran on the head and on `$BASE`; every finding carries
evidence and a concrete fix; every proposed test was proven green on the head
and red on `$BASE`; and the PR's branch is untouched (reviewer) or holds
one commit per fork id that passes every check (author).

## Supervised runs

`review.py` gives every independent review the same inputs, the same output
shape and the same checks, whichever agent does it:

```bash
python .claude/skills/miles-review/review.py <N> --python <venv>/bin/python \
    [--agent codex] [--board coord/BOARD.md] [--prior <earlier review.md>] [--out <dir>] \
    [--deep --context <repo path>... --evidence <file or dir>...]
```

`--deep` adds section 5. `--context` inlines more files at the head (the code
the change calls or feeds, such as the loss or the weight sync); `--evidence`
inlines the logs and result files behind the PR's numbers, as `evidence/<name>`.

It needs `gh`, `uv`, the agent's CLI logged in, a Python with pytest, aiohttp,
modal and pyyaml, and a full (not partial or shallow) clone.

1. It checks out the PR head in a throwaway clone, runs Setup and the Ledger,
   Lint and Extension-tests checks on head and base, proves each new test
   (section 3: run on the head, then with the base's production files), and
   writes a bundle.
2. It starts the agent read-only, with no shell, and hands it everything in
   one prompt: this file, the rules it points to, the bundle, and every changed
   file in full (only the diff when that would be too long). The agent judges;
   it runs nothing. A PR whose body starts "Stacked on #N." is diffed against
   #N's head.
3. It checks the answer against the rules below; on a break it reruns the agent
   once with the breaks listed, and a second break is printed at the top of the
   report.
4. It prints three lines (verdict and counts, protocol status, report path) and
   writes `review.md` and `review.json` under `<out>/pr<N>-<MMDD-HHMM>/`. The
   report header names the agent, model, effort and a hash of this file and the
   schema; two reviews compare only when those match.

**The bundle** stands in for Setup and the required checks: `pr.json` (title,
body, head), `commits.txt` (log with trailers), `diff.patch`, `diffstat.txt`,
`files.txt`, `ledger_list.txt` (every fork id at the head),
`ids_on_branches.txt` (ids used on any branch, merged or not; they count as
taken),
`reserved_ids.txt` (id reservations from the coordination board, given with
`--board`; a reserved id counts as taken), `ledger.txt`,
`precommit.txt` (failing hooks and exit code) and `pytest.txt` (head and base),
`test_proofs.txt` (each new test on the head and on the base's production
files, with times; `error` when it cannot run on this host),
`comment_blocks.txt` (blocks of three or more lines touching added lines),
`ratios.txt`, `attribution.txt` (lines that look like attribution), and
`prior_review.md` when given (check each of its findings against the head).

**The reviewer** does sections 1-4 from the prompt alone. A smaller test it
proposes goes in `fix` marked unproven; the PR's owner proves it. Upstream's
`REVIEW.md` is for upstream PRs; on this fork, this file wins.

**The answer** is one JSON object matching `output.schema.json`:

- one finding per problem; a problem repeated across files is one finding
  anchored at the first location, listing the others in `evidence`;
- `location` is a path from `files.txt` (or a `--context` file in a deep
  review), `PR body` or `commit <sha8>`; `line` is 1-based at the head, 0 for
  a whole file, the body or a commit;
- `evidence` quotes the text, or the bundle file and lines it comes from;
  `fix` is the exact replacement text or a diff, never advice;
- `rule` names the broken rule, `severity` follows Severity, and `verdict`
  follows from them (any blocking: blocked; any should-fix: ready after the
  should-fixes; else ready);
- `summary` is at most three sentences a non-specialist can read; technical
  doubts go in `handoffs`, except in a deep review, where they are findings.

**The supervisor rejects** an answer whose verdict does not follow from its
severities; a finding off the changed files, the body or a commit, past the end
of its file, or without evidence and an exact fix; a failing ledger on the head
without a blocking `ledger` finding; attribution lines without an `attribution`
finding; and a new test that passes on the base without a `test` finding
naming it.

**The thread that runs it** reads only the printed lines and the report's top,
posts the report as the review with the verdict first, says so when the
protocol check failed, and never re-reviews or adds findings: a disagreement
goes to the PR owner as a question citing the finding.

To add an agent, add an entry to `AGENTS` in `review.py`: the command that runs
it non-interactively and read-only with the prompt on stdin and the schema, and
writes its final message to a file.
