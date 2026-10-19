# Codex reviewer

The independent review in `../SKILL.md`, run by Codex under a fixed protocol. A Claude
supervisor starts it and posts the result; it does not review the PR itself.

| File | What it pins |
| --- | --- |
| `prompt.md` | the instructions Codex gets, filled with the PR number, head, base and bundle path |
| `schema.json` | the output Codex must return: verdict, findings (severity, rule, location, line, evidence, exact fix), test proofs, checks, handoffs |
| `review.py` | the supervisor: checks, bundle, Codex call, protocol check, one retry, report |

The model (`gpt-6-astra`, effort `high`), Codex version and a hash of the prompt and schema
are recorded at the top of every report, so two reviews are comparable only when that line
matches.

## Run

On a host with `codex` logged in, `gh`, `uv` and a Python that has pytest, aiohttp, modal
and pyyaml, from a full (not partial or shallow) clone of this repository:

```bash
python .claude/skills/miles-review/codex/review.py <N> --python <venv>/bin/python \
    [--prior <earlier review.md>] [--out <dir>]
```

It prints three lines (verdict and counts, protocol status, report path) and writes
`review.md` and `review.json` under `<out>/pr<N>-<MMDD-HHMM>/`.

What it does:

1. Checks out the PR head in a throwaway local clone and finds the merge base with `main`.
2. Runs the mechanical checks on head and base: `ledger.py check`, `pre-commit` on the changed
   files, `miles_ext/tests`; lists added comment blocks of three or more lines, the
   test/comment ratios and attribution-looking lines in the body and commits.
3. Runs `codex exec` in a `workspace-write` sandbox on that clone (no network) with the
   pinned prompt and schema. Codex reads the rulebook and diff itself and proves each new
   test green on head and red on base.
4. Checks the output against the protocol (below). On a violation it reruns Codex once with
   the violations appended; if the second run still breaks a rule, the report carries a
   "Protocol check failed" block.

## Protocol check

A review passes when all of these hold:

- the verdict follows from the severities (any blocking: blocked; any should-fix: ready
  after the should-fixes; else ready);
- every finding sits on a changed file (with a line inside it), the PR body, or a commit,
  and has evidence and an exact fix, not advice;
- a failing `ledger check` on the head has a blocking `ledger` finding;
- attribution lines in the body or commits have an `attribution` finding;
- every new `test_*` function has a proof entry, and one that passes on the base has a
  `test` finding.

## Supervisor (Claude)

Per PR, the reviewing Claude thread:

1. Runs `review.py` and reads only its three printed lines and the report's first lines.
2. Posts the report as the review (or attaches it), with the verdict line first.
3. On "Protocol check failed", says so in the post; it does not rewrite the findings.
4. Does not re-review, rephrase or add findings. Disagreements go to the PR owner as a
   question, citing the finding.
