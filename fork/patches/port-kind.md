# port-kind: the Port kind, and SGLang changes as patch files only

## Why
GPU-delta weight sync (radixark/miles#3894, sgl-project/sglang#42210) is two open upstream drafts that full-parameter
runs in split placement need now. The fork's kinds had no place for them: an Ext lives under `miles_ext/` only (the PR's
files import each other by their `miles.*` paths, so a verbatim vendored copy there cannot load), a Fix or Hook is the
smallest edit, and SGLang patches were capped at about 30 lines with no features. The rules also said to switch to a
`listar2000/sglang` fork past about five patches; we carry nine, and the user prefers patch files to a second repo.

## Decision
- A fifth kind, **Port** (`port-NNN`): an open upstream PR carried byte for byte from its head commit; Miles files at the
  PR's own paths, SGLang as one `port-NNN-<slug>.patch` of any size. Trailers `Touches:`, `Upstream-PR: <url>@<sha>`,
  `Remove-When:`. Conflicts are resolved toward our code and listed on the port's page; our adaptations are separate
  fix, hook or ext commits; the sync that brings the merged PR retires it.
- A refresh to a newer PR head reuses the id; its `Touches:` replace the earlier ones (a refresh may drop a file).
- Files a port adds or deletes need no marker (added ones stay the PR's bytes); files it edits do. In the image, `ledger.py markers`
  still requires each added SGLang file to exist and each deleted one to be gone.
- A fix or hook on top of a port may exceed the ~30-line SGLang limit when the port's feature needs it (e.g. a model
  mapping), with the user's OK on its PR. No SGLang fork: patch files only, however many.
- Rejected: a `listar2000/sglang` fork (a second sync per release); waiting for the upstream merges (our progress
  would wait on others); vendoring under `miles_ext/` (does not import); one huge Fix (hides that it is upstream's code).

## How it is applied
`fork/tools/ledger.py` (kind, id format, trailers, refresh, marker exemption, image check), `fork/README.md`
"Kinds of change" and "SGLang patches",
`.claude/skills/miles-fork/SKILL.md`. Tests: `miles_ext/tests/test_ledger_port.py`.

## Remove when
Never on its own; the kind stays while any `port-NNN` is live. Approved by the user on 2026-11-03.
