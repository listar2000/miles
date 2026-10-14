---
name: miles-fork
description: "How to change this fork of Miles (listar2000/miles) without making it harder to sync with upstream. Use for any change on a `fork/<tag>` branch: fixing a bug in upstream Miles code, adding a hook or an extension under miles_ext/, porting a file from a newer Miles or Harbor, writing a patch page, rebasing onto a new upstream release, or cutting a release image. Not needed for work on upstream's own `main`."
---

# Working with the Miles fork

This repository is upstream Miles (radixark/miles) at a release tag plus an
ordered series of our commits on a branch `fork/<tag>`, e.g. `fork/v0.1.2`.
It is permanent: it gets rebased onto every new upstream release, and other
people and agents contribute to it. Every rule below exists to keep that
rebase mechanical.

The rulebook is `fork/README.md`. Read it before your first change; this skill
is the procedure and does not repeat its reasoning. Upstream's own rules in
`.claude/rules/` still apply to any code you write here.

## Before you start

```bash
git fetch origin && git checkout fork/v0.1.2 && git pull --ff-only
python fork/tools/ledger.py list     # the series: id, kind, subject
python fork/tools/ledger.py check    # must say "ledger check: ok" before and after your change
cat fork/patches/README.md           # the changes that have pages
```

Work on a branch off `fork/<tag>` and land it as a fast-forward. Never merge
upstream into a fork branch; never commit to `main` (a mirror of upstream).

## Pick the kind of change

Each commit is exactly one kind and carries exactly one `Fork-<Kind>:` trailer.
Take the next free number from `ledger.py list` (numbers are never reused).

| You need to | Kind | Where | Trailer |
| --- | --- | --- | --- |
| fix a bug in upstream code | Fix | in place, in the upstream file | `Fork-Fix: fix-NNN` |
| add an extension point upstream lacks | Hook | in place, smallest possible edit | `Fork-Hook: hook-NNN` |
| add a capability (harness, sandbox provider, recipe, reward, platform API) | Ext | new files under `miles_ext/` | `Fork-Ext: ext-NNN` |
| use a file from a newer Miles/Harbor/dependency | Ext, vendored | verbatim copy under `miles_ext/`, entry in `fork/vendored.json` | `Fork-Ext: ext-NNN` |
| change tooling, image build, pins, docs, this skill | Infra | `fork/`, `miles_ext/stack.py`, `FORK.md`, `.claude/skills/miles-fork/` | `Fork-Infra: infra-NNN` |

Prefer the kinds lower in the table: an Ext cannot conflict on a sync, a Fix
can. If an extension needs upstream to call it somewhere it doesn't, write a
separate Hook commit; never edit upstream code from an Ext commit.

## Fix or Hook: editing an upstream file

1. Make the smallest change. Prefer inserting lines over rewriting upstream
   lines; an insertion survives most upstream edits around it.
2. Mark every touched spot with a comment containing `[fork fix-NNN]` (or
   `[fork hook-NNN]`).
3. Commit with these trailers (one block, no blank line between them, after
   the body):

   ```
   Fork-Fix: fix-004
   Touches: miles/path/one.py, miles/path/two.py
   Evidence: <the failure: error text, run id, log reference>
   Upstream-Status: not filed | <PR link> | merged in vX.Y.Z
   Remove-When: <the upstream condition that makes it unnecessary>
   ```

4. Fixes and hooks come first in the series. On an existing branch, place the
   new commit after the last fix/hook with
   `git rebase -i` (or rebuild the series with `git cherry-pick`), then rerun
   `ledger.py check`, which fails on wrong order.
5. File it upstream when you can: every fix upstream accepts is one the next
   sync drops.

## Ext: new code in miles_ext/

- New files only, plugged in through Miles' flags
  (`--custom-generate-function-path`, `--custom-agent-function-path`,
  `--custom-rm-path`, filter paths) and Harbor's `import_path` for agents and
  environments.
- Upstream code outside the `miles` package (e.g.
  `examples/experimental/harbor/`) is loaded from the tree by path, as
  `miles_ext/harbor/upstream.py` does, not copied, with a check for the names
  you use.
- `miles_ext.platform` is an API other services call. Any change to it is
  caller-facing: say so in the next release note.

## Porting an upstream file (vendoring)

When the code you need exists upstream only at a newer version:

1. Copy the file byte for byte into `miles_ext/` (e.g. `miles_ext/harbor/<name>/`).
2. Add an entry to `fork/vendored.json`: source repo, commit, upstream path,
   git blob (`git hash-object <file>`), the fork id, `remove_when`.
3. Put every adaptation in a subclass or wrapper next to it. Never edit the
   vendored file; `ledger.py check` fails if its blob changes.
4. Load it through an extension point (Harbor: `import_path`), not by patching
   the installed package.
5. Write a patch page (always required for vendored files).

If what you need is an edit inside an installed dependency rather than a whole
file, stop and read `fork/README.md` "Harbor": that is the trigger for a
dependency fork, not a patch in this repo.

## Patch pages

Required for every Hook, every vendored file, and any change with a design
choice someone may revisit. Optional for a small, obvious fix.

1. Create `fork/patches/<fork-id>-<slug>.md`:

   ```markdown
   # <fork-id>: <one line: what it does>

   ## Why
   <the failure or gap, with evidence>

   ## Decision
   <what we chose; the alternatives and why they lost>

   ## How it is applied
   <files, mechanism, how to verify it works, what to do on a sync>

   ## Remove when
   <upstream condition>; upstream status: <not filed | PR | merged in vX>
   ```

   Keep it to about one screen. Facts, not narrative.
2. Add a row to `fork/patches/README.md`:
   `| ext-003 | ext | <what> | active | [page](ext-003-harbor-sail.md) |`.
   Status is `active` or the remove-when condition once it is met.
3. Put `Patch page: fork/patches/<file>` in the commit body. A later commit
   that changes the patch updates the page in the same commit.

## Check before pushing

```bash
python fork/tools/ledger.py check      # kinds, trailers, markers, order, touch points, vendored blobs
git log --format='%h %s%n%(trailers:only)' <base>..HEAD | head -40   # eyeball the trailers
```

Then run the change: the launcher's dry check builds an image from your
pushed commit without a release.

```bash
export MILES_DATA=<task sets dir> MILES_DOTENV_DIR=<dir with .env> \
       MILES_RECIPE=miles_ext/recipes/configs/<recipe>.yaml MILES_FORK_COMMIT=$(git rev-parse HEAD)
modal run -m miles_ext.launch.modal_app --run-id <id> --check-only
```

A change to training code also needs a short run (a two-step smoke with
`--save-interval 1`, then the same run id again to test resume). A change to a
recipe or harness needs that recipe's run.

## Syncing to a new upstream release

```bash
git fetch upstream --tags
python fork/tools/sync.py v0.1.3 --dry-run   # which fixes may conflict; upstream diff of what miles_ext uses
python fork/tools/sync.py v0.1.3             # new branch fork/v0.1.3, rebase, stack bump, ledger check
```

On a conflict (only fixes and hooks can conflict), read the commit message and
its patch page. If upstream fixed it, `git rebase --skip` and mark the page's
status as removed. Otherwise re-apply the same intent, keep the markers,
`git rebase --continue`, then `sync.py v0.1.3 --finish`. Then refresh
`HARBOR_COMMIT`, recheck vendored files against the new upstream (drop any
that upstream now ships), and validate as in `fork/README.md` "Sync" before
releasing.

## Cutting a release

Follow `fork/README.md` "Releases". In short: ledger check, push, `python
fork/docker/build.py --release fork-<tag>-rN`, record `FORK_IMAGE` and
`FORK_RELEASE` in `miles_ext/stack.py` in a `Fork-Infra: release-fork-<tag>-rN`
commit, annotated tag `fork-<tag>-rN` on that commit, push both, dry check the
published image.

- The tag message is the release note. Always state whether anything
  caller-facing changed: the `miles_ext.platform` API, the recipe/run-spec
  format, the image ref, Modal resource names, teardown.
- If you changed the Docker Hub token, delete `<workdir>/docker` before a
  `build.py --workdir` retry; crane keeps the old login there.
- Release tags start with `fork-`, so they never trigger upstream's
  `v[0-9]+.*` workflows.

## Don'ts

- Don't edit upstream files from Infra or Ext commits, even "just one line".
- Don't edit a vendored file; subclass it.
- Don't merge upstream; rebase onto a tag.
- Don't add `miles_ext` to upstream's `setup.py`; the image puts
  `/root/miles` on `PYTHONPATH`.
- Don't commit credentials; the launcher reads `.env` from
  `MILES_DOTENV_DIR` at launch time.
