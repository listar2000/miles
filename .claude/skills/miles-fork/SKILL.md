---
name: miles-fork
description: "How to change miles-mobius, our Miles (listar2000/miles, branch main), without making it harder to sync with upstream. Use for any change here: fixing a bug in upstream Miles code, adding a hook or an extension under miles_ext/, porting a file from a newer Miles or Harbor, writing a patch page, merging a new upstream release, landing on main and building the image, or marking a version stable. Not needed for work in upstream radixark/miles itself."
---

# Working with the Miles fork

This repository is miles-mobius: upstream Miles (radixark/miles) plus our
commits on one running branch, `main`. Upstream releases come in by merge,
and other people and agents contribute to it. Every rule below exists to keep
those merges mechanical: our code sits in new files, and the few edits to
upstream files are small and marked.

The rulebook is `fork/README.md`. Read it before your first change; this skill
is the procedure and does not repeat its reasoning. The decision behind it is
`fork/rfcs/0001-main-and-stable-versions.md`. Upstream's own rules in
`.claude/rules/` still apply to any code you write here.

## Before you start

```bash
git fetch origin --tags && git checkout main && git pull --ff-only
python fork/tools/ledger.py list     # our commits since the upstream tag: id, kind, subject
python fork/tools/ledger.py check    # must say "ledger check: ok" before and after your change
cat fork/patches/README.md           # the changes that have pages
git checkout -b <topic>              # a work branch off main
```

Work on a branch `<topic>` off `main`, land it as a fast-forward, then delete
the work branch. Never force-push `main`, never move or delete a tag, and never
push upstream branches or tags here (the allowed refs are in `fork/README.md`
"Branches, versions and tags"). `fork/v0.1.2` and the `fork-v0.1.2-rN` tags are
frozen history: don't branch from them. If `upstream` is not a remote yet:
`git remote add upstream https://github.com/radixark/miles`.

## Pick the kind of change

Each commit is exactly one kind and carries exactly one `Fork-<Kind>:` trailer.
Take the highest number of that kind ever taken (in `ledger.py list`, including ids on unmerged
branches or retired), plus one. Never fill a gap, never reserve a block, and never reuse a number,
even one whose PR closed.

| You need to | Kind | Where | Trailer |
| --- | --- | --- | --- |
| fix a bug in upstream code | Fix | in place, in the upstream file | `Fork-Fix: fix-NNN` |
| add an extension point upstream lacks | Hook | in place, smallest possible edit | `Fork-Hook: hook-NNN` |
| fix a bug in SGLang, or add a hook there | Fix or Hook | `fork/deps/sglang/<id>-<slug>.patch` (below) | `Fork-Fix: fix-NNN` / `Fork-Hook: hook-NNN` |
| add a capability (harness, sandbox provider, recipe, reward, platform API) | Ext | new files under `miles_ext/` | `Fork-Ext: ext-NNN` |
| use a file from a newer Miles/Harbor/dependency | Ext, vendored | verbatim copy under `miles_ext/`, entry in `fork/vendored.json` | `Fork-Ext: ext-NNN` |
| change tooling, image build, pins, docs, this skill | Infra | `fork/`, `miles_ext/stack.py`, `FORK.md`, `.claude/skills/miles-fork/`, `.claude/skills/miles-review/` | `Fork-Infra: <kebab id>` (`infra-NNN`, `sync-<tag>`, `stable-YYYY-MM-DD`, or a short name) |

Prefer Ext and Infra: they cannot conflict on a sync, a Fix or Hook can. If an extension needs upstream to call it somewhere it doesn't, write a
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

4. There is no ordering rule (merges have no replay order): the commit lands
   on `main` like any other. Never reorder or rewrite `main`.
5. Add its row to `fork/UPSTREAM.md` (the upstream candidate list), or say in
   the PR why it is not a candidate. Every fix upstream accepts is one the next
   sync drops.

## Fix or Hook in SGLang

Rules and reasons: `fork/README.md` "SGLang patches". Only a bug fix or a
small hook that copies a pattern SGLang already has; never a runtime patch
from `miles_ext`.

```bash
SGL=$(python -c 'from miles_ext import stack; print(stack.SGLANG_COMMIT)')
git clone --filter=blob:none https://github.com/sgl-project/sglang /tmp/sglang && git -C /tmp/sglang checkout $SGL
# edit /tmp/sglang, with a `[fork fix-NNN]` comment on an added line in every file you touch
git -C /tmp/sglang diff > fork/deps/sglang/fix-NNN-<slug>.patch && git -C /tmp/sglang checkout -- .
python fork/deps/sglang/apply.py /tmp/sglang $SGL --check
git -C /tmp/sglang checkout origin/main && git -C /tmp/sglang apply --check "$PWD"/fork/deps/sglang/fix-NNN-<slug>.patch
git add fork/deps/sglang/fix-NNN-<slug>.patch
git commit   # trailers as below, with Touches: sglang:python/sglang/<path>, ...
```

`Upstream-Status:` records the last command: the SGLang PR, or `not filed;
also on sgl-project/sglang main <sha9>`. Add the patch page (always required)
in its own commit, then run the launcher's `--check-only` from the pushed
commit: its output lists the patch's marker in the installed SGLang. The
evidence is a GPU run on the patched image beside the same run without it.

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
  caller-facing: say so in the next stable version's release note.

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
file: for SGLang, see "Fix or Hook in SGLang" above; for anything else, stop
and read `fork/README.md` "Harbor": that is the trigger for a dependency fork,
not a patch in this repo.

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
python fork/tools/ledger.py check      # kinds, trailers, markers, touch points, vendored blobs
git log --format='%h %s%n%(trailers:only)' <base>..HEAD | head -40   # eyeball the trailers
```

Then run the change: the launcher's dry check builds an image on Modal from
your pushed work-branch commit, without touching the registry.

```bash
export MILES_DATA=<task sets dir> MILES_DOTENV_DIR=<dir with .env> \
       MILES_RECIPE=miles_ext/recipes/configs/<recipe>.yaml MILES_FORK_COMMIT=$(git rev-parse HEAD)
modal run -m miles_ext.launch.modal_app --run-id <id> --check-only
```

A change to training code also needs a short run (a two-step smoke with
`--save-interval 1`, then the same run id again to test resume). A change to a
recipe or harness needs that recipe's run.

## W&B projects

Every recipe sets `tracking.wandb_project: miles-<campaign>` (entity
`mobiuslab`); a new purpose, model family or sandbox provider gets a new
project. Use run ids starting with `smoke-` or `debug-` for smoke and debug
runs: they go to `miles-smoke`. Never log to `miles-swe-rl` (archive). Details:
`fork/README.md` "W&B projects".

## Syncing to a new upstream release

Merge, never rebase. Details: `fork/README.md` "Sync".

```bash
git fetch upstream --tags
python fork/tools/sync.py v0.1.3 --dry-run   # which fixes and hooks may conflict; upstream diff of what miles_ext uses
git checkout -b sync-v0.1.3 main
git merge --no-commit v0.1.3                # resolve conflicts, then:
git commit -m "sync: merge upstream v0.1.3" --trailer "Fork-Infra: sync-v0.1.3"
# in miles_ext/stack.py set UPSTREAM_TAG, UPSTREAM_BASE and BASE_IMAGE (by digest)
git commit -am "stack: base image for v0.1.3" --trailer "Fork-Infra: sync-v0.1.3"
python fork/tools/ledger.py check
```

Between releases, merge an upstream `main` commit only when we need a change
there, and only one a published upstream image was built from
(`radixark/miles:dev`): trailer `sync-<sha9>`, `UPSTREAM_BASE` = that commit,
`BASE_IMAGE` = that image, `UPSTREAM_TAG` unchanged.

Don't run `sync.py` without `--dry-run`: it still does the old rebase onto a
`fork/<tag>` branch. On a conflict (only in files fixes and hooks touch), read
the commit message and its patch page. If upstream fixed it, take upstream's
side, mark the page's status as removed, move its `fork/UPSTREAM.md` row to
"Fixed upstream" and add `--trailer "Retires: <id>"` to the stack commit. Otherwise re-apply the same intent
and keep the markers. Then refresh `HARBOR_COMMIT`, recheck vendored files
against the new upstream (drop any that upstream now ships), land, build and
validate as in `fork/README.md` "Sync".

## Landing

Follow `fork/README.md` "Landing". In short: `ledger.py check` on the work
branch, fast-forward `main` to it, push `main`, delete the work branch, then
`python fork/docker/build.py --main` (pushes `starlimobius/miles-mobius:<sha10>`
and moves `:main`, which is `latest`) and the launcher's `--check-only` on the
new image.

- A PR description follows `.github/pull_request_template.md` (its comment says what to leave out).
- Before asking for a merge, run `.claude/skills/miles-review/SKILL.md` on the PR as its author.
- A doc-only landing skips the build; `latest` stays on the last build.
- `--main` builds only `origin/main`'s tip. If `main` moved, rebase your own
  work branch onto it and land again; never force-push `main`.
- Nothing is committed after the build (no `stack: publish` commit), and
  `stack.FORK_RELEASE` / `stack.FORK_IMAGE` stay on r6: the platform still
  reads them until it resolves versions.
- If you changed the Docker Hub token, delete `<workdir>/docker` before a
  `build.py --workdir` retry; crane keeps the old login there.

## Marking a version stable

Only after the platform's owner reports that a `latest` build passed staging
(smoke test on `rl-staging`, then one recipe trained to the end). Follow
`fork/README.md` "Stable versions":

1. `python fork/docker/build.py --mark-stable stable-YYYY-MM-DD --image docker.io/starlimobius/miles-mobius@sha256:<digest>`
   (refuses to move an existing tag; a second one that day is `.2`).
2. An annotated git tag `stable-YYYY-MM-DD` on the image's
   `MILES_MOBIUS_COMMIT`, whose message is the release note; push the tag.
3. An entry `{name, commit, image, date, note}` in `miles_ext/stable.json`, in
   a commit with `Fork-Infra: stable-YYYY-MM-DD`, landed on `main`.
4. A GitHub Release for the tag with the same note.

- The release note always states whether anything caller-facing changed: the
  `miles_ext.platform` API, the recipe/run-spec format, the image ref, Modal
  resource names, teardown. Write "no caller-facing changes" when there are
  none.
- `stable-*` tags never match upstream's `v[0-9]+.*` workflows.

## Don'ts

- Don't edit upstream files from Infra or Ext commits, even "just one line".
- Don't edit a vendored file; subclass it.
- Don't patch SGLang at runtime (monkeypatching, appliers) from `miles_ext`; carry a patch file.
- Don't rebase `main` onto upstream; merge the tag. Don't force-push `main` or move a tag.
- Don't add `miles_ext` to upstream's `setup.py`; the image puts
  `/root/miles` on `PYTHONPATH`.
- Don't commit credentials; the launcher reads `.env` from
  `MILES_DOTENV_DIR` at launch time.
