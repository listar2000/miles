# This fork

`listar2000/miles` is miles-mobius: upstream Miles plus our commits on one
running branch, `main`. Upstream releases come in by merge. This file is the
rulebook for our commits; the record of what is in them is the history itself
(`python fork/tools/ledger.py list`). The important changes also have a short
page each (Patch pages, below). Agents: `.claude/skills/miles-fork/SKILL.md`
is the step-by-step workflow. The decision behind this layout is
[RFC 0001](rfcs/0001-main-and-stable-versions.md).

## Branches, versions and tags

This repository is standalone, not a GitHub fork: it holds only our refs, and
upstream is reached as a git remote (`git remote add upstream
https://github.com/radixark/miles`). Only these refs may exist here:

| Ref | What |
| --- | --- |
| `main` | the running line and default branch: the newest upstream release we merged plus our commits |
| `<topic>` | a work branch off `main`; lands on `main` by fast-forward, then is deleted |
| tag `stable-YYYY-MM-DD` | a stable version: annotated, on the commit whose tree is in the image; the image has the same tag. A second one on the same day is `stable-YYYY-MM-DD.2` |
| tag `<upstream tag>` | an upstream release we merged, e.g. `v0.1.2` (same object as upstream's) |
| (no ref) | an untagged upstream commit we merged between releases is named only by `stack.UPSTREAM_BASE` |
| `fork/v0.1.2`, tags `fork-v0.1.2-r1..r6`, `archive/*` | the old release line and its releases, frozen: no new commits or tags, never moved or deleted |

No upstream branches and no other upstream tags: fetch those from `upstream`
when needed, never push them here. Never force-push `main` and never move or
delete a tag: callers pin commits and images, and a stable version is a
promise.

A run uses one of two trainer versions, never a raw revision:

- **`latest`**: the newest build of `main` (image `starlimobius/miles-mobius:main`).
  The platform resolves it once when a run is created and keeps that image for
  every segment, so a resume never changes build.
- **`stable-YYYY-MM-DD`**: a build of `main` that passed the platform's staging
  validation (Stable versions, below), listed in `miles_ext/stable.json`.

`miles_ext/platform/versions.py` (`resolve(trainer)`) turns either into
`{trainer, name, commit, image}`.

Upstream's tag-triggered workflow (`release-docker.yml`) fires on
`v[0-9]+.*`, so `stable-*` tags match nothing. GitHub Actions are off for
this repository; we run no workflows.

## Four kinds of change

Every fork commit has exactly one trailer saying what it is.

| Kind (trailer) | What it is | Where it lives | Can it conflict on a sync? |
| --- | --- | --- | --- |
| **Fix** (`Fork-Fix: fix-NNN`) | a bug fix to upstream code | in place, in the upstream file; in SGLang, a patch file (SGLang patches, below) | yes: it is the only thing that can (with hooks) |
| **Hook** (`Fork-Hook: hook-NNN`) | an extension point upstream lacks, so an extension can plug in | in place, in the upstream file; in SGLang, a patch file | yes |
| **Infra** (`Fork-Infra: <id>`) | fork machinery: image build, tools, the stack pins, docs | `fork/`, `miles_ext/stack.py`, `FORK.md`, `.claude/skills/miles-fork/`, `.claude/skills/miles-review/`, `.claude/skills/miles-use/`, `.claude/rules/fork.md`, `.github/ISSUE_TEMPLATE/` | no (new files only) |
| **Ext** (`Fork-Ext: ext-NNN`) | new capability: harnesses, recipes, algorithms, sandbox providers | `miles_ext/` | no (new files only) |

Rules for **fixes and hooks** (enforced by `ledger.py check`):

- One commit per fix on `main`, as small as the bug allows. Prefer inserting lines over
  rewriting upstream lines: an insertion survives most upstream edits around it.
  During review, fix with `git commit --fixup=<commit>` instead of rewriting the
  branch; the merge desk folds those commits (Landing).
- Mark every touched spot with `[fork fix-NNN]` (or `hook-NNN`) in a comment,
  so `grep -rn "\[fork "` finds every place we changed.
- Trailers: `Touches:` (the upstream files), `Evidence:` (the failure it fixes),
  `Upstream-Status:` (not filed / PR link / merged in vX), `Remove-When:`.
- Add a row to [UPSTREAM.md](UPSTREAM.md) in the same PR, or say in the PR why
  the fix is not an upstream candidate.
- No ordering rule: with merges there is no replay order, so a fix lands
  on `main` like any other commit.
- Removing one of our fixes or hooks is itself a fix or hook commit (a new id). It
  names what it removes: `Retires: <id>` when that change is gone entirely,
  `Releases: <id> <file>` when its code left only some files. Where the commit only
  deleted lines it needs no marker of its own.

Rules for **extensions**: new files under `miles_ext/` only, plugged in
through Miles' own extension points (`--custom-generate-function-path`,
`--custom-agent-function-path`, `--custom-rm-path`, filter paths) and
Harbor's `import_path` for agents and environments. If an extension needs a
hook that does not exist, add a Hook commit with the smallest possible change
rather than editing upstream code from the extension's commit.

`python fork/tools/ledger.py markdown` prints our commits since the upstream base as a table.

## Flags, recipe keys and env vars

Every switch doubles the runs someone has to reason about, so the fork adds one
only when a real run needs its other value. `miles/utils/fork_arguments.py`
holds every Miles flag the fork adds, registered by one call in
`add_miles_arguments` (hook-014). The one exception is hook-006's LoRA flag
(`--custom-lora-spec-path`), which stays in
`miles/utils/lora/arguments.py` because upstream tests build that parser alone; recipe keys and their defaults live in
`miles_ext/recipes/recipe.py`.

- **A fix has no switch.** Turning it off brings the bug back. If upstream code
  needs the old behavior (a test, an option the fix cannot serve), the fix
  refuses that case or falls back for it alone, as fix-059 does for
  `--async-retry-aborted-tries` with a group reward.
- **A new flag, recipe key or env var needs a second real user**: a run, recipe
  or test that sets the value other than the default. Name it in the PR body
  (the template's "New flags" line). Without one, write the value as a
  constant.
- **Values that move together are one key or a preset.** Derive the rest from
  that owner in `recipe.py` or `presets.py`; never ask a run to set several
  flags in step.
- **The default is what runs use.** A key every recipe sets the same way
  should not need setting.
- **Tuning constants are module constants, not env vars.** An env var read
  inside upstream code is a switch nobody sees in a recipe.
- **Debug switches** live under the recipe's `obs.*`, off by default.
- **No compatibility shims.** A removed flag or key is gone; an old run
  reproduces from its image plus its `recipe-*.json`, not from a shim.
- **At each sync** ("Sync" below), list the recipe keys every config under
  `miles_ext/recipes/configs/` leaves at the default and the fork flags no
  recipe passes, and remove them in the sync PR or say why each stays.

## Patch pages

`ledger.py` lists every change; a few deserve an explanation a reader can
find without reading commit messages. Those get a page.

- **Where:** `fork/patches/<fork-id>-<slug>.md`, e.g. `fork/patches/ext-003-harbor-sail.md`,
  and one row in the index `fork/patches/README.md`.
- **When:** required for every Hook, every SGLang patch, every vendored upstream file (below) and
  any change that needed a design choice someone may revisit (several
  alternatives, a workaround, a dependency-of-a-dependency problem). Optional
  for a small, obvious fix: its commit message is enough.
- **What:** four short sections. *Why* (the failure or gap, with evidence),
  *Decision* (the choice, the alternatives and why they lost), *How it is
  applied* (files, mechanism, how to verify, what to do on a sync), *Remove
  when*. Aim for one screen.
- **Link:** the commit that adds or changes the patch has a body line
  `Patch page: fork/patches/<file>`. When a later commit changes the patch,
  update the page in the same commit.

## Porting an upstream file

When we need code that exists upstream (in Miles, Harbor or another
dependency) but not at the version we pin, carry the file **verbatim** in
`miles_ext/` and record it in `fork/vendored.json`: source repo, commit,
upstream path, git blob, the fork id that brought it, and when to remove it.
`ledger.py check` and `ledger.py markers` fail if a vendored file's blob no
longer matches, so a vendored file is never edited in place. Adapt it in a
subclass or wrapper next to it. Load it through an extension point (for
Harbor, `import_path`), not by patching the installed package. A vendored
file is an Ext change and always gets a patch page.

Edits to installed dependency code (not whole files) are done in this repo
only for SGLang, as patch files (SGLang patches, below). For any other
dependency, see Harbor below for when that becomes a dependency fork.

## Layout

```
FORK.md                 the front door: using the fork (fork/guide/) and changing it (this file)
fork/
  README.md             this rulebook
  guide/                the user guide: quickstart, recipes, extending, running, limits, help
  docker/build.py       build main into an image (--main) and mark an image stable (--mark-stable)
  tools/ledger.py       list / audit our commits (touch points, markers, trailers)
  tools/sync.py         upstream sync helper (still the old rebase flow: use --dry-run only; see Sync)
  patches/README.md     index of patch pages; patches/<fork-id>-<slug>.md, one page each
  rfcs/README.md        index of RFCs (process and design proposals); rfcs/NNNN-<slug>.md
  vendored.json         upstream files carried verbatim in miles_ext/ (blob-checked)
  UPSTREAM.md           our fixes worth sending to upstream Miles or SGLang, and whether upstream still has each bug
  deps/harbor/          offline Harbor patches (task-set generation only; see Harbor)
  deps/sglang/          SGLang patches (<fix|hook>-NNN-<slug>.patch) and apply.py, applied at image build
.claude/skills/miles-fork/  the workflow for agents changing this fork
.claude/skills/miles-review/  the review checklist every fork PR goes through
.claude/skills/miles-use/  the workflow for agents training with this fork
.claude/rules/fork.md   tells every agent this is a fork and where to start
.github/ISSUE_TEMPLATE/ the issue forms of fork/guide/help.md
miles_ext/              extensions (top-level package next to miles/, on PYTHONPATH in the image)
  stack.py              pins: upstream tag, base image digest, Harbor commit, mini-swe-agent, Sail SDK; image repo
  stable.json           the stable versions: name, commit, image digest, date, note
  harbor/               agent function, in-sandbox and host-side mini-swe-agent, reward, session proxy;
                        harbor/sail/: Harbor's Sail provider (vendored) and the fork's subclass;
                        modal_env.py: Harbor's Modal provider for no-network tasks (ext-041);
                        trial_procs.py: trials forked from a preloaded server (ext-042)
  recipes/              recipe YAML -> Miles argv + Harbor env; model presets
  launch/modal_app.py   test launcher for Modal (not platform API)
  monitor/              read-only run monitor web page (ext-039)
  obs/                  observers: GPU telemetry, rollout loop watchdog; sandbox_profile + sandbox_bench (ext-040)
  platform/             the platform API (ext-006): schema, validate, launch, observe, teardown, secret, weights, checkpoints
```

`miles_ext` is not added to upstream's `setup.py` (that would be a touch point
for no benefit): the image sets `PYTHONPATH=/root/miles`.

Upstream code that lives outside the `miles` package, such as Miles' Harbor
agent function under `examples/experimental/harbor/`, is loaded from the tree
by path (`miles_ext/harbor/upstream.py`) instead of being copied, so a sync
brings the new version and API drift fails at import, not silently.

## The image

`fork/docker/build.py --main` publishes `docker.io/starlimobius/miles-mobius:<sha10>`
(`stack.IMAGE_REPO`) and moves `:main` to it: upstream's release image (amd64, every layer untouched) plus two layers.

| Layer | Contents | Made by |
| --- | --- | --- |
| upstream (117 layers) | CUDA, torch, SGLang, Megatron, compiled extensions, `/root/miles` at the upstream tag | radixark; mounted from `radixark/miles` on Docker Hub, never re-uploaded |
| deps | Harbor at `HARBOR_COMMIT`, mini-swe-agent and the Sail SDK (`stack.DEPS_INSTALL`): exactly the files of the distributions `uv pip install` added or replaced (their `RECORD`s), whiteouts for the versions it removed; plus the SGLang files our SGLang patches change | a Modal container of the base image runs the install and `fork/deps/sglang/apply.py` |
| src | the built commit of `main` (`git archive`) over `/root/miles`, whiteouts for files the commit deletes | local `git` |

The image env adds `PYTHONPATH=/root/miles` (so the top-level `miles_ext`
imports; upstream's `setup.py` is not edited), `MILES_MOBIUS_VERSION`
(`main@<sha10>`) and `MILES_MOBIUS_COMMIT`, plus the older names
`MILES_FORK_RELEASE` and `MILES_FORK_COMMIT` for readers of that age. OCI
labels (`miles.mobius.*` and the standard ones) record the source commit, base
digest and the Harbor, mini-swe-agent and Sail SDK pins. `/root/miles/.git`
still describes upstream's checkout, so read the commit from
`MILES_MOBIUS_COMMIT`, not `git rev-parse`.

Why layers appended with `crane` instead of a Dockerfile: this machine has no
Docker daemon, CI with registry secrets was not an option, and Docker inside a
Modal Sandbox could not mount the 117-layer base (overlay mount "invalid
argument"; small images worked). Appending also keeps every upstream layer
byte-identical and shared, so hosts that already cached upstream's image
(Modal does) pull only the two new layers. With a normal Docker host the
equivalent Dockerfile is `FROM` the base, `RUN` the install line in
`build.py`, `COPY` the tree over `/root/miles`, and the same `ENV`.

For publishing it, see Landing.

**Docker Hub cleanup.** Test builds of unmerged branches go to
`starlimobius/miles` under `pr-<topic>-<sha10>` tags (`build.py --release`),
so tags pile up. A `pr-` tag older than 14 days that no recorded run used (the
run monitor and the launch log name each run's image) is a deletion
candidate. List candidates with tag, age, size and last-used run, and delete
each one only after the user's OK. Never delete `main`, a `<sha10>` build of
`main`, a `stable-*` tag, or the frozen `fork-v0.1.2-rN` tags. Most of each
image's bytes are upstream layers shared with every other tag, so a deletion
frees only that build's two small layers.

## Landing

A change lands on `main` by fast-forward from its work branch, then the work
branch is deleted. Build an image when something that runs in it changed (a
fix, an extension, the deps layer); a doc-only landing can skip the build,
and `latest` then stays on the last build (which is why a run reads its
commit from the image, not from git). A pull request for the work branch
follows `.github/pull_request_template.md`.

**Review and merge.** While a PR is in review its author adds `fixup!`
commits (`git commit --fixup=<commit>`) rather than rewriting pushed commits,
so each review round reads as a plain diff; `ledger.py check` counts a fixup
as part of the commit it names. An author may force-push only their own topic
branch, and only to rebase it onto `main`, restack it, or autosquash it;
never `main`, a tag, or another author's branch. On the user's merge go
naming the PRs, the merge desk (the one thread that pushes `main`) folds and
lands them; it is the one exception, and may force-push a PR's topic branch
only to autosquash and restack it for that merge. Make fixups with
`git commit --fixup=<commit>`, which writes the subject form the ledger
matches (`fixup! <sha>` subjects are not matched):

```bash
git fetch origin && git checkout <topic>          # for a stack: the top PR's branch
GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash --update-refs origin/main   # folds fixups, restacks onto main
python fork/tools/ledger.py check --merge         # must pass: no fixup left, one commit per fix or hook id
git push --force-with-lease origin <topic> [<lower stack branches>]           # GitHub then sees each PR's head on main
# then fast-forward main to the approved PR's head, as below
```

- A stack is restacked onto `main` before its bottom PR merges; `--update-refs`
  moves every branch of the stack in the same rebase.
- If the rebase stops on a conflict, abort it and hand the PR back to its
  author: the merge desk does not resolve conflicts.
- When nothing but fixups changed, the folded tree equals the reviewed head's
  (`git diff --quiet <reviewed head> <topic>`); otherwise `main` moved, and the
  ledger check on the folded branch is the gate.

```bash
python fork/tools/ledger.py check                 # must pass, on the work branch
git checkout main && git pull --ff-only
git merge --ff-only <topic> && git push origin main
git push origin --delete <topic>
set -a; . <dir with .env>/.env; set +a            # DOCKERHUB_USERNAME / DOCKERHUB_TOKEN (read-write)
python fork/docker/build.py --main                # pushes :<sha10>, then moves :main (= latest) to it
modal run -m miles_ext.launch.modal_app --run-id <id> --check-only   # dry check on the new image
```

- `--main` builds only `origin/main`'s tip, so `:main` only moves forward.
  If `main` moved on in the meantime, fetch and land on top: a fast-forward
  that fails means rebase the work branch (it is yours, unpushed to `main`),
  never force-push `main`.
- Nothing is committed after the build: the registry holds the digests, so
  there is no `stack: publish ...` commit any more.
- `build.py --workdir DIR` reuses the deps layer across attempts, but crane
  also keeps its Docker Hub login in `DIR/docker`: delete that directory after
  changing the token, or the old credentials are sent and the push fails with
  401 "incorrect username or password".
- Without a registry push the launcher can build the same image on Modal from
  a pushed commit (`MILES_FORK_COMMIT=<sha>`); use it for testing, not for
  anything a platform run will use.

## Stable versions

The Mobius platform validates a `latest` build on `rl-staging` (a smoke test,
then one recipe trained to the end). Only after its owner reports that this
passed is the build marked stable, in four steps:

```bash
python fork/docker/build.py --mark-stable stable-YYYY-MM-DD \
    --image docker.io/starlimobius/miles-mobius@sha256:<digest>   # adds the tag; refuses to move one
git tag -a stable-YYYY-MM-DD <commit> -F <release note file>        # <commit> = MILES_MOBIUS_COMMIT of that image
git push origin stable-YYYY-MM-DD
# append {name, commit, image, date, note} to miles_ext/stable.json, then on a work branch:
git commit -am "stable: stable-YYYY-MM-DD" --trailer "Fork-Infra: stable-YYYY-MM-DD"   # land on main as usual
```

Then publish a GitHub Release for the tag with the same note.

- The git tag goes on the commit whose tree is in the image (the image's
  `MILES_MOBIUS_COMMIT`), not on the later commit that edits `stable.json`.
- `versions.resolve` reads `stable.json` from `main` on GitHub (falling back
  to the packaged copy), so a stable version exists for the platform as soon
  as that commit is on `main`.
- The tag message is the release note: what changed since the last stable
  version, and explicitly any change to what callers depend on (the
  `miles_ext.platform` API, the recipe/run-spec format, the image ref, Modal
  resource names, teardown). Write "no caller-facing changes" when there are
  none.
- `stable-2026-10-15` is r6 (commit 28dcfeb141, image in the old
  `starlimobius/miles` repository). `--mark-stable` copies an image from
  another repository into `stack.IMAGE_REPO` when needed.

## W&B projects

This repo's maintainers log to the `mobiuslab` entity (users set their own
`tracking.wandb_entity`; unset, W&B uses the API key's default entity), **one
project per campaign**: a set of runs
with one purpose, compared with each other (e.g. `miles-fork-validation` for
release checks, `miles-sail` for the Sail sandbox rounds, `miles-dsv41` for
DeepSeek-V4.1). A different purpose, model family or sandbox provider means a
new project.

- The recipe names it: `tracking.wandb_project: miles-<campaign>`. There is no
  default; `recipe.load` rejects a recipe without one, so nothing lands in a
  project by accident.
- Smoke and debug runs go to `miles-smoke`: give them a run id starting with
  `smoke-` or `debug-` and the launcher switches the project (recipes cannot
  name `miles-smoke` themselves).
- `miles-swe-rl` holds everything from before this rule. It is an archive:
  no new runs, and nothing in it is moved or deleted without the owner's OK.
- The W&B group is the run id (`<recipe>-<yyyymmdd-hhmmss>` by hand unless you
  pass `--run-id`; the platform's run UUID); the run id is also what resumes a
  run. The display name is `tracking.wandb_name` when set (the platform sets
  `<short id> · <run name>`), else Miles' default `<run id>_<random>-RANK_0`.
- Platform runs log to `miles-mobius-<env>-<project slug>`, one project per
  platform project and environment; the platform names it, the campaign rule
  above is for hand-launched runs.
- Tags are set for every run: `recipe:`, `model:`, `sandbox:`, `harness:`,
  `miles-mobius:<trainer>` (`latest`, `stable-YYYY-MM-DD`, or the image's version
  when launched by hand), `upstream:<tag>` (`<tag>+<sha10>` when `main` is past the tag) and `commit:<sha10>`, plus any in
  `tracking.tags` (the platform adds `env:` and `project:`). Runs before
  2026-10-15 carry `fork:<release>` instead of `miles-mobius:` and `upstream:`.
  Filter by these instead of encoding them in run names.

## Sync (merging a newer upstream release)

`main` takes upstream by merge, never by rebase: a rebase rewrites every
commit callers pinned, and `main` cannot be force-pushed.

```bash
git fetch upstream --tags
python fork/tools/sync.py v0.1.3 --dry-run   # which fixes and hooks may conflict; upstream diff of REVIEW_PATHS
git checkout -b sync-v0.1.3 main
git merge --no-commit v0.1.3                # conflicts only where fixes and hooks touch
git commit -m "sync: merge upstream v0.1.3" --trailer "Fork-Infra: sync-v0.1.3"
# in miles_ext/stack.py: UPSTREAM_TAG and UPSTREAM_BASE = "v0.1.3", BASE_IMAGE = the tag's image by digest
git commit -am "stack: base image for v0.1.3" --trailer "Fork-Infra: sync-v0.1.3"
python fork/tools/ledger.py check
```

**Between releases.** Prefer release tags; take upstream `main` only when a
change we need is there, and then only a commit that a published upstream
image was built from (today the `radixark/miles:dev` builds; the commit is in
the image's `/root/miles/.git`), so code and image match. The merge carries
`Fork-Infra: sync-<sha9>`; `UPSTREAM_BASE` becomes that commit and
`BASE_IMAGE` that image by digest, while `UPSTREAM_TAG` stays the newest
merged release tag. The ledger and the image's whiteouts diff against
`UPSTREAM_BASE`; image labels and W&B tags show `UPSTREAM_LABEL`
(`v0.1.2+af4413b74f`). The 2026-10-15 merge of upstream `af4413b74` is the
example (`sync-af4413b74`).

Only `--dry-run` of `sync.py` fits this flow; without it the tool still runs
the old rebase onto a new `fork/<tag>` branch (see "Not yet in the tools").

On a conflict, read the fix's commit message and patch page. If the new
release fixed the bug, take upstream's side and drop the markers, mark the
patch page removed, and add a `Retires: <id>` trailer to the sync's stack
commit (one per retired fix or hook; `ledger.py` then stops asking for its
files and markers and lists it as retired); otherwise re-apply the same intent to the new code and
keep the markers. Prune switches no run uses ("Flags, recipe keys and env
vars"). After the merge: set `HARBOR_COMMIT` from the new tag's
`examples/experimental/harbor/README.md`, recheck vendored files (drop any
that upstream now ships), recheck SGLang patches when the base image's SGLang
moved (SGLang patches, "On a sync"), read the `REVIEW_PATHS` diff for API drift in what
`miles_ext` builds on, then land and build as usual and run the launcher's
`--check-only`, a save/stop/resume smoke and one recipe. The build becomes
`latest`; staging decides whether it becomes stable.

## Not yet in the tools

RFC 0001 changes some tools as well as these rules. Until those changes land:

- `sync.py` without `--dry-run` rebases and rewrites `FORK_RELEASE` and
  `FORK_IMAGE`; do the merge by hand as above. Its `--dry-run` still compares
  against `UPSTREAM_TAG`, so after a between-releases sync it also lists
  upstream changes `main` already has.
- `stack.FORK_RELEASE` and `stack.FORK_IMAGE` still name r6 and stay
  frozen: they are what the platform runs until it resolves versions, and
  `validate`'s deep check, `finalize_hf` and the weights download still use
  `FORK_IMAGE`. Landings do not change them.

## SGLang patches

The image's SGLang is upstream's: `BASE_IMAGE` ships the `sglang-miles`
branch of sgl-project/sglang, installed editable at `stack.SGLANG_SRC`
(`/sgl-workspace/sglang`) at commit `stack.SGLANG_COMMIT`. Upstream Miles
changes its dependencies in two ways: commits on a branch it keeps in the
dependency (`sglang-miles`, Megatron-LM `miles-main`), and `.patch` files
in `docker/patch/` (flash-linear-attention, TileLang, TileKernels) that the
Dockerfile applies with `patch --fuzz=0`, each with its upstream PR and a
"drop when" line, failing the build when one does not apply. We cannot commit
to `sglang-miles`, so we carry SGLang changes the second way.

**What may be patched.** A Fix or a Hook, under the same rules as in Miles
code: a bug fix, or the smallest extension point an extension needs. It
copies a pattern SGLang already uses for the same job (name it on the patch
page) and stays small: about 30 changed lines; a larger one needs the user's
OK on its PR. No new server flags, kernels, model support or features: those
go to SGLang. Never change SGLang at runtime from `miles_ext`
(monkeypatching, a script that edits the installed files): what is not a patch
file is not in the image's record. A runtime applier is fine on a throwaway
image during an investigation; it never lands.

**How.**

- One patch per id: `fork/deps/sglang/<id>-<slug>.patch`, the `git diff`
  of the SGLang checkout at `SGLANG_COMMIT` (paths from the SGLang repo
  root). Ids are the ordinary `fix-NNN` / `hook-NNN` series.
- Every file it edits gets an added line carrying `[fork <id>]`, so grep finds
  it in the image.
- One commit, touching only the patch file, with the usual trailers.
  `Touches:` lists the SGLang files as `sglang:python/sglang/...`;
  `Upstream-Status:` is the SGLang PR, or `not filed; also on sgl-project/sglang
  main <sha9>` (or `not on main <sha9>`); `Remove-When:` names the SGLang
  release or `sglang-miles` commit that has the fix.
- A patch page, always, in its own `patch-<id>` infra commit as usual. Its
  *Decision* names the SGLang code it mirrors; *Remove when* repeats the
  upstream status checked against SGLang main.
- Every SGLang patch is also an upstream candidate: list it wherever the
  project tracks fixes worth sending upstream.

`ledger.py check` enforces the shape: the file name, the `sglang:` touches
matching the files the patch edits, a marker in each, and that only the
patch's own commit (or a `sync-` commit) adds, changes or deletes it.

**Build.** `fork/deps/sglang/apply.py` applies every patch with one `git
apply` (exact context, no fuzz, all or none). When there is at least one
patch, the checkout must be at `SGLANG_COMMIT`. `build.py --main` runs it in
the deps layer's container and puts the changed files in the deps layer; the
image label `miles.mobius.sglang-patches` lists the patches. The launcher's
build-from-commit image (`MILES_FORK_COMMIT`) runs it after the checkout,
then `ledger.py markers`, which finds each patch's marker in the installed
SGLang. A patch that no longer applies, or an SGLang that moved, fails both
builds.

**Tests.**

1. `python fork/tools/ledger.py check`.
2. The patch applies at the pin and, for the upstream status, on SGLang main
   (CPU, seconds):

   ```bash
   git clone --filter=blob:none https://github.com/sgl-project/sglang /tmp/sglang
   git -C /tmp/sglang checkout <SGLANG_COMMIT>
   python fork/deps/sglang/apply.py /tmp/sglang <SGLANG_COMMIT> --check
   git -C /tmp/sglang checkout origin/main && git -C /tmp/sglang apply --check "$PWD"/fork/deps/sglang/<patch>
   ```

3. The launcher's `--check-only` from the pushed commit: the image builds
   and `ledger.py markers` lists the patch.
4. Evidence that it fixes what it says: a GPU run on the patched image beside
   the same run without the patch, both cited with numbers in the patch page.
   Run SGLang's own unit test of the area too when one runs on CPU.

**On a sync.** When the new `BASE_IMAGE` ships another SGLang commit, set
`SGLANG_COMMIT` from it (`git -C /sgl-workspace/sglang rev-parse HEAD` in the
image) and run step 2. A patch whose fix SGLang now has is deleted and its
page marked removed; one that no longer applies is redone against the new
code with the same id and markers. Both happen in the `sync-` commit.

**When to switch to a fork.** More than about five live SGLang patches, or one
that needs redoing on most syncs: propose a `listar2000/sglang` branch off the
pinned `sglang-miles` commit (as for Harbor below), so the patches become
commits that merge like ours.

## Harbor

The fork depends on Harbor's `harbor-miles-*` branch, the one Miles documents
for each release, pinned by commit in `miles_ext/stack.py`. We do not fork
Harbor today because we carry no change to installed Harbor code:

- our agents (`SandboxMiniSweAgent`, `HostMiniSweAgent`) are Harbor agents
  loaded by `import_path`; Harbor resolves environments the same way
  (`EnvironmentConfig.import_path`), so a new sandbox provider is an
  extension in `miles_ext/`, not a Harbor change (Sail is one: Harbor's own
  provider, newer than our pin, carried verbatim; fork/patches/ext-003-harbor-sail.md);
- the one Harbor patch we have (`deps/harbor/0001-…`) is in the SWE-Gym
  adapter, which generates task directories offline and is not part of the
  installed package.

The switch: the first change we need in installed Harbor code goes into a
`listar2000/harbor` fork branched from the pinned commit, with the same
four-kinds rules, and `HARBOR_REPO`/`HARBOR_COMMIT` point at it. Until then a
fork would only add a second sync to every upstream release.
