# This fork

`listar2000/miles` carries a small, ordered series of commits on top of an
upstream Miles release. This file is the rulebook for that series; the record
of what is in it is the commit series itself (`python fork/tools/ledger.py list`).
The important changes also have a short page each (Patch pages, below).
Agents: `.claude/skills/miles-fork/SKILL.md` is the step-by-step workflow.

## Branches and releases

This repository is standalone, not a GitHub fork: it holds only our refs, and
upstream is reached as a git remote (`git remote add upstream
https://github.com/radixark/miles`). Only these refs may exist here:

| Ref | What |
| --- | --- |
| `fork/<upstream tag>` | a release line: that upstream tag plus our series, e.g. `fork/v0.1.2`; the newest is the default branch |
| `fork/<upstream tag>-<topic>` | a work branch off a release line; deleted once it lands (fast-forward) |
| tag `fork-<upstream tag>-r<N>` | a released commit; the Docker Hub image has the same tag |
| tag `<upstream tag>` | the upstream tag a release line is based on, e.g. `v0.1.2` (same object as upstream's) |
| tag `archive/<name>` | history we replaced but keep reachable, e.g. `archive/fork-v0.1.2-dev` |

No upstream branches (not even `main`) and no other upstream tags: fetch those
from `upstream` when needed, never push them here.

Upstream is never merged into a release line. Moving to a newer upstream
release creates a new line by rebasing the series (see Sync). Release tags
start with `fork-` so they cannot match upstream's tag-triggered workflows
(`release-docker.yml` fires on `v[0-9]+.*`).

## Four kinds of change

Every fork commit has exactly one trailer saying what it is.

| Kind (trailer) | What it is | Where it lives | Can it conflict on a sync? |
| --- | --- | --- | --- |
| **Fix** (`Fork-Fix: fix-NNN`) | a bug fix to upstream code | in place, in the upstream file | yes: it is the only thing that can |
| **Hook** (`Fork-Hook: hook-NNN`) | an extension point upstream lacks, so an extension can plug in | in place, in the upstream file | yes |
| **Infra** (`Fork-Infra: <id>`) | fork machinery: image build, tools, the stack pins, docs | `fork/`, `miles_ext/stack.py`, `FORK.md`, `.claude/skills/miles-fork/` | no (new files only) |
| **Ext** (`Fork-Ext: ext-NNN`) | new capability: harnesses, recipes, algorithms, sandbox providers | `miles_ext/` | no (new files only) |

Rules for **fixes and hooks** (enforced by `ledger.py check`):

- One commit per fix, as small as the bug allows. Prefer inserting lines over
  rewriting upstream lines: an insertion survives most upstream edits around it.
- Mark every touched spot with `[fork fix-NNN]` (or `hook-NNN`) in a comment,
  so `grep -rn "\[fork "` finds every place we changed.
- Trailers: `Touches:` (the upstream files), `Evidence:` (the failure it fixes),
  `Upstream-Status:` (not filed / PR link / merged in vX), `Remove-When:`.
- They come first in the series, so a rebase meets every possible conflict
  before the commits that cannot conflict.

Rules for **extensions**: new files under `miles_ext/` only, plugged in
through Miles' own extension points (`--custom-generate-function-path`,
`--custom-agent-function-path`, `--custom-rm-path`, filter paths) and
Harbor's `import_path` for agents and environments. If an extension needs a
hook that does not exist, add a Hook commit with the smallest possible change
rather than editing upstream code from the extension's commit.

`python fork/tools/ledger.py markdown` prints the current series as a table.

## Patch pages

`ledger.py` lists every change; a few deserve an explanation a reader can
find without reading commit messages. Those get a page.

- **Where:** `fork/patches/<fork-id>-<slug>.md`, e.g. `fork/patches/ext-003-harbor-sail.md`,
  and one row in the index `fork/patches/README.md`.
- **When:** required for every Hook, every vendored upstream file (below) and
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

Edits to installed dependency code (not whole files) are not done in this
repo; see Harbor below for when that becomes a dependency fork.

## Layout

```
FORK.md                 pointer to this file
fork/
  README.md             this rulebook
  docker/build.py       publish the fork image: upstream image + deps layer + our source layer
  tools/ledger.py       list / audit the series (touch points, markers, trailers, order)
  tools/sync.py         rebase onto a newer upstream tag + stack bump
  patches/README.md     index of patch pages; patches/<fork-id>-<slug>.md, one page each
  rfcs/README.md        index of RFCs (process and design proposals); rfcs/NNNN-<slug>.md
  vendored.json         upstream files carried verbatim in miles_ext/ (blob-checked)
  deps/harbor/          offline Harbor patches (task-set generation only; see Harbor)
.claude/skills/miles-fork/  the workflow for agents changing this fork
miles_ext/              extensions (top-level package next to miles/, on PYTHONPATH in the image)
  stack.py              pins: base image digest, Harbor commit, mini-swe-agent, Sail SDK; the published fork image
  harbor/               agent function, in-sandbox and host-side mini-swe-agent, reward, session proxy;
                        harbor/sail/: Harbor's Sail provider (vendored) and the fork's subclass
  recipes/              recipe YAML -> Miles argv + Harbor env; model presets
  launch/modal_app.py   test launcher for Modal (not platform API)
  platform/             the platform API (ext-006): schema, validate, launch, observe, teardown, secret, weights, checkpoints
```

`miles_ext` is not added to upstream's `setup.py` (that would be a touch point
for no benefit): the image sets `PYTHONPATH=/root/miles`.

Upstream code that lives outside the `miles` package, such as Miles' Harbor
agent function under `examples/experimental/harbor/`, is loaded from the tree
by path (`miles_ext/harbor/upstream.py`) instead of being copied, so a sync
brings the new version and API drift fails at import, not silently.

## The image

`fork/docker/build.py` publishes `docker.io/<user>/miles:fork-<tag>-r<N>`:
upstream's release image (amd64, every layer untouched) plus two layers.

| Layer | Contents | Made by |
| --- | --- | --- |
| upstream (117 layers) | CUDA, torch, SGLang, Megatron, compiled extensions, `/root/miles` at the upstream tag | radixark; mounted from `radixark/miles` on Docker Hub, never re-uploaded |
| deps | Harbor at `HARBOR_COMMIT`, mini-swe-agent and the Sail SDK (`stack.DEPS_INSTALL`): exactly the files of the distributions `uv pip install` added or replaced (their `RECORD`s), whiteouts for the versions it removed | a Modal container of the base image runs the install |
| src | the released fork commit (`git archive`) over `/root/miles`, whiteouts for files the commit deletes | local `git` |

The image env adds `PYTHONPATH=/root/miles` (so the top-level `miles_ext`
imports; upstream's `setup.py` is not edited), `MILES_FORK_RELEASE` and
`MILES_FORK_COMMIT`; OCI labels record the source commit, base digest and the
Harbor, mini-swe-agent and Sail SDK pins. `/root/miles/.git` still describes upstream's
checkout, so read the commit from `MILES_FORK_COMMIT`, not `git rev-parse`.

Why layers appended with `crane` instead of a Dockerfile: this machine has no
Docker daemon, CI with registry secrets was not an option, and Docker inside a
Modal Sandbox could not mount the 117-layer base (overlay mount "invalid
argument"; small images worked). Appending also keeps every upstream layer
byte-identical and shared, so hosts that already cached upstream's image
(Modal does) pull only the two new layers. With a normal Docker host the
equivalent Dockerfile is `FROM` the base, `RUN` the install line in
`build.py`, `COPY` the tree over `/root/miles`, and the same `ENV`.

For publishing it, see Releases.

## Releases

A release is an image plus a tag. Cut one when something that runs in the
image changed (a fix, an extension, the deps layer), not for doc-only commits.

```bash
set -a; . <dir with .env>/.env; set +a            # DOCKERHUB_USERNAME / DOCKERHUB_TOKEN (read-write)
python fork/tools/ledger.py check                 # must pass
git push origin fork/v0.1.2                       # the commit must be on GitHub before the build
python fork/docker/build.py --release fork-v0.1.2-rN   # prints FORK_IMAGE = "docker.io/...@sha256:..."
# set FORK_RELEASE and FORK_IMAGE in miles_ext/stack.py, then:
git commit -am "stack: publish fork-v0.1.2-rN image" --trailer "Fork-Infra: release-fork-v0.1.2-rN"
git tag -a fork-v0.1.2-rN -m "<release note>" && git push origin fork/v0.1.2 fork-v0.1.2-rN
modal run -m miles_ext.launch.modal_app --run-id <id> --check-only   # dry check on the published image
```

- The tag goes on the commit that records the digest; the image is built from
  its parent, so the two trees differ only in that line.
- The tag message is the release note: what changed since the last release,
  and explicitly any change to what callers depend on (the `miles_ext.platform`
  API, the recipe/run-spec format, the image ref, Modal resource names,
  teardown). Write "no caller-facing changes" when there are none.
- `build.py --workdir DIR` reuses the deps layer across attempts, but crane
  also keeps its Docker Hub login in `DIR/docker`: delete that directory after
  changing the token, or the old credentials are sent and the push fails with
  401 "incorrect username or password".
- Without a registry push the launcher can build the same image on Modal from
  a pushed commit (`MILES_FORK_COMMIT=<sha>`); use it for testing, not releases.

## W&B projects

Runs go to the `mobiuslab` entity, **one project per campaign**: a set of runs
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
  when launched by hand), `upstream:<tag>` and `commit:<sha10>`, plus any in
  `tracking.tags` (the platform adds `env:` and `project:`). Runs before
  2026-10-15 carry `fork:<release>` instead of `miles-mobius:` and `upstream:`.
  Filter by these instead of encoding them in run names.

## Sync (moving to a newer upstream release)

```bash
git fetch upstream --tags
python fork/tools/sync.py v0.1.3 --dry-run   # series, which fixes may conflict, upstream diff of REVIEW_PATHS
python fork/tools/sync.py v0.1.3             # branch fork/v0.1.3, rebase --onto, stack bump, ledger check
```

On a conflict (fixes and hooks only), the commit message says why the change
exists and when it can go. If the new release fixed the bug, `git rebase
--skip`; otherwise re-apply the same intent to the new code, keep the markers,
`git rebase --continue`, then `sync.py v0.1.3 --finish`. After that: set
`HARBOR_COMMIT` from the new tag's `examples/experimental/harbor/README.md`,
read the `REVIEW_PATHS` diff for API drift in what `miles_ext` builds on,
build the image, and run the launcher's `--check-only`, a save/stop/resume
smoke and one recipe before tagging `fork-v0.1.3-r1`.

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
fork would only add a second rebase to every sync.
