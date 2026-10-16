# RFC 0001: one running `main`, `latest` or a stable version

Status: accepted 2026-10-15; steps 1-3 of Order done (Actions off, `main` default, `stable-2026-10-15`).
Folded into the rulebook (`fork/README.md`) and skill
(`.claude/skills/miles-fork/SKILL.md`) on 2026-10-15; the ledger rewrite is
done, the sync tool change is still open (rulebook, "Not yet in the tools"). Nothing here deletes or moves a tag or an
image, and the platform keeps running r6 (`stack.FORK_IMAGE`) until it opts in.

## What is decided

From the user (2026-10-15): our Miles is "miles-mobius", and "fork" leaves
everything the platform sees. One running `main`; the image defaults to
`main`'s latest build. A run picks `latest` or a past stable version, never a
revision. Stable versions get dated tags, `stable-2026-10-15`, on the commit
and on the image, after a staging smoke test and one recipe trained to the
end. `latest` is frozen per run when the run is created. Specs get a
`trainer: latest | stable-...` field. r6 is the first stable version. Modal
names move to option A (`rl-dev-0bebf81d`, `rl-dev-0bebf81d-sbx`,
`rl-dev-results`, environments `rl-dev`, `rl-staging`).

Settled here: `main` takes upstream by **merge**, not rebase. The reason is
that a shared `main` cannot be force-pushed (a rebase rewrites every commit
callers pinned). Tags would keep old commits reachable either way, so
reachability is not the reason.

## Repo

| Ref | What |
| --- | --- |
| `main` | the running line, default branch; created from `fork/v0.1.2` at its tip |
| `<topic>` work branches | off `main`, land by fast-forward, then deleted (`fork/` prefix dropped) |
| tag `stable-YYYY-MM-DD` | a stable version; the image has the same tag. A second one on the same day: `stable-YYYY-MM-DD.2` |
| `fork/v0.1.2`, tags `fork-v0.1.2-r1..r6`, `archive/*` | kept as they are, frozen; no new commits or tags |
| tag `v0.1.2` and later upstream tags we merge | the upstream base, as today |

The stable git tag goes on the commit whose tree is in the image
(`MILES_FORK_COMMIT`): for r6 that is 28dcfeb141, the parent of the r6 tag
commit a2dd311cfe, which only recorded the digest.

**Upstream sync by merge.** `git merge v0.1.3` on a work branch; conflicts
(fixes and hooks only, as now) are resolved in the merge commit, which carries
`Fork-Infra: sync-v0.1.3`. Then the usual checks, land, build.

**Between releases (amendment, 2026-10-15).** Upstream tags every few weeks
and its `main` moves daily. A sync may take an untagged upstream commit, but
only one that a published upstream image was built from (today the
`radixark/miles:dev` builds; the commit is in the image's `/root/miles/.git`),
and `BASE_IMAGE` moves to that image by digest in the same change. Code and
image then match commit for commit, as with a release. The merge commit
carries `Fork-Infra: sync-<sha9>`. `stack.UPSTREAM_TAG` stays the newest
merged release tag (v0.1.2's lockfile commit sits only on upstream's release
branch, so it stays in our history); `stack.UPSTREAM_BASE` names the merged
commit, and the ledger and the image build diff against it. Prefer release
tags; take `main` only when a change we need is there.

## Ledger and rulebook changes

- `ledger.py` lists `<stack.UPSTREAM_BASE>..main`, not `<tag>..fork/<tag>`,
  leaving out commits reachable from `stack.UPSTREAM_TAG` (upstream's release
  branch commits) and merge commits. Done 2026-10-15.
- The "fixes and hooks first" order rule goes: with merges there is no replay
  order (done 2026-10-15). Markers, trailers, `ID_FORMATS`, one commit per fix and vendored blob
  checks stay as they are.
- `sync.py` becomes a merge helper; its `FORK_RELEASE`/`FORK_IMAGE` rewrites
  go, since no version lives in the tree any more.
- Release section: no more `stack: publish ...-rN` commits, since the tag no longer
  goes on a commit that records the digest. The registry holds the digests and
  `miles_ext/stable.json` holds the stable list.
- SKILL.md: work branches off `main`, "rebase onto a tag" becomes "merge a tag",
  and "Cutting a release" becomes "Landing" plus "Marking a version stable".

## Image build and release workflow

Every push to `main` runs `build.py --main` in GitHub Actions
(`.github/workflows/miles-mobius-image.yml`, also startable by hand); the deps
layer still installs on Modal. The job summary names the pushed digest. It
needs the repository secrets `MILES_MOBIUS_DOCKERHUB_TOKEN` (Docker Hub,
read-write) and `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` (workspace `mobius`).
A local `build.py --main` still works the same way.

1. **Landing.** Fast-forward `main` and push; the workflow then runs
   `build.py --main`. It pushes `main-<sha12>` (immutable) and moves the `main`
   tag (= latest) to it, only forward and only to `origin/main`'s tip. A
   doc-only landing can skip the build; `latest` then stays on the last build.
   That is why a run reads its commit from the image, not from git.
2. **Staging.** Run the smoke test on `latest` in `rl-staging`, then train one
   recipe to the end.
3. **Marking it stable.** Run
   `build.py --mark-stable stable-YYYY-MM-DD --image <repo>@sha256:...`. This adds
   the tag to the existing image and refuses to move one. Then add an annotated
   git tag on the build commit, whose message is the release note. Finally,
   append an entry to `miles_ext/stable.json` in a commit on `main`, with
   `Fork-Infra: stable-YYYY-MM-DD`.

The release note keeps its rule: say explicitly what changed for callers.

## What the platform reads (done on this branch, off by default)

- `versions.resolve(trainer)` returns `{trainer, name, commit, image}`.
  For `latest` it reads the registry's `main` tag and the commit from the image
  config. For a stable name it reads the stable list from `main` on GitHub,
  falling back to the packaged copy. It reads from GitHub because the
  platform's installed `miles_ext` may itself be an older stable version.
  Checked live against r6: it reads digest `a3715f98...` and commit 28dcfeb141.
- At run creation the platform calls it once and stores the result. It then
  passes `launch(image=...)` to every segment, so a resume never changes build.
- Not yet pinned: `validate`'s deep check, `finalize_hf` and the weights
  download still run in `stack.FORK_IMAGE`. They need an image argument before
  a stable version older than the platform's library is offered.

## "Fork" names that reach the platform

Next stable version. The old names stay readable for a transition.

| Today | Becomes |
| --- | --- |
| image env `MILES_FORK_RELEASE`, `MILES_FORK_COMMIT` | `MILES_MOBIUS_VERSION`, `MILES_MOBIUS_COMMIT` (old ones still set) |
| W&B tag `fork:<release>`, launch record `stack` | `trainer:<name>`, `trainer` |
| image labels `miles.fork.*` | `miles.mobius.*` |
| Docker Hub `starlimobius/miles` | `starlimobius/miles-mobius` from the first `main` build (old repo kept) |
| launcher defaults: app `miles-fork`, volume `miles-fork-results`, queue `miles-fork-completion` | `miles-mobius`, `miles-mobius-results`, `miles-mobius-completion` |
| W&B project `miles-fork-validation` | `miles-mobius-validation` |

Repo-internal names (`fork/` folder, `Fork-*` trailers, `[fork fix-NNN]`
markers) describe the patch series and are not platform-facing. They stay.

## Modal names, option A (next stable version; on `main` since 2026-10-15)

- The trainer accepts the `rl-` namespace next to `mdev-`, `mstg-` and `mrl-`.
  The old prefixes go only after dev's cleanup.
- Target takes a short name id (8 characters) alongside the run UUID. The GPU
  app is `rl-<env>-<short>`, the sandbox app `<app>-sbx` (a suffix, where today
  it is prefix + run id) and the secret `<app>-secret`. The shared names are
  `rl-<env>-results`, `rl-<env>-weights` and `rl-<env>-progress`.
- The run UUID still keys files on the volume and the W&B run, group and resume.
- Cleanup keeps matching exact names only. The platform thread owns refusing
  a reused short id.

## Found while reading: GitHub Actions

`listar2000/miles` runs upstream's workflows, with Actions enabled for all.
The scheduled `Docker Build & Push`, `Build Wheels` and `Runner Utilization`
workflows have run 10 times since 2026-10-14 on `fork/v0.1.2`. Build Wheels
fails; Docker Build is gated off. A branch named `main` adds push triggers:
`pre-commit` on every push, and `docker-build` (which pushes upstream images)
when an upstream merge changes `docker/Dockerfile`. We use no Actions, so
turning Actions off for the repo is the clean fix. Editing the workflow files
would add upstream touch points. Tag triggers: `release-docker` fires on
`v[0-9]+.*` only, so `stable-*` and `main-*` match nothing.

## Order

1. **Needs OK:** turn off GitHub Actions on `listar2000/miles` (Settings → Actions → Disable).
2. **Needs OK:** create `main` from this branch once reviewed and make it the default branch.
   `fork/v0.1.2` stays, frozen.
3. **Needs OK** (approved in principle, as r6 = first stable):
   - push git tag `stable-2026-10-15` on 28dcfeb141;
   - run `build.py --mark-stable stable-2026-10-15 --image starlimobius/miles@sha256:a3715f98...`.
4. Next change set on `main`, which becomes the next stable candidate after staging:
   - option A names;
   - the `rl-` namespace;
   - the "fork" renames above;
   - image args for the platform functions;
   - ledger, sync and rulebook rewrite.
5. The platform adds `trainer` to specs and calls `versions.resolve` at run creation.
