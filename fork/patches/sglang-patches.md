# sglang-patches: bug fixes and small hooks to SGLang as patch files

## Why

The fork could not change installed dependency code at all (only whole vendored
files), so SGLang bugs that bite our runs were worked around with runtime appliers
on investigation images. The first case is DSpark speculative decoding dropping the
verify forward's routing (R3) rows: every decoded token's row was zero, while EAGLE
and DFlash pass them through. The user allowed SGLang patches on 2026-10-21 provided
we patch the way Miles does.

## Decision

- **Patch files applied at image build**, as upstream Miles does for dependencies it
  cannot commit to (`docker/patch/*.patch`, `patch --fuzz=0`, an upstream link and a
  drop condition each). Upstream's other way, commits on `sglang-miles`, is not open
  to us. Rejected: runtime monkeypatching (invisible in the image, breaks silently on
  an SGLang bump); a `listar2000/sglang` fork now (a second sync per release for a few
  lines; it becomes the route past about five patches).
- **Under `fork/deps/sglang/`, not `docker/patch/`**: `docker/` is an upstream path the
  ledger reserves for marked fixes, and our image is not built by upstream's
  Dockerfile but by `fork/docker/build.py` on top of upstream's image.
- **Ordinary `fix-NNN` / `hook-NNN` ids** with `Touches: sglang:<path>`: one id space,
  one set of trailers, and `ledger.py markdown` lists them with the rest. Rejected: a
  new kind, which would add a fifth set of rules for the same thing.
- **`git apply`, exact context, all or none**, and the SGLang commit pinned in
  `stack.SGLANG_COMMIT` once any patch exists: a moved SGLang or a stale patch stops
  the build instead of shipping unpatched.

## How it is applied

`fork/deps/sglang/apply.py` (one `git apply`, pin check), run by `fork/docker/build.py`
in the deps layer's container (the changed files join the deps layer; image label
`miles.mobius.sglang-patches`) and by `miles_ext/platform/images.py` in the
build-from-commit image, followed there by `ledger.py markers`, which finds each
patch's marker in the installed SGLang. `ledger.py check` enforces names, `sglang:`
touches and markers. Rules for authors and reviewers: `fork/README.md` "SGLang
patches", the miles-fork and miles-review skills. With no patch, builds are unchanged.

## Remove when

Never as a mechanism; replaced by a `listar2000/sglang` branch if the patches outgrow
files (fork/README.md "SGLang patches", "When to switch to a fork").
