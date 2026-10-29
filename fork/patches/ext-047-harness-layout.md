# ext-047: harnesses and sandbox providers out of `miles_ext/harbor/`

## Why
`miles_ext/harbor/` held three different things: Harbor as the trial runner, the sandbox providers
(Modal, Sail) and four agent harnesses. A harness such as dsh is not Harbor-specific, so its pins and
assets read as misplaced there, and each new harness or provider made the folder harder to scan.

## Decision
Split by role; nothing changes in behaviour, and recipe harness names stay:
- `miles_ext/harbor/`: the trial runner (agent function, trial processes, task hashes, reward,
  session proxy, the upstream loader).
- `miles_ext/harnesses/<harness>/`: one package per harness (`mini_swe_agent/`, `terminus/`, `deepseek/`),
  holding its Harbor agent class with its pins and assets; `harnesses/common.py` holds what the
  in-sandbox ones share (`PublicSetup`, the install retry budget).
- `miles_ext/sandboxes/`: `modal_env.py`, `sail/` (vendored file included), `image_refs.py`.
- Each harness package keeps a thin Harbor agent class, since Harbor is our only trial runner.
- Rejected: splitting each harness again into a Harbor adapter and a runner-free core (two files per
  harness for a second runner we do not have).

## How it is applied
`git mv` of the files; imports, Harbor import paths (`_IMPORT_PATHS`, `SAIL_IMPORT_PATH`,
`modal_env.IMPORT_PATH`), tests, patch pages, the guide and `fork/vendored.json` follow.
`miles_ext/harbor/sail.py` and `miles_ext/harbor/modal_env.py` re-export the two provider classes under
their old import paths, for run specs that name them. Check: `miles_ext/tests` and
`python fork/tools/ledger.py check` (vendored blob at its new path).

## Remove when
The two old-path aliases go after the next stable version; the layout itself stays.
