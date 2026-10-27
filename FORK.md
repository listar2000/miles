# listar2000/miles: a fork of radixark/miles

`main` is miles-mobius: an upstream Miles release, merged in, plus our
commits: bug fixes and hooks in upstream files (marked `[fork fix-NNN]` /
`[fork hook-NNN]`), fork machinery under `fork/` and extensions under
`miles_ext/` (recipes, Harbor harnesses, the Modal launcher). It runs Miles RL
on Modal from any machine with your own credentials.

## Use it: train with the fork

[fork/guide/](fork/guide/) is for people and agents who run training and plug in
algorithms, harnesses, rewards or data:

| Page | For |
| --- | --- |
| [quickstart.md](fork/guide/quickstart.md) | a fresh machine and Modal account to a first run |
| [recipes.md](fork/guide/recipes.md) | what a run is, and where every knob is defined |
| [extending.md](fork/guide/extending.md) | where new algorithm, harness, reward and data code goes |
| [running.md](fork/guide/running.md) | logs, outputs, W&B, stop and resume, first checks on failure |
| [limits.md](fork/guide/limits.md) | what is known not to work yet |
| [help.md](fork/guide/help.md) | the responsibility boundary, issues and PRs |

Agents: [.claude/skills/miles-use](.claude/skills/miles-use/SKILL.md).

## Change it: maintain the fork

Rules, versions, image build and sync: [fork/README.md](fork/README.md).
Pages for the important changes: [fork/patches/](fork/patches/README.md).
Agents changing the fork: [.claude/skills/miles-fork](.claude/skills/miles-fork/SKILL.md).

    python fork/tools/ledger.py list    # what the fork changes, from the commit trailers
    python fork/tools/ledger.py check   # touch-point audit against the upstream tag
