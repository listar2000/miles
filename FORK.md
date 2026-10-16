# listar2000/miles: a fork of radixark/miles

`main` is miles-mobius: an upstream Miles release, merged in, plus our
commits: bug fixes and hooks in upstream files (marked `[fork fix-NNN]` /
`[fork hook-NNN]`), fork machinery under `fork/` and extensions under
`miles_ext/`. Rules, versions, image build and sync: [fork/README.md](fork/README.md).
Pages for the important changes: [fork/patches/](fork/patches/README.md).
Agents changing the fork: [.claude/skills/miles-fork](.claude/skills/miles-fork/SKILL.md).

    python fork/tools/ledger.py list    # what the fork changes, from the commit trailers
    python fork/tools/ledger.py check   # touch-point audit against the upstream tag
