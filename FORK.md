# listar2000/miles: a fork of radixark/miles

This branch is an upstream Miles release plus a short, ordered series of fork
commits: bug fixes and hooks in upstream files (marked `[fork fix-NNN]` /
`[fork hook-NNN]`), then fork machinery under `fork/` and extensions under
`miles_ext/`. Rules, layout, image build and sync: [fork/README.md](fork/README.md).

    python fork/tools/ledger.py list    # what the fork changes, from the commit trailers
    python fork/tools/ledger.py check   # touch-point audit against the upstream tag
