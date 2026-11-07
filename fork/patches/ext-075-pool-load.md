# ext-075: pool load reads SGLang's /v1/loads

## Why
A two-node split run on real SGLang (image `migration-a49eeadbd0`) reported `running: null` and
`waiting: null` for every engine in `pool:<run_id>`, while its engine logs showed about 13 running requests each. A
launcher could not scale from that. At the pinned SGLang (3a938213c1), `/server_info`'s `internal_states` does not carry
these counts. `/v1/loads` does, per DP rank.

## Decision
Read `/v1/loads?include=core` and sum the DP ranks, so an engine stays one entry in `load`. An engine that does not
answer, or a reply without the fields, stays `null`. Rejected: Prometheus `/metrics`, which needs `--enable-metrics`.

## How it is applied
`miles_ext/node/rollout.py` (`_load`), test in `miles_ext/tests/test_node_rollout.py`. The change is checked against a
fake engine in the shape of SGLang's `entrypoints/v1_loads.py`; no real engine has run it yet. On an SGLang bump, recheck
that route.

## Remove when
With ext-063.
