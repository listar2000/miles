# ext-065: every role and job as a node command

## Why
RFC 0005: a caller with no Modal needs a command for everything a node does.

## Decision
`python -m miles_ext.node` runs each role and job; `--help` lists them. A role refuses to start when the contract's env
is missing and names what is missing (`contract.missing`), since a GPU node costs money from its first second. SIGTERM
exits through `finally` on nodes without a supervisor, so a worker still commits. `modal_app` and `sft/modal_*` call the
same functions in process: a subprocess would lose the supervisor's main-thread signal handler.

`miles_ext/sft/job.py` is `sft/modal_prepare` and `sft/modal_train`'s bodies, moved as they were, with the commit passed
in; the Modal apps keep their own volumes until they leave.

## How it is applied
On a sync: nothing. Verify: `test_node_contract.py` and the SFT tests; a 2-step run through `modal_app` runs the same
trainer role.

## Remove when
Never while the image has callers. Not an upstream candidate.
