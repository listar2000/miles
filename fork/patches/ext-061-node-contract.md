# ext-061: the image's node entry, `python -m miles_ext.node`

## Why
RFC 0005: a launcher should learn what a run needs without parsing a recipe or importing Modal. Today that knowledge
is `recipe.shape` read by `modal_app`, plus `platform.secrets.required_keys`.

## Decision
`miles_ext/node/contract.py` is the one owner of the contract's names and of `manifest`; `platform.secrets.required_keys`
is now an alias of `contract.required_secrets`. The package imports no provider SDK; `test_node_contract.py` runs the
CLI with `import modal` failing. `manifest` carries `cluster.*` as advice: placement keys stay in the recipe.

## How it is applied
`python -m miles_ext.node manifest --spec <file>`. Other roles join in the phases that build them. On a sync: nothing.

## Remove when
Never while the image has callers. Not an upstream candidate.
