# ext-001: host-side mini-swe-agent calls the model through the OpenAI client

## Why
`mini-swe-agent-host` (`miles_ext/harbor/host_agent.py`) runs every episode's agent loop in the rollout process, one
thread per episode. mini-swe-agent's `LitellmModel` sends through litellm's sync path, whose calls queue for a free
connection in one shared pool once ~100 calls are in flight. In a CPU benchmark against a mock server (script not in
the repo) litellm added +4.7 s per call at p90 with 128 threads and +4.6 s at p50 with 256; the OpenAI client on one
httpx pool added about 0.01 s. At 256 agents a pilot with the litellm model did not finish in 400 s; with the OpenAI
client it finished in 98 s at 13 ms p50 overhead per call. The in-sandbox harness is unaffected: each sandbox runs one
agent.

## Decision
`_OpenAIClientModel`, a `LitellmModel` subclass that overrides `_query` to call `openai.OpenAI.chat.completions.create`
on one process-wide `httpx.Client` (no pool limit; `--async-max-concurrent-samples` bounds the calls in flight), with
`max_retries=0` so mini-swe-agent's own retry loop stays the only one. Everything else (message formatting, tool-call
parsing, retries, cost tracking off) stays mini-swe-agent's.

- Request body: the OpenAI client's own parameters (`_OPENAI_PARAMS`) are passed as-is; any other sampling parameter
  (`top_k`, `min_p`, `repetition_penalty`, ...) goes in `extra_body`, which puts it at the top level of the JSON body.
  That is what litellm 1.106 sends for the `openai/` provider with `drop_params`; checked by recording both bodies for
  the same kwargs against one server (identical apart from `tools`). `drop_params` itself is consumed.
- 4xx answers (`BadRequestError`, `NotFoundError`, `PermissionDeniedError`, `AuthenticationError`,
  `UnprocessableEntityError`) are added to `abort_exceptions`: they repeat on retry.
- Alternative not taken: tuning litellm's connection pool (not benchmarked). The OpenAI client is the one litellm
  itself wraps for this provider, so dropping the litellm layer leaves the request body as the only thing to match.

## How it is applied
- `HostMiniSweAgent._build_agent` builds the model from `_openai_client_model_class()`.
- Verify: `python -m pytest miles_ext/tests/test_host_agent.py`
  (`test_openai_client_model_sends_what_litellm_sends` pins the body).
- On a mini-swe-agent upgrade: `LitellmModel._query` is private; re-run that test first.

## Remove when
litellm's sync path no longer stalls with 256 concurrent calls in one process, or the host harness is retired
(host-side vs in-sandbox harness A/B in progress).
