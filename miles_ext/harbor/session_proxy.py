"""Reverse proxy that lets agents inside Modal Sandboxes reach Miles' session servers.

Sandboxes cannot dial the trainer container's private address, so the launcher
publishes this proxy through one Modal tunnel (``modal.forward``) and the agent
function rewrites each session URL from ``http://<head-ip>:<port>/sessions/...``
to ``<tunnel-url>/<token>/<port>/sessions/...``.

The token keeps the public tunnel from being an open relay, and only the
session-server port range is forwarded (never the SGLang engines or Ray).

    python session_proxy.py --listen-port 29999 --upstream-host 10.0.0.1 \
        --ports 30000-30031 --token <secret>
"""

import argparse

from aiohttp import ClientSession, ClientTimeout, web

# hop-by-hop headers and the ones aiohttp recomputes
_DROP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "content-encoding"}


def make_app(upstream_host: str, ports: range, token: str) -> web.Application:
    async def on_startup(app):
        # one generated turn can take minutes; the agent's own timeout governs
        app["client"] = ClientSession(timeout=ClientTimeout(total=1800), auto_decompress=False)

    async def on_cleanup(app):
        await app["client"].close()

    async def health(request):
        return web.Response(text="ok")

    async def forward(request: web.Request) -> web.StreamResponse:
        if request.match_info["token"] != token:
            raise web.HTTPNotFound()
        port = int(request.match_info["port"])
        if port not in ports:
            raise web.HTTPForbidden()
        url = f"http://{upstream_host}:{port}/{request.match_info['tail']}"
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP}
        body = await request.read()
        async with request.app["client"].request(
            request.method, url, params=request.query, headers=headers, data=body
        ) as upstream:
            payload = await upstream.read()
            out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _DROP}
            return web.Response(status=upstream.status, body=payload, headers=out_headers)

    app = web.Application(client_max_size=256 * 1024**2)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/healthz", health)
    app.router.add_route("*", r"/{token}/{port:\d+}/{tail:.*}", forward)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-port", type=int, required=True)
    ap.add_argument("--upstream-host", required=True)
    ap.add_argument("--ports", required=True, help="first-last session server port, inclusive")
    ap.add_argument("--token", required=True)
    args = ap.parse_args()
    first, last = (int(p) for p in args.ports.split("-"))
    web.run_app(make_app(args.upstream_host, range(first, last + 1), args.token),
                host="0.0.0.0", port=args.listen_port, access_log=None)


if __name__ == "__main__":
    main()
