"""Fake OpenAI-compatible server with tunable TTFT and token rate.

Exists so the harness can be verified without a GPU: we know the ground-truth
latencies we injected, so we can assert the measured numbers match.
"""

import asyncio
import json
import sys

from aiohttp import web

TTFT_S = 0.20
ITL_S = 0.010
_inflight = 0


async def completions(request: web.Request) -> web.StreamResponse:
    global _inflight
    body = await request.json()
    max_tokens = body.get("max_tokens", 16)

    _inflight += 1
    # Crude batching effect: more concurrent requests slow each one down.
    slowdown = 1.0 + 0.05 * (_inflight - 1)

    resp = web.StreamResponse(
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    )
    await resp.prepare(request)

    try:
        await asyncio.sleep(TTFT_S * slowdown)
        for i in range(max_tokens):
            if i > 0:
                await asyncio.sleep(ITL_S * slowdown)
            chunk = {"choices": [{"text": " tok", "index": 0}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())

        usage = {
            "choices": [],
            "usage": {"prompt_tokens": 128, "completion_tokens": max_tokens},
        }
        await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
    finally:
        _inflight -= 1

    return resp


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    web.run_app(app, port=port, print=None)


if __name__ == "__main__":
    main()
