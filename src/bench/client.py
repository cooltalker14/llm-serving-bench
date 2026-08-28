"""Async client that issues streaming completions and records per-token timing.

Timing definitions used throughout this repo:
  ttft_s  - time from request send to first token received
  itl_s   - inter-token latency, the gap between consecutive tokens after the first
  e2e_s   - total wall time for the request

We measure on the client side, on purpose. Server-side metrics hide queueing
delay, which is exactly the thing that degrades under load.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field

import aiohttp


@dataclass
class RequestResult:
    ok: bool
    ttft_s: float = 0.0
    e2e_s: float = 0.0
    itls_s: list[float] = field(default_factory=list)
    output_tokens: int = 0
    prompt_tokens: int = 0
    error: str = ""

    @property
    def mean_itl_s(self) -> float:
        return sum(self.itls_s) / len(self.itls_s) if self.itls_s else 0.0


async def stream_one(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    timeout_s: float = 600.0,
) -> RequestResult:
    """Send one streaming request and time each token as it arrives."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    ttft = None
    last_tok_time = None
    itls: list[float] = []
    n_out = 0
    n_prompt = 0

    try:
        async with session.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(ok=False, error=f"HTTP {resp.status}: {body[:200]}")

            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break

                chunk = json.loads(data)

                # Usage-only chunk arrives last when include_usage is set.
                usage = chunk.get("usage")
                if usage:
                    n_prompt = usage.get("prompt_tokens", 0)
                    n_out = usage.get("completion_tokens", n_out)

                choices = chunk.get("choices") or []
                if not choices or not choices[0].get("text"):
                    continue

                now = time.perf_counter()
                if ttft is None:
                    ttft = now - start
                else:
                    itls.append(now - last_tok_time)
                last_tok_time = now
                n_out += 1

    except asyncio.TimeoutError:
        return RequestResult(ok=False, error="timeout")
    except aiohttp.ClientError as e:
        return RequestResult(ok=False, error=f"client error: {e}")

    if ttft is None:
        return RequestResult(ok=False, error="no tokens returned")

    return RequestResult(
        ok=True,
        ttft_s=ttft,
        e2e_s=time.perf_counter() - start,
        itls_s=itls,
        output_tokens=n_out,
        prompt_tokens=n_prompt,
    )
