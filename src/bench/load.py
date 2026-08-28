"""Drive a fixed concurrency level against a served model and aggregate results.

Closed-loop model: N workers each send a request, wait for it to finish, then
send the next. This measures the server under sustained pressure and is the
regime that matters for batch/offline workloads and for finding the knee in
the throughput curve.
"""

import asyncio
import time
from dataclasses import asdict, dataclass

import aiohttp
import numpy as np

from .client import RequestResult, stream_one


@dataclass
class LoadResult:
    concurrency: int
    n_ok: int
    n_failed: int
    duration_s: float
    output_tok_per_s: float
    total_tok_per_s: float
    req_per_s: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    itl_p50_ms: float
    itl_p90_ms: float
    itl_p99_ms: float
    e2e_p50_s: float
    e2e_p90_s: float
    mean_output_tokens: float
    errors: list[str]


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def summarize(results: list[RequestResult], concurrency: int, duration_s: float) -> LoadResult:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    ttfts = [r.ttft_s * 1000 for r in ok]
    itls = [x * 1000 for r in ok for x in r.itls_s]
    e2es = [r.e2e_s for r in ok]

    out_tokens = sum(r.output_tokens for r in ok)
    in_tokens = sum(r.prompt_tokens for r in ok)

    # Keep only distinct error strings so the summary stays readable.
    seen: list[str] = []
    for r in failed:
        if r.error not in seen:
            seen.append(r.error)

    return LoadResult(
        concurrency=concurrency,
        n_ok=len(ok),
        n_failed=len(failed),
        duration_s=round(duration_s, 2),
        output_tok_per_s=round(out_tokens / duration_s, 1) if duration_s else 0.0,
        total_tok_per_s=round((out_tokens + in_tokens) / duration_s, 1) if duration_s else 0.0,
        req_per_s=round(len(ok) / duration_s, 3) if duration_s else 0.0,
        ttft_p50_ms=round(_pct(ttfts, 50), 1),
        ttft_p90_ms=round(_pct(ttfts, 90), 1),
        ttft_p99_ms=round(_pct(ttfts, 99), 1),
        itl_p50_ms=round(_pct(itls, 50), 2),
        itl_p90_ms=round(_pct(itls, 90), 2),
        itl_p99_ms=round(_pct(itls, 99), 2),
        e2e_p50_s=round(_pct(e2es, 50), 3),
        e2e_p90_s=round(_pct(e2es, 90), 3),
        mean_output_tokens=round(out_tokens / len(ok), 1) if ok else 0.0,
        errors=seen[:5],
    )


async def run_load(
    base_url: str,
    model: str,
    prompts: list[str],
    concurrency: int,
    n_requests: int,
    max_tokens: int,
    warmup: int = 0,
) -> LoadResult:
    """Run n_requests through `concurrency` parallel workers."""
    conn = aiohttp.TCPConnector(limit=concurrency * 2)

    async with aiohttp.ClientSession(connector=conn) as session:

        async def drain(queue: asyncio.Queue, sink: list | None):
            async def worker():
                while True:
                    try:
                        idx = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    r = await stream_one(
                        session, base_url, model, prompts[idx % len(prompts)], max_tokens
                    )
                    if sink is not None:
                        sink.append(r)

            await asyncio.gather(*[worker() for _ in range(concurrency)])

        if warmup:
            wq: asyncio.Queue[int] = asyncio.Queue()
            for i in range(warmup):
                wq.put_nowait(i)
            await drain(wq, None)

        q: asyncio.Queue[int] = asyncio.Queue()
        for i in range(n_requests):
            q.put_nowait(i)

        results: list[RequestResult] = []
        start = time.perf_counter()
        await drain(q, results)
        duration = time.perf_counter() - start

    return summarize(results, concurrency, duration)


def result_to_dict(r: LoadResult) -> dict:
    return asdict(r)
