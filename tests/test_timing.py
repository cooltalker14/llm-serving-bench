"""Verify the harness measures what we think it measures.

Ground truth is injected by tests/mock_server.py (TTFT_S=0.20, ITL_S=0.010).
If these assertions fail, every number the repo produces is suspect.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.load import run_load  # noqa: E402

PORT = 8931
BASE = f"http://127.0.0.1:{PORT}"


def start_server():
    p = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "mock_server.py"), str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)
    return p


def main():
    server = start_server()
    failures = []
    try:
        r = asyncio.run(
            run_load(
                base_url=BASE,
                model="mock",
                prompts=["hello world"],
                concurrency=1,
                n_requests=6,
                max_tokens=20,
                warmup=2,
            )
        )

        print(f"  ttft_p50={r.ttft_p50_ms}ms  itl_p50={r.itl_p50_ms}ms")
        print(f"  out_tok/s={r.output_tok_per_s}  ok={r.n_ok} failed={r.n_failed}")

        # TTFT was injected at 200ms; allow scheduler jitter.
        if not 190 <= r.ttft_p50_ms <= 260:
            failures.append(f"ttft_p50 {r.ttft_p50_ms}ms outside [190,260]")

        # ITL was injected at 10ms.
        if not 8 <= r.itl_p50_ms <= 20:
            failures.append(f"itl_p50 {r.itl_p50_ms}ms outside [8,20]")

        # Usage block reports 20 completion tokens.
        if r.mean_output_tokens != 20:
            failures.append(f"mean_output_tokens {r.mean_output_tokens} != 20")

        if r.n_ok != 6 or r.n_failed != 0:
            failures.append(f"expected 6 ok / 0 failed, got {r.n_ok}/{r.n_failed}")

        # Concurrency 4 should raise throughput above concurrency 1.
        r4 = asyncio.run(
            run_load(BASE, "mock", ["hello world"], 4, 12, 20, warmup=2)
        )
        print(f"  c=4: out_tok/s={r4.output_tok_per_s} (c=1 was {r.output_tok_per_s})")
        if r4.output_tok_per_s <= r.output_tok_per_s:
            failures.append("throughput did not increase with concurrency")

    finally:
        server.terminate()
        server.wait(timeout=5)

    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nPASS: timing harness verified against known ground truth")


if __name__ == "__main__":
    main()
