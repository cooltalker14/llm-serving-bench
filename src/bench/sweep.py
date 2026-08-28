"""Run a concurrency sweep against a served model and write results to JSON.

Usage:
    python -m bench.sweep --config configs/small.yaml --tag colab-t4
"""

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .load import result_to_dict, run_load
from .prompts import build_prompts


def _sh(cmd: str) -> str:
    """Best-effort shell capture; provenance is nice-to-have, not critical."""
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return ""


def capture_env() -> dict:
    """Record everything needed to reproduce or defend these numbers later.

    This matters more than it looks: a benchmark without hardware and version
    provenance is not citable, and reviewers will say so.
    """
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": _sh("nvidia-smi --query-gpu=name,memory.total,driver_version "
                  "--format=csv,noheader") or "none detected",
        "cuda": _sh("nvcc --version | tail -1"),
        "vllm": _sh("python -c 'import vllm; print(vllm.__version__)' 2>/dev/null"),
        "torch": _sh("python -c 'import torch; print(torch.__version__)' 2>/dev/null"),
        "slurm_job_id": _sh("echo $SLURM_JOB_ID"),
    }


async def run_sweep(cfg: dict, tag: str, out_dir: Path) -> Path:
    prompts = build_prompts(
        n=cfg["workload"]["n_prompts"],
        target_tokens=cfg["workload"]["prompt_tokens"],
        seed=cfg["workload"].get("seed", 0),
    )

    base_url = cfg["server"]["base_url"]
    model = cfg["server"]["model"]
    max_tokens = cfg["workload"]["max_tokens"]

    rows = []
    for c in cfg["sweep"]["concurrency"]:
        n_req = max(c * cfg["sweep"]["requests_per_worker"], cfg["sweep"]["min_requests"])
        print(f"[sweep] concurrency={c:>4}  requests={n_req}", flush=True)

        t0 = time.perf_counter()
        r = await run_load(
            base_url=base_url,
            model=model,
            prompts=prompts,
            concurrency=c,
            n_requests=n_req,
            max_tokens=max_tokens,
            warmup=cfg["sweep"].get("warmup", 0),
        )
        row = result_to_dict(r)
        rows.append(row)

        print(
            f"         -> {row['output_tok_per_s']:>8.1f} out tok/s | "
            f"TTFT p50 {row['ttft_p50_ms']:>7.1f}ms p99 {row['ttft_p99_ms']:>7.1f}ms | "
            f"ITL p50 {row['itl_p50_ms']:>6.2f}ms | "
            f"{row['n_failed']} failed | {time.perf_counter()-t0:.0f}s",
            flush=True,
        )

        if row["n_failed"] > 0:
            print(f"         !! errors: {row['errors']}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{tag}_{stamp}.json"
    path.write_text(
        json.dumps(
            {"tag": tag, "config": cfg, "env": capture_env(), "rows": rows},
            indent=2,
        )
    )
    return path


def main():
    ap = argparse.ArgumentParser(description="LLM serving concurrency sweep")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--tag", required=True, help="label for this run, e.g. a100-fp16")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--base-url", help="override server.base_url from config")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.base_url:
        cfg["server"]["base_url"] = args.base_url

    path = asyncio.run(run_sweep(cfg, args.tag, args.out))
    print(f"\n[sweep] wrote {path}")


if __name__ == "__main__":
    main()
