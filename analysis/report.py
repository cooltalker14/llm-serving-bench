"""Turn results/*.json into the markdown tables and figures used in the README.

Usage:
    python analysis/report.py results/*.json --out analysis/out
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(paths: list[Path]) -> list[dict]:
    runs = []
    for p in paths:
        d = json.loads(p.read_text())
        d["_file"] = p.name
        runs.append(d)
    return runs


def sweep_table(run: dict) -> str:
    """Per-concurrency table for a single run."""
    head = (
        "| Concurrency | Output tok/s | Req/s | TTFT p50 (ms) | TTFT p99 (ms) "
        "| ITL p50 (ms) | E2E p90 (s) | Failed |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for r in run["rows"]:
        lines.append(
            f"| {r['concurrency']} | {r['output_tok_per_s']:.1f} | {r['req_per_s']:.2f} "
            f"| {r['ttft_p50_ms']:.0f} | {r['ttft_p99_ms']:.0f} "
            f"| {r['itl_p50_ms']:.1f} | {r['e2e_p90_s']:.2f} | {r['n_failed']} |"
        )
    return head + "\n".join(lines)


def comparison_table(runs: list[dict]) -> str:
    """Peak-throughput comparison across configurations (e.g. quantization schemes)."""
    head = (
        "| Config | Peak output tok/s | At concurrency | TTFT p50 there (ms) "
        "| ITL p50 there (ms) | Best TTFT p50 (ms) |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for run in runs:
        rows = run["rows"]
        peak = max(rows, key=lambda r: r["output_tok_per_s"])
        best_ttft = min(r["ttft_p50_ms"] for r in rows if r["n_ok"] > 0)
        lines.append(
            f"| {run['tag']} | {peak['output_tok_per_s']:.1f} | {peak['concurrency']} "
            f"| {peak['ttft_p50_ms']:.0f} | {peak['itl_p50_ms']:.1f} | {best_ttft:.0f} |"
        )
    return head + "\n".join(lines)


def knee(run: dict, ttft_budget_ms: float = 1000.0) -> str:
    """Highest throughput reachable while keeping TTFT p99 under a latency SLO.

    This is the number that actually answers 'how many users can we serve',
    which raw peak throughput does not.
    """
    ok = [r for r in run["rows"] if r["ttft_p99_ms"] <= ttft_budget_ms and r["n_ok"] > 0]
    if not ok:
        return f"{run['tag']}: no concurrency level met TTFT p99 <= {ttft_budget_ms:.0f}ms"
    best = max(ok, key=lambda r: r["output_tok_per_s"])
    return (
        f"{run['tag']}: {best['output_tok_per_s']:.0f} out tok/s at concurrency "
        f"{best['concurrency']} while holding TTFT p99 <= {ttft_budget_ms:.0f}ms"
    )


def plot_tradeoff(runs: list[dict], out: Path):
    """Throughput vs TTFT: the curve that shows the cost of pushing utilization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for run in runs:
        rows = run["rows"]
        c = [r["concurrency"] for r in rows]
        tps = [r["output_tok_per_s"] for r in rows]
        ttft = [r["ttft_p50_ms"] for r in rows]

        axes[0].plot(c, tps, marker="o", label=run["tag"])
        axes[1].plot(tps, ttft, marker="o", label=run["tag"])

    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Concurrency")
    axes[0].set_ylabel("Output tokens/sec")
    axes[0].set_title("Throughput scaling")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Output tokens/sec")
    axes[1].set_ylabel("TTFT p50 (ms)")
    axes[1].set_title("Latency cost of throughput")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("analysis/out"))
    ap.add_argument("--ttft-budget-ms", type=float, default=1000.0)
    args = ap.parse_args()

    runs = load(args.results)
    args.out.mkdir(parents=True, exist_ok=True)

    md = ["# Benchmark results\n"]

    if len(runs) > 1:
        md.append("## Configuration comparison\n")
        md.append(comparison_table(runs) + "\n")

    md.append(f"## Throughput under a TTFT p99 budget of {args.ttft_budget_ms:.0f}ms\n")
    for run in runs:
        md.append(f"- {knee(run, args.ttft_budget_ms)}")
    md.append("")

    for run in runs:
        md.append(f"## {run['tag']}\n")
        env = run["env"]
        md.append(f"GPU: `{env['gpu']}` | vLLM: `{env['vllm'] or 'n/a'}` "
                  f"| torch: `{env['torch'] or 'n/a'}` | run: `{env['timestamp_utc']}`\n")
        md.append(sweep_table(run) + "\n")

    plot_tradeoff(runs, args.out / "tradeoff.png")
    md.append("![throughput vs latency](out/tradeoff.png)\n")

    report = args.out / "RESULTS.md"
    report.write_text("\n".join(md))
    print(f"wrote {report}")
    print(f"wrote {args.out / 'tradeoff.png'}")


if __name__ == "__main__":
    main()
