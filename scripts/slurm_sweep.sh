#!/bin/bash
#SBATCH --job-name=llm-bench
#SBATCH --account=def-hsajjad    # <-- change this
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Concurrency sweep for one serving configuration.
#   sbatch scripts/slurm_sweep.sh fp16
#   sbatch scripts/slurm_sweep.sh awq   TheBloke/Qwen2.5-7B-Instruct-AWQ awq
#
# Alliance compute nodes have no internet, so models must be pre-downloaded on
# a login node into $SCRATCH/hf_cache first. See scripts/prefetch_model.sh.

set -euo pipefail

TAG="${1:?usage: slurm_sweep.sh TAG [MODEL] [QUANT]}"
MODEL="${2:-Qwen/Qwen2.5-7B-Instruct}"
QUANT="${3:-none}"

module load python/3.11 cuda/12.2
source "$SCRATCH/venvs/bench/bin/activate"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export OUTLINES_CACHE_DIR="$SLURM_TMPDIR/outlines"

mkdir -p logs results

QUANT_ARG=""
if [ "$QUANT" != "none" ]; then
  QUANT_ARG="--quantization $QUANT"
fi

echo "[job] starting vLLM: model=$MODEL quant=$QUANT"
vllm serve "$MODEL" \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --disable-log-requests \
  $QUANT_ARG &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "[job] waiting for server to become ready..."
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "[job] server ready after ${i}0s"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[job] FATAL: vLLM died during startup" >&2
    exit 1
  fi
  sleep 10
done

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

PYTHONPATH=src python -m bench.sweep \
  --config configs/full.yaml \
  --tag "$TAG"

echo "[job] done"
