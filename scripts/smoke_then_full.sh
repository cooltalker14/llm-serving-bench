#!/bin/bash
#SBATCH --job-name=llm-bench
#SBATCH --account=def-hsajjad
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:45:00
#SBATCH --output=logs/%x-%j.out

# Smoke test with configs/small.yaml, then the full sweep only if it passed.
#
#   sbatch scripts/smoke_then_full.sh bf16
#   sbatch scripts/smoke_then_full.sh awq
#
# Both configs serve the SAME base model under the SAME workload; only the
# weight precision changes. That is what makes the comparison controlled.

set -euo pipefail

TAG="${1:-bf16}"

CACHE=/home/swap1411/.cache/huggingface/hub

case "$TAG" in
  bf16|fp16)
    MODEL_PATH="$CACHE/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    QUANT_ARGS=""
    ;;
  awq)
    MODEL_PATH="$CACHE/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/b25037543e9394b818fdfca67ab2a00ecc7dd641"
    # awq_marlin is the faster Ampere kernel; vLLM selects it automatically,
    # but naming it explicitly keeps the run reproducible across versions.
    QUANT_ARGS="--quantization awq_marlin"
    ;;
  *)
    echo "unknown tag '$TAG' (expected: bf16 | awq)" >&2
    exit 1
    ;;
esac

SERVED_NAME="qwen7b"

module load python/3.11 cuda/12.2 gcc opencv
source /home/swap1411/projects/def-hsajjad/swap1411/llm-serving-bench/venvs/bin/activate

# Keep ~/.local out of sys.path so the venv's +computecanada torch wins.
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs results

echo "[job] node=$(hostname) tag=$TAG"
echo "[job] model=$MODEL_PATH"
echo "[job] quant_args=${QUANT_ARGS:-none}"
python -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__)"

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "[job] FATAL: no config.json at $MODEL_PATH" >&2
  exit 1
fi

vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  $QUANT_ARGS &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "[job] waiting for server..."
READY=0
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    READY=1
    echo "[job] server ready after ~$((i*10))s"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[job] FATAL: vLLM exited during startup" >&2
    exit 1
  fi
  sleep 10
done

if [ "$READY" -ne 1 ]; then
  echo "[job] FATAL: server not ready after 15 minutes" >&2
  exit 1
fi

curl -s http://127.0.0.1:8000/v1/models
echo ""
# KV cache size depends on how much memory the weights left behind, so record
# it per-configuration: it is the main reason quantization changes throughput.
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

echo ""
echo "=== SMOKE TEST (configs/small.yaml) ==="
PYTHONPATH=src python -m bench.sweep --config configs/small.yaml --tag "${TAG}-smoke"

LATEST=$(ls -t results/${TAG}-smoke_*.json | head -1)
FAILED=$(python -c "
import json
rows = json.load(open('$LATEST'))['rows']
print(sum(r['n_failed'] for r in rows))
")

if [ "$FAILED" -ne 0 ]; then
  echo "[job] FATAL: smoke test had $FAILED failed requests, skipping full sweep" >&2
  python -c "
import json
for r in json.load(open('$LATEST'))['rows']:
    if r['errors']: print(' c=%s errors=%s' % (r['concurrency'], r['errors']))
"
  exit 1
fi

echo "[job] smoke test clean, proceeding to full sweep"
echo ""
echo "=== FULL SWEEP (configs/full.yaml) ==="
PYTHONPATH=src python -m bench.sweep --config configs/full.yaml --tag "$TAG"

echo "[job] done"