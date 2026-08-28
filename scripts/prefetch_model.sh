#!/bin/bash
# Run on a LOGIN node (compute nodes have no internet on Alliance clusters).
#   bash scripts/prefetch_model.sh Qwen/Qwen2.5-7B-Instruct
set -euo pipefail
MODEL="${1:?usage: prefetch_model.sh MODEL_ID}"
export HF_HOME="$SCRATCH/hf_cache"
mkdir -p "$HF_HOME"
python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$MODEL')
print('cached at', p)
"
