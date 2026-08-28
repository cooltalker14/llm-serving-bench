#!/bin/bash
# One-time environment setup on an Alliance cluster (run on a login node).
set -euo pipefail
module load python/3.11 cuda/12.2
mkdir -p "$SCRATCH/venvs"
python -m venv "$SCRATCH/venvs/bench"
source "$SCRATCH/venvs/bench/bin/activate"
pip install --no-index --upgrade pip
# vLLM is not in the Alliance wheelhouse; pull from PyPI on the login node.
pip install vllm aiohttp pyyaml numpy matplotlib huggingface_hub
python -c "import vllm; print('vllm', vllm.__version__)"
