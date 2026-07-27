#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export MAX_JOBS="${MAX_JOBS:-22}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"

python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
if sys.version_info[:2] != (3, 12):
    raise SystemExit("Expected Python 3.12")
if not torch.__version__.startswith("2.8."):
    raise SystemExit("Expected PyTorch 2.8.x")
if torch.version.cuda != "12.8":
    raise SystemExit("Expected the cu128 PyTorch build")
PY

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

# Build against the already installed PyTorch instead of an isolated build
# environment that might pull a mismatched torch wheel.
# The production checkpoint path is validated against TE 2.15.0. Never let a
# routine environment repair silently replace it with the experimental 2.16
# A/B candidate; benchmark_te216_isolated.sh owns that isolated environment.
python -m pip install --no-build-isolation "transformer-engine[pytorch]==2.15.0"
FLASH_ATTN_CUDA_ARCHS=120 python -m pip install --no-build-isolation flash-attn

python check_environment.py --strict --full-model
