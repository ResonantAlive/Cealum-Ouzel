#!/usr/bin/env bash
set -euo pipefail

# One-variable Transformer Engine 2.15 -> 2.16 A/B. The 2.16 package is
# installed into a local virtual environment that can see the image's existing
# PyTorch/CUDA stack; the known-good global TE installation is never modified.

cd "$(dirname "$0")"
export MAX_JOBS="${MAX_JOBS:-22}"
export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}"
export NVTE_FRAMEWORK=pytorch

base_python="$(command -v python)"
mkdir -p server-results

"${base_python}" - <<'PY'
import sys
import torch
import transformer_engine

print("stable python:", sys.executable)
print("stable torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("stable TE:", transformer_engine.__version__)
if not torch.__version__.startswith("2.8."):
    raise SystemExit("The base image is no longer PyTorch 2.8.x")
if torch.version.cuda != "12.8":
    raise SystemExit("The base image is no longer the cu128 build")
PY

# This is intentionally tiny and synthetic. It never touches the training
# dataset or checkpoint directory.
"${base_python}" probe_te_grouped_linear.py \
  --warmup 3 \
  --steps 10 \
  --precision both \
  --legacy-checkpoint-out server-results/te215_grouped_linear_fp8_state.pt \
  --json-out server-results/te_grouped_linear_stable.json
"${base_python}" probe_te_grouped_linear.py \
  --warmup 3 \
  --steps 10 \
  --precision fp8 \
  --fp8-recipe mxfp8 \
  --json-out server-results/te_grouped_linear_stable_mxfp8.json

"${base_python}" -m venv --system-site-packages .venv-te216
source .venv-te216/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation "transformer-engine[pytorch]==2.16.0"

python - <<'PY'
import sys
import torch
import transformer_engine

print("probe python:", sys.executable)
print("probe torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("probe TE:", transformer_engine.__version__)
if not torch.__version__.startswith("2.8."):
    raise SystemExit("TE probe environment changed PyTorch away from 2.8.x")
if torch.version.cuda != "12.8":
    raise SystemExit("TE probe environment changed the CUDA build away from cu128")
if transformer_engine.__version__ != "2.16.0":
    raise SystemExit("The isolated environment did not resolve TE 2.16.0")
PY

python probe_te_grouped_linear.py \
  --warmup 3 \
  --steps 10 \
  --precision both \
  --legacy-checkpoint-in server-results/te215_grouped_linear_fp8_state.pt \
  --json-out server-results/te_grouped_linear_216.json
python probe_te_grouped_linear.py \
  --warmup 3 \
  --steps 10 \
  --precision fp8 \
  --fp8-recipe mxfp8 \
  --json-out server-results/te_grouped_linear_216_mxfp8.json

echo "A/B complete. Compare server-results/te_grouped_linear_stable.json and te_grouped_linear_216.json."
