#!/usr/bin/env bash
# Run INSIDE the pod (Jupyter terminal / SSH). Sets up the persistent layout on the network volume, installs deps,
# pulls the repo, then runs Gate 0. Safe to re-run: nothing is deleted except the vendor clone gate0.py refreshes.
set -euo pipefail
ROOT=${CG_ROOT:-/workspace/vjepa-causal-geometry}
echo "== persistence check"; df -h /workspace | tail -1; mount | grep -E " /workspace " || echo "WARN: /workspace is not a separate mount (data may not persist)"
mkdir -p "$ROOT"/{data/{manifests,clips},features/{pooled,object_window,full_tokens},vectors/{jacobians,metrics},results,reports,checkpoints,logs,vendor,cache}
export TORCH_HOME="$ROOT/cache/torch" HF_HOME="$ROOT/cache/hf" PIP_CACHE_DIR="$ROOT/cache/pip"   # weights/caches persist too
if [ -d "$ROOT/repo/.git" ]; then git -C "$ROOT/repo" pull --ff-only; else git clone https://github.com/arion-a/V-JEPA-causal-geometry "$ROOT/repo"; fi
pip install -q timm einops numpy pytest
cd "$ROOT/repo" && python -m pytest -q tests | tail -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
CG_ROOT="$ROOT" python scripts/gate0.py 2>&1 | tee "$ROOT/logs/gate0.log"
echo "Done. Results: $ROOT/results/gate0/gate0.json  (log: $ROOT/logs/gate0.log)"
