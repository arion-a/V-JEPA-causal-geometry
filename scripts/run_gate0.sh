#!/usr/bin/env bash
# Run Gate 0 on the pod and print a pasteable summary. No GitHub token needed (pulls the public repo).
set -uo pipefail
export CG_ROOT=${CG_ROOT:-/workspace/vjepa-causal-geometry}
export TORCH_HOME="$CG_ROOT/cache/torch" HF_HOME="$CG_ROOT/cache/hf"
cd "$CG_ROOT/repo"
python scripts/gate0.py > "$CG_ROOT/logs/gate0.log" 2>&1
python scripts/gate0_summary.py | tee "$CG_ROOT/logs/gate0_summary.txt"
