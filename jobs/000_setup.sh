# timeout: 900
# Create the persistent layout, install deps, record the environment. Small facts go to $JOB_OUT.
set -e
R="$CG_ROOT"
mkdir -p "$R"/{data/{manifests,clips},features/{pooled,object_window,full_tokens},vectors/{jacobians,metrics},results,reports,checkpoints,logs,vendor,cache}
export TORCH_HOME="$R/cache/torch" HF_HOME="$R/cache/hf" PIP_CACHE_DIR="$R/cache/pip"
pip install -q timm einops numpy pytest
{ df -h /workspace | tail -1; mount | grep -E " /workspace " || echo "WARN: /workspace not a separate mount"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; python --version; python -c "import torch;print('torch',torch.__version__,torch.cuda.is_available())"; } | tee "$JOB_OUT/env.txt"
python -m pytest -q tests | tail -2
