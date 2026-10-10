# timeout: 3600
# Dev-world preview at frozen manifest settings: split-half reliability + cross-context cosines (global vs union-window pooling), vs trajectory baseline.
export TORCH_HOME="$CG_ROOT/cache/torch" HF_HOME="$CG_ROOT/cache/hf"
python scripts/dev_derivatives.py
cp "$CG_ROOT/results/dev_derivatives/dev_derivatives.json" "$JOB_OUT/dev_derivatives.json"
