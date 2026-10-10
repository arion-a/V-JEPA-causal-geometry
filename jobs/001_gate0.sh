# timeout: 3000
# Gate 0 on the real model. Heavy artifacts stay on the volume; summary JSON is committed via $JOB_OUT.
export TORCH_HOME="$CG_ROOT/cache/torch" HF_HOME="$CG_ROOT/cache/hf"
python scripts/gate0.py
cp "$CG_ROOT/results/gate0/gate0.json" "$JOB_OUT/gate0.json"
