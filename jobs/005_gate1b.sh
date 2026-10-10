# timeout: 6000
# Gate 1b: diagnose Gate-1 failure on DEV worlds: render config x step multiplier x pooling; predeclared stability rule in scripts/gate1b.py.
export TORCH_HOME="$CG_ROOT/cache/torch" HF_HOME="$CG_ROOT/cache/hf"
python scripts/gate1b.py
cp "$CG_ROOT/results/gate1b/gate1b.json" "$JOB_OUT/gate1b.json"
