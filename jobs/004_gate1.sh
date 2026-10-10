# timeout: 2400
# Gate 1: fp32 features for the 20-clip subset, jitter noise floor, half-delta linearity, predeclared SNR criterion.
export TORCH_HOME="$CG_ROOT/cache/torch" HF_HOME="$CG_ROOT/cache/hf"
python scripts/gate1.py
cp "$CG_ROOT/results/gate1/gate1.json" "$JOB_OUT/gate1.json"
