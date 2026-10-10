# timeout: 1800
# Close open Gate 0 items: checkpoint hash, early-layer token map, fp32 vs bf16 per layer, native-input memory.
export TORCH_HOME="$CG_ROOT/cache/torch" HF_HOME="$CG_ROOT/cache/hf"
python scripts/gate0b.py
cp "$CG_ROOT/results/gate0b/gate0b.json" "$JOB_OUT/gate0b.json"
