# timeout: 2400
# Gate 1 v2 on the 15 DEV worlds per scene (rule declared in configs/pilot.json gate1_v2 and scripts/gate1_v2.py before running).
export TORCH_HOME="$CG_ROOT/cache/torch" HF_HOME="$CG_ROOT/cache/hf"
python scripts/gate1_v2.py
cp "$CG_ROOT/results/gate1_v2/gate1_v2.json" "$JOB_OUT/gate1_v2.json"
