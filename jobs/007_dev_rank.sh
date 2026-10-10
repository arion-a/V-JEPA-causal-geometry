# timeout: 600
# Within-scene rank/cosine test on dev-world derivative features saved by job 006 (numpy only, no GPU).
python scripts/dev_rank.py
cp "$CG_ROOT/results/dev_rank/dev_rank.json" "$JOB_OUT/dev_rank.json"
