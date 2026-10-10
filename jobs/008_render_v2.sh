# timeout: 7000
# Render all 460 clips of manifest v2 (protocol A-pilot-v2) to $CG_ROOT/data/clips_v2, then verify. Disk usage is printed (volume is 40 GB).
export TORCH_HOME="$CG_ROOT/cache/torch"
df -h "$CG_ROOT" | tail -1 | tee "$JOB_OUT/disk_before.txt"
python -m pytest -q tests/test_sim_pb.py 2>&1 | tail -3 | tee "$JOB_OUT/pod_engine_tests.txt"
python scripts/render_clips.py --out "$CG_ROOT"
cp "$CG_ROOT/data/clip_hashes_v2.csv" "$JOB_OUT/clip_hashes_v2.csv"
python scripts/verify_render.py "$JOB_OUT/render_verification.json"
df -h "$CG_ROOT" | tail -1 | tee "$JOB_OUT/disk_after.txt"
