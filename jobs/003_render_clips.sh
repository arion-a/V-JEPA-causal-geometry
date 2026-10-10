# timeout: 3600
# Stage 1 on the pod: install PyBullet, re-run engine tests on THIS platform, render all 200 clips to the volume, verify.
export TORCH_HOME="$CG_ROOT/cache/torch"
pip install -q pybullet 2>&1 | tail -2
python -c "import pybullet; print('pybullet ok')"
python -m pytest -q tests/test_sim_pb.py 2>&1 | tail -3 | tee "$JOB_OUT/pod_engine_tests.txt"
python scripts/render_clips.py --out "$CG_ROOT"
cp "$CG_ROOT/data/clip_hashes.csv" "$JOB_OUT/clip_hashes.csv"
python scripts/verify_render.py "$JOB_OUT/render_verification.json"
