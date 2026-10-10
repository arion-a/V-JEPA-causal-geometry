#!/usr/bin/env bash
# Supervisor: restarts the pod worker if it dies. Stops when the worker exits with code 3 (jobs/STOP) or when this script is killed.
# Token comes from $GH_TOKEN or the protected file $CG_ROOT/secrets/gh_token (see scripts/set_token.sh) - no paste needed on restart.
cd "$(dirname "$0")/.." || exit 1
export CG_ROOT=${CG_ROOT:-/workspace/vjepa-causal-geometry}
mkdir -p "$CG_ROOT/logs"
while true; do
  git pull -q --ff-only origin main 2>/dev/null || true
  python3 scripts/pod_worker.py >> "$CG_ROOT/logs/worker.log" 2>&1; rc=$?
  echo "[supervisor] worker exited rc=$rc at $(date -u +%FT%TZ)" >> "$CG_ROOT/logs/worker.log"
  [ "$rc" = "3" ] && break
  sleep 10
done
