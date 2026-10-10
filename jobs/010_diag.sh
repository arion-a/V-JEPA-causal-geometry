# timeout: 300
# Diagnostic: report pod state so the agent can see why job 009 produced no result. Read-only; writes small text files to $JOB_OUT.
{
  echo "== date"; date -u +%FT%TZ
  echo "== processes (worker/render/gate1)"; ps -eo pid,etimes,pcpu,cmd | grep -E "pod_worke[r]|run_worke[r]|gate1_v[2]|render_clip[s]|009_gate1" | cut -c1-140
  echo "== gpu"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
  echo "== disk"; df -h "$CG_ROOT" | tail -1
  echo "== worker.log tail (token never logged)"; tail -25 "$CG_ROOT/logs/worker.log" 2>&1
  echo "== gate1_v2 volume outputs"; ls -la "$CG_ROOT/results/gate1_v2" "$CG_ROOT/features/pooled" 2>&1 | tail -12
  echo "== local job dirs"; ls -la "$(dirname "$0")/../results/jobs/009_gate1_v2" 2>&1 | tail -6
  echo "== recent python errors"; grep -rhE "Traceback|Error|OOM|out of memory|Killed" "$CG_ROOT/logs" 2>/dev/null | tail -8
} > "$JOB_OUT/diag.txt" 2>&1
cat "$JOB_OUT/diag.txt"
