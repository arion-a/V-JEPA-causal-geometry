"""Print a compact Gate 0 summary (safe to paste: no secrets)."""
import json, os
p = os.path.join(os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry"), "results/gate0/gate0.json")
g = json.load(open(p))
for k in ("env", "weights_host", "load_model", "module_tree_and_input_spec", "hooks_shapes_determinism", "token_to_space_map", "throughput"):
    s = g["steps"].get(k, {}); print("==", k, "ok=", s.get("ok"))
    print(json.dumps(s.get("result") or s.get("error"))[:1500])
