"""DEV-WORLD within-scene rank test (pod, numpy only; reads features saved by job 006): are v_g and v_e (bounce) / v_mu (incline) distinguishable
directions (H1) or one shared 'motion' axis? Global vs union-window pooling, 4 hierarchical taps, plus the tracked-trajectory baseline. Dev only."""
import csv, json, os, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CG = os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry")
sys.path.insert(0, ROOT)
from cg import devstats as DS
Z = np.load(f"{CG}/features/pooled/dev_derivs.npz"); idx = {c: i for i, c in enumerate(Z["ids"])}; traj = np.load(f"{ROOT}/data/manifests/trajectories.npz")
rows = [r for r in csv.DictReader(open(f"{ROOT}/data/manifests/pilot_manifest.csv")) if r["family"] in ("bounce", "incline") and r["split"] == "dev"]
worlds = {}
for r in rows: worlds.setdefault((r["kind"], r["world_id"]), {})[r["variant"]] = r["clip_id"]
out = {}
for kind, par in (("bounce", "e"), ("incline", "mu")):
    ws = [vs for (k, _), vs in worlds.items() if k == kind]
    for pool, key in (("global", "normed"), ("window", "normed_win")):
        for tap, name in enumerate((5, 11, 17, 23)):
            d = lambda p: np.array([Z[key][idx[v[f"{p}+"]], tap] - Z[key][idx[v[f"{p}-"]], tap] for v in ws])
            out[f"{kind}|g~{par}|{pool}|norm{name}"] = DS.within_scene(d("g"), d(par))
    tr = lambda p: np.array([(traj[v[f"{p}+"]] - traj[v[f"{p}-"]]).ravel() for v in ws])
    out[f"{kind}|g~{par}|tracked_trajectory"] = DS.within_scene(tr("g"), tr(par))
os.makedirs(f"{CG}/results/dev_rank", exist_ok=True); json.dump(out, open(f"{CG}/results/dev_rank/dev_rank.json", "w"), indent=1)
for k, v in out.items(): print(k.ljust(38), "cos_mean=%+.2f" % v["cos_mean"], "eff_rank=%.2f" % v["eff_rank_mean"])
