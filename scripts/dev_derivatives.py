"""DEV-WORLD PREVIEW (pod): per-world derivatives at the FROZEN manifest settings (ss1, x1, m=1, 64 frames, fp32) for all dev worlds of the
3 primary scenes, pooled globally AND over a union object window (union of all variants' trajectories, so every variant of a world shares
one window). Reports split-half reliability across worlds, cross-context cosines with a wrong-parameter control, and the same on the
tracked-trajectory baseline. HELD-OUT WORLDS ARE NOT TOUCHED. Descriptive; a preview, not a test."""
import csv, itertools, json, os, sys, time
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CG = os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry"); VENDOR = f"{CG}/vendor/vjepa2"
sys.path.insert(0, ROOT); sys.path.insert(0, VENDOR)
import importlib, torch
from cg import sim_pb as S, features as FT, devstats as DS
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
OUT = f"{CG}/results/dev_derivatives"; os.makedirs(OUT, exist_ok=True)
CFG = json.load(open(f"{ROOT}/configs/pilot.json")); D = CFG["delta"]
importlib.import_module("src.hub.backbones").VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
hub = importlib.import_module("hubconf"); enc = hub.vjepa2_1_vit_large_384(); enc = (enc[0] if isinstance(enc, (tuple, list)) else enc).cuda().eval()
rows = [r for r in csv.DictReader(open(f"{ROOT}/data/manifests/pilot_manifest.csv")) if r["family"] in S.KINDS and r["split"] == "dev"]
traj = np.load(f"{ROOT}/data/manifests/trajectories.npz")
worlds = {}
for r in rows: worlds.setdefault((r["kind"], r["world_id"]), {})[r["variant"]] = r["clip_id"]
feats = {}; t0 = time.time()
for (kind, wid), vs in worlds.items():
    mask = np.zeros(32 * 24 * 24, bool)
    for cid in vs.values(): mask |= S.object_token_mask(traj[cid], kind)                  # union window across all variants of this world
    mt = torch.from_numpy(mask).cuda()
    for var, cid in vs.items(): feats[cid] = FT.extract(enc, FT.to_tensor(np.load(f"{CG}/data/clips/{cid}.npy")), mask=mt)
    print("done", wid, f"{time.time() - t0:.0f}s", f"window_frac={mask.mean():.3f}", flush=True)
CASES = [("g", "freefall"), ("g", "bounce"), ("g", "incline"), ("e", "bounce"), ("mu", "incline")]


def deriv(kind, par, pool, tap, which="normed"):
    """V [n_dev_worlds, d]: (z(+)-z(-)) per world, taps standardised to one delta step (not divided by delta, so units are comparable)."""
    key = {"global": which, "window": which + "_win"}[pool]; V = []
    for (k, wid), vs in worlds.items():
        if k == kind: V.append(feats[vs[f"{par}+"]][key][tap] - feats[vs[f"{par}-"]][key][tap])
    return np.array(V)


def traj_deriv(kind, par):
    return np.array([(traj[vs[f"{par}+"]] - traj[vs[f"{par}-"]]).ravel() for (k, wid), vs in worlds.items() if k == kind])


res = dict(n_dev_worlds={k: sum(1 for (kk, _) in worlds if kk == k) for k in S.KINDS}, taps="hierarchical post-norm blocks 5,11,17,23", out={})
for pool in ("global", "window"):
    for tap, name in enumerate((5, 11, 17, 23)):
        r = {}
        for par, kind in CASES:
            V = deriv(kind, par, pool, tap); mean_rel = float(np.mean([np.linalg.norm(v) / np.linalg.norm(feats[vs["base"]]["normed" if pool == "global" else "normed_win"][tap]) for v, ((k, _), vs) in zip(V, [(kk, vv) for kk, vv in worlds.items() if kk[0] == kind])]))
            rel, worst = DS.split_half_reliability(V); r[f"{par}@{kind}"] = dict(split_half_cos=round(rel, 3), worst_split=round(worst, 3), mean_rel_signal=round(mean_rel, 5))
        gF, gB, gI = (deriv(k, "g", pool, tap) for k in S.KINDS)
        eB, mI = deriv("bounce", "e", pool, tap), deriv("incline", "mu", pool, tap)
        r["cross_context"] = {"freefall~bounce": DS.alignment_with_control(gF, gB, eB), "freefall~incline": DS.alignment_with_control(gF, gI, mI),
                              "bounce~incline": DS.alignment_with_control(gB, gI, mI)}
        res["out"][f"{pool}|norm{name}"] = r
tb = {k: traj_deriv(k, "g") for k in S.KINDS}; te = traj_deriv("bounce", "e"); tm = traj_deriv("incline", "mu")
res["tracked_trajectory_baseline"] = {"split_half_g": {k: round(DS.split_half_reliability(v)[0], 3) for k, v in tb.items()},
                                      "freefall~bounce": DS.alignment_with_control(tb["freefall"], tb["bounce"], te), "freefall~incline": DS.alignment_with_control(tb["freefall"], tb["incline"], tm),
                                      "bounce~incline": DS.alignment_with_control(tb["bounce"], tb["incline"], tm)}
res["note"] = "dev worlds only; 5 worlds/scene; descriptive preview. Cosines between world-mean derivative vectors (same standardised step units)."
np.savez_compressed(f"{CG}/features/pooled/dev_derivs.npz", ids=np.array(list(feats)), normed=np.stack([f["normed"] for f in feats.values()]), normed_win=np.stack([f["normed_win"] for f in feats.values()]),
                    raw=np.stack([f["raw"] for f in feats.values()]), raw_win=np.stack([f["raw_win"] for f in feats.values()]))
json.dump(res, open(f"{OUT}/dev_derivatives.json", "w"), indent=1); print(json.dumps(res["tracked_trajectory_baseline"], indent=1))
