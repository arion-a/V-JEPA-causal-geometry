"""GATE 1 (pod): fp32 pooled features for the 20-clip subset + extra renders (jitter noise floor, half-delta linearity).
Writes $CG_ROOT/features/pooled/gate1.npz and $CG_ROOT/results/gate1/gate1.json. Descriptive; n=1 world per family."""
import csv, importlib, json, os, sys, time, traceback
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CG = os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry"); VENDOR = f"{CG}/vendor/vjepa2"
sys.path.insert(0, ROOT); sys.path.insert(0, VENDOR)
import torch
from cg import sim_pb as S, features as FT, gate1_analysis as GA
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
OUT = f"{CG}/results/gate1"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{CG}/features/pooled", exist_ok=True)
CFG = json.load(open(f"{ROOT}/configs/pilot.json")); R = dict(started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), checks={})
spec = importlib.util.spec_from_file_location("bm", f"{ROOT}/scripts/build_manifest.py"); bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)

importlib.import_module("src.hub.backbones").VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
hub = importlib.import_module("hubconf"); enc = hub.vjepa2_1_vit_large_384(); enc = (enc[0] if isinstance(enc, (tuple, list)) else enc).cuda().eval()
rows = [r for r in csv.DictReader(open(f"{ROOT}/data/manifests/pilot_manifest.csv")) if r["gate1"] == "1"]
by_id = {r["clip_id"]: r for r in rows}

# preprocessing note: record the repo's own preprocessor so normalisation can be compared (not relied upon)
try:
    pre = hub.vjepa2_preprocessor(); R["checks"]["repo_preprocessor"] = dict(repr=str(pre)[:1500])
except Exception as e: R["checks"]["repo_preprocessor"] = dict(error=repr(e)[:300])

feats, secs = {}, []
def run_clip(cid, frames):
    t = time.time(); f = FT.extract(enc, FT.to_tensor(frames)); secs.append(time.time() - t); feats[cid] = f; return f

# 1) the 20 gate-1 clips from the volume
for cid in by_id: run_clip(cid, np.load(f"{CG}/data/clips/{cid}.npy"))
# 2) verify our hierarchical taps == model's own return_hierarchical output on one clip
cid0 = next(iter(by_id)); x = FT.to_tensor(np.load(f"{CG}/data/clips/{cid0}.npy"))
enc.return_hierarchical = True
with torch.no_grad(): hier = enc(x)
enc.return_hierarchical = False
mine = feats[cid0]["normed"]; theirs = np.stack([h.mean(1)[0].float().cpu().numpy() for h in hier])
R["checks"]["hierarchical_taps_match_model"] = dict(n=len(hier), max_abs_diff=float(np.abs(mine - theirs).max()), ok=bool(np.abs(mine - theirs).max() < 1e-4),
                                                   final_equals_norm23=bool(np.abs(feats[cid0]["final"] - mine[3]).max() < 1e-4))
# 3) extra renders: sub-pixel camera jitter (noise floor) and half-delta g (linearity), for freefall/bounce/incline dev worlds
extra = {}
for kind in ("freefall", "bounce", "incline"):
    r0 = next(r for r in rows if r["family"] == kind and r["variant"] == "base"); w = S.make_world(kind, int(r0["world_seed"])); wid = r0["world_id"]
    for tag, over, nz in (("jit005", {}, {"cam": 0.005}), ("jit020", {}, {"cam": 0.02}),
                          ("gp", {"g": S.BASE["g"] + CFG["delta"]["g"] / 2}, None), ("gm", {"g": S.BASE["g"] - CFG["delta"]["g"] / 2}, None)):
        cid = f"{wid}__{tag}"; run_clip(cid, S.simulate(w, over, nz)["frames"]); extra[cid] = tag
F = {c: GA.taps(f) for c, f in feats.items()}
worlds = sorted({c.split("__")[0] for c in F})
jitter = {w: f"{w}__jit005" for w in worlds if f"{w}__jit005" in F}; jitter2 = {w: f"{w}__jit020" for w in worlds if f"{w}__jit020" in F}
half = {(w, "g"): (f"{w}__gp", f"{w}__gm", CFG["delta"]["g"] / 2) for w in worlds if f"{w}__gp" in F}
comp = {}
for w in worlds:
    ids = {c.split("__")[1]: c for c in F if c.startswith(w + "__")}
    if "second+" in ids: comp[w] = {k: ids[k] for k in ("base", "g+", "second+", "both+")}
an = GA.analyse(F, CFG["delta"], jitter, half, comp, jitter2)
# predeclared Gate-1 criterion: signal >= 5x jitter floor in at least one hierarchical tap for each observable parameter
crit = {}
for par, kind in (("g", "freefall"), ("g", "bounce"), ("g", "incline"), ("e", "bounce"), ("mu", "incline")):
    w = next(w for w in worlds if w.endswith(f"_{kind}") and f"snr_{par}" in an["raw0"].get(w, {}))
    best = {t: an[t][w][f"snr_{par}"] for t in GA.NORM if f"snr_{par}" in an[t].get(w, {})}
    crit[f"{par}@{kind}"] = dict(world=w, snr_by_hier_tap={k: round(v, 2) for k, v in best.items()}, passes=bool(max(best.values()) >= 5))
R["criteria_snr_ge_5_at_some_hier_tap"] = crit
R["gate1_pass"] = all(v["passes"] for v in crit.values()) and R["checks"]["hierarchical_taps_match_model"]["ok"]
R["analysis"] = an; R["extra_clips"] = extra; R["seconds_per_clip_infer_fp32"] = round(float(np.mean(secs)), 2); R["n_clips_extracted"] = len(feats)
np.savez_compressed(f"{CG}/features/pooled/gate1.npz", ids=np.array(list(feats)), raw=np.stack([f["raw"] for f in feats.values()]),
                    normed=np.stack([f["normed"] for f in feats.values()]), final=np.stack([f["final"] for f in feats.values()]))
json.dump(R, open(f"{OUT}/gate1.json", "w"), indent=1, default=float)
print(json.dumps({k: R[k] for k in ("checks", "criteria_snr_ge_5_at_some_hier_tap", "gate1_pass", "seconds_per_clip_infer_fp32", "n_clips_extracted")}, indent=1, default=float)[:3500])
