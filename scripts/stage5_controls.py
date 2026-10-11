"""STAGE 5 (pod): controls on DEV worlds only. Representations compared on identical matched pairs (delta_g=1.0, e 0.05, mu 0.009; 15 dev worlds/scene):
  pretrained V-JEPA 2.1 (4 hierarchical taps x {global, union-window}), RANDOM-WEIGHT V-JEPA (same architecture, no training), PIXEL difference (16 frames, 32x32 RGB),
  TRACKED TRAJECTORY (simulator positions). Per representation: split-half reliability, sign-flip null and shuffled-pairing reliability, cross-scene gravity cosine with
  95% world-bootstrap CI and a wrong-parameter control, within-scene cos(g, e|mu) with CI. DESCRIPTIVE ONLY - no hypothesis verdict; held-out stays locked."""
import csv, importlib, json, os, sys, time
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CG = os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry"); VENDOR = f"{CG}/vendor/vjepa2"
sys.path.insert(0, ROOT); sys.path.insert(0, VENDOR)
import torch
from cg import sim_pb as S, features as FT, devstats as DS
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
OUT = f"{CG}/results/stage5"; os.makedirs(OUT, exist_ok=True)
importlib.import_module("src.hub.backbones").VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
hub = importlib.import_module("hubconf")
rows = [r for r in csv.DictReader(open(f"{ROOT}/data/manifests/pilot_manifest.csv")) if r["family"] in S.KINDS and r["split"] == "dev"]
assert rows and all(r["split"] == "dev" for r in rows), "held-out must never be loaded"
traj = np.load(f"{ROOT}/data/manifests/trajectories.npz"); worlds = {}
for r in rows: worlds.setdefault((r["kind"], r["world_id"]), {})[r["variant"]] = r["clip_id"]
NEED = {"freefall": ("g",), "bounce": ("g", "e"), "incline": ("g", "mu")}; NT = (5, 11, 17, 23)
CASES = [("g", "freefall"), ("g", "bounce"), ("g", "incline"), ("e", "bounce"), ("mu", "incline")]
need_ids = [vs[f"{p}{sg}"] for (k, _), vs in worlds.items() for p in NEED[k] for sg in "+-"]
t0 = time.time(); print("clips needed:", len(need_ids), flush=True)

# ---- pretrained features (saved by the Gate-1 run on the same clips/masks)
Z0 = np.load(f"{CG}/features/pooled/gate1_v3.npz"); idx0 = {c: i for i, c in enumerate(Z0["ids"])}
# ---- random-weight V-JEPA, same masks
torch.manual_seed(0)
enc = hub.vjepa2_1_vit_large_384(pretrained=False); enc = (enc[0] if isinstance(enc, (tuple, list)) else enc).cuda().eval()
Fr = {}
for (kind, wid), vs in worlds.items():
    mask = np.zeros(32 * 24 * 24, bool)
    for cid in vs.values(): mask |= S.object_token_mask(traj[cid], kind)
    mt = torch.from_numpy(mask).cuda()
    for p in NEED[kind]:
        for sg in "+-":
            cid = vs[f"{p}{sg}"]; Fr[cid] = FT.extract(enc, FT.to_tensor(np.load(f"{CG}/data/clips_v3/{cid}.npy")), mask=mt)
    print("random-weight done", wid, f"{time.time() - t0:.0f}s", flush=True)
del enc; torch.cuda.empty_cache()
# ---- pixel baseline
def pix(cid):
    fr = np.load(f"{CG}/data/clips_v3/{cid}.npy", mmap_mode="r")[::4].astype(np.float32)
    return fr.reshape(16, 32, 12, 32, 12, 3).mean((2, 4)).ravel()
Px = {cid: pix(cid) for cid in need_ids}; print("pixel baseline done", f"{time.time() - t0:.0f}s", flush=True)

def pair(rep, kind, par):
    """-> (P, M) arrays [n_worlds, d] for the matched +/- clips of one case."""
    P, M = [], []
    for (k, wid), vs in worlds.items():
        if k != kind: continue
        a, b = vs[f"{par}+"], vs[f"{par}-"]
        if rep == "trajectory": P.append(traj[a].ravel()); M.append(traj[b].ravel())
        elif rep == "pixel": P.append(Px[a]); M.append(Px[b])
        else:
            model, pool, tap = rep.split("|"); t = NT.index(int(tap[4:])); key = "normed" if pool == "global" else "normed_win"
            if model == "pretrained": P.append(Z0[key][idx0[a], t]); M.append(Z0[key][idx0[b], t])
            else: P.append(Fr[a][key][t]); M.append(Fr[b][key][t])
    return np.array(P, dtype=np.float64), np.array(M, dtype=np.float64)

reps = [f"{m}|{p}|norm{t}" for m in ("pretrained", "random") for p in ("global", "window") for t in NT] + ["pixel", "trajectory"]
res = dict(n_worlds={k: sum(1 for (kk, _) in worlds if kk == k) for k in S.KINDS}, note="dev only, descriptive; CIs are 95% world-bootstrap percentile intervals (biased slightly low near cos=1)", reps={})
for rep in reps:
    D = {c: pair(rep, c[1], c[0]) for c in CASES}; V = {c: D[c][0] - D[c][1] for c in CASES}; o = {"reliability": {}, "sign_flip_null": {}, "shuffled_pairing": {}, "cross_scene_g": {}, "within_scene": {}}
    for c in CASES:
        name = f"{c[0]}@{c[1]}"; o["reliability"][name] = round(DS.split_half_reliability(V[c])[0], 3)
        o["sign_flip_null"][name] = round(DS.sign_flip_null_reliability(V[c])[0], 3); o["shuffled_pairing"][name] = round(DS.shuffled_pairing_reliability(*D[c])[0], 3)
    wrong = {"bounce": ("e", "bounce"), "incline": ("mu", "incline")}
    for A, B in (("freefall", "bounce"), ("freefall", "incline"), ("bounce", "incline")):
        gA, gB = V[("g", A)], V[("g", B)]; wr = V[wrong[B]]
        o["cross_scene_g"][f"{A}~{B}"] = dict(g_g=[round(x, 3) for x in DS.bootstrap_mean_cos(gA, gB)], g_vs_wrong_param=[round(x, 3) for x in DS.bootstrap_mean_cos(gA, wr)])
    o["within_scene"]["bounce g~e"] = [round(x, 3) for x in DS.within_scene_ci(V[("g", "bounce")], V[("e", "bounce")])]
    o["within_scene"]["incline g~mu"] = [round(x, 3) for x in DS.within_scene_ci(V[("g", "incline")], V[("mu", "incline")])]
    res["reps"][rep] = o
json.dump(res, open(f"{OUT}/stage5.json", "w"), indent=1); print("STAGE5 DONE", f"{time.time() - t0:.0f}s", flush=True)
