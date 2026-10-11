"""STAGE 7 (pod): cause vs effect, free fall, DEV worlds only (held-out already consumed by Stage 6).
For each dev free-fall world: gravity pair g-/g+ (existing design, g = 9.8 -/+ 1.0) and an INITIAL-VELOCITY pair v-/v+ whose visible vertical trajectory is least-squares matched to the gravity pair
(vz chosen so z_base + vz*t best fits z_g(t) over all 64 frames, separately for g- and g+). Analytic expectation: ~25% rms residual (a linear-in-t change cannot reproduce a t^2 change).
Measures: cosine between the gravity-derivative V_g = z(g+)-z(g-) and the velocity-derivative V_v = z(v+)-z(v-) in V-JEPA 2.1 block 23 (global, window), random-weight, pixel, trajectory.
Same visible effect -> cosine near the noise ceiling; different causes represented differently -> lower. DESCRIPTIVE ONLY. Thresholds declared in configs/pilot.json 'stage7'."""
import csv, importlib, json, os, sys, time
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CG = os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry"); sys.path.insert(0, ROOT); sys.path.insert(0, f"{CG}/vendor/vjepa2")
import torch
from cg import sim_pb as S, features as FT, devstats as DS
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
importlib.import_module("src.hub.backbones").VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
hub = importlib.import_module("hubconf")
CFG = json.load(open(f"{ROOT}/configs/pilot.json")); DG = CFG["delta"]["g"]; T = np.arange(S.N_FRAMES) / S.FPS
OUT = f"{CG}/results/stage7"; os.makedirs(OUT, exist_ok=True); CLIPS = f"{CG}/data/clips_s7"; os.makedirs(CLIPS, exist_ok=True)
rows = [r for r in csv.DictReader(open(f"{ROOT}/data/manifests/pilot_manifest.csv")) if r["family"] == "freefall" and r["split"] == "dev" and r["variant"] == "base"]
assert len(rows) == 15 and all(r["split"] == "dev" for r in rows), "Stage 7 uses dev freefall worlds only"
t0 = time.time(); W = {}
# ---- calibration: simulate (no render) g-, g+, base; solve vz; verify matched trajectories numerically
for r in rows:
    w = S.make_world("freefall", int(r["world_seed"])); base = S.simulate(w, {}, None, render=False)["pos"][:, 2]; d = {}
    for sg, dg in (("-", -DG), ("+", +DG)):
        zg = S.simulate(w, {"g": S.BASE["g"] + dg}, None, render=False)["pos"][:, 2]
        vz = float(((zg - base) @ T) / (T @ T))                        # least squares through the origin: min sum (base + vz t - zg)^2
        zv = S.simulate(w, {"vz": vz}, None, render=False)["pos"][:, 2]
        eff = zg - base; res = zv - zg
        d[sg] = dict(vz=vz, rms_resid_over_rms_effect=float(np.sqrt((res ** 2).mean() / (eff ** 2).mean())), peak_resid_m=float(np.abs(res).max()), peak_effect_m=float(np.abs(eff).max()))
    W[r["world_id"]] = dict(seed=int(r["world_seed"]), cal=d)
cal = {k: [W[w]["cal"][k]["rms_resid_over_rms_effect"] for w in W] for k in "-+"}
print("calibration done", f"{time.time() - t0:.0f}s", "rms residual/effect mean", {k: round(float(np.mean(v)), 3) for k, v in cal.items()}, flush=True)
# ---- render v-/v+ clips; build the four clip set per world {g-, g+, v-, v+}
P, CL = {}, {}
for wid, d in W.items():
    w = S.make_world("freefall", d["seed"]); P[wid] = {}
    for name, th in (("g-", {"g": S.BASE["g"] - DG}), ("g+", {"g": S.BASE["g"] + DG}), ("v-", {"vz": d["cal"]["-"]["vz"]}), ("v+", {"vz": d["cal"]["+"]["vz"]})):
        o = S.simulate(w, th, None, render=True); P[wid][name] = o["pos"]
        fp = f"{CLIPS}/{wid}_{name}.npy"; np.save(fp, o["frames"]); CL[(wid, name)] = fp
    print("rendered", wid, f"{time.time() - t0:.0f}s", flush=True)
# ---- features
F = {}
for mname, pre in (("pt", True), ("rw", False)):
    torch.manual_seed(0)
    enc = hub.vjepa2_1_vit_large_384(pretrained=pre); enc = (enc[0] if isinstance(enc, (tuple, list)) else enc).cuda().eval()
    for wid in W:
        mask = np.zeros(32 * 24 * 24, bool)
        for name in ("g-", "g+", "v-", "v+"): mask |= S.object_token_mask(P[wid][name], "freefall")
        mt = torch.from_numpy(mask).cuda()
        for name in ("g-", "g+", "v-", "v+"):
            f = FT.extract(enc, FT.to_tensor(np.load(CL[(wid, name)])), mask=mt); F[(f"{mname}_g", wid, name)] = f["normed"][3]; F[(f"{mname}_w", wid, name)] = f["normed_win"][3]
        print(mname, wid, f"{time.time() - t0:.0f}s", flush=True)
    del enc; torch.cuda.empty_cache()
def pix(fp):
    fr = np.load(fp, mmap_mode="r")[::4].astype(np.float32); return fr.reshape(16, 16, 24, 16, 24, 3).mean((2, 4)).ravel()
for (wid, name), fp in CL.items(): F[("px", wid, name)] = pix(fp)
for wid in W:
    for name in ("g-", "g+", "v-", "v+"): F[("traj", wid, name)] = P[wid][name].ravel()
# ---- analysis
ids = list(W); unit = lambda v: v / max(np.linalg.norm(v), 1e-12)
def Vd(rep, kind): return np.array([F[(rep, w, f"{kind}+")] - F[(rep, w, f"{kind}-")] for w in ids], dtype=np.float64)
sb = lambda r: 2 * r / (1 + r) if r > 0 else float("nan")
def pcos(A, B, idx): return float(unit(A[idx].mean(0)) @ unit(B[idx].mean(0)))
rng = np.random.default_rng(0); res = dict(n_worlds=len(ids), calibration={k: dict(mean_rms_resid_over_effect=round(float(np.mean(v)), 3), max=round(float(np.max(v)), 3)) for k, v in cal.items()},
                                              peak_resid_m_max=round(max(W[w]["cal"][k]["peak_resid_m"] for w in W for k in "-+"), 4), peak_effect_m_mean=round(float(np.mean([W[w]["cal"][k]["peak_effect_m"] for w in W for k in "-+"])), 3), reps={})
for rep in ("pt_g", "pt_w", "rw_g", "rw_w", "px", "traj"):
    A, B = Vd(rep, "g"), Vd(rep, "v"); full = np.arange(len(ids)); obs = pcos(A, B, full)
    bs = [pcos(A, B, (ix := rng.integers(0, len(ids), len(ids)))) for _ in range(2000)]
    rg, rv = DS.split_half_reliability(A)[0], DS.split_half_reliability(B)[0]; ceil = float(np.sqrt(sb(rg) * sb(rv))) if rg > 0 and rv > 0 else float("nan")
    pw = float(np.mean([unit(a) @ unit(b) for a, b in zip(A, B)])); pw_shuf = float(np.mean([unit(a) @ unit(B[(i + 1 + rng.integers(0, len(ids) - 1)) % len(ids)]) for i, a in enumerate(A) for _ in range(1)]))
    res["reps"][rep] = dict(cos_world_mean=round(obs, 3), ci95=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], reliability_g=round(rg, 3), reliability_v=round(rv, 3),
                            ceiling=round(ceil, 3), cos_over_ceiling=round(obs / ceil, 3) if ceil == ceil else None,
                            per_world_cos_mean=round(pw, 3), per_world_cos_other_world=round(pw_shuf, 3))
json.dump(res, open(f"{OUT}/stage7.json", "w"), indent=1); print(json.dumps(res["reps"], indent=1)); print("STAGE7 DONE", f"{time.time() - t0:.0f}s", flush=True)
