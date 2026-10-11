"""GATE 1 on the FINAL v3 dev manifest (same declared rule as Gate 1 v2; delta_g=1.0 per amendment A6). DEV worlds only (15/scene x 3 scenes), manifest-v2 settings (ss1, x1, delta from configs/pilot.json), fp32, 64 frames.
RULE DECLARED BEFORE THE RUN (configs/pilot.json -> gate1_v2): for EACH of g@freefall, g@bounce, g@incline, e@bounce, mu@incline,
split-half reliability of world-level derivatives >= 0.5 at >=2 of the 4 hierarchical post-norm taps, for at least one of the two poolings
(global mean, union object window); AND the hierarchical-tap consistency check passes. 2000 random splits of the 15 worlds, fixed seed.
Held-out worlds are never loaded (hard assertion). Descriptive reports (cross-context, within-scene rank, trajectory baseline) are included, not gating."""
import csv, importlib, json, os, sys, time
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CG = os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry"); VENDOR = f"{CG}/vendor/vjepa2"
sys.path.insert(0, ROOT); sys.path.insert(0, VENDOR)
import torch
from cg import sim_pb as S, features as FT, devstats as DS
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
OUT = f"{CG}/results/gate1_v3"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{CG}/features/pooled", exist_ok=True)
CFG = json.load(open(f"{ROOT}/configs/pilot.json")); D = CFG["delta"]
importlib.import_module("src.hub.backbones").VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
hub = importlib.import_module("hubconf"); enc = hub.vjepa2_1_vit_large_384(); enc = (enc[0] if isinstance(enc, (tuple, list)) else enc).cuda().eval()
rows = [r for r in csv.DictReader(open(f"{ROOT}/data/manifests/pilot_manifest.csv")) if r["family"] in S.KINDS]
assert all(r["split"] in ("dev", "heldout") for r in rows); rows = [r for r in rows if r["split"] == "dev"]
assert rows and all(r["split"] == "dev" for r in rows), "held-out worlds must never be loaded in Gate 1"
traj = np.load(f"{ROOT}/data/manifests/trajectories.npz")
worlds = {}
for r in rows: worlds.setdefault((r["kind"], r["world_id"]), {})[r["variant"]] = r["clip_id"]
NEED = {"freefall": ("g",), "bounce": ("g", "e"), "incline": ("g", "mu")}
feats, t0, R = {}, time.time(), dict(started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
for (kind, wid), vs in worlds.items():
    mask = np.zeros(32 * 24 * 24, bool)
    for cid in vs.values(): mask |= S.object_token_mask(traj[cid], kind)                 # union window over ALL variants of this world
    mt = torch.from_numpy(mask).cuda()
    for par in NEED[kind]:
        for sg in "+-":
            cid = vs[f"{par}{sg}"]; assert cid.startswith(wid)
            feats[cid] = FT.extract(enc, FT.to_tensor(np.load(f"{CG}/data/clips_v3/{cid}.npy")), mask=mt)
    print("done", wid, f"{time.time() - t0:.0f}s", flush=True)
    open(f"{OUT}/progress.txt", "w").write(f"{time.strftime('%H:%M:%S')} done {wid} ({len(feats)} clips, {time.time() - t0:.0f}s)\n")      # visible to the diag job if this ever hangs
# hierarchical consistency check (model concatenates the 4 taps on channels -> chunk)
cid0 = next(iter(feats)); enc.return_hierarchical = True
with torch.no_grad(): hier = enc(FT.to_tensor(np.load(f"{CG}/data/clips_v3/{cid0}.npy")))
enc.return_hierarchical = False
theirs = np.stack([h.mean(1)[0].float().cpu().numpy() for h in hier.chunk(4, dim=2)]); diff = float(np.abs(theirs - feats[cid0]["normed"]).max())
R["hier_check"] = dict(max_abs_diff=diff, ok=bool(diff < 1e-3), final_equals_norm23=bool(np.abs(feats[cid0]["final"] - feats[cid0]["normed"][3]).max() < 1e-4))

def V(kind, par, pool, tap):
    key = {"global": "normed", "window": "normed_win"}[pool]
    return np.array([feats[vs[f"{par}+"]][key][tap] - feats[vs[f"{par}-"]][key][tap] for (k, _), vs in worlds.items() if k == kind])
CASES = [("g", "freefall"), ("g", "bounce"), ("g", "incline"), ("e", "bounce"), ("mu", "incline")]; NT = (5, 11, 17, 23)
rel, verdict = {}, {}
for par, kind in CASES:
    case = f"{par}@{kind}"; verdict[case] = {}
    for pool in ("global", "window"):
        rs = [DS.split_half_reliability(V(kind, par, pool, k)) for k in range(4)]
        rel[f"{case}|{pool}"] = {f"norm{NT[k]}": dict(mean=round(rs[k][0], 3), worst=round(rs[k][1], 3)) for k in range(4)}
        verdict[case][pool] = dict(taps_ge_0p5=sum(r[0] >= 0.5 for r in rs), passes=sum(r[0] >= 0.5 for r in rs) >= 2)
    verdict[case]["case_passes"] = any(verdict[case][p]["passes"] for p in ("global", "window"))
R["reliability"] = rel; R["verdict"] = verdict
R["gate1_v2_pass"] = bool(all(v["case_passes"] for v in verdict.values()) and R["hier_check"]["ok"]); R["n_worlds"] = {k: sum(1 for (kk, _) in worlds if kk == k) for k in S.KINDS}
# descriptive, non-gating: within-scene separability (cos g~e, g~mu), cross-context g~g with wrong-parameter control, trajectory baseline
desc = {}
for kind, par in (("bounce", "e"), ("incline", "mu")):
    for pool in ("global", "window"):
        for k in (1, 3): desc[f"within|{kind}|g~{par}|{pool}|norm{NT[k]}"] = DS.within_scene(V(kind, "g", pool, k), V(kind, par, pool, k))
    tj = lambda p: np.array([(traj[vs[f"{p}+"]] - traj[vs[f"{p}-"]]).ravel() for (kk, _), vs in worlds.items() if kk == kind])
    desc[f"within|{kind}|g~{par}|tracked_trajectory"] = DS.within_scene(tj("g"), tj(par))
for pool in ("global", "window"):
    for k in (1, 3):
        gF, gB, gI = (V(q, "g", pool, k) for q in S.KINDS); eB, mI = V("bounce", "e", pool, k), V("incline", "mu", pool, k)
        desc[f"cross|{pool}|norm{NT[k]}"] = {"freefall~bounce": DS.alignment_with_control(gF, gB, eB), "freefall~incline": DS.alignment_with_control(gF, gI, mI), "bounce~incline": DS.alignment_with_control(gB, gI, mI)}
R["descriptive_not_gating"] = desc
np.savez_compressed(f"{CG}/features/pooled/gate1_v3.npz", ids=np.array(list(feats)), normed=np.stack([f["normed"] for f in feats.values()]), normed_win=np.stack([f["normed_win"] for f in feats.values()]),
                    raw=np.stack([f["raw"] for f in feats.values()]), raw_win=np.stack([f["raw_win"] for f in feats.values()]))
json.dump(R, open(f"{OUT}/gate1_v3.json", "w"), indent=1, default=float)
print(json.dumps(dict(hier_check=R["hier_check"], verdict=verdict, gate1_v2_pass=R["gate1_v2_pass"], n_worlds=R["n_worlds"]), indent=1)[:3000])
