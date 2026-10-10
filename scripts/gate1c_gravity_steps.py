"""GATE 1c (pod): does a larger gravity step make the g-derivative reliable? DEV worlds only (15/scene), frozen manifest settings except delta_g.
Steps: m in {1,2,4} -> delta_g = m*0.5 (g = 9.8 -/+ m*0.5). Poolings: global mean, union object window (union over ALL variants of the world incl. the new ones).
RULE DECLARED BEFORE THE RUN (configs/pilot.json -> gate1c): for each m and each case g@freefall, g@bounce, g@incline: split-half reliability (2000 random splits
of the 15 worlds) >= 0.5 at >=2 of the 4 hierarchical post-norm taps for at least one pooling. The adoption candidate is the SMALLEST m for which all three cases pass.
Reported, not gating: dose-response cosine between world-mean derivatives at m and 2m, relative signal size, in-view / transition-frame validity of the new variants.
Held-out worlds are never loaded (hard assertion)."""
import csv, importlib, json, os, sys, time
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CG = os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry"); VENDOR = f"{CG}/vendor/vjepa2"
sys.path.insert(0, ROOT); sys.path.insert(0, VENDOR)
import torch
from cg import sim_pb as S, features as FT, devstats as DS
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
OUT = f"{CG}/results/gate1c"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{CG}/features/pooled", exist_ok=True)
CFG = json.load(open(f"{ROOT}/configs/pilot.json")); D = CFG["delta"]; G0 = CFG["base_theta"]["g"]
importlib.import_module("src.hub.backbones").VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
hub = importlib.import_module("hubconf"); enc = hub.vjepa2_1_vit_large_384(); enc = (enc[0] if isinstance(enc, (tuple, list)) else enc).cuda().eval()
rows = [r for r in csv.DictReader(open(f"{ROOT}/data/manifests/pilot_manifest.csv")) if r["family"] in S.KINDS]
rows = [r for r in rows if r["split"] == "dev"]; assert rows and all(r["split"] == "dev" for r in rows), "held-out must never be loaded"
traj = np.load(f"{ROOT}/data/manifests/trajectories.npz")
worlds = {}
for r in rows: worlds.setdefault((r["kind"], r["world_id"]), {})[r["variant"]] = r
MULT = (1, 2, 4); NT = (5, 11, 17, 23); feats, validity, t0 = {}, [], time.time()
for (kind, wid), vs in worlds.items():
    w = S.make_world(kind, int(next(iter(vs.values()))["world_seed"])); frames, pos = {}, {}
    for m in MULT:
        for sg, tag in ((1, "+"), (-1, "-")):
            key = (m, tag)
            if m == 1: frames[key] = np.load(f"{CG}/data/clips_v2/{vs['g' + tag]['clip_id']}.npy")              # already rendered (identical to a re-render)
            else:
                o = S.simulate(w, {"g": G0 + sg * m * D["g"]}, render=True); frames[key], pos[key] = o["frames"], o["pos"]
                validity.append(dict(world=wid, m=m, sign=tag, in_view=o["in_view"], transition=o.get("transition_frame"), impacts=len(o["impacts"])))
    mask = np.zeros(32 * 24 * 24, bool)
    for r in vs.values(): mask |= S.object_token_mask(traj[r["clip_id"]], kind)
    for p_ in pos.values(): mask |= S.object_token_mask(p_, kind)
    mt = torch.from_numpy(mask).cuda()
    for key, fr in frames.items(): feats[(wid, key)] = FT.extract(enc, FT.to_tensor(fr), mask=mt)
    del frames
    open(f"{OUT}/progress.txt", "w").write(f"{time.strftime('%H:%M:%S')} done {wid} ({len(feats)} clips, {time.time() - t0:.0f}s)\n"); print("done", wid, f"{time.time() - t0:.0f}s", flush=True)


def V(kind, m, pool, tap):
    key = {"global": "normed", "window": "normed_win"}[pool]
    return np.array([feats[(wid, (m, "+"))][key][tap] - feats[(wid, (m, "-"))][key][tap] for (k, wid) in worlds if k == kind])


R = dict(started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), n_worlds={k: sum(1 for (kk, _) in worlds if kk == k) for k in S.KINDS}, reliability={}, verdict={}, dose_response={}, validity={})
for kind in S.KINDS:
    for m in MULT:
        case = f"g@{kind}|m{m}(dg={m * D['g']})"; per_pool = {}
        for pool in ("global", "window"):
            rs = [DS.split_half_reliability(V(kind, m, pool, k)) for k in range(4)]
            R["reliability"][f"{case}|{pool}"] = {f"norm{NT[k]}": dict(mean=round(rs[k][0], 3), worst=round(rs[k][1], 3)) for k in range(4)}
            per_pool[pool] = dict(taps_ge_0p5=sum(r[0] >= 0.5 for r in rs), passes=sum(r[0] >= 0.5 for r in rs) >= 2)
        R["verdict"][case] = dict(per_pool, case_passes=any(v["passes"] for v in per_pool.values()))
    for pool in ("global", "window"):
        for k in (1, 3):
            mean = {m: V(kind, m, pool, k).mean(0) for m in MULT}; cs = lambda a, b: float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
            R["dose_response"][f"g@{kind}|{pool}|norm{NT[k]}"] = dict(cos_m1_m2=round(cs(mean[1], mean[2]), 3), cos_m2_m4=round(cs(mean[2], mean[4]), 3),
                                                                    rel_signal={m: round(float(np.mean([np.linalg.norm(x) for x in V(kind, m, pool, k)]) / np.mean([np.linalg.norm(feats[(wid, (1, '+'))]['normed' if pool == 'global' else 'normed_win'][k]) for (kk, wid) in worlds if kk == kind])), 5) for m in MULT})
R["adoption"] = {}
for m in MULT:
    ok = {kind: R["verdict"][f"g@{kind}|m{m}(dg={m * D['g']})"]["case_passes"] for kind in S.KINDS}
    R["adoption"][f"m{m}"] = dict(cases=ok, all_three_pass=all(ok.values()))
cand = [m for m in MULT if R["adoption"][f"m{m}"]["all_three_pass"]]; R["adoption_candidate_smallest_m"] = cand[0] if cand else None
for kind in S.KINDS:
    vv = [v for v in validity if kind in v["world"]]
    R["validity"][kind] = dict(n=len(vv), not_in_view=sum(not v["in_view"] for v in vv), transition_outside_12_50=sum(1 for v in vv if v["transition"] is not None and not (12 <= v["transition"] <= 50)), no_impact=sum(1 for v in vv if kind == "bounce" and v["impacts"] < 1))
np.savez_compressed(f"{CG}/features/pooled/gate1c.npz", keys=np.array([f"{w}|{k[0]}{k[1]}" for (w, k) in feats]), normed=np.stack([f["normed"] for f in feats.values()]), normed_win=np.stack([f["normed_win"] for f in feats.values()]))
json.dump(R, open(f"{OUT}/gate1c.json", "w"), indent=1, default=float); print(json.dumps(dict(adoption=R["adoption"], candidate=R["adoption_candidate_smallest_m"], validity=R["validity"]), indent=1))
