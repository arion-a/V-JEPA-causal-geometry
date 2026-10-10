"""GATE 1b (pod): why did Gate 1 fail, and which render/step/pooling settings give a STABLE finite-difference derivative?
DEV WORLDS ONLY (the 3 Gate-1 base worlds); held-out worlds untouched. Sweep: render config (supersampling x object scale) x step multiplier m in {1,2,4}
(delta = m * configs/pilot.json delta) x pooling {global mean, object window}.
PREDECLARED ADOPTION RULE (written before the run): a (render config, pooling) is a candidate iff for EACH of the 5 cases
g@freefall, g@bounce, g@incline, e@bounce, mu@incline there is a step pair (m,2m) such that at >=2 of the 4 hierarchical post-norm taps
cos(v(m),v(2m)) >= 0.8 and 0.7 <= |v(2m)|/|v(m)| <= 1.4, where v(m)=(z(theta+m d)-z(theta-m d))/(2 m d).
The Gate-1 'signal >= 5x camera-jitter floor' number is still reported. Choosing a candidate after seeing dev data is an amendment, logged as such."""
import csv, importlib, itertools, json, os, sys, time
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CG = os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry"); VENDOR = f"{CG}/vendor/vjepa2"
sys.path.insert(0, ROOT); sys.path.insert(0, VENDOR)
import torch
from cg import sim_pb as S, features as FT
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
OUT = f"{CG}/results/gate1b"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{CG}/features/pooled", exist_ok=True)
CFG = json.load(open(f"{ROOT}/configs/pilot.json")); D = CFG["delta"]; B = CFG["base_theta"]
importlib.import_module("src.hub.backbones").VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
hub = importlib.import_module("hubconf"); enc = hub.vjepa2_1_vit_large_384(); enc = (enc[0] if isinstance(enc, (tuple, list)) else enc).cuda().eval()
rows = [r for r in csv.DictReader(open(f"{ROOT}/data/manifests/pilot_manifest.csv")) if r["gate1"] == "1" and r["variant"] == "base" and r["family"] in S.KINDS]
CONFIGS = [dict(name=f"ss{ss}_x{int(sc)}", ss=ss, scale=sc) for ss, sc in itertools.product((1, 2), (1.0, 2.0))]
MULT = (1, 2, 4); CASES = [("g", "freefall"), ("g", "bounce"), ("g", "incline"), ("e", "bounce"), ("mu", "incline")]
POOL = ("global", "window"); NT = [5, 11, 17, 23]
feats, timing = {}, dict(render=[], infer=[])


def clip(key, w, over, nz, c, mask):
    t = time.time(); out = S.simulate(w, over, nz, ss=c["ss"], size_scale=c["scale"]); timing["render"].append(time.time() - t)
    t = time.time(); f = FT.extract(enc, FT.to_tensor(out["frames"]), mask=mask); timing["infer"].append(time.time() - t); feats[key] = f
    return out


def vec(f, pool, tap):  # hierarchical post-norm tap k (0..3)
    return f["normed" if pool == "global" else "normed_win"][tap]


def cosr(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)), float(np.linalg.norm(b) / max(np.linalg.norm(a), 1e-12))


for c in CONFIGS:
    for r in rows:
        w = S.make_world(r["kind"], int(r["world_seed"])); kind = r["kind"]
        base = S.simulate(w, ss=1, size_scale=c["scale"], render=False)
        mask = torch.from_numpy(S.object_token_mask(base["pos"], kind)).cuda()
        tag = f"{c['name']}|{kind}"
        clip(f"{tag}|base", w, {}, None, c, mask); clip(f"{tag}|jit", w, {}, {"cam": 0.005}, c, mask)
        for par in S.OBSERVABLE[kind]:
            for m in MULT:
                for sg in (1, -1): clip(f"{tag}|{par}{'+' if sg > 0 else '-'}{m}", w, {par: B[par] + sg * m * D[par]}, None, c, mask)
        print("done", tag, flush=True)

res = dict(configs=[c["name"] for c in CONFIGS], rule=__doc__.split("PREDECLARED ADOPTION RULE")[1].split("The Gate-1")[0].strip(), table={}, candidates=[])
for c in CONFIGS:
    for pool in POOL:
        passed = {}
        for par, kind in CASES:
            tag = f"{c['name']}|{kind}"; best = None; detail = {}
            for m in (1, 2):
                ok_taps = 0
                for k in range(4):
                    v = {mm: (vec(feats[f"{tag}|{par}+{mm}"], pool, k) - vec(feats[f"{tag}|{par}-{mm}"], pool, k)) / (2 * mm * D[par]) for mm in (m, 2 * m)}
                    cs, rt = cosr(v[m], v[2 * m]); z0 = vec(feats[f"{tag}|base"], pool, k)
                    sig = float(np.linalg.norm(vec(feats[f"{tag}|{par}+{m}"], pool, k) - vec(feats[f"{tag}|{par}-{m}"], pool, k)) / np.linalg.norm(z0))
                    jit = float(np.linalg.norm(vec(feats[f"{tag}|jit"], pool, k) - z0) / np.linalg.norm(z0))
                    detail[f"m{m}->{2 * m}|norm{NT[k]}"] = dict(cos=round(cs, 3), ratio=round(rt, 3), rel_signal=round(sig, 5), jitter_floor=round(jit, 5), snr=round(sig / max(jit, 1e-12), 2))
                    ok_taps += int(cs >= 0.8 and 0.7 <= rt <= 1.4)
                if ok_taps >= 2 and best is None: best = m
            passed[f"{par}@{kind}"] = dict(stable_from_m=best, passes=best is not None)
            res["table"][f"{c['name']}|{pool}|{par}@{kind}"] = detail
        allp = all(v["passes"] for v in passed.values())
        res.setdefault("summary", {})[f"{c['name']}|{pool}"] = dict(cases=passed, all_five_pass=allp)
        if allp: res["candidates"].append(dict(config=c["name"], pooling=pool, max_m=max(v["stable_from_m"] for v in passed.values())))
res["candidates"].sort(key=lambda x: (CONFIGS[[q["name"] for q in CONFIGS].index(x["config"])]["ss"] * 10 + CONFIGS[[q["name"] for q in CONFIGS].index(x["config"])]["scale"], x["max_m"]))
res["seconds_per_clip"] = dict(render=round(float(np.mean(timing["render"])), 2), infer=round(float(np.mean(timing["infer"])), 2)); res["n_clips"] = len(feats)
np.savez_compressed(f"{CG}/features/pooled/gate1b.npz", keys=np.array(list(feats)), raw=np.stack([f["raw"] for f in feats.values()]), normed=np.stack([f["normed"] for f in feats.values()]),
                    raw_win=np.stack([f["raw_win"] for f in feats.values()]), normed_win=np.stack([f["normed_win"] for f in feats.values()]))
json.dump(res, open(f"{OUT}/gate1b.json", "w"), indent=1)
print(json.dumps(dict(summary=res["summary"], candidates=res["candidates"], timing=res["seconds_per_clip"], n=res["n_clips"]), indent=1)[:3500])
