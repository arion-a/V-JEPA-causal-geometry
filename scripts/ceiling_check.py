"""Noise-ceiling check on Stage 5 output (dev worlds, descriptive). CPU only, reads results/stage5/stage5.json.
r_full = 2r/(1+r) (Spearman-Brown, split-half -> full 15 worlds); ceiling(A,B) = sqrt(r_full_A * r_full_B); corrected = observed / ceiling."""
import json, math, os
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
D = json.load(open(f"{ROOT}/results/stage5/stage5.json"))["reps"]
sb = lambda r: 2 * r / (1 + r) if r > 0 else float("nan")
PAIRS = {"freefall~bounce": ("g@freefall", "g@bounce"), "freefall~incline": ("g@freefall", "g@incline"), "bounce~incline": ("g@bounce", "g@incline")}
WITHIN = {"bounce g~e": ("g@bounce", "e@bounce"), "incline g~mu": ("g@incline", "mu@incline")}
out = {}
for rep, v in D.items():
    rf = {k: sb(x) for k, x in v["reliability"].items()}
    o = dict(r_full={k: round(x, 3) for k, x in rf.items()}, cross_scene={}, within_scene={})
    for p, (a, b) in PAIRS.items():
        c = math.sqrt(rf[a] * rf[b]) if rf[a] == rf[a] and rf[b] == rf[b] else float("nan")
        obs, lo, hi = v["cross_scene_g"][p]["g_g"]
        o["cross_scene"][p] = dict(obs=obs, ceiling=round(c, 3), corrected=round(obs / c, 3) if c == c else None, corrected_ci=[round(lo / c, 3), round(hi / c, 3)] if c == c else None)
    for p, (a, b) in WITHIN.items():
        c = math.sqrt(rf[a] * rf[b]) if rf[a] == rf[a] and rf[b] == rf[b] else float("nan")
        obs, lo, hi = v["within_scene"][p]
        o["within_scene"][p] = dict(obs=obs, ceiling=round(c, 3), corrected=round(obs / c, 3) if c == c else None)
    out[rep] = o
os.makedirs(f"{ROOT}/results/ceiling", exist_ok=True)
json.dump(out, open(f"{ROOT}/results/ceiling/ceiling.json", "w"), indent=1)
for rep in ("pretrained|global|norm23", "pretrained|window|norm23", "pretrained|global|norm17", "random|window|norm23", "pixel", "trajectory"):
    o = out[rep]; print(rep, "r_full", o["r_full"])
    for p, x in o["cross_scene"].items(): print("   ", p, x)
    for p, x in o["within_scene"].items(): print("   ", p, x)
