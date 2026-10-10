"""Pipeline smoke test on the toy encoder. Output is NOT a scientific result."""
import argparse, json, numpy as np, torch
from cg import simulate as S, extract as E, analysis as A

ap = argparse.ArgumentParser(); ap.add_argument("--worlds", type=int, default=3); a = ap.parse_args()
enc = E.HookedEncoder(*(lambda m: (m, m.blocks))(E.ToyVideoEncoder()))
kinds, feats, valid = {}, {}, []
for kind in ("freefall", "bounce", "incline"):
    for s in range(a.worlds):
        w = S.make_world(kind, 100 + s); wid = f"{kind}{s}"; kinds[wid] = kind
        v = {"base": {}}; variants = {"base": {}, "cam": dict(nuis=dict(cam=.3)), "color": dict(nuis=dict(color=1 - w.color))}
        for p in S.observable(kind):
            for sg, d in (("+", 1), ("-", -1)): variants[f"{p}{sg}"] = dict(theta_over={p: S.BASE[p] + d * S.DELTA[p]})
        sec = [p for p in S.observable(kind) if p != "g"]
        if sec: variants["comp"] = dict(theta_over={"g": S.BASE["g"] + S.DELTA["g"], sec[0]: S.BASE[sec[0]] + S.DELTA[sec[0]]})
        clips = {k: S.clip(w, **kw) for k, kw in variants.items()}
        # validity gate: the intervened parameter must change the physical trajectory/event
        ok = all(np.abs(clips[f"{p}+"][1] - clips[f"{p}-"][1]).max() > 1e-3 for p in S.observable(kind)); valid.append(ok)
        x = torch.from_numpy(np.stack([c[0] for c in clips.values()]))
        L = enc(x)                                                     # [V, layers, D]
        feats[wid] = {k: [L[i, l] for l in range(L.shape[1])] for i, k in enumerate(clips)}
rep = {l: A.layer_report(feats, kinds, l) for l in range(4)}
print(json.dumps(dict(valid_worlds=f"{sum(valid)}/{len(valid)}", layers=rep), indent=1, default=float)[:3000])
