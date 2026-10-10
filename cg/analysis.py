"""Layerwise mechanism-geometry analysis from stored features.

features[world_id][variant][layer] -> 1-D vector. Variants: 'base','g+','g-','e+','e-','mu+','mu-',
'both_g+e+' style composition keys 'comp', 'cam', 'color'.
"""
import numpy as np
from . import geometry as G
from .simulate import DELTA, observable


def world_jacobian(feat, kind, layer, norm=False):
    f = lambda v: (v / max(np.linalg.norm(v), 1e-12) if norm else v)
    cols = {p: G.central_diff(f(feat[f"{p}+"][layer]), f(feat[f"{p}-"][layer]), DELTA[p]) * DELTA[p] for p in observable(kind)}
    return cols            # columns scaled to "one perturbation step" (standardised)


def nuisance_cols(feat, layer):
    return np.stack([feat["cam"][layer] - feat["base"][layer], feat["color"][layer] - feat["base"][layer]], 1)


def layer_report(feats, kinds, layer, norm=False):
    """feats: dict world->variants. Returns primary statistics for a layer (per-world arrays then CI)."""
    out, vg = {}, {}
    for w, k in kinds.items():
        vg.setdefault(k, []).append(world_jacobian(feats[w], k, layer, norm)["g"])
    ks = list(vg)
    # cross-context gravity alignment: mean cosine between world means of different scenarios vs shuffled null
    out["cross_context_cos"] = {}
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = np.array(vg[ks[i]]), np.array(vg[ks[j]])
            ma, mb = a.mean(0), b.mean(0)
            obs = float(ma @ mb / max(np.linalg.norm(ma) * np.linalg.norm(mb), 1e-12))
            null = G.shuffled_cosine_null(a, b[:len(a)] if len(b) >= len(a) else b)
            out["cross_context_cos"][f"{ks[i]}~{ks[j]}"] = dict(obs=obs, null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)))
    # per-world mechanism rank, mechanism-vs-nuisance angle, composition error
    rk, ang, comp = [], [], []
    for w, k in kinds.items():
        cols = world_jacobian(feats[w], k, layer, norm)
        J = np.stack(list(cols.values()), 1)
        rk.append(G.effective_rank(J)[0])
        N = nuisance_cols(feats[w], layer) if "cam" in feats[w] else None
        if N is not None: ang.append(np.degrees(G.principal_angles(J, N)).min())
        if "comp" in feats[w] and len(cols) >= 2:
            dz = lambda key: feats[w][key][layer] - feats[w]["base"][layer]
            comp.append(G.composition_error(dz("comp"), dz("g+"), dz(f"{[p for p in observable(k) if p != 'g'][0]}+"))[0])
    out["eff_rank"] = G.bootstrap_worlds(rk)
    if ang: out["min_principal_angle_deg"] = G.bootstrap_worlds(ang)
    if comp: out["composition_error"] = G.bootstrap_worlds(comp)
    return out
