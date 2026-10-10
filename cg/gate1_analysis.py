"""Gate 1 statistics from per-clip pooled features. Descriptive only (n=1 world per family): not evidence for any hypothesis."""
import numpy as np
from . import geometry as G

RAW = [f"raw{i}" for i in range(24)]; NORM = [f"norm{i}" for i in (5, 11, 17, 23)]


def taps(f):
    """feature dict -> {tap_name: vector}"""
    d = {f"raw{i}": f["raw"][i] for i in range(f["raw"].shape[0])}
    d.update({f"norm{i}": f["normed"][k] for k, i in enumerate((5, 11, 17, 23))})
    return d


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def analyse(F, delta, jitter, half, comp, jitter2=None):
    jitter2 = jitter2 or {}
    """F: clip_id -> taps dict. delta: {g,e,mu}. jitter: {world: clip_id of base+tiny camera shift}.
    half: {(world,'g'): (clip_plus_half, clip_minus_half, delta_half)}. comp: world -> dict ids base,g+,second+,both+."""
    out = {}
    for tap in RAW + NORM:
        row = {}
        for w in {c.split("__")[0] for c in F}:
            ids = {c.split("__")[1]: c for c in F if c.startswith(w + "__")}
            if "base" not in ids: continue
            z0 = F[ids["base"]][tap]; r = {}
            for par in ("g", "e", "mu"):
                if f"{par}+" in ids and f"{par}-" in ids:
                    r[f"signal_{par}"] = rel(F[ids[f"{par}+"]][tap], F[ids[f"{par}-"]][tap])
            if w in jitter: r["jitter_floor"] = rel(F[jitter[w]][tap], z0)
            if "cam" in ids: r["cam_shift_rel"] = rel(F[ids["cam"]][tap], z0)
            if "color" in ids: r["color_shift_rel"] = rel(F[ids["color"]][tap], z0)
            if "jitter_floor" in r:
                for par in ("g", "e", "mu"):
                    if f"signal_{par}" in r: r[f"snr_{par}"] = r[f"signal_{par}"] / max(r["jitter_floor"], 1e-12)
            if w in jitter2: r["jitter2_rel"] = rel(F[jitter2[w]][tap], z0)
            if (w, "g") in half:
                p, m, dh = half[(w, "g")]
                v_full = G.central_diff(F[ids["g+"]][tap], F[ids["g-"]][tap], delta["g"]); v_half = G.central_diff(F[p][tap], F[m][tap], dh)
                r["linearity_cos_g"] = float(v_full @ v_half / max(np.linalg.norm(v_full) * np.linalg.norm(v_half), 1e-12))
                r["linearity_norm_ratio_g"] = float(np.linalg.norm(v_half) / max(np.linalg.norm(v_full), 1e-12))
            if w in comp:
                c = comp[w]; dz = lambda k: F[c[k]][tap] - F[c["base"]][tap]
                r["composition_error"] = G.composition_error(dz("both+"), dz("g+"), dz("second+"))[0]
            row[w] = r
        out[tap] = row
    return out
