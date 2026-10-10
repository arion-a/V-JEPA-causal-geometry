"""Dev-world statistics on per-world derivative vectors V [n_worlds, d]: split-half reliability and cross-context cosine."""
import itertools
import numpy as np


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def split_half_reliability(V):
    """Mean over all (floor(n/2), rest) splits of cos(mean of half A, mean of half B). 1 = every world agrees; ~0 = noise."""
    n = len(V); idx = range(n); k = n // 2; out = []
    for A in itertools.combinations(idx, k):
        B = [i for i in idx if i not in A]
        out.append(float(_unit(V[list(A)].mean(0)) @ _unit(V[B].mean(0))))
    return float(np.mean(out)), float(np.min(out))


def mean_cos(VA, VB):
    return float(_unit(VA.mean(0)) @ _unit(VB.mean(0)))


def alignment_with_control(VgA, VgB, VotherB):
    """cos(g in A, g in B) vs cos(g in A, other parameter in B): gravity-specific alignment gap."""
    return dict(g_g=mean_cos(VgA, VgB), g_other=mean_cos(VgA, VotherB), gap=mean_cos(VgA, VgB) - mean_cos(VgA, VotherB))


def within_scene(Vg, Vp):
    """Per-world cosine between the g-derivative and the second parameter's derivative, and the effective rank (1..2) of the unit-column pair.
    cos near +-1 -> one shared axis (rank ~1, 'amount of motion change'); cos near 0 -> two distinguishable directions (rank ~2)."""
    cos, er = [], []
    for a, b in zip(Vg, Vp):
        ua, ub = _unit(a), _unit(b); c = float(ua @ ub); cos.append(c)
        sv = np.linalg.svd(np.stack([ua, ub], 1), compute_uv=False); p = sv / sv.sum(); p = p[p > 1e-12]
        er.append(float(np.exp(-(p * np.log(p)).sum())))
    return dict(cos_mean=float(np.mean(cos)), cos_per_world=[round(c, 3) for c in cos], eff_rank_mean=float(np.mean(er)))
