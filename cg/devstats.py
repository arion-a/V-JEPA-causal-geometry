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
