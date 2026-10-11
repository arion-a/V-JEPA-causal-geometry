"""Dev-world statistics on per-world derivative vectors V [n_worlds, d]: split-half reliability and cross-context cosine."""
import itertools
import numpy as np


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def split_half_reliability(V, max_splits=2000, seed=0):
    """Mean over (floor(n/2), rest) splits of cos(mean of half A, mean of half B). 1 = every world agrees; ~0 = noise.
    Exhaustive when C(n, n//2) <= max_splits, otherwise `max_splits` random splits (fixed seed)."""
    from math import comb
    n = len(V); k = n // 2; out = []
    if comb(n, k) <= max_splits: splits = [list(A) for A in itertools.combinations(range(n), k)]
    else:
        rng = np.random.default_rng(seed); splits = [list(rng.permutation(n)[:k]) for _ in range(max_splits)]
    for A in splits:
        B = [i for i in range(n) if i not in A]
        out.append(float(_unit(V[A].mean(0)) @ _unit(V[B].mean(0))))
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


def bootstrap_mean_cos(VA, VB, n=2000, seed=0):
    """cos between world-mean derivatives of two groups; 95% CI by resampling worlds within each group (the world is the independent unit)."""
    rng = np.random.default_rng(seed); obs = mean_cos(VA, VB); xs = []
    for _ in range(n):
        xs.append(mean_cos(VA[rng.integers(0, len(VA), len(VA))], VB[rng.integers(0, len(VB), len(VB))]))
    return float(obs), float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))


def within_scene_ci(Vg, Vp, n=2000, seed=0):
    """mean per-world cos(g-derivative, other-parameter derivative) with a 95% bootstrap CI over worlds."""
    per = np.array([_unit(a) @ _unit(b) for a, b in zip(Vg, Vp)]); rng = np.random.default_rng(seed)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n)]
    return float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def shuffled_pairing_reliability(Zplus, Zminus, n=20, seed=0):
    """Null: pair world i's '+' clip with a DIFFERENT world's '-' clip (destroys the matched-pair structure), then split-half reliability. Should be ~0."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        perm = rng.permutation(len(Zplus))
        while (perm == np.arange(len(perm))).any(): perm = rng.permutation(len(Zplus))
        out.append(split_half_reliability(Zplus - Zminus[perm], max_splits=300)[0])
    return float(np.mean(out)), float(np.max(out))


def sign_flip_null_reliability(V, n=20, seed=0):
    """Negative control: flip the sign of each world's derivative at random (=shuffled intervention labels), then split-half reliability. Should be ~0 for any V."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        s = rng.choice([-1.0, 1.0], size=(len(V), 1)); out.append(split_half_reliability(V * s, max_splits=300)[0])
    return float(np.mean(out)), float(np.max(np.abs(out)))
