"""Geometry of mechanism interventions: Jacobian columns, rank, induced metric, principal angles, composition.

All functions take per-world arrays so the *base world* stays the resampling unit (protocol sec. 2, 6).
Features are [n_dim] vectors for a single (world, layer); stack worlds on a leading axis where noted.
"""
import numpy as np

EPS = 1e-12


def central_diff(z_plus, z_minus, delta):
    """v_j = [z(theta+delta e_j) - z(theta-delta e_j)] / (2 delta). delta is in *parameter units*."""
    return (np.asarray(z_plus) - np.asarray(z_minus)) / (2.0 * delta)


def standardized_columns(cols, deltas, scales):
    """Express each column per one 'scale' of its parameter (e.g. 1 sd of the prior) so entries of G compare."""
    return np.stack([c * s for c, s in zip(cols, scales)], 1)  # [n_dim, n_params]


def effective_rank(J):
    s = np.linalg.svd(J, compute_uv=False)
    p = s / max(s.sum(), EPS)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())), s


def induced_metric(J):
    return J.T @ J


def cosine_matrix(J):
    n = np.linalg.norm(J, axis=0, keepdims=True)
    Jn = J / np.maximum(n, EPS)
    return Jn.T @ Jn


def principal_angles(A, B):
    """Principal angles (radians) between column spaces of A [d,p] and B [d,q]."""
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.arccos(np.clip(s, -1, 1))


def composition_error(dz_both, dz_a, dz_b, eps=1e-8):
    """E = ||dz_both - (dz_a + dz_b)||^2 / max(||dz_both||^2, eps); also the absolute residual."""
    r = float(np.sum((dz_both - (dz_a + dz_b)) ** 2))
    return r / max(float(np.sum(dz_both ** 2)), eps), r


def metric_distance(G1, G2):
    """Scale-free distance between induced metrics: ||G1/tr G1 - G2/tr G2||_F."""
    return float(np.linalg.norm(G1 / max(np.trace(G1), EPS) - G2 / max(np.trace(G2), EPS)))


def bootstrap_worlds(stat_per_world, n_boot=2000, seed=0):
    """Percentile 95% CI of the mean over base worlds (the independent unit)."""
    x = np.asarray(stat_per_world, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def shuffled_cosine_null(v_a, v_b, n=2000, seed=0):
    """Null for cross-context alignment: cosine between v_a[i] and v_b[perm(i)] (pairs broken across worlds)."""
    rng = np.random.default_rng(seed)
    nrm = lambda v: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), EPS)
    a, b = nrm(v_a), nrm(v_b)
    return np.array([(a * b[rng.permutation(len(b))]).sum(1).mean() for _ in range(n)])
