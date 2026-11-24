"""Geometry of mechanism interventions: differences, rank, induced metric, angles, composition.

Features are 1-D vectors for one (world, layer); stack worlds on a leading axis where noted.
float32 and float64 inputs are both supported and computed in the dtype given. Random draws come
from a generator built inside each function from an explicit integer ``seed``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from cg.arrays import Float

EPS = 1e-12
_CI_LOW, _CI_HIGH = 2.5, 97.5


class EffectiveRank(NamedTuple):
    value: float
    singular_values: Float


class CompositionError(NamedTuple):
    relative: float
    absolute: float


class MeanCI(NamedTuple):
    mean: float
    low: float
    high: float


def central_diff(z_plus: Float, z_minus: Float, delta: float) -> Float:
    """[z(theta + delta e_j) - z(theta - delta e_j)] / (2 delta); delta in parameter units."""
    return (z_plus - z_minus) / (2.0 * delta)


def effective_rank(j: Float) -> EffectiveRank:
    """exp of the entropy of the normalised singular values (1 for one direction)."""
    s = np.linalg.svd(j, compute_uv=False)
    p = s / max(s.sum(), EPS)
    p = p[p > 0]
    return EffectiveRank(float(np.exp(-(p * np.log(p)).sum())), s)


def induced_metric(j: Float) -> Float:
    return j.T @ j


def principal_angles(a: Float, b: Float) -> Float:
    """Principal angles (radians) between the column spaces of a [d, p] and b [d, q]."""
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.arccos(np.clip(s, -1, 1))


def composition_error(
    dz_both: Float, dz_a: Float, dz_b: Float, eps: float = 1e-8
) -> CompositionError:
    """||dz_both - (dz_a + dz_b)||^2 / max(||dz_both||^2, eps), and the absolute residual."""
    residual = float(np.sum((dz_both - (dz_a + dz_b)) ** 2))
    return CompositionError(residual / max(float(np.sum(dz_both**2)), eps), residual)


def metric_distance(g1: Float, g2: Float) -> float:
    """Scale-free distance between induced metrics: ||G1 / tr G1 - G2 / tr G2||_F."""
    return float(np.linalg.norm(g1 / max(np.trace(g1), EPS) - g2 / max(np.trace(g2), EPS)))


def bootstrap_worlds(
    stat_per_world: Float | list[float], *, n_boot: int = 2000, seed: int
) -> MeanCI:
    """Mean over base worlds (the independent unit) with a 95% percentile bootstrap interval."""
    x = np.asarray(stat_per_world, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return MeanCI(
        float(x.mean()), float(np.percentile(means, _CI_LOW)), float(np.percentile(means, _CI_HIGH))
    )


def _unit_rows(v: Float) -> Float:
    unit: Float = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), EPS)
    return unit


def shuffled_cosine_null(v_a: Float, v_b: Float, *, n_permutations: int = 2000, seed: int) -> Float:
    """Cosine between v_a[i] and v_b[perm(i)] with the pairing broken, per permutation."""
    if v_a.shape != v_b.shape:
        raise ValueError(f"need equal shapes, got {v_a.shape} and {v_b.shape}")
    rng = np.random.default_rng(seed)
    a, b = _unit_rows(v_a), _unit_rows(v_b)
    return np.array([(a * b[rng.permutation(len(b))]).sum(1).mean() for _ in range(n_permutations)])
