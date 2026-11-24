"""Statistics on per-world derivative arrays.

An array has shape ``(n_worlds, d)``: one row per independent world, ``d`` pooled feature (or
trajectory) dimensions, in feature units per intervention step. float32 and float64 inputs are
both supported and every computation runs in the dtype it is given.

Random draws come from a generator built inside each function from an explicit integer
``seed``, so a result never depends on how many draws other calls made before it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

type Derivatives = NDArray[np.float32] | NDArray[np.float64]

_NORM_FLOOR = 1e-12
_PROB_FLOOR = 1e-12
_CI_LOW, _CI_HIGH = 2.5, 97.5
# The null statistics below sample their inner split-half reliability with a fixed seed of 0,
# independent of their own seed argument; changing it changes published numbers.
_NULL_SPLIT_SEED = 0
_NULL_MAX_SPLITS = 300


class Reliability(NamedTuple):
    mean: float
    worst: float


class Interval(NamedTuple):
    estimate: float
    low: float
    high: float


class ShuffledPairing(NamedTuple):
    mean: float
    max: float


class SignFlipNull(NamedTuple):
    mean: float
    max_abs: float


@dataclass(frozen=True)
class AlignmentGap:
    g_g: float
    g_other: float
    gap: float


@dataclass(frozen=True)
class WithinScene:
    cos_mean: float
    cos_per_world: tuple[float, ...]
    eff_rank_mean: float


def _require_worlds(v: Derivatives, min_worlds: int) -> None:
    if v.ndim != 2 or len(v) < min_worlds:
        raise ValueError(f"need shape (n_worlds >= {min_worlds}, d), got {v.shape}")


def _require_same_shape(a: Derivatives, b: Derivatives) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")


def _unit(v: Derivatives) -> Derivatives:
    return v / max(np.linalg.norm(v), _NORM_FLOOR)


def _half_splits(n: int, max_splits: int, seed: int) -> list[list[int]]:
    k = n // 2
    if comb(n, k) <= max_splits:
        return [list(half) for half in itertools.combinations(range(n), k)]
    rng = np.random.default_rng(seed)
    return [list(rng.permutation(n)[:k]) for _ in range(max_splits)]


def _split_cos(v: Derivatives, half: list[int]) -> float:
    rest = [i for i in range(len(v)) if i not in half]
    return float(_unit(v[half].mean(0)) @ _unit(v[rest].mean(0)))


def split_half_reliability(v: Derivatives, *, max_splits: int = 2000, seed: int) -> Reliability:
    """Mean and minimum over splits of cos(mean of half A, mean of half B).

    1 means every world agrees, about 0 means noise. Exhaustive when C(n, n//2) <= max_splits,
    otherwise ``max_splits`` random splits drawn from ``seed``.
    """
    _require_worlds(v, 2)
    cos = [_split_cos(v, half) for half in _half_splits(len(v), max_splits, seed)]
    return Reliability(float(np.mean(cos)), float(np.min(cos)))


def _mean_cos(va: Derivatives, vb: Derivatives) -> float:
    return float(_unit(va.mean(0)) @ _unit(vb.mean(0)))


def mean_cos(va: Derivatives, vb: Derivatives) -> float:
    """cos between the world-mean derivatives of two groups of worlds."""
    _require_worlds(va, 1)
    _require_worlds(vb, 1)
    return _mean_cos(va, vb)


def alignment_with_control(
    vg_a: Derivatives, vg_b: Derivatives, v_other_b: Derivatives
) -> AlignmentGap:
    """cos(g in A, g in B) against cos(g in A, other parameter in B)."""
    g_g = mean_cos(vg_a, vg_b)
    g_other = mean_cos(vg_a, v_other_b)
    return AlignmentGap(g_g, g_other, g_g - g_other)


def _effective_rank(ua: Derivatives, ub: Derivatives) -> float:
    sv = np.linalg.svd(np.stack([ua, ub], 1), compute_uv=False)
    p = sv / sv.sum()
    p = p[p > _PROB_FLOOR]
    return float(np.exp(-(p * np.log(p)).sum()))


def _pair_cos_and_rank(a: Derivatives, b: Derivatives) -> tuple[float, float]:
    ua, ub = _unit(a), _unit(b)
    return float(ua @ ub), _effective_rank(ua, ub)


def within_scene(vg: Derivatives, vp: Derivatives) -> WithinScene:
    """Per-world cosine between two parameters' derivatives and the effective rank (1 to 2).

    Cosine near +-1 means one shared axis (rank near 1); near 0 means two directions (rank near 2).
    """
    _require_worlds(vg, 1)
    _require_same_shape(vg, vp)
    stats = [_pair_cos_and_rank(a, b) for a, b in zip(vg, vp, strict=True)]
    cos = [c for c, _ in stats]
    return WithinScene(
        float(np.mean(cos)),
        tuple(round(c, 3) for c in cos),
        float(np.mean([r for _, r in stats])),
    )


def _percentile_interval(estimate: float, samples: Derivatives) -> Interval:
    return Interval(
        estimate,
        float(np.percentile(samples, _CI_LOW)),
        float(np.percentile(samples, _CI_HIGH)),
    )


def _resampled_mean_cos(va: Derivatives, vb: Derivatives, rng: np.random.Generator) -> float:
    ia = rng.integers(0, len(va), len(va))
    ib = rng.integers(0, len(vb), len(vb))
    return _mean_cos(va[ia], vb[ib])


def bootstrap_mean_cos(
    va: Derivatives, vb: Derivatives, *, n_resamples: int = 2000, seed: int
) -> Interval:
    """cos between world-mean derivatives with a 95% percentile interval.

    Worlds are resampled with replacement independently within each group.
    """
    _require_worlds(va, 1)
    _require_worlds(vb, 1)
    rng = np.random.default_rng(seed)
    samples = [_resampled_mean_cos(va, vb, rng) for _ in range(n_resamples)]
    return _percentile_interval(mean_cos(va, vb), np.array(samples))


def within_scene_ci(
    vg: Derivatives, vp: Derivatives, *, n_resamples: int = 2000, seed: int
) -> Interval:
    """Mean per-world cos(g derivative, other derivative) with a 95% bootstrap interval."""
    _require_worlds(vg, 1)
    _require_same_shape(vg, vp)
    per_world = np.array([_unit(a) @ _unit(b) for a, b in zip(vg, vp, strict=True)])
    rng = np.random.default_rng(seed)
    means = [
        per_world[rng.integers(0, len(per_world), len(per_world))].mean()
        for _ in range(n_resamples)
    ]
    return _percentile_interval(float(per_world.mean()), np.array(means))


def _null_reliability(v: Derivatives) -> float:
    return split_half_reliability(v, max_splits=_NULL_MAX_SPLITS, seed=_NULL_SPLIT_SEED).mean


def _derangement(n: int, rng: np.random.Generator) -> NDArray[np.intp]:
    perm = rng.permutation(n)
    while (perm == np.arange(n)).any():
        perm = rng.permutation(n)
    return perm


def shuffled_pairing_reliability(
    z_plus: Derivatives, z_minus: Derivatives, *, n_repeats: int = 20, seed: int
) -> ShuffledPairing:
    """Pair each world's plus clip with a different world's minus clip, then split-half reliability.

    Removes the matched-pair structure; a plus/minus effect common to all worlds survives.
    """
    _require_worlds(z_plus, 2)
    _require_same_shape(z_plus, z_minus)
    rng = np.random.default_rng(seed)
    out = [
        _null_reliability(z_plus - z_minus[_derangement(len(z_plus), rng)])
        for _ in range(n_repeats)
    ]
    return ShuffledPairing(float(np.mean(out)), float(np.max(out)))


def sign_flip_null_reliability(v: Derivatives, *, n_repeats: int = 20, seed: int) -> SignFlipNull:
    """Flip each world's derivative sign at random (shuffled labels): expect about 0."""
    _require_worlds(v, 2)
    rng = np.random.default_rng(seed)
    out = [
        _null_reliability(v * rng.choice([-1.0, 1.0], size=(len(v), 1))) for _ in range(n_repeats)
    ]
    return SignFlipNull(float(np.mean(out)), float(np.max(np.abs(out))))
