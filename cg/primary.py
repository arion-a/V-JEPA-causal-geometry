"""Confirmatory primary endpoint (configs/pilot.json, block "confirmatory").

Input per representation is one float64 array of shape ``(n_worlds, d)`` for each
(parameter, scene) case: the plus-minus feature (or trajectory) derivative of every world.

G(rep) is the mean over the three scene pairs of cos(world-mean g in A, world-mean g in B).
W(rep) is the same with scene B's g replaced by that scene's other parameter.
Every bootstrap draw resamples worlds within each scene once and applies the same indices
to every representation, so differences between representations are paired.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from cg.json_types import JsonValue
from cg.scenes import Param, Scene

type Array = NDArray[np.float64]


class Pooling(StrEnum):
    GLOBAL = "global"
    WINDOW = "window"


class Verdict(StrEnum):
    ABOVE = "V-JEPA above trajectory baseline"
    BELOW = "trajectory baseline stronger"
    UNDISTINGUISHED = "not distinguished from kinematics"


class Claim(StrEnum):
    MECHANISM_LIKE = "mechanism-like geometry (both poolings pass, controls pass)"
    CONTROL_FAILS = "primary passes for both poolings but a control fails: not supported"
    POOLING_DEPENDENT = "pooling-dependent, not supported"
    BELOW = "trajectory baseline stronger"
    UNDISTINGUISHED = "not distinguished from kinematics"


type Case = tuple[Param, Scene]
type RepDerivatives = Mapping[Case, Array]

SCENES = (Scene.FREEFALL, Scene.BOUNCE, Scene.INCLINE)
PAIRS = (
    (Scene.FREEFALL, Scene.BOUNCE),
    (Scene.FREEFALL, Scene.INCLINE),
    (Scene.BOUNCE, Scene.INCLINE),
)
WRONG_PARAM = {Scene.BOUNCE: Param.E, Scene.INCLINE: Param.MU}
REQUIRED_CASES: tuple[Case, ...] = (
    (Param.G, Scene.FREEFALL),
    (Param.G, Scene.BOUNCE),
    (Param.G, Scene.INCLINE),
    (Param.E, Scene.BOUNCE),
    (Param.MU, Scene.INCLINE),
)
# 97.5% two-sided interval per endpoint: two co-primary endpoints, Bonferroni split of 5%.
LEVEL_LOW, LEVEL_HIGH = 1.25, 98.75
LEVEL_LABEL = "97.5% two-sided paired world bootstrap"
_NORM_FLOOR = 1e-12


@dataclass(frozen=True)
class Endpoint:
    """Representations compared for one pooling choice."""

    pooling: Pooling
    vjepa: str
    trajectory: str
    pixel: str
    random: str


@dataclass(frozen=True)
class Bounds:
    low: float
    high: float


@dataclass(frozen=True)
class PairCosines:
    freefall_bounce: float
    freefall_incline: float
    bounce_incline: float


@dataclass(frozen=True)
class RepSummary:
    g: float
    g_wrong: float
    pairs: PairCosines


@dataclass(frozen=True)
class EndpointResult:
    vjepa_g: float
    trajectory_g: float
    diff_vs_trajectory: float
    diff_vs_trajectory_interval: Bounds
    gravity_minus_wrong: float
    gravity_minus_wrong_interval: Bounds
    pixel_minus_vjepa: float
    pixel_minus_vjepa_interval: Bounds
    random_minus_vjepa: float
    random_minus_vjepa_interval: Bounds
    primary_success: bool
    wrong_param_ok: bool
    baselines_ok: bool
    verdict_this_pooling: Verdict


@dataclass(frozen=True)
class PrimaryResult:
    n_worlds: Mapping[Scene, int]
    per_rep: Mapping[str, RepSummary]
    endpoints: Mapping[Pooling, EndpointResult]
    claim: Claim


@dataclass(frozen=True)
class _Samples:
    gravity: Array
    wrong: Array


def _unit(v: Array) -> Array:
    return v / max(np.linalg.norm(v), _NORM_FLOOR)


def _pair_cosines(
    rep: RepDerivatives, idx: Mapping[Scene, NDArray[np.intp]], *, wrong: bool
) -> list[float]:
    out = []
    for a, b in PAIRS:
        param_b = WRONG_PARAM[b] if wrong else Param.G
        va = _unit(rep[(Param.G, a)][idx[a]].mean(0))
        vb = _unit(rep[(param_b, b)][idx[b]].mean(0))
        out.append(float(va @ vb))
    return out


def _statistic(rep: RepDerivatives, idx: Mapping[Scene, NDArray[np.intp]], *, wrong: bool) -> float:
    return float(np.mean(_pair_cosines(rep, idx, wrong=wrong)))


def _validate(
    reps: Mapping[str, RepDerivatives], endpoints: Sequence[Endpoint], n_boot: int
) -> dict[Scene, int]:
    if not reps or not endpoints or n_boot < 1:
        raise ValueError("need at least one representation, one endpoint and one resample")
    for name, rep in reps.items():
        if any(case not in rep for case in REQUIRED_CASES):
            raise ValueError(f"{name!r} is missing one of {REQUIRED_CASES}")
    used = {n for e in endpoints for n in (e.vjepa, e.trajectory, e.pixel, e.random)}
    if not used <= reps.keys():
        raise ValueError(f"endpoints refer to unknown representations {sorted(used - reps.keys())}")
    sizes = {s: len(next(iter(reps.values()))[(Param.G, s)]) for s in SCENES}
    if any(len(rep[case]) != sizes[case[1]] for rep in reps.values() for case in REQUIRED_CASES):
        raise ValueError("every representation needs the same number of worlds per scene")
    return sizes


def _draw_indices(
    sizes: Mapping[Scene, int], rng: np.random.Generator
) -> dict[Scene, NDArray[np.intp]]:
    return {s: rng.integers(0, sizes[s], sizes[s]) for s in SCENES}


def _bootstrap(
    reps: Mapping[str, RepDerivatives], sizes: Mapping[Scene, int], n_boot: int, seed: int
) -> dict[str, _Samples]:
    rng = np.random.default_rng(seed)
    gravity: dict[str, list[float]] = {name: [] for name in reps}
    wrong: dict[str, list[float]] = {name: [] for name in reps}
    for _ in range(n_boot):
        idx = _draw_indices(sizes, rng)
        for name, rep in reps.items():
            gravity[name].append(_statistic(rep, idx, wrong=False))
            wrong[name].append(_statistic(rep, idx, wrong=True))
    return {name: _Samples(np.array(gravity[name]), np.array(wrong[name])) for name in reps}


def _percentiles(x: Array) -> Bounds:
    return Bounds(float(np.percentile(x, LEVEL_LOW)), float(np.percentile(x, LEVEL_HIGH)))


def _rounded(b: Bounds) -> Bounds:
    return Bounds(round(b.low, 3), round(b.high, 3))


def _verdict(diff: Bounds) -> Verdict:
    if diff.low > 0:
        return Verdict.ABOVE
    if diff.high < 0:
        return Verdict.BELOW
    return Verdict.UNDISTINGUISHED


@dataclass(frozen=True)
class _Differences:
    trajectory: Bounds
    wrong: Bounds
    pixel: Bounds
    random: Bounds


def _differences(e: Endpoint, boot: Mapping[str, _Samples]) -> _Differences:
    vj = boot[e.vjepa]
    return _Differences(
        _percentiles(vj.gravity - boot[e.trajectory].gravity),
        _percentiles(vj.gravity - vj.wrong),
        _percentiles(boot[e.pixel].gravity - vj.gravity),
        _percentiles(boot[e.random].gravity - vj.gravity),
    )


def _endpoint_result(
    e: Endpoint,
    obs: Mapping[str, float],
    obs_wrong: Mapping[str, float],
    boot: Mapping[str, _Samples],
) -> EndpointResult:
    vj, tr, px, rw = obs[e.vjepa], obs[e.trajectory], obs[e.pixel], obs[e.random]
    d = _differences(e, boot)
    return EndpointResult(
        vjepa_g=round(vj, 3),
        trajectory_g=round(tr, 3),
        diff_vs_trajectory=round(vj - tr, 3),
        diff_vs_trajectory_interval=_rounded(d.trajectory),
        gravity_minus_wrong=round(vj - obs_wrong[e.vjepa], 3),
        gravity_minus_wrong_interval=_rounded(d.wrong),
        pixel_minus_vjepa=round(px - vj, 3),
        pixel_minus_vjepa_interval=_rounded(d.pixel),
        random_minus_vjepa=round(rw - vj, 3),
        random_minus_vjepa_interval=_rounded(d.random),
        primary_success=d.trajectory.low > 0,
        wrong_param_ok=d.wrong.low > 0,
        baselines_ok=d.pixel.low <= 0 and d.random.low <= 0,
        verdict_this_pooling=_verdict(d.trajectory),
    )


def _claim(results: Sequence[EndpointResult]) -> Claim:
    successes = [r.primary_success for r in results]
    controls_ok = all(r.wrong_param_ok and r.baselines_ok for r in results)
    if all(successes):
        return Claim.MECHANISM_LIKE if controls_ok else Claim.CONTROL_FAILS
    if any(successes):
        return Claim.POOLING_DEPENDENT
    if all(r.verdict_this_pooling is Verdict.BELOW for r in results):
        return Claim.BELOW
    return Claim.UNDISTINGUISHED


def _summaries(
    reps: Mapping[str, RepDerivatives], sizes: Mapping[Scene, int]
) -> dict[str, RepSummary]:
    full = {s: np.arange(sizes[s]) for s in SCENES}
    out = {}
    for name, rep in reps.items():
        pairs = [round(c, 3) for c in _pair_cosines(rep, full, wrong=False)]
        out[name] = RepSummary(
            round(_statistic(rep, full, wrong=False), 3),
            round(_statistic(rep, full, wrong=True), 3),
            PairCosines(*pairs),
        )
    return out


def run(
    reps: Mapping[str, RepDerivatives],
    endpoints: Sequence[Endpoint],
    *,
    n_boot: int = 2000,
    seed: int,
) -> PrimaryResult:
    """Observed statistics, paired-bootstrap intervals and the claim for every endpoint."""
    sizes = _validate(reps, endpoints, n_boot)
    full = {s: np.arange(sizes[s]) for s in SCENES}
    obs = {name: _statistic(rep, full, wrong=False) for name, rep in reps.items()}
    obs_wrong = {name: _statistic(rep, full, wrong=True) for name, rep in reps.items()}
    boot = _bootstrap(reps, sizes, n_boot, seed)
    results = {e.pooling: _endpoint_result(e, obs, obs_wrong, boot) for e in endpoints}
    return PrimaryResult(sizes, _summaries(reps, sizes), results, _claim(list(results.values())))


def _bounds_json(b: Bounds) -> JsonValue:
    return [b.low, b.high]


def _endpoint_json(r: EndpointResult) -> JsonValue:
    return {
        "vjepa_G": r.vjepa_g,
        "trajectory_G": r.trajectory_g,
        "diff_vs_trajectory": r.diff_vs_trajectory,
        "diff_vs_trajectory_interval": _bounds_json(r.diff_vs_trajectory_interval),
        "gravity_minus_wrong": r.gravity_minus_wrong,
        "gravity_minus_wrong_interval": _bounds_json(r.gravity_minus_wrong_interval),
        "pixel_minus_vjepa": r.pixel_minus_vjepa,
        "pixel_minus_vjepa_interval": _bounds_json(r.pixel_minus_vjepa_interval),
        "random_minus_vjepa": r.random_minus_vjepa,
        "random_minus_vjepa_interval": _bounds_json(r.random_minus_vjepa_interval),
        "primary_success": r.primary_success,
        "wrong_param_ok": r.wrong_param_ok,
        "baselines_ok": r.baselines_ok,
        "verdict_this_pooling": r.verdict_this_pooling.value,
    }


def _summary_json(s: RepSummary) -> JsonValue:
    return {
        "G": s.g,
        "G_wrong": s.g_wrong,
        "pairs": {
            "freefall~bounce": s.pairs.freefall_bounce,
            "freefall~incline": s.pairs.freefall_incline,
            "bounce~incline": s.pairs.bounce_incline,
        },
    }


def to_json(result: PrimaryResult) -> dict[str, JsonValue]:
    """The result in the layout of results/primary_*/primary.json (same keys, same order)."""
    return {
        "n_worlds": {s.value: n for s, n in result.n_worlds.items()},
        "level": LEVEL_LABEL,
        "per_rep": {name: _summary_json(s) for name, s in result.per_rep.items()},
        "endpoints": {p.value: _endpoint_json(r) for p, r in result.endpoints.items()},
        "claim": result.claim.value,
    }
