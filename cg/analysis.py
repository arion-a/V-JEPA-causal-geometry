"""Layerwise mechanism-geometry analysis from stored features (synthetic scaffold pipeline).

``feats[world][variant][layer]`` is a 1-D vector. Variants: base, g+, g-, e+, e-, mu+, mu-, comp
(both parameters raised together), cam, color.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cg import geometry as geo
from cg.arrays import Float
from cg.json_types import JsonValue
from cg.scenes import Param, Scene
from cg.simulate import DELTA, observable

type Feats = Mapping[str, Mapping[str, Sequence[Float]]]

_FLOOR = 1e-12
_BOOTSTRAP_SEED = 0
_NULL_SEED = 0
_NULL_PERCENTILE = 95


@dataclass(frozen=True)
class CrossContext:
    obs: float
    null_mean: float
    null_p95: float


@dataclass(frozen=True)
class LayerReport:
    cross_context_cos: Mapping[str, CrossContext]
    eff_rank: geo.MeanCI
    min_principal_angle_deg: geo.MeanCI | None
    composition_error: geo.MeanCI | None


def _normalise(v: Float) -> Float:
    unit: Float = v / max(np.linalg.norm(v), _FLOOR)
    return unit


def world_jacobian(
    feat: Mapping[str, Sequence[Float]], kind: Scene, layer: int, *, norm: bool = False
) -> dict[Param, Float]:
    """One column per observable parameter, scaled to one perturbation step."""
    cols: dict[Param, Float] = {}
    for p in observable(kind):
        plus, minus = feat[f"{p}+"][layer], feat[f"{p}-"][layer]
        if norm:
            plus, minus = _normalise(plus), _normalise(minus)
        cols[p] = geo.central_diff(plus, minus, DELTA.get(p)) * DELTA.get(p)
    return cols


def nuisance_cols(feat: Mapping[str, Sequence[Float]], layer: int) -> Float:
    base = feat["base"][layer]
    return np.stack([feat["cam"][layer] - base, feat["color"][layer] - base], 1)


def _g_columns(
    feats: Feats, kinds: Mapping[str, Scene], layer: int, norm: bool
) -> dict[Scene, list[Float]]:
    out: dict[Scene, list[Float]] = {}
    for w, k in kinds.items():
        out.setdefault(k, []).append(world_jacobian(feats[w], k, layer, norm=norm)[Param.G])
    return out


def _cross_context_pair(a: Float, b: Float) -> CrossContext:
    ma, mb = a.mean(0), b.mean(0)
    obs = float(ma @ mb / max(np.linalg.norm(ma) * np.linalg.norm(mb), _FLOOR))
    n = min(len(a), len(b))
    null = geo.shuffled_cosine_null(a[:n], b[:n], seed=_NULL_SEED)
    return CrossContext(obs, float(null.mean()), float(np.percentile(null, _NULL_PERCENTILE)))


def _cross_context(columns: Mapping[Scene, list[Float]]) -> dict[str, CrossContext]:
    scenes = list(columns)
    return {
        f"{a}~{b}": _cross_context_pair(np.array(columns[a]), np.array(columns[b]))
        for i, a in enumerate(scenes)
        for b in scenes[i + 1 :]
    }


def _composition_error(feat: Mapping[str, Sequence[Float]], kind: Scene, layer: int) -> float:
    second = next(p for p in observable(kind) if p is not Param.G)

    def dz(key: str) -> Float:
        return feat[key][layer] - feat["base"][layer]

    return geo.composition_error(dz("comp"), dz("g+"), dz(f"{second}+")).relative


def _world_statistics(
    feat: Mapping[str, Sequence[Float]], kind: Scene, layer: int, norm: bool
) -> tuple[float, float | None, float | None]:
    cols = world_jacobian(feat, kind, layer, norm=norm)
    j = np.stack(list(cols.values()), 1)
    angle = None
    if "cam" in feat:
        angle = float(np.degrees(geo.principal_angles(j, nuisance_cols(feat, layer))).min())
    comp = _composition_error(feat, kind, layer) if "comp" in feat and len(cols) >= 2 else None
    return geo.effective_rank(j).value, angle, comp


def _summarise(values: Sequence[float | None]) -> geo.MeanCI | None:
    present = [v for v in values if v is not None]
    return geo.bootstrap_worlds(present, seed=_BOOTSTRAP_SEED) if present else None


def layer_report(
    feats: Feats, kinds: Mapping[str, Scene], layer: int, *, norm: bool = False
) -> LayerReport:
    """Cross-context gravity alignment, rank, mechanism-nuisance angle and composition error."""
    stats = [_world_statistics(feats[w], k, layer, norm) for w, k in kinds.items()]
    rank = geo.bootstrap_worlds([s[0] for s in stats], seed=_BOOTSTRAP_SEED)
    return LayerReport(
        _cross_context(_g_columns(feats, kinds, layer, norm)),
        rank,
        _summarise([s[1] for s in stats]),
        _summarise([s[2] for s in stats]),
    )


def _ci_json(ci: geo.MeanCI) -> JsonValue:
    return [ci.mean, [ci.low, ci.high]]


def to_json(report: LayerReport) -> dict[str, JsonValue]:
    """The layout printed by scripts/run_smoke.py; statistics a layer lacks are left out."""
    out: dict[str, JsonValue] = {
        "cross_context_cos": {
            pair: {"obs": c.obs, "null_mean": c.null_mean, "null_p95": c.null_p95}
            for pair, c in report.cross_context_cos.items()
        },
        "eff_rank": _ci_json(report.eff_rank),
    }
    if report.min_principal_angle_deg is not None:
        out["min_principal_angle_deg"] = _ci_json(report.min_principal_angle_deg)
    if report.composition_error is not None:
        out["composition_error"] = _ci_json(report.composition_error)
    return out
