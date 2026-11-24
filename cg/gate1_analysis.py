"""Gate 1 statistics from per-clip pooled features (first 20-clip smoke run).

Descriptive only (one world per family): not evidence for any hypothesis. A clip id is
"<world>__<variant>"; variants are base, g+/-, e+/-, mu+/-, cam, color, jitter and half-step clips.
Features are float32 or float64 vectors and are computed in the dtype given.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from cg import geometry as geo
from cg.arrays import Float

if TYPE_CHECKING:
    from cg.features import Features

type Taps = Mapping[str, Float]

NORM_BLOCKS = (5, 11, 17, 23)
RAW = tuple(f"raw{i}" for i in range(24))
NORM = tuple(f"norm{i}" for i in NORM_BLOCKS)
PARAMS = ("g", "e", "mu")
_FLOOR = 1e-12


@dataclass(frozen=True)
class HalfStep:
    plus: str
    minus: str
    delta: float


@dataclass(frozen=True)
class CompositionClips:
    base: str
    g_plus: str
    second_plus: str
    both_plus: str


@dataclass(frozen=True)
class Plan:
    """Which extra clips exist for which world, keyed by world id."""

    delta_g: float
    jitter: Mapping[str, str]
    jitter2: Mapping[str, str]
    half: Mapping[str, HalfStep]
    composition: Mapping[str, CompositionClips]


@dataclass(frozen=True)
class WorldTapStats:
    signal_g: float | None = None
    signal_e: float | None = None
    signal_mu: float | None = None
    jitter_floor: float | None = None
    cam_shift_rel: float | None = None
    color_shift_rel: float | None = None
    snr_g: float | None = None
    snr_e: float | None = None
    snr_mu: float | None = None
    jitter2_rel: float | None = None
    linearity_cos_g: float | None = None
    linearity_norm_ratio_g: float | None = None
    composition_error: float | None = None


type Analysis = Mapping[str, Mapping[str, WorldTapStats]]


def taps(f: Features) -> dict[str, Float]:
    """Tap name -> vector: every raw block output and the four post-norm hierarchical taps."""
    out = {f"raw{i}": f["raw"][i] for i in range(f["raw"].shape[0])}
    out.update({f"norm{b}": f["normed"][k] for k, b in enumerate(NORM_BLOCKS)})
    return out


def rel(a: Float, b: Float) -> float:
    """||a - b|| / ||b||."""
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), _FLOOR))


def _world(clip_id: str) -> str:
    return clip_id.split("__")[0]


def _variants_by_world(clips: Mapping[str, Taps]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for clip_id in clips:
        out.setdefault(_world(clip_id), {})[clip_id.split("__")[1]] = clip_id
    return out


def _linearity(
    clips: Mapping[str, Taps], ids: Mapping[str, str], half: HalfStep, delta_g: float, tap: str
) -> tuple[float, float]:
    v_full = geo.central_diff(clips[ids["g+"]][tap], clips[ids["g-"]][tap], delta_g)
    v_half = geo.central_diff(clips[half.plus][tap], clips[half.minus][tap], half.delta)
    norms = np.linalg.norm(v_full) * np.linalg.norm(v_half)
    cos = float(v_full @ v_half / max(norms, _FLOOR))
    return cos, float(np.linalg.norm(v_half) / max(np.linalg.norm(v_full), _FLOOR))


def _composition(clips: Mapping[str, Taps], c: CompositionClips, tap: str) -> float:
    def dz(clip_id: str) -> Float:
        return clips[clip_id][tap] - clips[c.base][tap]

    return geo.composition_error(dz(c.both_plus), dz(c.g_plus), dz(c.second_plus)).relative


def _signals(clips: Mapping[str, Taps], ids: Mapping[str, str], tap: str) -> dict[str, float]:
    return {
        f"signal_{p}": rel(clips[ids[f"{p}+"]][tap], clips[ids[f"{p}-"]][tap])
        for p in PARAMS
        if f"{p}+" in ids and f"{p}-" in ids
    }


def _optional_rel(
    clips: Mapping[str, Taps], clip_id: str | None, z0: Float, tap: str
) -> float | None:
    return None if clip_id is None else rel(clips[clip_id][tap], z0)


def _world_tap_stats(
    clips: Mapping[str, Taps], ids: Mapping[str, str], world: str, plan: Plan, tap: str
) -> WorldTapStats:
    z0 = clips[ids["base"]][tap]
    values: dict[str, float | None] = dict(_signals(clips, ids, tap))
    values["jitter_floor"] = _optional_rel(clips, plan.jitter.get(world), z0, tap)
    values["cam_shift_rel"] = _optional_rel(clips, ids.get("cam"), z0, tap)
    values["color_shift_rel"] = _optional_rel(clips, ids.get("color"), z0, tap)
    floor = values["jitter_floor"]
    for p in PARAMS:
        signal = values.get(f"signal_{p}")
        if floor is not None and signal is not None:
            values[f"snr_{p}"] = signal / max(floor, _FLOOR)
    values["jitter2_rel"] = _optional_rel(clips, plan.jitter2.get(world), z0, tap)
    if world in plan.half:
        cos, ratio = _linearity(clips, ids, plan.half[world], plan.delta_g, tap)
        values["linearity_cos_g"], values["linearity_norm_ratio_g"] = cos, ratio
    if world in plan.composition:
        values["composition_error"] = _composition(clips, plan.composition[world], tap)
    return WorldTapStats(**values)


def _validate(
    clips: Mapping[str, Taps], by_world: Mapping[str, Mapping[str, str]], plan: Plan
) -> None:
    named = [*plan.jitter.values(), *plan.jitter2.values()]
    named += [c for h in plan.half.values() for c in (h.plus, h.minus)]
    named += [
        c for k in plan.composition.values() for c in (k.base, k.g_plus, k.second_plus, k.both_plus)
    ]
    if missing := [c for c in named if c not in clips]:
        raise ValueError(f"plan refers to clips that were not extracted: {missing}")
    if bad := [w for w in plan.half if not {"g+", "g-"} <= by_world.get(w, {}).keys()]:
        raise ValueError(f"half-step linearity needs g+ and g- clips for worlds {bad}")


def analyse(clips: Mapping[str, Taps], plan: Plan) -> Analysis:
    """tap -> world -> statistics, for worlds that have a base clip."""
    by_world = _variants_by_world(clips)
    _validate(clips, by_world, plan)
    worlds = sorted(w for w, ids in by_world.items() if "base" in ids)
    return {
        tap: {w: _world_tap_stats(clips, by_world[w], w, plan, tap) for w in worlds}
        for tap in RAW + NORM
    }


def to_json(analysis: Analysis) -> dict[str, dict[str, dict[str, float]]]:
    """The layout written to gate1.json: only the statistics a world actually has."""
    return {
        tap: {
            world: {f.name: v for f in fields(stats) if (v := getattr(stats, f.name)) is not None}
            for world, stats in worlds.items()
        }
        for tap, worlds in analysis.items()
    }
