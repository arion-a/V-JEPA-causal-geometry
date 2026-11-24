"""Deterministic numpy scenarios with exact event handling and a numpy renderer.

A pipeline-validation backend with no engine dependency; the protocol runs use cg.sim_pb. Scenarios:
freefall (g), bounce (g, e; needs an impact), incline (g, mu; needs sliding). Positions are (x, z)
in metres, time in seconds, frames float32 in [0, 1] with shape (T, 3, res, res).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from cg.scenes import OBSERVABLE, Param, Scene

type Array = NDArray[np.float64]

T, FPS = 16, 16.0
_SUBSTEPS = 20


@dataclass(frozen=True)
class Theta:
    g: float
    e: float
    mu: float

    def get(self, param: Param) -> float:
        return float(getattr(self, param.value))


BASE = Theta(g=9.8, e=0.6, mu=0.2)
DELTA = Theta(g=1.0, e=0.1, mu=0.05)


@dataclass(frozen=True, eq=False)
class World:
    kind: Scene
    seed: int
    x0: float
    z0: float
    vx: float
    vz: float
    angle: float
    color: Array


@dataclass(frozen=True, eq=False)
class Nuisance:
    color: Array | None = None
    cam_shift: float = 0.0


@dataclass(frozen=True, eq=False)
class Clip:
    frames: NDArray[np.float32]
    xz: Array
    impacts: tuple[int, ...]


NO_NUISANCE = Nuisance()


def observable(kind: Scene) -> tuple[Param, ...]:
    return OBSERVABLE[Scene(kind)]


def make_world(kind: Scene, seed: int) -> World:
    r = np.random.default_rng(seed)
    x0 = r.uniform(-0.5, 0.5)
    z0 = r.uniform(3.0, 4.0)
    vx = r.uniform(-0.5, 0.5)
    vz = r.uniform(-0.5, 0.5)
    angle = r.uniform(0.25, 0.45)
    return World(Scene(kind), seed, x0, z0, vx, vz, angle, r.uniform(0.4, 1.0, 3))


def with_overrides(theta: Theta, overrides: Mapping[Param, float]) -> Theta:
    return replace(theta, **{param.value: value for param, value in overrides.items()})


def _freefall(w: World, theta: Theta, t: Array) -> Array:
    return np.stack([w.x0 + w.vx * t, w.z0 + w.vz * t - 0.5 * theta.g * t**2], 1)


def _bounce(w: World, theta: Theta) -> tuple[Array, list[int]]:
    x, z, vz = w.x0, w.z0, w.vz
    out: list[tuple[float, float]] = []
    hits: list[int] = []
    dt = 1 / (FPS * _SUBSTEPS)
    for i in range(T * _SUBSTEPS):
        if i % _SUBSTEPS == 0:
            out.append((x, z))
        vz -= theta.g * dt
        z += vz * dt
        x += w.vx * dt
        if z <= 0 and vz < 0:
            z, vz = -z * theta.e, -vz * theta.e
            hits.append(i // _SUBSTEPS)
    return np.array(out), hits


def _incline(w: World, theta: Theta, t: Array) -> tuple[Array, list[int]]:
    """Sphere slides without rolling along a plane of angle a with Coulomb friction mu."""
    a = w.angle
    acc = theta.g * (np.sin(a) - theta.mu * np.cos(a))
    s0 = 0.5
    s = s0 + np.maximum(0, 0.5 * acc * t**2) if acc > 0 else np.full_like(t, s0)
    return np.stack([s * np.cos(a), 3.0 - s * np.sin(a)], 1), [0] if acc > 0 else []


def trajectory(w: World, theta: Theta) -> tuple[Array, list[int]]:
    """(x, z) positions [T, 2] and the frames at which an impact happened."""
    t = np.arange(T) / FPS
    if w.kind is Scene.FREEFALL:
        return _freefall(w, theta, t), []
    if w.kind is Scene.BOUNCE:
        return _bounce(w, theta)
    return _incline(w, theta, t)


def render(w: World, xz: Array, res: int = 64, nuis: Nuisance = NO_NUISANCE) -> NDArray[np.float32]:
    """Gaussian disc on a ground line; world x in [-3, 3], z in [-0.5, 5.5]."""
    col = w.color if nuis.color is None else nuis.color
    px = (xz[:, 0] + 3 + nuis.cam_shift) / 6 * res
    pz = (5.5 - xz[:, 1]) / 6 * res
    g = np.arange(res)
    dist2 = (g[None, None, :] - px[:, None, None]) ** 2 + (
        g[None, :, None] - pz[:, None, None]
    ) ** 2
    blob = np.exp(-0.5 * (dist2 / 1.8**2))
    img = np.full((len(xz), 3, res, res), 0.15, np.float32)
    img[:, :, int((5.5 - 0) / 6 * res) :, :] = 0.35 if w.kind is not Scene.FREEFALL else 0.15
    return np.clip(img + blob[:, None] * col[None, :, None, None], 0, 1).astype(np.float32)


def clip(
    w: World,
    theta_over: Mapping[Param, float] | None = None,
    nuis: Nuisance = NO_NUISANCE,
    res: int = 64,
) -> Clip:
    xz, hits = trajectory(w, with_overrides(BASE, theta_over or {}))
    return Clip(render(w, xz, res, nuis), xz, tuple(hits))
