"""PyBullet scenarios for Protocol A: free fall (g), bounce (g, e), incline (g, mu).

Headless and deterministic (DIRECT + TinyRenderer). Frames: 64 at 64 fps (1.0 s), physics step
1/256 s (4 substeps per frame), 384x384 RGB. Units: metres and seconds, z up. Bullet combines
contact coefficients as a product, so the plane carries 1.0 and the body carries the parameter.
Engine error against the closed form is O(g t dt) (symplectic Euler); tolerances are in the tests.

The engine is an imperative handle, so this module is an adapter: the pure parts are the world
sampling, the summaries of a finished run and the pixel and token geometry.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Final, TypedDict

import numpy as np
import pybullet as p
from numpy.typing import NDArray

from cg.scenes import OBSERVABLE as _OBSERVABLE
from cg.scenes import Scene

type Array = NDArray[np.float64]
type Color = tuple[float, float, float]
type Matrix = Sequence[float]

FPS, N_FRAMES, SUB, RES = 64, 64, 4, 384
DT = 1.0 / (FPS * SUB)
KINDS = tuple(Scene)
OBSERVABLE = {kind.value: tuple(p.value for p in params) for kind, params in _OBSERVABLE.items()}
_VIEW_MARGIN = 0.05
_IMPACT_SPEED = 1e-3


class Theta(TypedDict):
    """Physical parameters; vz is the initial vertical speed (m/s), used only by Stage 7."""

    g: float
    e: float
    mu: float
    vz: float


class ThetaOverride(TypedDict, total=False):
    g: float
    e: float
    mu: float
    vz: float


class NuisanceSpec(TypedDict, total=False):
    cam: float
    color: Color


class SimResult(TypedDict):
    frames: NDArray[np.uint8] | None
    pos: Array
    impacts: list[int]
    slide_m: float | None
    transition_frame: int | None
    apex_after_first_impact: float | None
    in_view: bool


BASE: Final[Theta] = {"g": 9.8, "e": 0.6, "mu": 0.25, "vz": 0.0}


@dataclass(frozen=True)
class World:
    kind: Scene
    seed: int
    h0: float  # initial height (m) for freefall and bounce, distance along the slope for incline
    x0: float
    vx: float
    angle: float
    size: float
    color: Color


def derive_seed(root: str, role: str, index: int) -> int:
    """Protocol seed policy: first 4 bytes of SHA256(root|role|index), big endian."""
    return int.from_bytes(hashlib.sha256(f"{root}|{role}|{index}".encode()).digest()[:4], "big")


def _height(kind: Scene, r: np.random.Generator) -> float:
    if kind is Scene.FREEFALL:
        return float(r.uniform(3.0, 4.0))
    return float(r.uniform(0.9, 1.2) if kind is Scene.BOUNCE else r.uniform(0.45, 0.55))


def make_world(kind: Scene, seed: int) -> World:
    kind = Scene(kind)
    r = np.random.default_rng(seed)
    h0 = _height(kind, r)
    x0 = float(r.uniform(-0.3, 0.3))
    vx = float(r.uniform(-0.3, 0.3))
    angle = float(np.radians(r.uniform(38, 42)))
    size = float(r.uniform(0.10, 0.14) if kind is Scene.INCLINE else r.uniform(0.12, 0.18))
    r_, g_, b_ = (float(c) for c in r.uniform(0.2, 0.95, 3))
    return World(kind, int(seed), h0, x0, vx, angle, size, (r_, g_, b_))


def resolve_theta(override: ThetaOverride | None) -> Theta:
    o = override or {}
    return Theta(
        g=o.get("g", BASE["g"]),
        e=o.get("e", BASE["e"]),
        mu=o.get("mu", BASE["mu"]),
        vz=o.get("vz", BASE["vz"]),
    )


def _camera(kind: Scene, shift: float = 0.0) -> tuple[Matrix, Matrix]:
    target = {
        Scene.FREEFALL: (0, 0, 2.0),
        Scene.BOUNCE: (0, 0, 0.9),
        Scene.INCLINE: (0.25, 0, 0.25),
    }[kind]
    dist = {Scene.FREEFALL: 9.0, Scene.BOUNCE: 4.4, Scene.INCLINE: 3.5}[kind]
    view = p.computeViewMatrixFromYawPitchRoll(
        [target[0] + shift, target[1], target[2]], dist, 0, 0, 0, 2
    )
    return view, p.computeProjectionMatrixFOV(50, 1.0, 0.1, 50)


@contextmanager
def _engine() -> Iterator[int]:
    cid = p.connect(p.DIRECT)
    try:
        yield cid
    finally:
        p.disconnect(cid)


def _build_incline(cid: int, w: World, theta: Theta) -> int:
    """Slab descending toward +x meeting a flat run-out at x = 0, so g and mu are separable."""
    ang = w.angle
    q = p.getQuaternionFromEuler([0, ang, 0])
    nrm = np.array([np.sin(ang), 0, np.cos(ang)])
    up = np.array([-np.cos(ang), 0, np.sin(ang)])
    length = 3.0
    flat = p.createCollisionShape(p.GEOM_BOX, halfExtents=[4, 2, 0.05], physicsClientId=cid)
    fid = int(p.createMultiBody(0, flat, basePosition=[4, 0, -0.05], physicsClientId=cid))
    slab = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[length / 2, 2, 0.05], physicsClientId=cid
    )
    pos = (up * length / 2 - nrm * 0.05).tolist()
    sid = int(p.createMultiBody(0, slab, basePosition=pos, baseOrientation=q, physicsClientId=cid))
    for plate in (fid, sid):
        p.changeDynamics(plate, -1, lateralFriction=1.0, restitution=0.0, physicsClientId=cid)
    shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[w.size, w.size, 0.05], physicsClientId=cid
    )
    start = (up * w.h0 + nrm * 0.05).tolist()
    bid = int(
        p.createMultiBody(0.5, shape, basePosition=start, baseOrientation=q, physicsClientId=cid)
    )
    p.changeDynamics(
        bid, -1, lateralFriction=theta["mu"], restitution=0.0, linearDamping=0, angularDamping=0,
        rollingFriction=0, spinningFriction=0, physicsClientId=cid,
    )  # fmt: skip
    return bid


def _build_ball(cid: int, w: World, theta: Theta) -> int:
    """Sphere with an initial velocity; only bounce has a floor, free fall has no scored contact."""
    if w.kind is Scene.BOUNCE:
        plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid)
        floor = p.createMultiBody(0, plane, physicsClientId=cid)
        p.changeDynamics(floor, -1, restitution=1.0, lateralFriction=0.0, physicsClientId=cid)
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=w.size, physicsClientId=cid)
    start = [w.x0, 0, w.h0 + w.size]
    bid = int(p.createMultiBody(0.5, shape, basePosition=start, physicsClientId=cid))
    restitution = theta["e"] if w.kind is Scene.BOUNCE else 0.0
    p.changeDynamics(
        bid, -1, restitution=restitution, lateralFriction=0.0, linearDamping=0, angularDamping=0,
        physicsClientId=cid,
    )  # fmt: skip
    p.resetBaseVelocity(bid, [w.vx, 0, theta["vz"]], physicsClientId=cid)
    return bid


def _add_reference_poles(cid: int, kind: Scene) -> None:
    """Static collision-free poles give motion cues, identical in every variant of a world."""
    for px in (-1.2, 1.2):
        shape = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.03, length=8.0, rgbaColor=[0.35, 0.35, 0.4, 1],
            physicsClientId=cid,
        )  # fmt: skip
        height = 3.0 if kind is Scene.FREEFALL else 1.0
        p.createMultiBody(0, -1, shape, basePosition=[px, 0.8, height], physicsClientId=cid)


def _snapshot(cid: int, ss: int, view: Matrix, proj: Matrix) -> NDArray[np.uint8]:
    img = p.getCameraImage(
        RES * ss, RES * ss, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=cid
    )[2]
    im = np.asarray(img, np.uint8).reshape(RES * ss, RES * ss, 4)[..., :3]
    if ss == 1:
        return im
    mean = im.reshape(RES, ss, RES, ss, 3).astype(np.float32).mean((1, 3))
    return np.rint(mean).astype(np.uint8)


@dataclass(frozen=True)
class _Run:
    frames: list[NDArray[np.uint8]]
    pos: Array
    impacts: list[int]


@dataclass(frozen=True)
class _Body:
    cid: int
    bid: int
    kind: Scene


@dataclass(frozen=True)
class _Rendering:
    render: bool
    ss: int


def _simulate_frames(body: _Body, camera: tuple[Matrix, Matrix], rendering: _Rendering) -> _Run:
    cid, bid = body.cid, body.bid
    pos: list[Sequence[float]] = []
    frames: list[NDArray[np.uint8]] = []
    impacts: list[int] = []
    vz_prev = 0.0
    for f in range(N_FRAMES):
        if rendering.render:
            frames.append(_snapshot(cid, rendering.ss, camera[0], camera[1]))
        pos.append(p.getBasePositionAndOrientation(bid, physicsClientId=cid)[0])
        for _ in range(SUB):
            p.stepSimulation(physicsClientId=cid)
            vz = p.getBaseVelocity(bid, physicsClientId=cid)[0][2]
            if body.kind is Scene.BOUNCE and vz_prev < -_IMPACT_SPEED and vz > _IMPACT_SPEED:
                impacts.append(f)
            vz_prev = vz
    return _Run(frames, np.array(pos), impacts)


def _summarise(run: _Run, kind: Scene, camera: tuple[Matrix, Matrix]) -> SimResult:
    pos = run.pos
    moved = pos[:, 0] > 0
    on_incline = kind is Scene.INCLINE
    return SimResult(
        frames=np.stack(run.frames) if run.frames else None,
        pos=pos,
        impacts=run.impacts,
        slide_m=float(np.linalg.norm(pos[-1] - pos[0])) if on_incline else None,
        transition_frame=int(np.argmax(moved)) if on_incline and moved.any() else None,
        apex_after_first_impact=float(pos[run.impacts[0] :, 2].max()) if run.impacts else None,
        in_view=_in_view(pos, *camera),
    )


def simulate(  # noqa: PLR0913 - public signature used by every script
    w: World,
    theta: ThetaOverride | None = None,
    nuis: NuisanceSpec | None = None,
    render: bool = True,
    ss: int = 1,
    size_scale: float = 1.0,
) -> SimResult:
    """Run one clip. ``ss`` supersamples (render RES*ss, then area-average).

    ``size_scale`` multiplies the object size.

    frames are uint8 [64, 384, 384, 3] or None; pos is [64, 3]; impacts are frame indices.
    """
    th = resolve_theta(theta)
    nz = nuis or {}
    if size_scale != 1.0:
        w = replace(w, size=w.size * size_scale)
    with _engine() as cid:
        p.setGravity(0, 0, -th["g"], physicsClientId=cid)
        p.setTimeStep(DT, physicsClientId=cid)
        p.setPhysicsEngineParameter(
            numSolverIterations=50, deterministicOverlappingPairs=1, physicsClientId=cid
        )
        bid = _build_incline(cid, w, th) if w.kind is Scene.INCLINE else _build_ball(cid, w, th)
        _add_reference_poles(cid, w.kind)
        rgba = [*nz.get("color", w.color), 1]
        p.changeVisualShape(bid, -1, rgbaColor=rgba, physicsClientId=cid)
        camera = _camera(w.kind, nz.get("cam", 0.0))
        run = _simulate_frames(_Body(cid, bid, w.kind), camera, _Rendering(render, ss))
    return _summarise(run, w.kind, camera)


def _homogeneous(pos: Array, view: Matrix, proj: Matrix) -> Array:
    v = np.array(view).reshape(4, 4).T
    pr = np.array(proj).reshape(4, 4).T
    return np.asarray(np.c_[pos, np.ones(len(pos))] @ v.T @ pr.T, dtype=np.float64)


def _in_view(pos: Array, view: Matrix, proj: Matrix, margin: float = _VIEW_MARGIN) -> bool:
    h = _homogeneous(pos, view, proj)
    ndc = h[:, :2] / h[:, 3:4]
    return bool((np.abs(ndc) < 1 - margin).all())


def frames_sha256(frames: NDArray[np.uint8]) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def project_pixels(pos: Array, kind: Scene, cam_shift: float = 0.0) -> Array:
    """World positions [T, 3] -> pixel (col, row) in the RES x RES image, as seen by simulate."""
    view, proj = _camera(Scene(kind), cam_shift)
    h = _homogeneous(pos, view, proj)
    ndc = h[:, :2] / h[:, 3:4]
    return np.asarray(np.c_[(ndc[:, 0] + 1) / 2 * RES, (1 - ndc[:, 1]) / 2 * RES], dtype=np.float64)


def object_token_mask(
    pos: Array, kind: Scene, dilate: int = 2, patch: int = 16, tubelet: int = 2
) -> NDArray[np.bool_]:
    """Boolean mask over the encoder token grid in (t, h, w) order covering the tracked path.

    The path is dilated by ``dilate`` tokens (Chebyshev). It is built from one reference trajectory
    so all variants of a world share the same window.
    """
    g = RES // patch
    mask = np.zeros((len(pos) // tubelet, g, g), bool)
    for f, (c, r) in enumerate(project_pixels(pos, kind)):
        ci = int(np.clip(c // patch, 0, g - 1))
        ri = int(np.clip(r // patch, 0, g - 1))
        rows = slice(max(0, ri - dilate), ri + dilate + 1)
        cols = slice(max(0, ci - dilate), ci + dilate + 1)
        mask[f // tubelet, rows, cols] = True
    return mask.reshape(-1)
