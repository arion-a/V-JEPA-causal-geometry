"""Deterministic physical scenarios with exact event handling, plus a numpy renderer.

Pipeline-validation backend (pure numpy, no engine dependency). Protocol A calls for MuJoCo/PyBullet in the
GPU stage: implement the same `Scenario.trajectory(theta, world)` interface there and keep these as
analytic cross-checks. Intervened parameters theta=(g, e, mu); nuisance eta=(camera shift, colour).

Scenarios: freefall (g), bounce (g,e; needs >=1 impact), incline (g,mu; needs contact sliding).
"""
from dataclasses import dataclass
import numpy as np

T, FPS = 16, 16.0          # frames and fps for the clip window (1.0 s)
PARAMS = ("g", "e", "mu")


@dataclass
class World:
    kind: str
    seed: int
    x0: float; z0: float; vx: float; vz: float; angle: float
    color: np.ndarray
    cam_shift: float = 0.0


def make_world(kind, seed):
    r = np.random.default_rng(seed)
    return World(kind, seed, r.uniform(-.5, .5), r.uniform(3.0, 4.0), r.uniform(-.5, .5), r.uniform(-.5, .5),
                 r.uniform(.25, .45), r.uniform(.4, 1.0, 3))


def observable(kind):
    return {"freefall": ("g",), "bounce": ("g", "e"), "incline": ("g", "mu")}[kind]


def trajectory(w, theta):
    """(x,z) positions [T,2] and impact frame list. theta: dict g,e,mu."""
    t = np.arange(T) / FPS
    g, e, mu = theta["g"], theta["e"], theta["mu"]
    if w.kind == "freefall":
        return np.stack([w.x0 + w.vx * t, w.z0 + w.vz * t - .5 * g * t ** 2], 1), []
    if w.kind == "bounce":
        x, z, vz, out, hits = w.x0, w.z0, w.vz, [], []
        dt = 1 / (FPS * 20)
        for i in range(T * 20):
            if i % 20 == 0: out.append((x, z))
            vz -= g * dt; z += vz * dt; x += w.vx * dt
            if z <= 0 and vz < 0: z, vz = -z * e, -vz * e; hits.append(i // 20)
        return np.array(out), hits
    # incline: sphere slides (no rolling) along a plane of angle a, Coulomb friction mu; stays on the plane.
    a = w.angle
    acc = g * (np.sin(a) - mu * np.cos(a))
    s0 = 0.5
    s = s0 + np.maximum(0, .5 * acc * t ** 2) if acc > 0 else np.full_like(t, s0)
    hit = [0] if acc > 0 else []
    return np.stack([s * np.cos(a), 3.0 - s * np.sin(a)], 1), hit


def render(w, xz, res=64, nuis_color=None, cam_shift=0.0):
    """float32 [T,3,res,res]; world x in [-3,3], z in [-0.5,5.5]; gaussian disc + ground/plane line."""
    col = w.color if nuis_color is None else nuis_color
    px = (xz[:, 0] + 3 + cam_shift) / 6 * res; pz = (5.5 - xz[:, 1]) / 6 * res
    g = np.arange(res)
    bl = np.exp(-.5 * (((g[None, None, :] - px[:, None, None]) ** 2 + (g[None, :, None] - pz[:, None, None]) ** 2) / 1.8 ** 2))
    img = np.full((len(xz), 3, res, res), .15, np.float32)
    img[:, :, int((5.5 - 0) / 6 * res):, :] = .35 if w.kind != "freefall" else .15
    return np.clip(img + bl[:, None] * col[None, :, None, None], 0, 1).astype(np.float32)


BASE = dict(g=9.8, e=0.6, mu=0.2)
DELTA = dict(g=1.0, e=0.1, mu=0.05)       # symmetric perturbation (parameter units); validated at Gate 1


def clip(w, theta_over=None, nuis=None, res=64):
    th = {**BASE, **(theta_over or {})}
    xz, hits = trajectory(w, th)
    n = nuis or {}
    return render(w, xz, res, n.get("color"), n.get("cam", 0.0)), xz, hits
