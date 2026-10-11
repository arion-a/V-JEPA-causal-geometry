"""PyBullet scenarios for Protocol A: free fall (g), bounce (g,e), incline (g,mu). Headless, deterministic (DIRECT + TinyRenderer).

Frames: 64 @ 64 fps (1.0 s), physics dt=1/256 (4 substeps/frame), 384x384 RGB. World units metres, z up.
Combined-coefficient rule in Bullet is a product, so the plane carries 1.0 and the body carries the parameter.
Engine tolerance vs closed form is O(g*t*dt) (symplectic Euler); documented in tests, not the 1e-6 of the numpy cross-check.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import pybullet as p

FPS, N_FRAMES, SUB, RES = 64, 64, 4, 384
DT = 1.0 / (FPS * SUB)
BASE = dict(g=9.8, e=0.6, mu=0.25, vz=0.0)      # vz: initial vertical speed (m/s), used only by Stage 7 (freefall/bounce); 0 for every earlier clip
KINDS = ("freefall", "bounce", "incline")
OBSERVABLE = {"freefall": ("g",), "bounce": ("g", "e"), "incline": ("g", "mu")}


@dataclass
class World:
    kind: str; seed: int
    h0: float          # initial height (m) (bounce/freefall) or distance along slope (incline)
    x0: float; vx: float; angle: float; size: float
    color: tuple


def derive_seed(root, role, index):
    """Protocol seed policy: first 4 bytes of SHA256(root|role|index), big endian."""
    return int.from_bytes(hashlib.sha256(f"{root}|{role}|{index}".encode()).digest()[:4], "big")


def make_world(kind, seed):
    r = np.random.default_rng(seed)
    return World(kind, int(seed),
                 h0=float(r.uniform(3.0, 4.0) if kind == "freefall" else r.uniform(0.9, 1.2) if kind == "bounce" else r.uniform(0.45, 0.55)),
                 x0=float(r.uniform(-0.3, 0.3)), vx=float(r.uniform(-0.3, 0.3)), angle=float(np.radians(r.uniform(38, 42))),
                 size=float(r.uniform(0.12, 0.18) if kind != "incline" else r.uniform(0.10, 0.14)), color=tuple(float(c) for c in r.uniform(0.2, 0.95, 3)))


def _camera(kind, shift=0.0):
    tgt = {"freefall": (0, 0, 2.0), "bounce": (0, 0, 0.9), "incline": (0.25, 0, 0.25)}[kind]
    dist = {"freefall": 9.0, "bounce": 4.4, "incline": 3.5}[kind]
    view = p.computeViewMatrixFromYawPitchRoll([tgt[0] + shift, tgt[1], tgt[2]], dist, 0, 0, 0, 2)
    proj = p.computeProjectionMatrixFOV(50, 1.0, 0.1, 50)
    return view, proj


def simulate(w, theta=None, nuis=None, render=True, ss=1, size_scale=1.0):
    """ss: supersampling factor (render RES*ss then area-average). size_scale: multiplies object size.
    -> dict(frames uint8[64,384,384,3] or None, pos [64,3], impacts [frame idx], slide_m, apex). Deterministic."""
    th = {**BASE, **(theta or {})}; nz = nuis or {}
    if size_scale != 1.0:
        from dataclasses import replace as _rep; w = _rep(w, size=w.size * size_scale)
    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -th["g"], physicsClientId=cid); p.setTimeStep(DT, physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=50, deterministicOverlappingPairs=1, physicsClientId=cid)
        col = nz.get("color", w.color)
        if w.kind == "incline":
            ang = w.angle                                   # slab descends toward +x, meets a flat run-out at x=0 (v2 design: g and mu separable)
            q = p.getQuaternionFromEuler([0, ang, 0]); nrm = np.array([np.sin(ang), 0, np.cos(ang)]); up = np.array([-np.cos(ang), 0, np.sin(ang)])
            L = 3.0
            flat = p.createCollisionShape(p.GEOM_BOX, halfExtents=[4, 2, 0.05], physicsClientId=cid)
            fid = p.createMultiBody(0, flat, basePosition=[4, 0, -0.05], physicsClientId=cid)
            slab = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L / 2, 2, 0.05], physicsClientId=cid)
            sid = p.createMultiBody(0, slab, basePosition=(up * L / 2 - nrm * 0.05).tolist(), baseOrientation=q, physicsClientId=cid)
            for b_ in (fid, sid): p.changeDynamics(b_, -1, lateralFriction=1.0, restitution=0.0, physicsClientId=cid)
            body_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[w.size, w.size, 0.05], physicsClientId=cid)
            bid = p.createMultiBody(0.5, body_col, basePosition=(up * w.h0 + nrm * 0.05).tolist(), baseOrientation=q, physicsClientId=cid)
            p.changeDynamics(bid, -1, lateralFriction=th["mu"], restitution=0.0, linearDamping=0, angularDamping=0,
                             rollingFriction=0, spinningFriction=0, physicsClientId=cid)
        else:
            if w.kind == "bounce":                          # free fall has NO floor (no scored contact)
                fl = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid); floor = p.createMultiBody(0, fl, physicsClientId=cid)
                p.changeDynamics(floor, -1, restitution=1.0, lateralFriction=0.0, physicsClientId=cid)
            body_col = p.createCollisionShape(p.GEOM_SPHERE, radius=w.size, physicsClientId=cid)
            bid = p.createMultiBody(0.5, body_col, basePosition=[w.x0, 0, w.h0 + w.size], physicsClientId=cid)
            p.changeDynamics(bid, -1, restitution=th["e"] if w.kind == "bounce" else 0.0, lateralFriction=0.0, linearDamping=0, angularDamping=0, physicsClientId=cid)
            p.resetBaseVelocity(bid, [w.vx, 0, th["vz"]], physicsClientId=cid)
        # static, collision-free reference poles give motion cues (same in every variant of a world)
        for px in (-1.2, 1.2):
            pv = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=8.0, rgbaColor=[.35, .35, .4, 1], physicsClientId=cid)
            p.createMultiBody(0, -1, pv, basePosition=[px, 0.8, 3.0 if w.kind == "freefall" else 1.0], physicsClientId=cid)
        p.changeVisualShape(bid, -1, rgbaColor=list(col) + [1], physicsClientId=cid)
        view, proj = _camera(w.kind, nz.get("cam", 0.0))
        pos, frames, impacts, vz_prev = [], [], [], 0.0
        for f in range(N_FRAMES):
            if render:
                img = p.getCameraImage(RES * ss, RES * ss, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=cid)[2]
                im = np.asarray(img, np.uint8).reshape(RES * ss, RES * ss, 4)[..., :3]
                frames.append(im if ss == 1 else np.rint(im.reshape(RES, ss, RES, ss, 3).astype(np.float32).mean((1, 3))).astype(np.uint8))
            pos.append(p.getBasePositionAndOrientation(bid, physicsClientId=cid)[0])
            for _ in range(SUB):
                p.stepSimulation(physicsClientId=cid)
                vz = p.getBaseVelocity(bid, physicsClientId=cid)[0][2]
                if w.kind == "bounce" and vz_prev < -1e-3 and vz > 1e-3: impacts.append(f)
                vz_prev = vz
        pos = np.array(pos)
        out = dict(frames=np.stack(frames) if render else None, pos=pos, impacts=impacts)
        out["slide_m"] = float(np.linalg.norm(pos[-1] - pos[0])) if w.kind == "incline" else None
        out["transition_frame"] = (int(np.argmax(pos[:, 0] > 0)) if (pos[:, 0] > 0).any() else None) if w.kind == "incline" else None
        out["apex_after_first_impact"] = float(pos[impacts[0]:, 2].max()) if impacts else None
        out["in_view"] = bool(_in_view(pos, view, proj))
        return out
    finally:
        p.disconnect(cid)


def _in_view(pos, view, proj, margin=0.05):
    V = np.array(view).reshape(4, 4).T; P = np.array(proj).reshape(4, 4).T
    h = np.c_[pos, np.ones(len(pos))] @ V.T @ P.T
    ndc = h[:, :2] / h[:, 3:4]
    return bool((np.abs(ndc) < 1 - margin).all())


def frames_sha256(frames):
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def project_pixels(pos, kind, cam_shift=0.0):
    """World positions [T,3] -> pixel (col,row) in the RES x RES image, same camera as simulate()."""
    view, proj = _camera(kind, cam_shift)
    V = np.array(view).reshape(4, 4).T; P = np.array(proj).reshape(4, 4).T
    h = np.c_[pos, np.ones(len(pos))] @ V.T @ P.T; ndc = h[:, :2] / h[:, 3:4]
    return np.c_[(ndc[:, 0] + 1) / 2 * RES, (1 - ndc[:, 1]) / 2 * RES]


def object_token_mask(pos, kind, dilate=2, patch=16, tubelet=2):
    """Boolean mask over the encoder's token grid [(T/tubelet)*(RES/patch)^2], (t,h,w) order, covering the object's tracked path
    (+ `dilate` tokens, Chebyshev). Built from one reference trajectory so all variants of a world share the same window."""
    g = RES // patch; nt = len(pos) // tubelet; m = np.zeros((nt, g, g), bool); px = project_pixels(pos, kind)
    for f, (c, r) in enumerate(px):
        ci, ri = int(np.clip(c // patch, 0, g - 1)), int(np.clip(r // patch, 0, g - 1))
        m[f // tubelet, max(0, ri - dilate):ri + dilate + 1, max(0, ci - dilate):ci + dilate + 1] = True
    return m.reshape(-1)
