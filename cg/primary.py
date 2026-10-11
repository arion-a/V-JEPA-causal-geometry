"""Confirmatory primary endpoint (configs/pilot.json 'confirmatory'). Pure numpy; no model, no file access.
Input per representation: V[(param, scene)] = array [n_worlds, d] of (plus - minus) derivatives, one row per world.
Statistic G(rep) = mean over the 3 scene pairs of cos(world-mean g in A, world-mean g in B).
Wrong-parameter version W(rep): same, but the second scene's g is replaced by that scene's other parameter (bounce: e, incline: mu).
Paired world bootstrap: each iteration resamples worlds within each scene ONCE and applies the same indices to every representation."""
import numpy as np

PAIRS = (("freefall", "bounce"), ("freefall", "incline"), ("bounce", "incline"))
WRONG = {"bounce": "e", "incline": "mu"}
LEVEL = (1.25, 98.75)          # 97.5% two-sided interval per endpoint (2 co-primary endpoints, Bonferroni)


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def _stat(V, idx, wrong=False):
    m = {}
    out = []
    for A, B in PAIRS:
        a = _unit(V[("g", A)][idx[A]].mean(0))
        b = _unit(V[(WRONG[B], B) if wrong else ("g", B)][idx[B]].mean(0))
        out.append(float(a @ b))
    return float(np.mean(out)), out


def run(reps, primary_pairs, n_boot=2000, seed=0):
    """reps: {name: V}. primary_pairs: {pooling_label: (vjepa_name, trajectory_name, pixel_name, random_name)}.
    Returns per-pooling results incl. paired-difference intervals and the claim-rule verdict."""
    n = {s: len(next(v for k, v in next(iter(reps.values())).items() if k[1] == s)) for s in ("freefall", "bounce", "incline")}
    full = {s: np.arange(n[s]) for s in n}
    rng = np.random.default_rng(seed)
    names = list(reps)
    obs = {r: _stat(reps[r], full)[0] for r in names}; obs_w = {r: _stat(reps[r], full, True)[0] for r in names}
    obs_pairs = {r: dict(zip(("freefall~bounce", "freefall~incline", "bounce~incline"), [round(x, 3) for x in _stat(reps[r], full)[1]])) for r in names}
    B = {r: [] for r in names}; BW = {r: [] for r in names}
    for _ in range(n_boot):
        idx = {s: rng.integers(0, n[s], n[s]) for s in n}
        for r in names:
            B[r].append(_stat(reps[r], idx)[0]); BW[r].append(_stat(reps[r], idx, True)[0])
    B = {r: np.array(x) for r, x in B.items()}; BW = {r: np.array(x) for r, x in BW.items()}
    iv = lambda x: [round(float(np.percentile(x, LEVEL[0])), 3), round(float(np.percentile(x, LEVEL[1])), 3)]
    out = dict(n_worlds=n, level="97.5% two-sided paired world bootstrap", per_rep={r: dict(G=round(obs[r], 3), G_wrong=round(obs_w[r], 3), pairs=obs_pairs[r]) for r in names}, endpoints={})
    for label, (vj, tr, px, rw) in primary_pairs.items():
        d_traj = B[vj] - B[tr]; d_wrong = B[vj] - BW[vj]; d_px = B[px] - B[vj]; d_rw = B[rw] - B[vj]
        lo = lambda x: float(np.percentile(x, LEVEL[0])); hi = lambda x: float(np.percentile(x, LEVEL[1]))
        e = dict(vjepa_G=round(obs[vj], 3), trajectory_G=round(obs[tr], 3), diff_vs_trajectory=round(obs[vj] - obs[tr], 3), diff_vs_trajectory_interval=iv(d_traj),
                 gravity_minus_wrong=round(obs[vj] - obs_w[vj], 3), gravity_minus_wrong_interval=iv(d_wrong),
                 pixel_minus_vjepa=round(obs[px] - obs[vj], 3), pixel_minus_vjepa_interval=iv(d_px),
                 random_minus_vjepa=round(obs[rw] - obs[vj], 3), random_minus_vjepa_interval=iv(d_rw))
        e["primary_success"] = bool(lo(d_traj) > 0)
        e["wrong_param_ok"] = bool(lo(d_wrong) > 0)                      # gravity alignment clearly above wrong-parameter alignment
        e["baselines_ok"] = bool(lo(d_px) <= 0 and lo(d_rw) <= 0)         # a baseline 'exceeds' V-JEPA only if its paired lower bound > 0
        if lo(d_traj) > 0: e["verdict_this_pooling"] = "V-JEPA above trajectory baseline"
        elif hi(d_traj) < 0: e["verdict_this_pooling"] = "trajectory baseline stronger"
        else: e["verdict_this_pooling"] = "not distinguished from kinematics"
        out["endpoints"][label] = e
    ok = [e["primary_success"] for e in out["endpoints"].values()]
    controls = all(e["wrong_param_ok"] and e["baselines_ok"] for e in out["endpoints"].values())
    if all(ok) and controls: out["claim"] = "mechanism-like geometry (both poolings pass, controls pass)"
    elif all(ok): out["claim"] = "primary passes for both poolings but a control fails: not supported"
    elif any(ok): out["claim"] = "pooling-dependent, not supported"
    elif all(e["verdict_this_pooling"] == "trajectory baseline stronger" for e in out["endpoints"].values()): out["claim"] = "trajectory baseline stronger"
    else: out["claim"] = "not distinguished from kinematics"
    return out
