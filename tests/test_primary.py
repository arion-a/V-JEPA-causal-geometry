import numpy as np
from cg import primary as P


def mk(rng, d, shared, n=15, noise=0.3):
    """shared: direction used for g in every scene; e/mu are independent directions."""
    V = {}
    for s in ("freefall", "bounce", "incline"):
        V[("g", s)] = np.stack([shared * 3 + rng.normal(size=d) * noise for _ in range(n)])
    V[("e", "bounce")] = np.stack([rng.normal(size=d) * 3 for _ in range(n)]); V[("mu", "incline")] = np.stack([rng.normal(size=d) * 3 for _ in range(n)])
    return V


def sceneg(rng, d, n=15):
    """scene-specific gravity directions (no shared mechanism)"""
    V = {}
    for s in ("freefall", "bounce", "incline"):
        u = rng.normal(size=d); V[("g", s)] = np.stack([u * 3 + rng.normal(size=d) * 0.3 for _ in range(n)])
    V[("e", "bounce")] = np.stack([rng.normal(size=d) * 3 for _ in range(15)]); V[("mu", "incline")] = np.stack([rng.normal(size=d) * 3 for _ in range(15)])
    return V


def test_shared_direction_beats_scene_specific_baseline_and_claim_passes():
    rng = np.random.default_rng(0); d = 256; u = rng.normal(size=d)
    reps = {"vj_g": mk(rng, d, u), "vj_w": mk(rng, d, u), "traj": sceneg(rng, d), "px": sceneg(rng, d), "rw": sceneg(rng, d)}
    r = P.run(reps, {"global": ("vj_g", "traj", "px", "rw"), "window": ("vj_w", "traj", "px", "rw")}, n_boot=300)
    assert r["claim"].startswith("mechanism-like") and all(e["primary_success"] for e in r["endpoints"].values())


def test_no_shared_direction_is_not_a_success_and_baseline_stronger_is_detected():
    rng = np.random.default_rng(1); d = 256; u = rng.normal(size=d)
    reps = {"vj_g": sceneg(rng, d), "vj_w": sceneg(rng, d), "traj": mk(rng, d, u), "px": sceneg(rng, d), "rw": sceneg(rng, d)}
    r = P.run(reps, {"global": ("vj_g", "traj", "px", "rw"), "window": ("vj_w", "traj", "px", "rw")}, n_boot=300)
    assert r["claim"] == "trajectory baseline stronger" and not any(e["primary_success"] for e in r["endpoints"].values())


def test_one_pooling_only_is_pooling_dependent():
    rng = np.random.default_rng(2); d = 256; u = rng.normal(size=d)
    reps = {"vj_g": mk(rng, d, u), "vj_w": sceneg(rng, d), "traj": sceneg(rng, d), "px": sceneg(rng, d), "rw": sceneg(rng, d)}
    r = P.run(reps, {"global": ("vj_g", "traj", "px", "rw"), "window": ("vj_w", "traj", "px", "rw")}, n_boot=300)
    assert r["claim"] == "pooling-dependent, not supported"


def test_paired_bootstrap_is_deterministic():
    rng = np.random.default_rng(3); d = 64; u = rng.normal(size=d)
    reps = {"a": mk(rng, d, u), "t": sceneg(rng, d), "p": sceneg(rng, d), "r": sceneg(rng, d)}
    f = lambda: P.run(reps, {"x": ("a", "t", "p", "r")}, n_boot=100)
    assert f() == f()


def test_heldout_refused_without_valid_unlock():
    import subprocess, sys, os
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    for extra in ([], ["--unlock-hash", "0" * 64]):
        r = subprocess.run([sys.executable, f"{root}/scripts/heldout_primary.py", "--split", "heldout", *extra], capture_output=True, text=True)
        assert r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr)
