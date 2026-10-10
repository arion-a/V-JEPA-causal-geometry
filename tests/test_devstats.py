import numpy as np
from cg import devstats as DS


def test_reliability_high_for_shared_direction_low_for_noise():
    rng = np.random.default_rng(0); d = 512; u = rng.normal(size=d); u /= np.linalg.norm(u)
    shared = np.stack([u * 5 + rng.normal(size=d) * 0.05 for _ in range(5)]); noise = rng.normal(size=(5, d))
    assert DS.split_half_reliability(shared)[0] > 0.95 and abs(DS.split_half_reliability(noise)[0]) < 0.25


def test_alignment_gap_detects_specific_direction():
    rng = np.random.default_rng(1); d = 256; u = rng.normal(size=d); w = rng.normal(size=d)
    mk = lambda base: np.stack([base + rng.normal(size=d) * 0.2 for _ in range(5)])
    r = DS.alignment_with_control(mk(u), mk(u), mk(w)); assert r["g_g"] > 0.9 and abs(r["g_other"]) < 0.3 and r["gap"] > 0.6


def test_within_scene_rank():
    rng = np.random.default_rng(2); u = rng.normal(size=128); w = rng.normal(size=128)
    same = DS.within_scene(np.stack([u, u]), np.stack([-u, -2 * u])); assert same["cos_mean"] < -0.99 and same["eff_rank_mean"] < 1.01
    diff = DS.within_scene(np.stack([u]), np.stack([w])); assert abs(diff["cos_mean"]) < 0.4 and diff["eff_rank_mean"] > 1.7


def test_reliability_samples_splits_for_large_n_and_is_seed_stable():
    rng = np.random.default_rng(3); d = 256; u = rng.normal(size=d); V = np.stack([u + rng.normal(size=d) * 0.5 for _ in range(15)])
    a, b = DS.split_half_reliability(V), DS.split_half_reliability(V)
    assert a == b and 0.8 < a[0] <= 1.0                    # 15 worlds -> sampled splits, deterministic
