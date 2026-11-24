import numpy as np
import pytest

from cg import devstats as ds


def test_reliability_high_for_shared_direction_low_for_noise() -> None:
    rng = np.random.default_rng(0)
    d = 512
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    shared = np.stack([u * 5 + rng.normal(size=d) * 0.05 for _ in range(5)])
    noise = rng.normal(size=(5, d))
    assert ds.split_half_reliability(shared, seed=0).mean > 0.95
    assert abs(ds.split_half_reliability(noise, seed=0).mean) < 0.25


def test_alignment_gap_detects_specific_direction() -> None:
    rng = np.random.default_rng(1)
    d = 256
    u = rng.normal(size=d)
    w = rng.normal(size=d)

    def group(base: np.ndarray) -> np.ndarray:
        return np.stack([base + rng.normal(size=d) * 0.2 for _ in range(5)])

    r = ds.alignment_with_control(group(u), group(u), group(w))
    assert r.g_g > 0.9
    assert abs(r.g_other) < 0.3
    assert r.gap > 0.6


def test_within_scene_rank() -> None:
    rng = np.random.default_rng(2)
    u = rng.normal(size=128)
    w = rng.normal(size=128)
    same = ds.within_scene(np.stack([u, u]), np.stack([-u, -2 * u]))
    assert same.cos_mean < -0.99
    assert same.eff_rank_mean < 1.01
    diff = ds.within_scene(np.stack([u]), np.stack([w]))
    assert abs(diff.cos_mean) < 0.4
    assert diff.eff_rank_mean > 1.7


def test_reliability_samples_splits_for_large_n_and_is_seed_stable() -> None:
    rng = np.random.default_rng(3)
    d = 256
    u = rng.normal(size=d)
    v = np.stack([u + rng.normal(size=d) * 0.5 for _ in range(15)])
    a = ds.split_half_reliability(v, seed=0)
    b = ds.split_half_reliability(v, seed=0)
    assert a == b
    assert 0.8 < a.mean <= 1.0


def test_bootstrap_cos_ci_and_nulls() -> None:
    rng = np.random.default_rng(4)
    d = 128
    u = rng.normal(size=d)
    a = np.stack([u + rng.normal(size=d) * 0.3 for _ in range(15)])
    b = np.stack([u + rng.normal(size=d) * 0.3 for _ in range(15)])
    boot = ds.bootstrap_mean_cos(a, b, n_resamples=300, seed=0)
    assert boot.low > 0.8
    assert boot.high <= 1.0
    assert boot.low < boot.estimate
    zm = rng.normal(size=(15, d))
    zp = zm + u * 2
    matched = ds.split_half_reliability(zp - zm, seed=0).mean
    shuffled = ds.shuffled_pairing_reliability(zp, zm, n_repeats=10, seed=0).mean
    assert matched > 0.95
    assert shuffled <= matched
    assert abs(ds.sign_flip_null_reliability(zp - zm, seed=0).mean) < 0.3
    assert ds.within_scene_ci(a, -a, seed=0).estimate < -0.99


def test_results_do_not_depend_on_call_order() -> None:
    rng = np.random.default_rng(5)
    v = rng.normal(size=(15, 32))
    first = ds.bootstrap_mean_cos(v, v + 1.0, n_resamples=100, seed=9)
    ds.sign_flip_null_reliability(v, seed=1)
    assert ds.bootstrap_mean_cos(v, v + 1.0, n_resamples=100, seed=9) == first


def test_invalid_shapes_fail_loudly() -> None:
    one_world = np.ones((1, 8))
    with pytest.raises(ValueError):
        ds.split_half_reliability(one_world, seed=0)
    with pytest.raises(ValueError):
        ds.shuffled_pairing_reliability(one_world, one_world, seed=0)
    with pytest.raises(ValueError):
        ds.within_scene(np.ones((3, 8)), np.ones((2, 8)))
    with pytest.raises(ValueError):
        ds.mean_cos(np.ones(8), np.ones(8))
