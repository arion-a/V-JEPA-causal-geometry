import numpy as np
import pytest

from cg import geometry as g


def test_central_diff_exact_for_linear_and_rank() -> None:
    a = np.linalg.qr(np.random.default_rng(0).normal(size=(50, 3)))[0]
    cols = [
        g.central_diff(a @ (np.eye(3)[j] * 0.1), a @ (-np.eye(3)[j] * 0.1), 0.1) for j in range(3)
    ]
    j = np.stack(cols, 1)
    assert np.allclose(j, a)
    assert g.effective_rank(j).value > 2.99
    collapsed = np.outer(a[:, 0], [1, 1, 1])
    assert g.effective_rank(collapsed).value < 1.01
    assert np.allclose(g.induced_metric(j), np.eye(3), atol=1e-12)


def test_principal_angles() -> None:
    a = np.eye(6)[:, :2]
    b = np.eye(6)[:, 2:4]
    assert np.allclose(g.principal_angles(a, b), np.pi / 2)
    assert np.allclose(g.principal_angles(a, a), 0, atol=1e-7)


def test_composition_linear_zero_nonlinear_positive() -> None:
    w = np.random.default_rng(1).normal(size=(20, 2))

    def linear(t: np.ndarray) -> np.ndarray:
        out: np.ndarray = w @ t
        return out

    def nonlinear(t: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.tanh(w @ t) + 0.5 * t[0] * t[1] * np.ones(20)
        return out

    for f, is_additive in ((linear, True), (nonlinear, False)):
        z0 = f(np.zeros(2))
        both = f(np.array([1.0, 1.0])) - z0
        err = g.composition_error(both, f(np.array([1.0, 0.0])) - z0, f(np.array([0.0, 1.0])) - z0)
        assert (err.relative < 1e-20) if is_additive else (err.relative > 1e-3)


def test_metric_distance_scale_free() -> None:
    g1 = np.diag([1.0, 2.0, 3.0])
    assert g.metric_distance(g1, 7 * g1) < 1e-12


def test_bootstrap_and_null_are_seeded_and_order_independent() -> None:
    rng = np.random.default_rng(2)
    stat = rng.normal(size=15)
    first = g.bootstrap_worlds(stat, n_boot=200, seed=3)
    g.shuffled_cosine_null(
        rng.normal(size=(6, 8)), rng.normal(size=(6, 8)), n_permutations=20, seed=1
    )
    assert g.bootstrap_worlds(stat, n_boot=200, seed=3) == first
    assert first.low < first.mean < first.high


def test_shuffled_null_rejects_unequal_groups() -> None:
    with pytest.raises(ValueError):
        g.shuffled_cosine_null(np.ones((3, 4)), np.ones((2, 4)), seed=0)
