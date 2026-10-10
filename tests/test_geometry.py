import numpy as np
from cg import geometry as G, simulate as S, extract as E


def test_central_diff_exact_for_linear_and_rank():
    A = np.linalg.qr(np.random.default_rng(0).normal(size=(50, 3)))[0]
    f = lambda th: A @ th
    cols = [G.central_diff(f(np.eye(3)[j] * .1), f(-np.eye(3)[j] * .1), .1) for j in range(3)]
    J = np.stack(cols, 1)
    assert np.allclose(J, A)
    assert G.effective_rank(J)[0] > 2.99                                  # 3 distinguishable mechanisms
    J1 = np.outer(A[:, 0], [1, 1, 1])
    assert G.effective_rank(J1)[0] < 1.01                                 # collapsed to one 'motion' direction
    assert np.allclose(G.induced_metric(J), np.eye(3), atol=1e-12)


def test_principal_angles():
    A = np.eye(6)[:, :2]; B = np.eye(6)[:, 2:4]
    assert np.allclose(G.principal_angles(A, B), np.pi / 2)
    assert np.allclose(G.principal_angles(A, A), 0, atol=1e-7)


def test_composition_linear_zero_nonlinear_positive():
    rng = np.random.default_rng(1); W = rng.normal(size=(20, 2))
    lin = lambda t: W @ t
    nl = lambda t: np.tanh(W @ t) + 0.5 * t[0] * t[1] * np.ones(20)
    for f, zero in ((lin, True), (nl, False)):
        d = np.array([1.0, 1.0]); z0 = f(np.zeros(2))
        e = G.composition_error(f(d) - z0, f(np.array([1, 0.])) - z0, f(np.array([0, 1.])) - z0)[0]
        assert (e < 1e-20) if zero else (e > 1e-3)


def test_metric_distance_scale_free():
    G1 = np.diag([1., 2., 3.]); assert G.metric_distance(G1, 7 * G1) < 1e-12


def test_simulator_events_and_determinism():
    for kind in ("freefall", "bounce", "incline"):
        w = S.make_world(kind, 5)
        a, b = S.clip(w), S.clip(w)
        assert np.array_equal(a[0], b[0])                                  # deterministic duplicate render
        for p in S.observable(kind):
            hi, lo = (S.clip(w, {p: S.BASE[p] + d * S.DELTA[p]})[1] for d in (1, -1))
            assert np.abs(hi - lo).max() > 1e-3, (kind, p)                # intervention alters the physics
    # restitution without a collision is excluded: freefall has no e dependence
    assert "e" not in S.observable("freefall")
    assert S.clip(S.make_world("bounce", 5))[2], "bounce world must contain an impact"


def test_hooked_encoder_deterministic_layers():
    import torch
    m = E.ToyVideoEncoder(); h = E.HookedEncoder(m, m.blocks)
    x = torch.rand(2, 16, 3, 64, 64)
    a, b = h(x), h(x)
    assert a.shape[:2] == (2, 4) and np.array_equal(a, b)
