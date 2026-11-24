import numpy as np
import pytest
from golden_util import assert_matches, load, max_abs_difference, subtree

from cg import geometry as g

BASE = load("geometry_golden")
INPUTS = subtree(BASE, "in")
RTOL = ATOL = 1e-12


def arr(name: str) -> np.ndarray:
    return INPUTS[f"/{name}"]


def results() -> dict[str, object]:
    stat, va, vb = arr("stat"), arr("va"), arr("vb")
    zp, zm = arr("zp"), arr("zm")
    return {
        "central": g.central_diff(zp, zm, 0.25),
        "central_lists": g.central_diff(np.array(zp.tolist()), np.array(zm.tolist()), 0.5),
        "rank_J": g.effective_rank(arr("J")),
        "rank_J1": g.effective_rank(arr("J1")),
        "rank_zero": g.effective_rank(np.zeros((4, 3))),
        "metric": g.induced_metric(arr("J")),
        "angles": g.principal_angles(arr("J"), arr("B2")),
        "angles_same": g.principal_angles(arr("A"), arr("A")),
        "comp": g.composition_error(arr("dz/0"), arr("dz/1"), arr("dz/2")),
        "comp_eps": g.composition_error(np.zeros(5), np.ones(5), -np.ones(5), eps=1e-3),
        "mdist": g.metric_distance(arr("G1"), arr("G2")),
        "mdist_zero": g.metric_distance(np.zeros((3, 3)), arr("G2")),
        "boot0": g.bootstrap_worlds(stat, n_boot=500, seed=0),
        "boot4": g.bootstrap_worlds(list(stat), n_boot=300, seed=4),
        "null0": g.shuffled_cosine_null(va, vb, n_permutations=200, seed=0),
        "null2": g.shuffled_cosine_null(va, vb, n_permutations=100, seed=2),
    }


def expected(name: str) -> dict[str, np.ndarray]:
    return subtree(BASE, f"out/{name}")


def legacy_layout(name: str, value: object) -> object:
    """Old bootstrap layout: (mean, (low, high)); other results already match."""
    if isinstance(value, g.MeanCI):
        return (value.mean, (value.low, value.high))
    return value


@pytest.mark.parametrize("name", sorted(results()))
def test_matches_pre_refactor_baseline(name: str) -> None:
    value = legacy_layout(name, results()[name])
    assert_matches(value, {f"{k}": v for k, v in expected(name).items()}, rtol=RTOL, atol=ATOL)


def test_exact_agreement() -> None:
    worst = max(max_abs_difference(legacy_layout(n, v), expected(n)) for n, v in results().items())
    assert worst == 0.0
