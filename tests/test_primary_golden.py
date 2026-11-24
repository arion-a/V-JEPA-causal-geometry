import json
from pathlib import Path

import numpy as np
import pytest

from cg import primary as p
from cg.scenes import Param, Scene

GOLDEN = Path(__file__).parent / "golden"
BASELINE = json.loads((GOLDEN / "primary_baseline.json").read_text())["results"]
INPUTS = np.load(GOLDEN / "primary_inputs.npz")
INTERMEDIATES = np.load(GOLDEN / "primary_intermediates.npz")
SCENARIOS = sorted({key.split("|")[0] for key in INPUTS.files})
ENDPOINTS = [
    p.Endpoint(p.Pooling.GLOBAL, "pt_g", "traj", "px", "rw_g"),
    p.Endpoint(p.Pooling.WINDOW, "pt_w", "traj", "px", "rw_w"),
]
RTOL = ATOL = 1e-12
N_BOOT = 300


def reps_for(scenario: str) -> dict[str, dict[p.Case, p.Array]]:
    reps: dict[str, dict[p.Case, p.Array]] = {}
    for key in INPUTS.files:
        name, rep, param, scene = key.split("|")
        if name == scenario:
            reps.setdefault(rep, {})[(Param(param), Scene(scene))] = INPUTS[key]
    return reps


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_unrounded_statistics_and_bootstrap_draws_match_baseline(scenario: str) -> None:
    reps = reps_for(scenario)
    sizes = {s: len(next(iter(reps.values()))[(Param.G, s)]) for s in p.SCENES}
    full = {s: np.arange(sizes[s]) for s in p.SCENES}
    boot = p._bootstrap(reps, sizes, N_BOOT, seed=0)
    for name, rep in reps.items():
        base = f"{scenario}|{name}"
        np.testing.assert_allclose(
            p._pair_cosines(rep, full, wrong=False), INTERMEDIATES[f"{base}|pairs"], RTOL, ATOL
        )
        np.testing.assert_allclose(
            p._statistic(rep, full, wrong=False), INTERMEDIATES[f"{base}|G"][0], RTOL, ATOL
        )
        np.testing.assert_allclose(
            p._statistic(rep, full, wrong=True), INTERMEDIATES[f"{base}|W"][0], RTOL, ATOL
        )
        np.testing.assert_allclose(boot[name].gravity, INTERMEDIATES[f"{base}|boot_g"], RTOL, ATOL)
        np.testing.assert_allclose(boot[name].wrong, INTERMEDIATES[f"{base}|boot_w"], RTOL, ATOL)


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_result_json_is_identical_to_pre_refactor_output(scenario: str, seed: int) -> None:
    result = p.run(reps_for(scenario), ENDPOINTS, n_boot=N_BOOT, seed=seed)
    expected = BASELINE[f"{scenario}|seed{seed}"]
    assert json.dumps(p.to_json(result), indent=1) == json.dumps(expected, indent=1)


def test_baseline_covers_every_claim_and_verdict() -> None:
    claims = {r["claim"] for r in BASELINE.values()}
    verdicts = {
        e["verdict_this_pooling"] for r in BASELINE.values() for e in r["endpoints"].values()
    }
    assert claims == {c.value for c in p.Claim}
    assert verdicts == {v.value for v in p.Verdict}
