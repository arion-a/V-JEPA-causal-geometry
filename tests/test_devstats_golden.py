import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from cg import devstats as ds

GOLDEN = Path(__file__).parent / "golden"
BASELINE = json.loads((GOLDEN / "devstats_baseline.json").read_text())["cases"]
INPUTS = np.load(GOLDEN / "devstats_inputs.npz")
TOLERANCE = {"f64": 1e-12, "f32": 1e-6}
DTYPE = {"f64": np.float64, "f32": np.float32}

Arrays = dict[str, NDArray[np.float64] | NDArray[np.float32]]
Case = Callable[[Arrays], tuple[float, ...]]


def _flat(result: object) -> tuple[float, ...]:
    if isinstance(result, ds.WithinScene):
        return (result.cos_mean, *result.cos_per_world, result.eff_rank_mean)
    if isinstance(result, ds.AlignmentGap):
        return (result.g_g, result.g_other, result.gap)
    if isinstance(result, float):
        return (result,)
    assert isinstance(result, tuple)
    return tuple(result)


def _scalar(value: float | list[float]) -> float:
    assert not isinstance(value, list)
    return float(value)


def _series(value: float | list[float]) -> list[float]:
    assert isinstance(value, list)
    return value


def _baseline_flat(
    value: float | list[float] | dict[str, float | list[float]],
) -> tuple[float, ...]:
    if isinstance(value, dict):
        if "cos_mean" in value:
            return (
                _scalar(value["cos_mean"]),
                *_series(value["cos_per_world"]),
                _scalar(value["eff_rank_mean"]),
            )
        return (_scalar(value["g_g"]), _scalar(value["g_other"]), _scalar(value["gap"]))
    if isinstance(value, list):
        return tuple(value)
    return (value,)


CASES: dict[str, Case] = {
    "split_half_15": lambda i: _flat(ds.split_half_reliability(i["A15"], seed=0)),
    "split_half_15_max300": lambda i: _flat(
        ds.split_half_reliability(i["A15"], max_splits=300, seed=0)
    ),
    "split_half_6_exhaustive": lambda i: _flat(ds.split_half_reliability(i["A6"], seed=0)),
    "split_half_noise": lambda i: _flat(ds.split_half_reliability(i["N15"], seed=0)),
    "split_half_2": lambda i: _flat(ds.split_half_reliability(i["A2"], seed=0)),
    "mean_cos": lambda i: _flat(ds.mean_cos(i["A15"], i["B15"])),
    "alignment": lambda i: _flat(ds.alignment_with_control(i["A15"], i["B15"], i["C15"])),
    "within_15": lambda i: _flat(ds.within_scene(i["A15"], i["C15"])),
    "within_1": lambda i: _flat(ds.within_scene(i["A1"], i["B1"])),
    "within_2": lambda i: _flat(ds.within_scene(i["A2"], i["B2"])),
    "boot_s0": lambda i: _flat(ds.bootstrap_mean_cos(i["A15"], i["B15"], n_resamples=500, seed=0)),
    "boot_s7": lambda i: _flat(ds.bootstrap_mean_cos(i["A15"], i["C15"], n_resamples=500, seed=7)),
    "boot_1": lambda i: _flat(ds.bootstrap_mean_cos(i["A1"], i["B1"], n_resamples=50, seed=0)),
    "within_ci_s0": lambda i: _flat(
        ds.within_scene_ci(i["A15"], i["C15"], n_resamples=500, seed=0)
    ),
    "within_ci_s3": lambda i: _flat(
        ds.within_scene_ci(i["A15"], i["B15"], n_resamples=500, seed=3)
    ),
    "shuffled_s0": lambda i: _flat(
        ds.shuffled_pairing_reliability(i["Zp15"], i["Zm15"], n_repeats=8, seed=0)
    ),
    "shuffled_s5": lambda i: _flat(
        ds.shuffled_pairing_reliability(i["Zp15"], i["Zm15"], n_repeats=8, seed=5)
    ),
    "shuffled_2": lambda i: _flat(
        ds.shuffled_pairing_reliability(i["A2"], i["B2"], n_repeats=4, seed=1)
    ),
    "signflip_s0": lambda i: _flat(
        ds.sign_flip_null_reliability(i["Zp15"] - i["Zm15"], n_repeats=8, seed=0)
    ),
    "signflip_s5": lambda i: _flat(ds.sign_flip_null_reliability(i["A15"], n_repeats=8, seed=5)),
}


@pytest.mark.parametrize("precision", ["f64", "f32"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_pre_refactor_baseline(name: str, precision: str) -> None:
    arrays: Arrays = {k: INPUTS[k].astype(DTYPE[precision]) for k in INPUTS.files}
    expected = _baseline_flat(BASELINE[f"{name}|{precision}"])
    actual = CASES[name](arrays)
    np.testing.assert_allclose(
        actual, expected, rtol=TOLERANCE[precision], atol=TOLERANCE[precision]
    )


def test_result_json_is_unchanged_by_dataclass_results() -> None:
    a15, c15, b15 = INPUTS["A15"], INPUTS["C15"], INPUTS["B15"]
    within = json.dumps(dataclasses.asdict(ds.within_scene(a15, c15)), indent=1)
    gap = json.dumps(dataclasses.asdict(ds.alignment_with_control(a15, b15, c15)), indent=1)
    assert within == json.dumps(BASELINE["within_15|f64"], indent=1)
    assert gap == json.dumps(BASELINE["alignment|f64"], indent=1)
