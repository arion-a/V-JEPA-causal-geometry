import numpy as np
import pytest
from golden_util import Flat, assert_matches, load, max_abs_difference, subtree

from cg import analysis as an
from cg.scenes import Scene

BASE = load("analysis_golden")
INPUTS = subtree(BASE, "in")
LAYERS = 4


def feats() -> dict[str, dict[str, list[np.ndarray]]]:
    out: dict[str, dict[str, list[np.ndarray]]] = {}
    for key, value in sorted(subtree(INPUTS, "/feats").items()):
        _, world, variant, layer = key.split("/")
        layers = out.setdefault(world, {}).setdefault(variant, [])
        assert len(layers) == int(layer)
        layers.append(value)
    return out


def kinds() -> dict[str, Scene]:
    return {k.removeprefix("/"): Scene(str(v)) for k, v in subtree(INPUTS, "/kinds").items()}


def legacy(report: an.LayerReport) -> dict[str, object]:
    out: dict[str, object] = {
        "cross_context_cos": report.cross_context_cos,
        "eff_rank": (report.eff_rank.mean, (report.eff_rank.low, report.eff_rank.high)),
    }
    if report.min_principal_angle_deg is not None:
        m = report.min_principal_angle_deg
        out["min_principal_angle_deg"] = (m.mean, (m.low, m.high))
    if report.composition_error is not None:
        c = report.composition_error
        out["composition_error"] = (c.mean, (c.low, c.high))
    return out


def test_layer_reports_match_pre_refactor_baseline() -> None:
    value = {
        f"{layer}|{norm}": legacy(an.layer_report(feats(), kinds(), layer, norm=norm))
        for layer in range(LAYERS)
        for norm in (False, True)
    }
    expected: Flat = subtree(BASE, "out/report")
    assert_matches(value, expected, rtol=1e-12, atol=1e-12)
    assert max_abs_difference(value, expected) == 0.0


@pytest.mark.parametrize("norm", [False, True])
def test_jacobian_columns_match_pre_refactor_baseline(norm: bool) -> None:
    jac = an.world_jacobian(feats()["w_b0"], Scene.BOUNCE, 1, norm=norm)
    expected = subtree(BASE, "out/jac_n" if norm else "out/jac")
    assert_matches(jac, expected, rtol=1e-12, atol=1e-12)


def test_nuisance_columns_match_pre_refactor_baseline() -> None:
    value = an.nuisance_cols(feats()["w_b0"], 2)
    assert_matches(value, subtree(BASE, "out/nuis"), rtol=1e-12, atol=1e-12)
