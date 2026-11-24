import numpy as np
from golden_util import Flat, as_f32, assert_matches, load, max_abs_difference, subtree

from cg import gate1_analysis as ga
from cg.features import Features

BASE = load("gate1_analysis_golden")
INPUTS = subtree(BASE, "in")


def clips() -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for key, value in subtree(INPUTS, "/F").items():
        _, clip_id, tap = key.split("/")
        out.setdefault(clip_id, {})[tap] = value
    return out


def strings(prefix: str) -> dict[str, str]:
    return {k.removeprefix("/"): str(v) for k, v in subtree(INPUTS, prefix).items()}


def plan(*, full: bool) -> ga.Plan:
    delta_g = float(INPUTS["/delta/g"])
    if not full:
        return ga.Plan(delta_g, {}, {}, {}, {})
    half = {
        w: ga.HalfStep(
            str(INPUTS[f"/half/{w}/0"]), str(INPUTS[f"/half/{w}/1"]), float(INPUTS[f"/half/{w}/2"])
        )
        for w in ("wa_freefall", "wb_bounce", "wc_incline")
    }
    comp = ga.CompositionClips(
        *(str(INPUTS[f"/comp/wb_bounce/{k}"]) for k in ("base", "g+", "second+", "both+"))
    )
    return ga.Plan(delta_g, strings("/jitter"), strings("/jitter2"), half, {"wb_bounce": comp})


def test_analysis_matches_pre_refactor_baseline() -> None:
    for name, full in (("full", True), ("minimal", False)):
        value = ga.to_json(ga.analyse(clips(), plan(full=full)))
        expected: Flat = subtree(BASE, f"out/{name}")
        assert_matches(value, expected, rtol=1e-12, atol=1e-12)
        assert max_abs_difference(value, expected) == 0.0


def test_taps_match_pre_refactor_baseline() -> None:
    normed = as_f32(INPUTS["/feat/normed"])
    feature = Features(raw=as_f32(INPUTS["/feat/raw"]), normed=normed, final=normed[3])
    expected: Flat = {k: as_f32(v) for k, v in subtree(BASE, "out/taps").items()}
    assert_matches(ga.taps(feature), expected, rtol=0, atol=0)
