import numpy as np
import pytest
from golden_util import assert_matches, load, max_abs_difference, subtree

pytest.importorskip("pybullet")
from cg import sim_pb as sp  # noqa: E402
from cg.scenes import Scene  # noqa: E402

BASE = load("sim_pb_golden")
CASES: list[tuple[Scene, int, sp.ThetaOverride, sp.NuisanceSpec | None]] = [
    (Scene.FREEFALL, 11, {}, None),
    (Scene.FREEFALL, 12, {"g": 8.8, "vz": -0.3}, {"cam": 0.3}),
    (Scene.BOUNCE, 12, {"e": 0.4}, None),
    (Scene.BOUNCE, 14, {"g": 10.8, "e": 0.65}, {"color": (0.1, 0.5, 0.9)}),
    (Scene.INCLINE, 7, {"mu": 0.3}, None),
    (Scene.INCLINE, 9, {"g": 9.0, "mu": 0.26}, {"cam": 0.3}),
    (Scene.INCLINE, 9, {}, None),
]


def none_as_minus_one(x: float | None) -> np.ndarray:
    return np.array([-1.0 if x is None else x])


def results() -> dict[str, object]:
    out: dict[str, object] = {
        "seeds": [sp.derive_seed("root", r, i) for r in "ab" for i in (0, 1, 7)]
    }
    for i, (kind, seed, theta, nuis) in enumerate(CASES):
        w = sp.make_world(kind, seed)
        r = sp.simulate(w, theta, nuis, render=False)
        out[f"sim{i}"] = {
            "world": [w.h0, w.x0, w.vx, w.angle, w.size, np.array(w.color)],
            "pos": r["pos"],
            "impacts": np.array(r["impacts"], float),
            "slide": none_as_minus_one(r["slide_m"]),
            "trans": none_as_minus_one(r["transition_frame"]),
            "apex": none_as_minus_one(r["apex_after_first_impact"]),
            "in_view": np.array([float(r["in_view"])]),
        }
        if kind is Scene.FREEFALL or i == 4:
            rendered = sp.simulate(w, theta, nuis, render=True)
            assert rendered["frames"] is not None
            out[f"render{i}"] = {
                "sha": sp.frames_sha256(rendered["frames"]),
                "pos": rendered["pos"],
            }
        out[f"mask{i}"] = sp.object_token_mask(r["pos"], kind)
        out[f"px{i}"] = sp.project_pixels(r["pos"], kind, 0.1)
    w = sp.make_world(Scene.BOUNCE, 12)
    scaled = sp.simulate(w, None, None, render=False, size_scale=1.5)
    out["scaled"] = scaled["pos"]
    out["mask_d0"] = sp.object_token_mask(
        scaled["pos"], Scene.BOUNCE, dilate=0, patch=16, tubelet=2
    )
    return out


def test_matches_pre_refactor_baseline() -> None:
    value = results()
    expected = subtree(BASE, "out")
    assert_matches(value, expected, rtol=1e-12, atol=1e-12)
    assert max_abs_difference(value, expected) == 0.0
