import numpy as np
from golden_util import assert_matches, load, max_abs_difference, subtree

from cg import simulate as sim
from cg.scenes import Param, Scene

BASE = load("simulate_golden")


def results() -> dict[str, object]:
    out: dict[str, object] = {}
    for kind in Scene:
        for seed in (5, 9):
            w = sim.make_world(kind, seed)
            plain = sim.clip(w)
            theta = {"g": 11.0, "e": 0.4, "mu": 0.3}
            overrides = {p: v for p, v in zip(Param, theta.values(), strict=True)}
            noisy = sim.clip(w, overrides, sim.Nuisance(1 - w.color, 0.3), res=32)
            out[f"{kind}{seed}"] = {
                "world": [w.x0, w.z0, w.vx, w.vz, w.angle, w.color],
                "clip0": plain.frames,
                "xz0": plain.xz,
                "hits0": np.array(plain.impacts, float),
                "clip1": noisy.frames,
                "xz1": noisy.xz,
                "hits1": np.array(noisy.impacts, float),
            }
    return out


def test_matches_pre_refactor_baseline() -> None:
    value = results()
    expected = subtree(BASE, "out")
    assert_matches(value, expected, rtol=1e-12, atol=1e-12)
    assert max_abs_difference(value, expected) == 0.0
