import numpy as np
import pytest

from cg import simulate as sim
from cg.scenes import Param, Scene


@pytest.mark.parametrize("kind", list(Scene))
def test_simulator_events_and_determinism(kind: Scene) -> None:
    w = sim.make_world(kind, 5)
    a, b = sim.clip(w), sim.clip(w)
    assert np.array_equal(a.frames, b.frames)
    for p in sim.observable(kind):
        base = getattr(sim.BASE, p)
        step = getattr(sim.DELTA, p)
        hi, lo = (sim.clip(w, {p: base + d * step}).xz for d in (1, -1))
        assert np.abs(hi - lo).max() > 1e-3, (kind, p)


def test_restitution_needs_a_collision() -> None:
    assert Param.E not in sim.observable(Scene.FREEFALL)
    assert sim.clip(sim.make_world(Scene.BOUNCE, 5)).impacts


def test_overrides_do_not_mutate_the_base_parameters() -> None:
    before = sim.BASE
    sim.clip(sim.make_world(Scene.BOUNCE, 5), {Param.G: 12.0})
    assert before == sim.BASE
    assert sim.with_overrides(sim.BASE, {Param.MU: 0.5}).mu == 0.5
