import numpy as np
import pytest

from cg import gate1_analysis as ga

DELTA_G = 0.5


def linear_clips() -> dict[str, dict[str, np.ndarray]]:
    a = np.random.default_rng(0).normal(size=(16, 3))

    def clip(g: float = 0.0, e: float = 0.0, cam: float = 0.0) -> dict[str, np.ndarray]:
        return {t: a @ np.array([g, e, cam]) + 50.0 for t in ga.RAW + ga.NORM}

    return {
        "w1__base": clip(),
        "w1__g+": clip(g=0.5),
        "w1__g-": clip(g=-0.5),
        "w1__e+": clip(e=0.04),
        "w1__e-": clip(e=-0.04),
        "w1__cam": clip(cam=0.3),
        "w1__gp": clip(g=0.25),
        "w1__gm": clip(g=-0.25),
        "w1__jit": clip(cam=0.003),
        "w2c__base": clip(),
        "w2c__g+": clip(g=0.5),
        "w2c__second+": clip(e=0.04),
        "w2c__both+": clip(g=0.5, e=0.04),
    }


PLAN = ga.Plan(
    delta_g=DELTA_G,
    jitter={"w1": "w1__jit"},
    jitter2={},
    half={"w1": ga.HalfStep("w1__gp", "w1__gm", 0.25)},
    composition={"w2c": ga.CompositionClips("w2c__base", "w2c__g+", "w2c__second+", "w2c__both+")},
)


def test_analysis_recovers_known_linear_structure() -> None:
    result = ga.analyse(linear_clips(), PLAN)
    t = result["raw3"]["w1"]
    assert t.linearity_cos_g is not None and abs(t.linearity_cos_g - 1) < 1e-9
    assert t.linearity_norm_ratio_g is not None and abs(t.linearity_norm_ratio_g - 1) < 1e-9
    assert t.snr_g is not None and t.snr_g > 100
    assert t.jitter_floor is not None and t.signal_g is not None and t.jitter_floor < t.signal_g
    comp = result["raw3"]["w2c"].composition_error
    assert comp is not None and comp < 1e-20


def test_json_layout_omits_statistics_a_world_does_not_have() -> None:
    layout = ga.to_json(ga.analyse(linear_clips(), PLAN))
    assert set(layout["raw3"]["w2c"]) == {"composition_error"}
    assert "jitter_floor" in layout["raw3"]["w1"]
    assert list(layout["raw3"]["w1"])[:2] == ["signal_g", "signal_e"]


def test_world_order_is_deterministic() -> None:
    assert list(ga.analyse(linear_clips(), PLAN)["raw0"]) == ["w1", "w2c"]


def test_plan_referring_to_missing_clips_fails_loudly() -> None:
    bad = ga.Plan(DELTA_G, {"w1": "w1__nope"}, {}, {}, {})
    with pytest.raises(ValueError):
        ga.analyse(linear_clips(), bad)
    no_pair = ga.Plan(DELTA_G, {}, {}, {"w2c": ga.HalfStep("w1__gp", "w1__gm", 0.25)}, {})
    with pytest.raises(ValueError):
        ga.analyse({k: v for k, v in linear_clips().items() if k != "w2c__g-"}, no_pair)
