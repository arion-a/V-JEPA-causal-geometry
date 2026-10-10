import json, os
import numpy as np
import pytest
pytest.importorskip("pybullet")
from cg import sim_pb as S

CFG = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "pilot.json")))
T = np.arange(S.N_FRAMES) / S.FPS


def test_free_fall_matches_closed_form_and_has_no_contact():
    w = S.make_world("freefall", 11); a = S.simulate(w, render=False)
    z = w.h0 + w.size - 0.5 * 9.8 * T ** 2
    assert np.abs(a["pos"][:, 2] - z).max() < CFG["engine_tolerances"]["freefall_vs_closed_form_m"]
    assert a["impacts"] == [] and a["in_view"]


def test_incline_acceleration_matches_coulomb_formula():
    for mu in (0.15, 0.25, 0.35):
        w = S.make_world("incline", 13); r = S.simulate(w, {"mu": mu}, render=False)
        d = np.linalg.norm(r["pos"] - r["pos"][0], axis=1); acc = 2 * d[-1] / T[-1] ** 2
        exp = 9.8 * (np.sin(w.angle) - mu * np.cos(w.angle))
        assert abs(acc - exp) / exp < CFG["engine_tolerances"]["incline_accel_rel"]


def test_restitution_is_monotonic_and_near_e_squared():
    apex = {}
    for e in (0.4, 0.6, 0.8):
        w = S.make_world("bounce", 12); r = S.simulate(w, {"e": e}, render=False)
        apex[e] = (r["apex_after_first_impact"] - w.size) / w.h0
        assert abs(apex[e] - e * e) < CFG["engine_tolerances"]["restitution_apex_ratio_abs_vs_e2"]
    assert apex[0.4] < apex[0.6] < apex[0.8]


def test_interventions_change_the_relevant_physics():
    d = CFG["delta"]; thr = CFG["validity"]["min_peak_displacement_m"]
    for kind, par in (("freefall", "g"), ("bounce", "g"), ("bounce", "e"), ("incline", "g"), ("incline", "mu")):
        w = S.make_world(kind, S.derive_seed("t", kind, 3))
        hi = S.simulate(w, {par: S.BASE[par] + d[par]}, render=False)["pos"]; lo = S.simulate(w, {par: S.BASE[par] - d[par]}, render=False)["pos"]
        assert np.linalg.norm(hi - lo, axis=1).max() > thr, (kind, par)


def test_render_deterministic_and_matched_pairs_share_first_frame():
    w = S.make_world("bounce", 5)
    a, b = S.simulate(w), S.simulate(w)
    assert S.frames_sha256(a["frames"]) == S.frames_sha256(b["frames"]) and a["frames"].shape == (64, 384, 384, 3)
    g_hi = S.simulate(w, {"g": 10.3})
    assert np.array_equal(a["frames"][0], g_hi["frames"][0])                  # same initial state/appearance in a pair
    assert not np.array_equal(a["frames"][40], g_hi["frames"][40])            # physics differs later


def test_nuisance_changes_appearance_not_physics():
    w = S.make_world("freefall", 6); base = S.simulate(w)
    cam = S.simulate(w, nuis={"cam": 0.3}); col = S.simulate(w, nuis={"color": tuple(1 - c for c in w.color)})
    assert np.array_equal(base["pos"], cam["pos"]) and np.array_equal(base["pos"], col["pos"])
    assert not np.array_equal(base["frames"][10], cam["frames"][10]) and not np.array_equal(base["frames"][10], col["frames"][10])


def test_seed_derivation_is_stable():
    assert S.derive_seed("r", "role", 1) == S.derive_seed("r", "role", 1) != S.derive_seed("r", "role", 2)


def test_validity_gate_can_reject():
    import importlib.util, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    spec = importlib.util.spec_from_file_location("bm", os.path.join(os.path.dirname(__file__), "..", "scripts", "build_manifest.py"))
    bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)
    w = S.make_world("bounce", 5); w.h0 = 6.0                                   # drops too high: no impact inside the 1 s clip
    ok, why, _ = bm.check_world(w, "bounce", "bounce"); assert not ok and why in ("no_impact", "object_out_of_view")
    w = S.make_world("incline", 5); w.angle = np.radians(5)                       # shallow slope: object barely slides
    ok, why, _ = bm.check_world(w, "incline", "incline"); assert not ok and why in ("no_slide", "mu_no_effect", "g_no_effect")
    w = S.make_world("freefall", 5); w.x0 = 40.0                                  # starts far off-screen
    ok, why, _ = bm.check_world(w, "freefall", "freefall"); assert not ok and why == "object_out_of_view"
