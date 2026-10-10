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


def test_incline_slope_and_flat_phases_follow_coulomb_physics():
    """Slope phase: a1 = g(sin a - mu cos a). Flat run-out: deceleration ~ mu*g (corner transition adds bias, tolerance documented)."""
    for mu in (0.15, 0.25, 0.35):
        w = S.make_world("incline", S.derive_seed("t", "inc", 7)); r = S.simulate(w, {"mu": mu}, render=False)
        x, z, tr = r["pos"][:, 0], r["pos"][:, 2], r["transition_frame"]
        assert tr is not None and 12 < tr < 50
        e_ = tr - 10                                   # stop before the box's front edge reaches the flat (that biases the fit by ~7%)
        s = np.hypot(x[:e_ + 1] - x[0], z[:e_ + 1] - z[0]); a1 = 2 * s[e_] / T[e_] ** 2
        exp1 = 9.8 * (np.sin(w.angle) - mu * np.cos(w.angle)); assert abs(a1 - exp1) / exp1 < CFG["engine_tolerances"]["incline_slope_accel_rel"], (mu, a1, exp1)
        v = np.diff(x) * S.FPS; idx = np.arange(tr + 4, len(v)); mv = idx[v[idx] > 0.15]       # fit only frames where the box is still moving
        assert len(mv) >= 10, (mu, len(mv)); dec = -np.polyfit(mv / S.FPS, v[mv], 1)[0]
        assert abs(dec - mu * 9.8) / (mu * 9.8) < CFG["engine_tolerances"]["incline_flat_decel_rel"], (mu, dec)


def test_restitution_is_monotonic_and_near_e_squared():
    apex = {}
    for e in (0.4, 0.6, 0.8):
        w = S.make_world("bounce", 12); r = S.simulate(w, {"e": e}, render=False)
        apex[e] = (r["apex_after_first_impact"] - w.size) / w.h0
        assert abs(apex[e] - e * e) < CFG["engine_tolerances"]["restitution_apex_ratio_abs_vs_e2"]
    assert apex[0.4] < apex[0.6] < apex[0.8]


def test_interventions_change_the_relevant_physics():
    d = CFG["delta"]; thr = CFG["validity"]["min_peak_displacement_px"]
    for kind, par in (("freefall", "g"), ("bounce", "g"), ("bounce", "e"), ("incline", "g"), ("incline", "mu")):
        w = S.make_world(kind, S.derive_seed("t", kind, 3))
        hi = S.simulate(w, {par: S.BASE[par] + d[par]}, render=False)["pos"]; lo = S.simulate(w, {par: S.BASE[par] - d[par]}, render=False)["pos"]
        assert np.linalg.norm(S.project_pixels(hi, kind) - S.project_pixels(lo, kind), axis=1).max() >= thr, (kind, par)


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
    ok, why, _ = bm.check_world(w, "incline", "incline"); assert not ok and why in ("no_slide", "mu_no_effect", "g_no_effect", "bad_transition")
    w = S.make_world("freefall", 5); w.x0 = 40.0                                  # starts far off-screen
    ok, why, _ = bm.check_world(w, "freefall", "freefall"); assert not ok and why == "object_out_of_view"


def test_projection_matches_rendered_object_centroid_and_token_mask():
    for kind in ("freefall", "bounce", "incline"):
        w = S.make_world(kind, 5); out = S.simulate(w, nuis={"color": (1.0, 0.0, 1.0)})      # magenta object: unambiguous to segment
        px = S.project_pixels(out["pos"], kind); checked = 0
        for f in (0, 20, 50):
            fr = out["frames"][f].astype(int); m = (fr[..., 0] > 120) & (fr[..., 2] > 120) & (fr[..., 1] < 0.6 * np.minimum(fr[..., 0], fr[..., 2]))
            if m.sum() < 15: continue
            rr, cc = np.nonzero(m); checked += 1
            assert abs(cc.mean() - px[f, 0]) < 6 and abs(rr.mean() - px[f, 1]) < 6, (kind, f, (cc.mean(), rr.mean()), px[f])
        assert checked >= 2, kind
    mk = S.object_token_mask(out["pos"], "incline", dilate=1)
    assert mk.shape == (32 * 24 * 24,) and 0 < mk.mean() < 0.2


def test_supersampling_and_size_scale():
    w = S.make_world("freefall", 6); a = S.simulate(w, ss=2); b = S.simulate(w, ss=2)
    assert a["frames"].shape == (64, 384, 384, 3) and S.frames_sha256(a["frames"]) == S.frames_sha256(b["frames"])
    assert np.array_equal(S.simulate(w, render=False, size_scale=2.0)["pos"][:, :2], S.simulate(w, render=False)["pos"][:, :2])   # lateral motion unchanged


def test_incline_g_and_mu_are_distinguishable_in_trajectory_space():
    """v2 design goal: slope+flat gives two observables (a1, mu*g). The unit-column effective rank of (dg, dmu) must be well above the old slope-only value 1.0."""
    d = CFG["delta"]; ranks = []
    for k in range(5):
        w = S.make_world("incline", S.derive_seed("t", "rank", k))
        dg = S.simulate(w, {"g": 9.8 + d["g"]}, render=False)["pos"] - S.simulate(w, {"g": 9.8 - d["g"]}, render=False)["pos"]
        dm = S.simulate(w, {"mu": CFG["base_theta"]["mu"] + d["mu"]}, render=False)["pos"] - S.simulate(w, {"mu": CFG["base_theta"]["mu"] - d["mu"]}, render=False)["pos"]
        U = np.stack([dg.ravel() / np.linalg.norm(dg), dm.ravel() / np.linalg.norm(dm)], 1); sv = np.linalg.svd(U, compute_uv=False); p_ = sv / sv.sum()
        ranks.append(float(np.exp(-(p_ * np.log(p_)).sum())))
    assert min(ranks) >= CFG["validity"]["incline_min_effective_rank"], ranks
