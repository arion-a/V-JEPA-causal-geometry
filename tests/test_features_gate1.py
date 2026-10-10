import numpy as np, torch, torch.nn as nn
from cg import features as FT, gate1_analysis as GA


class ResBlock(nn.Module):
    """Like the real Block: returns the residual stream x + f(x)."""
    def __init__(s, d): super().__init__(); s.lin = nn.Linear(d, d)
    def forward(s, x): return x + torch.tanh(s.lin(x))


class StubEnc(nn.Module):
    """Mimics the V-JEPA 2.1 encoder API used by cg.features: .blocks, .norms_block, forward -> norms_block[-1](x)."""
    def __init__(s, depth=24, d=8):
        super().__init__(); torch.manual_seed(0)
        s.embed = nn.Linear(3, d); s.blocks = nn.ModuleList([ResBlock(d) for _ in range(depth)])
        s.norms_block = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
    def forward(s, x):
        t = s.embed(x.flatten(2).transpose(1, 2))
        for b in s.blocks: t = b(t)
        return s.norms_block[-1](t)


def test_extract_shapes_and_final_equals_norm23_tap():
    enc = StubEnc().eval(); x = torch.rand(1, 3, 2, 4, 4)
    f = FT.extract(enc, x)
    assert f["raw"].shape == (24, 8) and f["normed"].shape == (4, 8)
    assert np.allclose(f["final"], f["normed"][3], atol=1e-6)                     # default output == block-23 tap after its own norm
    assert np.array_equal(FT.extract(enc, x)["raw"], f["raw"])                     # deterministic


def test_to_tensor_normalisation_and_layout():
    fr = np.full((4, 6, 6, 3), 255, np.uint8); x = FT.to_tensor(fr, "cpu")
    assert x.shape == (1, 3, 4, 6, 6) and abs(float(x[0, 0].mean()) - (1 - 0.485) / 0.229) < 1e-5


def test_analysis_recovers_known_linear_structure():
    rng = np.random.default_rng(0); A = rng.normal(size=(16, 3)); dl = dict(g=0.5, e=0.04, mu=0.02)
    f = lambda th: A @ th + 50.0                               # linear encoder, big baseline
    mk = lambda g=0.0, e=0.0, cam=0.0: {t: f(np.array([g, e, cam])) for t in GA.RAW + GA.NORM}
    F = {"w1__base": mk(), "w1__g+": mk(g=.5), "w1__g-": mk(g=-.5), "w1__e+": mk(e=.04), "w1__e-": mk(e=-.04),
         "w1__cam": mk(cam=.3), "w1__gp": mk(g=.25), "w1__gm": mk(g=-.25), "w1__jit": mk(cam=.003),
         "w2c__base": mk(), "w2c__g+": mk(g=.5), "w2c__second+": mk(e=.04), "w2c__both+": mk(g=.5, e=.04)}
    r = GA.analyse(F, dl, jitter={"w1": "w1__jit"}, half={("w1", "g"): ("w1__gp", "w1__gm", 0.25)}, comp={"w2c": {"base": "w2c__base", "g+": "w2c__g+", "second+": "w2c__second+", "both+": "w2c__both+"}})
    t = r["raw3"]["w1"]
    assert abs(t["linearity_cos_g"] - 1) < 1e-9 and abs(t["linearity_norm_ratio_g"] - 1) < 1e-9     # exactly linear -> delta/2 agrees
    assert t["snr_g"] > 100 and t["jitter_floor"] < t["signal_g"]
    assert r["raw3"]["w2c"]["composition_error"] < 1e-20


def test_masked_window_taps():
    enc = StubEnc().eval(); x = torch.rand(1, 3, 2, 4, 4); n = 8   # stub tokens = T*H*W = 2*4*4 = 32 -> use first 8 as window
    mask = torch.zeros(32, dtype=torch.bool); mask[:n] = True
    f = FT.extract(enc, x, mask=mask)
    assert f["raw_win"].shape == (24, 8) and f["normed_win"].shape == (4, 8) and not np.allclose(f["raw_win"], f["raw"])
