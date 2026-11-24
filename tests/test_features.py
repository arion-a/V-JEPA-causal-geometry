import numpy as np
import pytest
import torch
from stubs import StubEnc

from cg import features as ft


def test_extract_shapes_and_final_equals_norm23_tap() -> None:
    enc = StubEnc().eval()
    x = torch.rand(1, 3, 2, 4, 4)
    f = ft.extract(enc, x)
    assert f["raw"].shape == (24, 8)
    assert f["normed"].shape == (4, 8)
    assert np.allclose(f["final"], f["normed"][3], atol=1e-6)
    assert np.array_equal(ft.extract(enc, x)["raw"], f["raw"])


def test_to_tensor_normalisation_and_layout() -> None:
    frames = np.full((4, 6, 6, 3), 255, np.uint8)
    x = ft.to_tensor(frames, "cpu")
    assert x.shape == (1, 3, 4, 6, 6)
    assert abs(float(x[0, 0].mean()) - (1 - 0.485) / 0.229) < 1e-5


def test_masked_window_taps() -> None:
    enc = StubEnc().eval()
    x = torch.rand(1, 3, 2, 4, 4)
    mask = torch.zeros(32, dtype=torch.bool)
    mask[:8] = True
    f = ft.extract(enc, x, mask=mask)
    assert f["raw_win"].shape == (24, 8)
    assert f["normed_win"].shape == (4, 8)
    assert not np.allclose(f["raw_win"], f["raw"])
    assert "raw_win" not in ft.extract(enc, x)


def test_hooks_are_removed_even_when_the_forward_pass_fails() -> None:
    enc = StubEnc().eval()
    with pytest.raises(RuntimeError):
        ft.extract(enc, torch.rand(1, 5, 2, 4, 4))
    assert all(len(b._forward_hooks) == 0 for b in enc.blocks)


def test_hier_blocks_outside_the_encoder_are_rejected() -> None:
    with pytest.raises(ValueError):
        ft.extract(StubEnc(depth=4).eval(), torch.rand(1, 3, 2, 4, 4))
