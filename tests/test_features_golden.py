import numpy as np
import torch
from golden_util import as_u8, assert_matches, load, max_abs_difference, subtree
from stubs import StubEnc

from cg import features as ft

BASE = load("features_golden")
INPUTS = subtree(BASE, "in")


def results() -> dict[str, object]:
    enc = StubEnc().eval()
    x = torch.from_numpy(INPUTS["/x"])
    mask = torch.from_numpy(INPUTS["/mask"])
    return {
        "f0": ft.extract(enc, x),
        "f1": ft.extract(enc, x, mask=mask),
        "f2": ft.extract(enc, x, hier=(3, 7, 11, 23)),
        "tensor": ft.to_tensor(as_u8(INPUTS["/frames"]), "cpu").numpy(),
    }


def test_matches_pre_refactor_baseline() -> None:
    for name, value in results().items():
        expected = subtree(BASE, f"out/{name}")
        assert_matches(value, expected, rtol=1e-6, atol=1e-6)
        assert max_abs_difference(value, expected) == 0.0
    assert np.asarray(BASE["meta/commit"]).size == 1
