import torch
from golden_util import assert_matches, load, max_abs_difference, subtree

from cg import extract as ex

BASE = load("extract_golden")


def test_activations_match_pre_refactor_baseline() -> None:
    x = torch.from_numpy(subtree(BASE, "in")["/x"])
    small = ex.ToyVideoEncoder().eval()
    other = ex.ToyVideoEncoder(d=32, depth=3, patch=16, seed=3).eval()
    value = {
        "acts": ex.layer_means(small, small.blocks, x),
        "acts2": ex.layer_means(other, other.blocks, x),
    }
    expected = subtree(BASE, "out")
    assert_matches(value, expected, rtol=1e-6, atol=1e-6)
    assert max_abs_difference(value, expected) == 0.0
