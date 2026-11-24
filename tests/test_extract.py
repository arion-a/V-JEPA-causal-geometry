import numpy as np
import torch

from cg import extract as ex


def test_layer_means_are_deterministic_and_shaped() -> None:
    m = ex.ToyVideoEncoder().eval()
    x = torch.rand(2, 16, 3, 64, 64)
    a, b = ex.layer_means(m, m.blocks, x), ex.layer_means(m, m.blocks, x)
    assert a.shape[:2] == (2, 4)
    assert np.array_equal(a, b)


def test_building_the_encoder_leaves_the_global_random_state_untouched() -> None:
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    ex.ToyVideoEncoder(seed=7)
    assert torch.equal(torch.rand(3), expected)


def test_encoder_weights_depend_only_on_the_seed() -> None:
    a, b = ex.ToyVideoEncoder(seed=1), ex.ToyVideoEncoder(seed=1)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True))
    assert not torch.equal(next(a.parameters()), next(ex.ToyVideoEncoder(seed=2).parameters()))


def test_hooks_do_not_outlive_the_call() -> None:
    m = ex.ToyVideoEncoder().eval()
    ex.layer_means(m, m.blocks, torch.rand(1, 16, 3, 32, 32))
    assert all(len(b._forward_hooks) == 0 for b in m.blocks)
