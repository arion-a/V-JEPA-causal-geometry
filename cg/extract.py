"""Per-block activation means for any torch encoder (synthetic scaffold pipeline)."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn


class ToyVideoEncoder(nn.Module):
    """Random-weight stand-in with a V-JEPA-like interface for smoke tests only; not a result."""

    def __init__(self, d: int = 64, depth: int = 4, patch: int = 8, seed: int = 0) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.pe = nn.Conv3d(3, d, (2, patch, patch), (2, patch, patch))
            self.blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(d, 2, 4 * d, 0.0, batch_first=True, norm_first=True)
                    for _ in range(depth)
                ]
            )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, 3, H, W] -> tokens [B, N, d]."""
        tokens = self.pe(x.transpose(1, 2)).flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        out: Tensor = tokens
        return out


def _token_mean(out: Tensor | tuple[Tensor, ...]) -> Tensor:
    tokens = out[0] if isinstance(out, tuple) else out
    return tokens.float().mean(1).cpu()


@torch.no_grad()
def layer_means(model: nn.Module, blocks: nn.ModuleList, x: Tensor) -> NDArray[np.float32]:
    """Token-mean output of every block, [B, n_blocks, d]; call model.eval() first."""
    acts: list[Tensor] = []
    hooks = [
        b.register_forward_hook(lambda _m, _i, out: acts.append(_token_mean(out))) for b in blocks
    ]
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()
    return np.stack([a.numpy() for a in acts], 1)
