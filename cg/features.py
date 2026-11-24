"""Pooled layer features from a V-JEPA 2.1 style encoder, computed in fp32.

Taps: every raw block output (before norm, mean over tokens) and the hierarchical taps (blocks in
``hier``, each through its own ``norms_block`` entry, then mean over tokens). With a token mask the
same taps are also averaged over the masked tokens only ("window").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NotRequired, Protocol, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_HIER = (5, 11, 17, 23)

type Float32 = NDArray[np.float32]


class Features(TypedDict):
    """Shapes: raw (n_blocks, d), normed (n_hier, d), final (d,); window taps only with a mask."""

    raw: Float32
    normed: Float32
    final: Float32
    raw_win: NotRequired[Float32]
    normed_win: NotRequired[Float32]


class Encoder(Protocol):
    blocks: nn.ModuleList
    norms_block: nn.ModuleList

    def __call__(self, x: Tensor) -> Tensor: ...


def to_tensor(frames_u8: NDArray[np.uint8], device: str = "cuda") -> Tensor:
    """uint8 [T, H, W, 3] -> ImageNet-normalised float [1, 3, T, H, W]."""
    x = torch.from_numpy(np.ascontiguousarray(frames_u8)).to(device).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 1, 3)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 1, 3)
    return ((x - mean) / std).permute(3, 0, 1, 2)[None].contiguous()


def _pool(tokens: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
    """Global mean and, with a mask, masked mean; both as float32 CPU vectors."""
    window = None if mask is None else tokens[0, mask].mean(0).float().cpu()
    return tokens.mean(1)[0].float().cpu(), window


@dataclass
class _Recorder:
    """Collects pooled block outputs while a forward pass runs."""

    hier: tuple[int, ...]
    norms: nn.ModuleList
    mask: Tensor | None
    raw: list[Tensor] = field(default_factory=list)
    raw_win: list[Tensor] = field(default_factory=list)
    normed: dict[int, Tensor] = field(default_factory=dict)
    normed_win: dict[int, Tensor] = field(default_factory=dict)

    def hook(self, block: int) -> Callable[[nn.Module, object, Tensor | tuple[Tensor, ...]], None]:
        def record(_module: nn.Module, _inputs: object, out: Tensor | tuple[Tensor, ...]) -> None:
            tokens = out[0] if isinstance(out, tuple) else out
            pooled, window = _pool(tokens, self.mask)
            self.raw.append(pooled)
            if window is not None:
                self.raw_win.append(window)
            if block in self.hier:
                self._record_normed(block, tokens)

        return record

    def _record_normed(self, block: int, tokens: Tensor) -> None:
        normed = self.norms[list(self.hier).index(block)](tokens)
        self.normed[block], window = _pool(normed, self.mask)
        if window is not None:
            self.normed_win[block] = window


def _assemble(rec: _Recorder, final: Tensor) -> Features:
    out = Features(
        raw=torch.stack(rec.raw).numpy(),
        normed=torch.stack([rec.normed[i] for i in rec.hier]).numpy(),
        final=final.mean(1)[0].float().cpu().numpy(),
    )
    if rec.mask is not None:
        out["raw_win"] = torch.stack(rec.raw_win).numpy()
        out["normed_win"] = torch.stack([rec.normed_win[i] for i in rec.hier]).numpy()
    return out


@torch.no_grad()
def extract(
    enc: Encoder,
    x: Tensor,
    hier: tuple[int, ...] = DEFAULT_HIER,
    mask: Tensor | None = None,
) -> Features:
    """Pooled taps for one clip ``x`` [1, 3, T, H, W]; ``mask`` is a bool tensor over tokens."""
    if any(not 0 <= b < len(enc.blocks) for b in hier):
        raise ValueError(f"hier blocks {hier} outside 0..{len(enc.blocks) - 1}")
    rec = _Recorder(hier, enc.norms_block, mask)
    hooks = [block.register_forward_hook(rec.hook(i)) for i, block in enumerate(enc.blocks)]
    try:
        final = enc(x)
    finally:
        for h in hooks:
            h.remove()
    return _assemble(rec, final)
