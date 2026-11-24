import torch
from torch import Tensor, nn


class ResBlock(nn.Module):
    """Like the real block: returns the residual stream x + f(x)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x: Tensor) -> Tensor:
        return x + torch.tanh(self.lin(x))


class StubEnc(nn.Module):
    """The encoder API cg.features uses: .blocks, .norms_block, forward -> norms_block[-1](x)."""

    def __init__(self, depth: int = 24, d: int = 8) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.embed = nn.Linear(3, d)
            self.blocks = nn.ModuleList([ResBlock(d) for _ in range(depth)])
            self.norms_block = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])

    def forward(self, x: Tensor) -> Tensor:
        t = self.embed(x.flatten(2).transpose(1, 2))
        for b in self.blocks:
            t = b(t)
        out: Tensor = self.norms_block[-1](t)
        return out
