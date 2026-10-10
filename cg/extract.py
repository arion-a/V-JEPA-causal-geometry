"""Layer-activation extraction. `HookedEncoder` wraps any torch encoder exposing `.blocks`;
`load_vjepa21()` is the Gate-0 loader for the real model (must be verified on GPU before use)."""
import hashlib, numpy as np, torch, torch.nn as nn


class HookedEncoder:
    """Captures mean-pooled token output of each block (pooled-token analysis; token-level is a follow-up)."""
    def __init__(self, enc, blocks):
        self.enc, self.acts = enc.eval(), []
        for p in enc.parameters(): p.requires_grad_(False)
        for b in blocks: b.register_forward_hook(lambda m, i, o: self.acts.append((o[0] if isinstance(o, tuple) else o).float().mean(1).cpu()))

    @torch.no_grad()
    def __call__(self, x):
        self.acts.clear(); self.enc(x)
        return np.stack([a.numpy() for a in self.acts], 1)       # [B, L, D]


class ToyVideoEncoder(nn.Module):
    """Stand-in with a V-JEPA-like interface for pipeline smoke tests ONLY. Random weights; not a result."""
    def __init__(self, d=64, depth=4, patch=8, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.pe = nn.Conv3d(3, d, (2, patch, patch), (2, patch, patch))
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, 2, 4 * d, 0.0, batch_first=True, norm_first=True) for _ in range(depth)])
    def forward(self, x):                                         # [B,T,3,H,W]
        t = self.pe(x.transpose(1, 2)).flatten(2).transpose(1, 2)
        for b in self.blocks: t = b(t)
        return t


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()


def load_vjepa21(repo="facebookresearch/vjepa2", name="vjepa2_1_vit_large", device="cuda"):
    """GATE 0: verify entrypoint name, checkpoint hash and block hook mapping against the official repo
    before any rental is spent on Gate 2. Returns (HookedEncoder, info). Not exercised in CI."""
    out = torch.hub.load(repo, name)
    enc = out[0] if isinstance(out, tuple) else out
    enc = enc.to(device)
    return HookedEncoder(enc, enc.blocks), dict(repo=repo, entry=name, n_blocks=len(enc.blocks))
