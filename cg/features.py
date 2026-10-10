"""Global-pooled layer features from a V-JEPA 2.1 style encoder (fp32). Taps: all raw block outputs (pre-norm, mean over tokens)
and the model's hierarchical taps (blocks in `hier`, each passed through its own norms_block[k], then mean over tokens)."""
import numpy as np
import torch

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)     # [VERIFY against repo preprocessor at Gate 1]


def to_tensor(frames_u8, device="cuda"):
    """uint8 [T,H,W,3] -> normalised float [1,3,T,H,W]."""
    x = torch.from_numpy(np.ascontiguousarray(frames_u8)).to(device).float() / 255.0
    m = torch.tensor(IMNET_MEAN, device=device).view(1, 1, 1, 3); s = torch.tensor(IMNET_STD, device=device).view(1, 1, 1, 3)
    return ((x - m) / s).permute(3, 0, 1, 2)[None].contiguous()


@torch.no_grad()
def extract(enc, x, hier=(5, 11, 17, 23), mask=None):
    """mask: optional bool tensor [N_tokens]; adds masked-mean ('window') taps next to the global-mean ones."""
    raw, normed, hooks, rawm, normm = [], {}, [], [], {}
    for i, b in enumerate(enc.blocks):
        def h(m, inp, out, i=i):
            o = out[0] if isinstance(out, tuple) else out
            raw.append(o.mean(1)[0].float().cpu())
            if mask is not None: rawm.append(o[0, mask].mean(0).float().cpu())
            if i in hier:
                no = enc.norms_block[list(hier).index(i)](o); normed[i] = no.mean(1)[0].float().cpu()
                if mask is not None: normm[i] = no[0, mask].mean(0).float().cpu()
        hooks.append(b.register_forward_hook(h))
    out = enc(x)
    for hk in hooks: hk.remove()
    res = dict(raw=torch.stack(raw).numpy(), normed=torch.stack([normed[i] for i in hier]).numpy(), final=out.mean(1)[0].float().cpu().numpy())
    if mask is not None: res.update(raw_win=torch.stack(rawm).numpy(), normed_win=torch.stack([normm[i] for i in hier]).numpy())
    return res
