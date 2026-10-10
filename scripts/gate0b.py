"""GATE 0b: close the open Gate 0 items. (1) SHA256 of the real checkpoint, (2) token->space map using EARLY layers,
(3) fp32-vs-bf16 error per layer, (4) memory/throughput at the native 64-frame input. Writes $CG_ROOT/results/gate0b/gate0b.json."""
import glob, hashlib, json, os, sys, time, traceback
ROOT = os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry"); VENDOR = f"{ROOT}/vendor/vjepa2"
OUT = f"{ROOT}/results/gate0b"; os.makedirs(OUT, exist_ok=True)
R = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": {}}
M = {}


def step(name):
    def deco(fn):
        t = time.time(); print("===", name, flush=True)
        try: R["steps"][name] = dict(ok=True, result=fn(), seconds=round(time.time() - t, 1))
        except Exception: R["steps"][name] = dict(ok=False, error=traceback.format_exc()[-1500:])
        print(json.dumps(R["steps"][name], default=str)[:1500], flush=True); json.dump(R, open(f"{OUT}/gate0b.json", "w"), indent=2, default=str)
    return deco


@step("checkpoint_sha256")
def _():
    out = []
    for p in glob.glob(f"{ROOT}/cache/torch/hub/checkpoints/*.pt"):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
        out.append(dict(file=os.path.basename(p), bytes=os.path.getsize(p), sha256=h.hexdigest()))
    return out


@step("load")
def _():
    import torch, importlib
    sys.path.insert(0, VENDOR); importlib.import_module("src.hub.backbones").VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
    hub = importlib.import_module("hubconf"); enc = hub.vjepa2_1_vit_large_384()
    enc = enc[0] if isinstance(enc, (tuple, list)) else enc
    M["enc"] = enc.cuda().eval()
    return dict(params_M=round(sum(p.numel() for p in enc.parameters()) / 1e6, 1), norms_block=type(enc.norms_block).__name__ if hasattr(enc, "norms_block") else None,
                norms_block_len=len(enc.norms_block) if hasattr(enc, "norms_block") else None)


def run_collect(x, dtype=None):
    import torch
    enc, acts = M["enc"], []
    hs = [b.register_forward_hook(lambda m, i, o: acts.append((o[0] if isinstance(o, tuple) else o).detach().float().cpu())) for b in enc.blocks]
    with torch.no_grad():
        if dtype is None: enc(x)
        else:
            with torch.autocast("cuda", dtype=dtype): enc(x)
    for h in hs: h.remove()
    return acts


@step("token_map_early_layers")
def _():
    """Bright 96x96 px square (6x6 patches) at pixel rows 96-191, cols 240-335 in all frames -> patch rows 6-11, cols 15-20."""
    import torch
    S, T = 384, 16
    base = torch.full((1, 3, T, S, S), .2, device="cuda"); obj = base.clone(); obj[:, :, :, 96:192, 240:336] = 1.0
    a, b = run_collect(base), run_collect(obj); nt, hp = T // 2, S // 16; res = {}
    for L in (0, 1, 3, 7, 11, 17, 23):
        d = (a[L][0] - b[L][0]).norm(dim=-1).view(nt, hp, hp)
        mask = torch.zeros(nt, hp, hp, dtype=torch.bool); mask[:, 6:12, 15:21] = True
        res[f"block{L}"] = dict(frac_mass_inside=round(float(d[mask].sum() / d.sum()), 3), inside_over_outside=round(float(d[mask].mean() / d[~mask].mean()), 2),
                                area_fraction=round(float(mask.float().mean()), 3))
    return dict(layers=res, note="localised if frac_mass_inside >> area_fraction in early blocks; ordering (t,h,w) confirmed then")


@step("fp32_vs_bf16_per_layer")
def _():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    g = torch.Generator(device="cpu").manual_seed(0); x = torch.rand(1, 3, 16, 384, 384, generator=g).cuda()
    f1, f2, bf = run_collect(x), run_collect(x), run_collect(x, torch.bfloat16)
    rel = lambda p, q: float((p - q).norm() / p.norm())
    return dict(fp32_repeat_max_abs=max(float((p - q).abs().max()) for p, q in zip(f1, f2)),
                bf16_rel_err_by_block=[round(rel(p, q), 5) for p, q in zip(f1, bf)])


@step("native_64_frames_memory_throughput")
def _():
    import torch
    out = {}
    for T, B in ((64, 1), (64, 2), (16, 4)):
        x = torch.rand(B, 3, T, 384, 384, device="cuda"); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            M["enc"](x); torch.cuda.synchronize(); t = time.time(); M["enc"](x); torch.cuda.synchronize()
        out[f"T{T}_B{B}_fp32"] = dict(clips_per_s=round(B / (time.time() - t), 2), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    return out


R["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); json.dump(R, open(f"{OUT}/gate0b.json", "w"), indent=2, default=str); print("GATE0B DONE")
