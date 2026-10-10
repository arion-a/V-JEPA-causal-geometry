"""GATE 0 (runs ON the GPU pod, non-interactive). Verifies facts the plan marks [VERIFY]; writes /workspace/out/*.
Does not assume the V-JEPA 2.1 entry point: it reads the official repo's hubconf and tries what it finds.
Output: $CG_ROOT/results/gate0/gate0.json (+ gate0.log when run via pod_bootstrap.sh). Everything lives on the persistent volume."""
import hashlib, json, os, re, subprocess, sys, time, traceback

ROOT = os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry")   # persistent network volume
VENDOR = f"{ROOT}/vendor/vjepa2"
OUT = f"{ROOT}/results/gate0"; os.makedirs(OUT, exist_ok=True)
R = dict(started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), steps={})


def save(): json.dump(R, open(f"{OUT}/gate0.json", "w"), indent=2, default=str)


def step(name):
    def deco(fn):
        t = time.time(); print(f"\n=== {name}", flush=True)
        try: R["steps"][name] = dict(ok=True, result=fn(), seconds=round(time.time() - t, 1))
        except Exception as e:
            R["steps"][name] = dict(ok=False, error=repr(e), tb=traceback.format_exc()[-2500:]); print(R["steps"][name]["tb"], flush=True)
        print(json.dumps(R["steps"][name], default=str)[:1800], flush=True); save()
        return fn
    return deco


def sh(c, timeout=900):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout); return p.returncode, (p.stdout + p.stderr)[-4000:]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


@step("env")
def _():
    import torch
    return dict(torch=torch.__version__, cuda=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
                nvidia_smi=sh("nvidia-smi --query-gpu=driver_version,name --format=csv,noheader")[1].strip())


@step("clone_repo")
def _():
    sh("pip install -q timm einops 2>&1 | tail -1")
    rc, o = sh(f"mkdir -p {ROOT}/vendor && cd {ROOT}/vendor && rm -rf vjepa2 && git clone --depth 1 https://github.com/facebookresearch/vjepa2.git 2>&1 | tail -2")
    rc2, commit = sh(f"cd {VENDOR} && git rev-parse HEAD && git log -1 --format=%cd")
    lic = open(f"{VENDOR}/LICENSE").read()[:120] if os.path.exists(f"{VENDOR}/LICENSE") else None
    return dict(rc=rc, commit=commit.strip(), license_head=lic)


@step("discover_entrypoints")
def _():
    """hubconf usually re-exports entry points by import, so list what the imported module exposes."""
    import importlib
    sys.path.insert(0, VENDOR)
    hub = importlib.import_module("hubconf")
    fns = sorted(n for n in dir(hub) if not n.startswith("_") and callable(getattr(hub, n)) and getattr(getattr(hub, n), "__module__", "") != "torch")
    R["entrypoints"] = fns
    rc, hits = sh(f"cd {VENDOR} && grep -rn --include=*.py -i -E 'vjepa2_1|vjepa2\\.1|vit_.*_384' . | head -40")
    rc, urls = sh(f"cd {VENDOR} && grep -rhoE --include=*.py 'https?://[^ \"\\']+' . | sort -u | head -60")
    rc, readme = sh(f"cd {VENDOR} && grep -n -i -E '2\\.1|2_1' README.md | head -25")
    return dict(entrypoints=fns, has_2_1=[f for f in fns if "2_1" in f or "21" in f], grep_2_1=hits, urls=urls.split("\n"), readme_2_1=readme)


MODEL = {}


@step("load_model")
def _():
    import torch
    sys.path.insert(0, VENDOR)
    E = R.get("entrypoints", [])
    cands = [f for f in E if "2_1" in f and "large" in f.lower()] or [f for f in E if "2_1" in f] or \
            [f for f in E if "large" in f.lower() and "vjepa" in f]
    R["note_loaded_is_2_1"] = bool(cands) and "2_1" in cands[0]
    R["load_candidates"] = cands; tried = []
    for name in cands:
        try:
            out = torch.hub.load(VENDOR, name, source="local")
            enc = out[0] if isinstance(out, (tuple, list)) else out
            MODEL.update(name=name, enc=enc.cuda().eval()); tried.append((name, "ok")); break
        except Exception as e: tried.append((name, repr(e)[:300]))
    if "enc" not in MODEL: raise RuntimeError(f"no entrypoint loaded: {tried}")
    ck = []
    for root in (os.path.expanduser("~/.cache/torch/hub/checkpoints"),):
        if os.path.isdir(root):
            for fn in os.listdir(root):
                p = os.path.join(root, fn); ck.append(dict(file=fn, bytes=os.path.getsize(p), sha256=sha256(p)))
    n = sum(p.numel() for p in MODEL["enc"].parameters())
    return dict(entry=MODEL["name"], tried=tried, params_M=round(n / 1e6, 1), checkpoints=ck, cls=type(MODEL["enc"]).__name__)


@step("module_tree_and_input_spec")
def _():
    enc = MODEL["enc"]
    attrs = {k: getattr(enc, k) for k in ("img_height", "img_width", "num_frames", "tubelet_size", "patch_size", "embed_dim", "depth", "num_heads") if hasattr(enc, k)}
    pe = [(n, type(m).__name__, str(getattr(m, "kernel_size", ""))) for n, m in enc.named_modules() if "patch_embed" in n or n.endswith("proj")][:6]
    return dict(attrs={k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in attrs.items()}, n_blocks=len(enc.blocks), block_type=type(enc.blocks[0]).__name__, patch_embed=pe,
                top_level=[n for n, _ in enc.named_children()])


@step("hooks_shapes_determinism")
def _():
    import torch
    enc = MODEL["enc"]; acts = []
    hs = [b.register_forward_hook(lambda m, i, o: acts.append(o[0] if isinstance(o, tuple) else o)) for b in enc.blocks]
    S = int(getattr(enc, "img_height", 256) if not isinstance(getattr(enc, "img_height", 256), (list, tuple)) else enc.img_height[0]); T = 16
    x = torch.rand(1, 3, T, S, S, device="cuda")
    res = {}
    with torch.no_grad():
        acts.clear(); y1 = enc(x); a1 = [a.float().cpu() for a in acts]
        acts.clear(); y2 = enc(x); a2 = [a.float().cpu() for a in acts]
        res["out_shape"] = list(y1.shape); res["n_hooked"] = len(a1); res["block_shape"] = list(a1[0].shape)
        res["repeat_max_abs_diff"] = max(float((p - q).abs().max()) for p, q in zip(a1, a2))
        acts.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16): enc(x)
        a3 = [a.float().cpu() for a in acts]
        res["bf16_rel_err_per_block_last"] = float((a3[-1] - a1[-1]).norm() / a1[-1].norm())
        res["bf16_rel_err_per_block_mid"] = float((a3[len(a1) // 2] - a1[len(a1) // 2]).norm() / a1[len(a1) // 2].norm())
    for h in hs: h.remove()
    res["input_used"] = [1, 3, T, S, S]; res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return res


@step("token_to_space_map")
def _():
    """Bright square at a known place; which tokens change? Verifies token order (t,h,w) and patch grid."""
    import torch
    enc = MODEL["enc"]; S = 256; T = 16; ps = 16; ts = int(getattr(enc, "tubelet_size", 2))
    base = torch.zeros(1, 3, T, S, S, device="cuda") + .2
    obj = base.clone(); r0, c0 = 64, 160; obj[:, :, :, r0:r0 + 32, c0:c0 + 32] = 1.0     # patches rows 4-5, cols 10-11, all frames
    with torch.no_grad(): a, b = enc(base).float()[0], enc(obj).float()[0]
    d = (a - b).norm(dim=-1)
    nt, hp = T // ts, S // ps
    d = d.view(nt, hp, hp) if d.numel() == nt * hp * hp else None
    if d is None: return dict(note="token count != t*h*w", n_tokens=int(a.shape[0]))
    top = (d[0] > d[0].max() * .5).nonzero().tolist()
    rows = sorted({r for r, c in top}); cols = sorted({c for r, c in top})
    return dict(grid=[nt, hp, hp], strong_rows=rows, strong_cols=cols, expected_rows=[4, 5], expected_cols=[10, 11],
                order_consistent=(set(rows) <= {3, 4, 5, 6} and set(cols) <= {9, 10, 11, 12}))


@step("throughput")
def _():
    import torch
    enc = MODEL["enc"]; S = 256; x = torch.rand(8, 3, 16, S, S, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        enc(x); torch.cuda.synchronize(); t = time.time()
        for _ in range(3): enc(x)
        torch.cuda.synchronize()
    return dict(clips_per_s_16frames=round(24 / (time.time() - t), 2), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))


R["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); save(); print("\nGATE0 DONE", flush=True)
