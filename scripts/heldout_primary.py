"""STAGE 6 primary analysis (pod, GPU). Runs the confirmatory endpoint declared in configs/pilot.json 'confirmatory'.
  --split dev      : dry run on DEV worlds (always allowed). Exercises exactly the same code path as the held-out run.
  --split heldout  : refused unless the ledger holds a decision node 'UNLOCK HELD-OUT ...' that lists THIS script and cg/primary.py as artifacts
                     with their current hashes, the chain verifies, and --unlock-hash names that node. Analysed once.
Representations (block 23 only): pretrained V-JEPA 2.1 {global, window}, random-weight V-JEPA {global, window}, pixel (16 frames, 16x16 RGB), tracked trajectory."""
import argparse, csv, importlib, json, os, sys, time
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CG = os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry"); sys.path.insert(0, ROOT); sys.path.insert(0, f"{CG}/vendor/vjepa2")
from cg import ledger as L, primary as P

ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=("dev", "heldout"), default="dev"); ap.add_argument("--unlock-hash", default=None); ap.add_argument("--n-boot", type=int, default=2000)
a = ap.parse_args()


def check_unlock(h):
    path = f"{ROOT}/log/ledger.jsonl"; ok, problems = L.verify(path, root=ROOT)
    if not ok: sys.exit(f"REFUSED: ledger does not verify: {problems[:3]}")
    e = next((x for x in L.read(path) if x["hash"] == h), None)
    if not e or e["kind"] != "decision" or not e["summary"].startswith("UNLOCK HELD-OUT"): sys.exit("REFUSED: --unlock-hash is not an 'UNLOCK HELD-OUT' decision node")
    need = {"scripts/heldout_primary.py", "cg/primary.py", "configs/pilot.json"}
    if not need <= set(e["artifacts"]): sys.exit(f"REFUSED: unlock node must list artifacts {sorted(need)}")
    if any(L.read(path)[i]["summary"].startswith("HELD-OUT PRIMARY RUN") for i in range(len(L.read(path)))): sys.exit("REFUSED: held-out was already analysed once")
    print("unlock verified:", h[:12], flush=True)


if a.split == "heldout":
    if not a.unlock_hash: sys.exit("REFUSED: held-out needs --unlock-hash")
    check_unlock(a.unlock_hash)
import torch
from cg import sim_pb as S, features as FT
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
importlib.import_module("src.hub.backbones").VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
hub = importlib.import_module("hubconf")
OUT = f"{CG}/results/primary_{a.split}"; os.makedirs(OUT, exist_ok=True); CACHE = f"{CG}/features/pooled/primary_{a.split}.npz"
rows = [r for r in csv.DictReader(open(f"{ROOT}/data/manifests/pilot_manifest.csv")) if r["family"] in S.KINDS and r["split"] == a.split]
assert rows and all(r["split"] == a.split for r in rows)
traj = np.load(f"{ROOT}/data/manifests/trajectories.npz"); worlds = {}
for r in rows: worlds.setdefault((r["kind"], r["world_id"]), {})[r["variant"]] = r["clip_id"]
NEED = {"freefall": ("g",), "bounce": ("g", "e"), "incline": ("g", "mu")}
ids = [vs[f"{p}{sg}"] for (k, _), vs in worlds.items() for p in NEED[k] for sg in "+-"]
missing = [c for c in ids if not os.path.exists(f"{CG}/data/clips_v3/{c}.npy")]
assert not missing, f"{len(missing)} clips not rendered (render first), e.g. {missing[:2]}"
print(a.split, "worlds", {k: sum(1 for (kk, _) in worlds if kk == k) for k in S.KINDS}, "clips", len(ids), flush=True); t0 = time.time()


def pix(cid):
    fr = np.load(f"{CG}/data/clips_v3/{cid}.npy", mmap_mode="r")[::4].astype(np.float32)
    return fr.reshape(16, 16, 24, 16, 24, 3).mean((2, 4)).ravel()


def extract_all():
    F = {}
    for name, pre in (("pt", True), ("rw", False)):
        torch.manual_seed(0)
        enc = hub.vjepa2_1_vit_large_384(pretrained=pre); enc = (enc[0] if isinstance(enc, (tuple, list)) else enc).cuda().eval()
        for (kind, wid), vs in worlds.items():
            mask = np.zeros(32 * 24 * 24, bool)
            for cid in vs.values(): mask |= S.object_token_mask(traj[cid], kind)
            mt = torch.from_numpy(mask).cuda()
            for p in NEED[kind]:
                for sg in "+-":
                    cid = vs[f"{p}{sg}"]; f = FT.extract(enc, FT.to_tensor(np.load(f"{CG}/data/clips_v3/{cid}.npy")), mask=mt)
                    F[f"{name}_g|{cid}"] = f["normed"][3]; F[f"{name}_w|{cid}"] = f["normed_win"][3]
            print(name, wid, f"{time.time() - t0:.0f}s", flush=True)
        del enc; torch.cuda.empty_cache()
    for c in ids: F[f"px|{c}"] = pix(c)
    np.savez(CACHE, **F); return F


if os.path.exists(CACHE): C = np.load(CACHE); F = {k: C[k] for k in C.files}; print("loaded cache", flush=True)
else: F = extract_all()
if a.split == "dev":      # consistency: freshly extracted pretrained block-23 features must equal the Gate-1 features of the same clips
    Z0 = np.load(f"{CG}/features/pooled/gate1_v3.npz"); i0 = {c: i for i, c in enumerate(Z0["ids"])}
    dif = max(float(np.abs(F[f"pt_g|{c}"] - Z0["normed"][i0[c], 3]).max()) for c in ids[:20]); print("consistency vs gate1 features, max abs diff:", dif, flush=True)

def V(rep, par, kind):
    out = []
    for (k, _), vs in worlds.items():
        if k != kind: continue
        p, m = vs[f"{par}+"], vs[f"{par}-"]
        if rep == "traj": out.append(traj[p].ravel() - traj[m].ravel())
        elif rep == "px": out.append(F[f"px|{p}"] - F[f"px|{m}"])
        else: out.append(F[f"{rep}|{p}"] - F[f"{rep}|{m}"])
    return np.array(out, dtype=np.float64)


CASES = [("g", "freefall"), ("g", "bounce"), ("g", "incline"), ("e", "bounce"), ("mu", "incline")]
reps = {r: {c: V(r, *c) for c in CASES} for r in ("pt_g", "pt_w", "rw_g", "rw_w", "px", "traj")}
res = P.run(reps, {"global": ("pt_g", "traj", "px", "rw_g"), "window": ("pt_w", "traj", "px", "rw_w")}, n_boot=a.n_boot)
res.update(split=a.split, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), config_confirmatory=json.load(open(f"{ROOT}/configs/pilot.json"))["confirmatory"]["statistic"])
json.dump(res, open(f"{OUT}/primary.json", "w"), indent=1); print(json.dumps({k: res[k] for k in ("claim", "endpoints")}, indent=1)); print("PRIMARY DONE", a.split, f"{time.time() - t0:.0f}s", flush=True)
