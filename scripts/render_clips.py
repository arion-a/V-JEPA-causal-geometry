"""Render manifest clips to $CG_ROOT/data/clips/<clip_id>.npy (uint8 [64,384,384,3]); resumable; records SHA256 per clip.
Usage: python scripts/render_clips.py [--only gate1] [--limit N]. Writes $OUT_CSV (default $CG_ROOT/data/clip_hashes.csv)."""
import argparse, csv, importlib.util, json, os, sys, time
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from cg import sim_pb as S
spec = importlib.util.spec_from_file_location("bm", f"{ROOT}/scripts/build_manifest.py"); bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)

ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None); ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default=os.environ.get("CG_ROOT", f"{ROOT}/_local")); a = ap.parse_args()
rows = list(csv.DictReader(open(f"{ROOT}/data/manifests/pilot_manifest.csv")))
if a.only == "gate1": rows = [r for r in rows if r["gate1"] == "1"]
if a.limit: rows = rows[:a.limit]
cdir = f"{a.out}/data/clips_v2"; os.makedirs(cdir, exist_ok=True)
hp = f"{a.out}/data/clip_hashes_v2.csv"
done = {r["clip_id"]: r for r in csv.DictReader(open(hp))} if os.path.exists(hp) else {}
for i, r in enumerate(rows):
    path = f"{cdir}/{r['clip_id']}.npy"
    if r["clip_id"] in done and os.path.exists(path): continue
    w = S.make_world(r["kind"], int(r["world_seed"])); over, nz = bm.variants(r["family"], r["kind"])[r["variant"]]
    if nz == "complement": nz = {"color": tuple(1 - c for c in w.color)}
    t = time.time(); out = S.simulate(w, over, nz, render=True)
    np.save(path + ".tmp.npy", out["frames"]); os.replace(path + ".tmp.npy", path)           # atomic
    done[r["clip_id"]] = dict(clip_id=r["clip_id"], sha256=S.frames_sha256(out["frames"]), seconds=round(time.time() - t, 2), in_view=int(out["in_view"]))
    with open(hp, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["clip_id", "sha256", "seconds", "in_view"]); wr.writeheader(); wr.writerows(done.values())
    print(f"[{i + 1}/{len(rows)}] {r['clip_id']} {done[r['clip_id']]['seconds']}s", flush=True)
print("rendered", len(done), "clips")
