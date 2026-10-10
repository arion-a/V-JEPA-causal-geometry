"""Pod-side render verification: all clips present with expected shape, in view, and reference hashes match the build host."""
import csv, json, os, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CG = os.environ.get("CG_ROOT"); rows = list(csv.DictReader(open(f"{ROOT}/data/manifests/pilot_manifest.csv")))
H = {r["clip_id"]: r for r in csv.DictReader(open(f"{CG}/data/clip_hashes.csv"))}
ref = json.load(open(f"{ROOT}/data/manifests/reference_hashes.json"))
bad_shape = [r["clip_id"] for r in rows if not os.path.exists(f"{CG}/data/clips/{r['clip_id']}.npy") or np.load(f"{CG}/data/clips/{r['clip_id']}.npy", mmap_mode="r").shape != (64, 384, 384, 3)]
res = dict(manifest_clips=len(rows), rendered=len(H), missing_or_bad_shape=bad_shape, out_of_view=[c for c, r in H.items() if r["in_view"] != "1"],
           reference_hash_match={k: (H.get(k, {}).get("sha256") == v) for k, v in ref["hashes"].items()}, build_host=ref["platform"],
           mean_render_s=round(float(np.mean([float(r["seconds"]) for r in H.values()])), 2) if H else None)
res["ok"] = (res["rendered"] == res["manifest_clips"] and not bad_shape and not res["out_of_view"])
res["cross_machine_bit_identical"] = all(res["reference_hash_match"].values())
json.dump(res, open(sys.argv[1], "w"), indent=1); print(json.dumps(res, indent=1))
