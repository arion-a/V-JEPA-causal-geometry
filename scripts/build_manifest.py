"""Build the 200-clip manifest (50 base worlds) with validity gating. Physics only, no rendering (fast, CPU).
Outputs data/manifests/{pilot_manifest.csv, rejected_worlds.csv, manifest_summary.json}. Deterministic given configs/pilot.json."""
import csv, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cg import sim_pb as S

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CFG = json.load(open(f"{ROOT}/configs/pilot.json")); V = CFG["validity"]; D = CFG["delta"]; B = CFG["base_theta"]
OUT = f"{ROOT}/data/manifests"; os.makedirs(OUT, exist_ok=True)


def variants(family, kind):
    """-> {variant_name: (theta_override, nuisance_spec)}"""
    th = lambda **k: {**B, **k}
    if family in ("freefall", "bounce", "incline"):
        v = {"base": ({}, None), "g-": ({"g": B["g"] - D["g"]}, None), "g+": ({"g": B["g"] + D["g"]}, None)}
        if kind == "bounce": v.update({"e-": ({"e": B["e"] - D["e"]}, None), "e+": ({"e": B["e"] + D["e"]}, None)})
        if kind == "incline": v.update({"mu-": ({"mu": B["mu"] - D["mu"]}, None), "mu+": ({"mu": B["mu"] + D["mu"]}, None)})
        return v
    if family == "nuisance":
        return {"base": ({}, None), "cam": ({}, {"cam": CFG["nuisance"]["cam_shift_m"]}), "color": ({}, "complement")}
    sec = CFG["allocation"]["composition"]["second"][kind]
    return {"base": ({}, None), "g+": ({"g": B["g"] + D["g"]}, None), "second+": ({sec: B[sec] + D[sec]}, None),
            "both+": ({"g": B["g"] + D["g"], sec: B[sec] + D[sec]}, None)}


def run(w, over, nuis):
    nz = None
    if nuis == "complement": nz = {"color": tuple(1 - c for c in w.color)}
    elif nuis: nz = nuis
    return S.simulate(w, over, nz, render=False)


def check_world(w, family, kind):
    """Validity gate. Returns (ok, reason, sims)."""
    sims = {n: run(w, o, z) for n, (o, z) in variants(family, kind).items()}
    if not all(s["in_view"] for s in sims.values()): return False, "object_out_of_view", sims
    if kind == "bounce" and any(len(s["impacts"]) < V["bounce_min_impacts_all_variants"] for s in sims.values()): return False, "no_impact", sims
    if kind == "incline":
        if any((s["slide_m"] or 0) < V["incline_min_slide_m"] for s in sims.values()): return False, "no_slide", sims
        lo_t, hi_t = V["incline_transition_frame_range"]
        if any(s["transition_frame"] is None or not (lo_t <= s["transition_frame"] <= hi_t) for s in sims.values()): return False, "bad_transition", sims
        if family == "incline":                                       # g and mu must be separable in trajectory space (unit-column effective rank)
            dg = (sims["g+"]["pos"] - sims["g-"]["pos"]).ravel(); dm = (sims["mu+"]["pos"] - sims["mu-"]["pos"]).ravel()
            if min(np.linalg.norm(dg), np.linalg.norm(dm)) < 1e-9: return False, "no_effect", sims
            sv = np.linalg.svd(np.stack([dg / np.linalg.norm(dg), dm / np.linalg.norm(dm)], 1), compute_uv=False); p_ = sv / sv.sum()
            if float(np.exp(-(p_ * np.log(p_)).sum())) < V["incline_min_effective_rank"]: return False, "g_mu_collinear", sims
    if kind == "freefall" and float(sims["base"]["pos"][:, 2].min()) < -50: return False, "diverged", sims
    px = lambda a, b: float(np.linalg.norm(S.project_pixels(a["pos"], kind) - S.project_pixels(b["pos"], kind), axis=1).max())   # peak pixel displacement
    for p in ("g", "e", "mu"):                                    # each intervened pair must move the object by >= min_peak_displacement_px
        lo, hi = sims.get(f"{p}-"), sims.get(f"{p}+")
        if lo and hi and px(hi, lo) < V["min_peak_displacement_px"]: return False, f"{p}_no_effect", sims
    if family == "composition":
        if px(sims["g+"], sims["base"]) < V["min_peak_displacement_px"] / 2: return False, "g_no_effect", sims
        if px(sims["second+"], sims["base"]) < V["min_peak_displacement_px"] / 4: return False, "second_no_effect", sims
    return True, "ok", sims


def main():
    alloc = CFG["allocation"]; plan = []
    for fam in ("freefall", "bounce", "incline"): plan += [(fam, fam)] * alloc[fam]["worlds"]
    for k, n in alloc["nuisance"]["split_across"].items(): plan += [("nuisance", k)] * n
    for k, n in alloc["composition"]["split_across"].items(): plan += [("composition", k)] * n
    rows, rej, counters = [], [], {}
    for fam, kind in plan:
        key = f"{fam}:{kind}"; counters.setdefault(key, 0)
        while True:
            idx = counters[key]; counters[key] += 1
            seed = S.derive_seed(CFG["root_seed"], f"world-{fam}-{kind}", idx)
            w = S.make_world(kind, seed); ok, why, sims = check_world(w, fam, kind)
            if ok: break
            rej.append(dict(family=fam, kind=kind, index=idx, seed=seed, reason=why))
            if sum(1 for r in rej if r["family"] == fam and r["kind"] == kind) > V["max_rejections_per_family"]:
                raise RuntimeError(f"{key}: >{V['max_rejections_per_family']} rejections (last reason {why}); fix the design or the gate, do not loop")
        rows.append((fam, kind, w, sims))
    # splits: per (family,kind) alternate dev / heldout in generation order (first half dev)
    seen, out = {}, []
    for wi, (fam, kind, w, sims) in enumerate(rows):
        seen[(fam, kind)] = seen.get((fam, kind), -1) + 1; seen[fam] = seen.get(fam, -1) + 1
        split = "dev" if seen[fam] % 2 == 0 else "heldout"            # parity of the within-family ordinal (v1 dev worlds stay dev)
        wid = f"{fam}_{kind}_{seen[(fam, kind)]:03d}"
        for var, (over, nz) in variants(fam, kind).items():
            th = {**B, **over}; s = sims[var]
            out.append(dict(clip_id=f"{wid}__{var}", world_id=wid, family=fam, kind=kind, split=split, variant=var, world_seed=w.seed,
                            g=th["g"], e=th["e"], mu=th["mu"], cam_shift=(nz or {}).get("cam", 0.0) if isinstance(nz, dict) else 0.0,
                            color_variant=int(nz == "complement"), n_impacts=len(s["impacts"]), first_impact_frame=(s["impacts"][0] if s["impacts"] else ""),
                            slide_m=("" if s["slide_m"] is None else round(s["slide_m"], 4)), gate1=0))
    # Gate-1 subset: exactly 20 clips covering all five families (first dev world of each family, all its variants)
    g1 = []
    for fam in ("freefall", "bounce", "incline", "nuisance", "composition"):
        wid = next(r["world_id"] for r in out if r["family"] == fam and r["split"] == "dev")
        g1 += [r["clip_id"] for r in out if r["world_id"] == wid]
    assert len(g1) == 20, len(g1)
    for r in out: r["gate1"] = int(r["clip_id"] in g1)
    with open(f"{OUT}/pilot_manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
    with open(f"{OUT}/rejected_worlds.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["family", "kind", "index", "seed", "reason"]); wr.writeheader(); wr.writerows(rej)
    traj = {r["clip_id"]: None for r in out}
    sims_by_world = {}
    seen2 = {}
    for wi, (fam, kind, w, sims) in enumerate(rows):
        seen2[(fam, kind)] = seen2.get((fam, kind), -1) + 1
        for var in sims: traj[f"{fam}_{kind}_{seen2[(fam, kind)]:03d}__{var}"] = sims[var]["pos"].astype(np.float64)
    np.savez_compressed(f"{OUT}/trajectories.npz", **traj)          # tracked-state baseline inputs (step 1.7)
    h = hashlib.sha256(open(f"{OUT}/pilot_manifest.csv", "rb").read()).hexdigest()
    by_fam = {}
    for r in out: by_fam[r["family"]] = by_fam.get(r["family"], 0) + 1
    summ = dict(clips=len(out), worlds=len(rows), clips_by_family=by_fam, rejected=len(rej),
                rejected_by_reason={k: sum(1 for r in rej if r["reason"] == k) for k in {r["reason"] for r in rej}},
                gate1_clips=sum(r["gate1"] for r in out), manifest_sha256=h,
                splits={s: len({r["world_id"] for r in out if r["split"] == s}) for s in ("dev", "heldout")})
    json.dump(summ, open(f"{OUT}/manifest_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
