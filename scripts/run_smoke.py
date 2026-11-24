"""Pipeline smoke test on the toy encoder. Output is NOT a scientific result."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cg import analysis as A  # noqa: E402
from cg import extract as E  # noqa: E402
from cg import simulate as S  # noqa: E402
from cg.scenes import Param, Scene  # noqa: E402


def variants(w: S.World) -> dict[str, tuple[dict[Param, float], S.Nuisance]]:
    """Clip variants of one world: base, camera and colour nuisance, +/- steps and one composition."""
    out = {"base": ({}, S.NO_NUISANCE), "cam": ({}, S.Nuisance(cam_shift=0.3)), "color": ({}, S.Nuisance(color=1 - w.color))}
    obs = S.observable(w.kind)
    for p in obs:
        for sign, d in (("+", 1), ("-", -1)):
            out[f"{p}{sign}"] = ({p: S.BASE.get(p) + d * S.DELTA.get(p)}, S.NO_NUISANCE)
    second = [p for p in obs if p is not Param.G]
    if second:
        both = {p: S.BASE.get(p) + S.DELTA.get(p) for p in (Param.G, second[0])}
        out["comp"] = (both, S.NO_NUISANCE)
    return out


def world_features(enc: E.ToyVideoEncoder, w: S.World) -> tuple[dict[str, list[np.ndarray]], bool]:
    clips = {k: S.clip(w, theta, nuis) for k, (theta, nuis) in variants(w).items()}
    # validity gate: the intervened parameter must change the physical trajectory
    valid = all(np.abs(clips[f"{p}+"].xz - clips[f"{p}-"].xz).max() > 1e-3 for p in S.observable(w.kind))
    x = torch.from_numpy(np.stack([c.frames for c in clips.values()]))
    layers = E.layer_means(enc, enc.blocks, x)
    return {k: [layers[i, layer] for layer in range(layers.shape[1])] for i, k in enumerate(clips)}, valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=3)
    n_worlds = ap.parse_args().worlds
    enc = E.ToyVideoEncoder().eval()
    kinds: dict[str, Scene] = {}
    feats: dict[str, dict[str, list[np.ndarray]]] = {}
    valid: list[bool] = []
    for kind in Scene:
        for s in range(n_worlds):
            wid = f"{kind}{s}"
            kinds[wid] = kind
            feats[wid], ok = world_features(enc, S.make_world(kind, 100 + s))
            valid.append(ok)
    rep = {layer: A.to_json(A.layer_report(feats, kinds, layer)) for layer in range(4)}
    print(json.dumps(dict(valid_worlds=f"{sum(valid)}/{len(valid)}", layers=rep), indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
