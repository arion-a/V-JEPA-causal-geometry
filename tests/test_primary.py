import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cg import primary as p

ROOT = Path(__file__).parent.parent
GRAVITY = [(p.Param.G, s) for s in p.SCENES]


def shared_gravity(rng: np.random.Generator, d: int, direction: p.Array) -> dict[p.Case, p.Array]:
    reps = {
        c: np.stack([direction * 3 + rng.normal(size=d) * 0.3 for _ in range(15)]) for c in GRAVITY
    }
    return {**reps, **independent_others(rng, d)}


def independent_others(rng: np.random.Generator, d: int) -> dict[p.Case, p.Array]:
    return {
        (param, scene): np.stack([rng.normal(size=d) * 3 for _ in range(15)])
        for param, scene in p.REQUIRED_CASES
        if param is not p.Param.G
    }


def scene_specific(rng: np.random.Generator, d: int) -> dict[p.Case, p.Array]:
    gravity = {
        c: np.stack([rng.normal(size=d) * 3 + rng.normal(size=d) * 0.3 for _ in range(15)])
        for c in GRAVITY
    }
    return {**gravity, **independent_others(rng, d)}


def endpoints(vjepa_global: str, vjepa_window: str) -> list[p.Endpoint]:
    return [
        p.Endpoint(p.Pooling.GLOBAL, vjepa_global, "traj", "px", "rw"),
        p.Endpoint(p.Pooling.WINDOW, vjepa_window, "traj", "px", "rw"),
    ]


def test_shared_direction_beats_scene_specific_baseline_and_claim_passes() -> None:
    rng = np.random.default_rng(0)
    u = rng.normal(size=256)
    reps = {
        "vj_g": shared_gravity(rng, 256, u),
        "vj_w": shared_gravity(rng, 256, u),
        "traj": scene_specific(rng, 256),
        "px": scene_specific(rng, 256),
        "rw": scene_specific(rng, 256),
    }
    r = p.run(reps, endpoints("vj_g", "vj_w"), n_boot=300, seed=0)
    assert r.claim is p.Claim.MECHANISM_LIKE
    assert all(e.primary_success for e in r.endpoints.values())


def test_no_shared_direction_is_not_a_success_and_baseline_stronger_is_detected() -> None:
    rng = np.random.default_rng(1)
    u = rng.normal(size=256)
    reps = {
        "vj_g": scene_specific(rng, 256),
        "vj_w": scene_specific(rng, 256),
        "traj": shared_gravity(rng, 256, u),
        "px": scene_specific(rng, 256),
        "rw": scene_specific(rng, 256),
    }
    r = p.run(reps, endpoints("vj_g", "vj_w"), n_boot=300, seed=0)
    assert r.claim is p.Claim.BELOW
    assert not any(e.primary_success for e in r.endpoints.values())


def test_one_pooling_only_is_pooling_dependent() -> None:
    rng = np.random.default_rng(2)
    u = rng.normal(size=256)
    reps = {
        "vj_g": shared_gravity(rng, 256, u),
        "vj_w": scene_specific(rng, 256),
        "traj": scene_specific(rng, 256),
        "px": scene_specific(rng, 256),
        "rw": scene_specific(rng, 256),
    }
    r = p.run(reps, endpoints("vj_g", "vj_w"), n_boot=300, seed=0)
    assert r.claim is p.Claim.POOLING_DEPENDENT


def test_paired_bootstrap_is_deterministic_and_seed_dependent() -> None:
    rng = np.random.default_rng(3)
    u = rng.normal(size=64)
    reps = {
        "a": shared_gravity(rng, 64, u),
        "traj": scene_specific(rng, 64),
        "px": scene_specific(rng, 64),
        "rw": scene_specific(rng, 64),
    }
    ep = [p.Endpoint(p.Pooling.GLOBAL, "a", "traj", "px", "rw")]
    first = p.run(reps, ep, n_boot=100, seed=0)
    assert first == p.run(reps, ep, n_boot=100, seed=0)
    assert first.endpoints != p.run(reps, ep, n_boot=100, seed=1).endpoints


def test_invalid_inputs_fail_loudly() -> None:
    rng = np.random.default_rng(4)
    rep = scene_specific(rng, 16)
    ep = [p.Endpoint(p.Pooling.GLOBAL, "a", "a", "a", "a")]
    with pytest.raises(ValueError):
        p.run({}, ep, n_boot=10, seed=0)
    with pytest.raises(ValueError):
        p.run(
            {"a": rep}, [p.Endpoint(p.Pooling.GLOBAL, "a", "a", "a", "missing")], n_boot=10, seed=0
        )
    with pytest.raises(ValueError):
        p.run({"a": {k: v for k, v in rep.items() if k != GRAVITY[0]}}, ep, n_boot=10, seed=0)
    with pytest.raises(ValueError):
        p.run({"a": rep}, ep, n_boot=0, seed=0)


@pytest.mark.parametrize("extra", [[], ["--unlock-hash", "0" * 64]])
def test_heldout_refused_without_valid_unlock(extra: list[str]) -> None:
    script = str(ROOT / "scripts" / "heldout_primary.py")
    r = subprocess.run(
        [sys.executable, script, "--split", "heldout", *extra], capture_output=True, text=True
    )
    assert r.returncode != 0
    assert "REFUSED" in r.stdout + r.stderr
