# Causal geometry of physical mechanisms in V-JEPA 2.1 (Protocol A pilot)

Question: is the latent space of a frozen V-JEPA 2.1 organised by the *mechanisms* (g, e, mu) that generate a
video, or only by the trajectories they produce? Design: `docs/01_VJEPA21_Pilot_Protocol.md`; theory:
`docs/02_Causal_Geometry_Theory.md`.

## Status
The pilot has been run end to end. The confirmatory test (H2, held-out worlds, run once) did **not** support a
mechanism-like geometry: V-JEPA's cross-scene gravity alignment was not distinguished from the tracked-trajectory
baseline. See `reports/pilot_results.html` (hypotheses, definitions, equations, conditions, results, observed /
unsupported / not-tested table). Every action and decision is a node in the hash-chained ledger `log/ledger.jsonl`
(verify with `python3 scripts/ledger.py verify --files`; `reports/ledger_live.html` is a browsable copy).
Frozen decisions and amendments A1-A6: `configs/pilot.json`.

Not done: nuisance separation (H3), composition (H4), effective rank / induced metric / principal angles on the final
data, step-size (curvature) tests, optical-flow baseline, pendulum / collision scenes. Two caveats are easy to miss: the
first Gate 1 test against the render-jitter floor failed and its rule was replaced (amendment A4), and the last layer
does not localise objects, so object-window pooling there is not shown to isolate the object (report sections 5.1b, 5.2).

`docs/01`-`docs/04` are the original pre-run design (protocol, theory, plans) and are kept unchanged on purpose;
where they say "no experiment has been run" they describe the state before the pilot.

What is in the code (`pytest tests`, 40 tests):
- `cg/sim_pb.py` - PyBullet scenes (free fall, bounce, incline), deterministic headless render, token masks.
- `cg/features.py` - fp32 extraction from the official V-JEPA 2.1 ViT-L 384 (4 hierarchical post-norm taps; global and
  object-window pooling). Weights and source come from the official `facebookresearch/vjepa2` release.
- `cg/devstats.py`, `cg/primary.py` - reliability, bootstrap, cross-scene alignment and the confirmatory endpoint.
- `cg/ledger.py` - the hash-chained ledger. `cg/gate1_analysis.py` - first-run Gate 1 statistics.
- `cg/geometry.py`, `cg/analysis.py`, `cg/simulate.py`, `cg/extract.py`, `scripts/run_smoke.py` - the early synthetic
  scaffold (numpy scenarios, toy encoder). Kept for tests; smoke output is not a result.
- `scripts/` - one script per gate or stage (`gate0.py` ... `stage7_cause_effect.py`, `heldout_primary.py`).

Data and features live on the GPU volume, not in git. Result files that matter are under `results/`.

## Pod worker (git-based, no open ports)
`scripts/pod_worker.py` runs inside the GPU pod: it `git pull`s, runs new `jobs/NNN_name.sh` scripts in order, and
commits `results/jobs/<name>/{status.json,log.txt,...}` back. Heavy artifacts live on the persistent volume
(`$CG_ROOT`, default `/workspace/vjepa-causal-geometry`). Kill switch: commit `jobs/STOP`.
Security: use a fine-grained GitHub token limited to this repo (Contents read/write); the worker keeps it out of
git config, logs and job environments. Anyone with push access can run code on the pod.
```
cd /workspace/vjepa-causal-geometry/repo && git pull && python3 scripts/pod_worker.py   # prompts for token (hidden)
```
