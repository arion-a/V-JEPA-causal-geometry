# Causal geometry of physical mechanisms in V-JEPA 2.1 (Protocol A pilot)

Question: is the latent space of a frozen V-JEPA 2.1 organised by the *mechanisms* (g, e, mu) that generate a
video, or only by the trajectories they produce? Design: `docs/01_VJEPA21_Pilot_Protocol.md`; theory:
`docs/02_Causal_Geometry_Theory.md`.

## Status
No experiment on V-JEPA 2.1 has been run. What exists and is tested (`pytest tests`):
- `cg/geometry.py` - central-difference Jacobian columns, effective rank, induced metric G = J^T J, principal
  angles, composition error, shuffled-pair null, bootstrap over base worlds. Verified against synthetic encoders
  with known answers (linear => rank 3, composition error 0; nonlinear => > 0).
- `cg/simulate.py` - numpy scenarios (freefall / bounce / incline) with validity gates (an intervention must
  change the physics; restitution excluded without impacts). Pipeline-validation backend; Protocol A requires
  MuJoCo/PyBullet in the GPU stage behind the same `trajectory(world, theta)` interface.
- `cg/extract.py` - per-block pooled-token hooks. `ToyVideoEncoder` is random weights for smoke tests only;
  `load_vjepa21()` is **unverified** (Gate 0 must confirm the entrypoint, checkpoint SHA256 and block mapping).
- `cg/analysis.py`, `scripts/run_smoke.py` - layerwise report; smoke output is NOT a result.

## Next (gates from the protocol)
0. On a 24-48 GB GPU: load real V-JEPA 2.1 L, record hashes, verify hooks and repeat-inference determinism.
1. 20-clip smoke with the physics engine; check interventions alter the relevant event.
2. 200 clips (50 worlds), extract all layers; measure throughput before approving more rental.
3. Layerwise analysis with baselines (pixel diff, trajectory features) and negative controls; cause-vs-effect
   (gravity vs initial-velocity matched-trajectory) extension.

Run: `PYTHONPATH=. python scripts/run_smoke.py`
