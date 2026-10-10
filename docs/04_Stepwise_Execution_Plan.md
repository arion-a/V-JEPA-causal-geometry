# Stepwise Execution Plan (Protocol A)

Status: plan, no results. Companion to `01` (protocol), `02` (theory), `03` (open questions). Every stage has numbered steps, a deliverable, an acceptance test that can fail, a stop rule, and a rollback. **Proposed** defaults are starting points to be frozen in Stage 0, not established values. **[VERIFY]** marks facts about V-JEPA 2/2.1 I have not confirmed against the official repo.

Rules that hold throughout:
- No analysis of real-model features before Stage 0 is committed (git tag `prereg-v1`). Later changes are logged amendments with reason and date.
- Held-out worlds stay untouched until the primary endpoint, layer summary and perturbation sizes are frozen.
- Failed clips, runs and gates stay in the record. A null or negative result is a valid outcome.
- One pod at a time; log cost and throughput; hard-stop at the budget cap.

---

## Stage 0 - Preregistration freeze (no GPU, about half a day)

| Step | Action | Output |
|---|---|---|
| 0.1 | Choose primary hypothesis and endpoint. Proposed: H2 (cross-context gravity alignment), statistic = mean cosine between world-level v_g across scenario pairs, minus the same statistic on the tracked-trajectory baseline, over a predeclared layer band | `configs/pilot.json` fields |
| 0.2 | Predeclare layer band (proposed: middle third of blocks) and primary representation (proposed: tokens within a window around the tracked object, mean-pooled; global mean and full-grid as sensitivity) | config |
| 0.3 | Fix perturbation rule (proposed: step size per parameter chosen so peak trajectory displacement is equal across g, e, mu; dev worlds test delta/2, delta, 2delta) | config |
| 0.4 | Write validity rules: intervened parameter must change its event (impact time/post-impact speed, sliding distance, fall height) by more than a threshold; minimum fraction of valid worlds per family | config |
| 0.5 | Define margins: what counts as "exceeds baseline" (proposed: base-world bootstrap 95% lower bound of the difference > 0) and what counts as a practical null | config |
| 0.6 | Write the claims table: observed / unsupported / not yet tested, empty at this point | `reports/claims.md` |
| 0.7 | Run a power/precision simulation on synthetic data: with 10 worlds per family and 5 held-out, what difference in cosine is detectable? Publish the sensitivity curve | `reports/power.md` |

Acceptance: every metric in `cg/analysis.py` has a frozen definition; the power curve exists. Stop: if the detectable effect is larger than any plausible effect, add worlds (update the budget) before proceeding. Rollback: n/a.

## Stage 1 - Simulator and dataset (CPU or small GPU, 1-2 days)

| Step | Action | Acceptance |
|---|---|---|
| 1.1 | MuJoCo scenes: free fall, bounce, incline; headless EGL render at the checkpoint's native size and frame count (decided at Stage 2 step 2.2, so stub with a config value) | Duplicate render is bit-identical; engine, version, timestep, seed logged |
| 1.2 | Analytic cross-checks: free fall and bounce against the closed forms already in `cg/simulate.py` | Position error under a stated tolerance (proposed 1e-3 m) |
| 1.3 | Event extraction (impact frames, sliding distance, speed) | Matches analytic events |
| 1.4 | Validity gate per intervention pair | Every family meets the 0.4 rule; rejected worlds counted by reason |
| 1.5 | Build the 200-clip manifest: 50 base worlds, 5 development + 5 held-out per primary scenario, nuisance worlds balanced across scenarios, composition worlds distinct from the others | Manifest hash recorded, counts exactly as protocol table |
| 1.6 | Nuisance calibration: camera shift and colour magnitudes chosen so pixel-space displacement is comparable to the physical perturbations | Calibration table |
| 1.7 | Tracked-state baseline features (object x, z, speed, impact times) per clip | Saved with the manifest |

Stop: if a scenario cannot yield valid worlds, redesign it before any model work. Deliverable: `data/manifests/pilot_manifest.csv`, 20-clip pilot subset, simulator report.

## Stage 2 - Gate 0: real model on GPU (about 2-6 hours, cost cap about $2)

| Step | Action | Acceptance |
|---|---|---|
| 2.1 | Launch one 24-48 GB pod via the connected RunPod account (after your approval of the spend); record GPU, driver, library versions | Pod ID and cost logged |
| 2.2 | Fetch official repo and weights; record commit, checkpoint SHA256, license; confirm the V-JEPA 2.1 entrypoint name **[VERIFY]**; read the native input spec (frames, resolution, tubelet size, normalization) | `reports/gate0.md` |
| 2.3 | Print module tree; verify hooks yield exactly one tensor per block with expected shape and token order | Shape table matches |
| 2.4 | Determinism: same clip twice gives identical features; fp32 vs bf16 difference measured | Within stated tolerance (proposed 1e-3 relative in bf16) |
| 2.5 | Token-to-space map: confirm which token indices correspond to which spatial-temporal cell, so object-window pooling is correct | A rendered test clip with a bright moving square lights the expected tokens |
| 2.6 | Positive control in pixel/latent: shifting the object by one patch moves the pooled-window feature in the expected direction | Pass |
| 2.7 | Update 1.1 to the native format and re-render the 20-clip subset | Rendered clips load without resizing |

Stop: any of 2.2-2.5 uncertain, or the model cannot be loaded. Do not start Stage 4. Rollback: stop the pod immediately after logging.

## Stage 3 - Gate 1: 20-clip smoke test (1-3 hours)

| Step | Action | Acceptance |
|---|---|---|
| 3.1 | Render and extract features for 20 clips covering all five families | All finite, shapes right |
| 3.2 | Noise floor: duplicate renders and tiny-delta repeats give the latent noise scale per layer | Table |
| 3.3 | Finite-difference check: signal ||z(+)-z(-)|| exceeds noise floor by a stated factor (proposed 5x) at some layers for observable parameters; forward vs central difference compared | Pass or stop |
| 3.4 | Dry-run the full analysis on the 20 clips (development worlds only) and confirm it produces every table | Tables present, no NaN |
| 3.5 | Measure clips/sec and storage per clip; project the 200-clip cost | Projection within the budget cap |

Stop: interventions do not alter the relevant physical event, or signal is indistinguishable from noise at every layer (then re-examine perturbation size and pooling on development worlds only, as an amendment). Deliverable: `reports/gate1.md`.

## Stage 4 - Gate 2: 200-clip extraction (2-8 hours)

| Step | Action | Acceptance |
|---|---|---|
| 4.1 | Approve spend from the Gate 1 projection | Written budget line |
| 4.2 | Batch inference, features streamed to disk (float16), resumable by clip hash | 200 clips, or failures documented |
| 4.3 | Store pooled global, object-window and, for a small subset, full-token features | Index file |
| 4.4 | Checkpoint progress every N clips; verify a random 5% subset by re-extraction | Matches within tolerance |
| 4.5 | Copy features off the pod, then shut the pod down | Pod stopped, final cost logged |

Stop: spend reaches 80% of the cap with under 80% of clips done: pause, reduce scope by amendment (drop the optional families first). 

## Stage 5 - Baselines and controls (CPU, can overlap with Stage 4)

| Step | Action | Purpose |
|---|---|---|
| 5.1 | Same Jacobian, cross-context cosine, rank, metric, composition analysis in pixel space | Image-statistics baseline |
| 5.2 | Same in optical-flow features | Motion baseline |
| 5.3 | Same in tracked-trajectory vector (x, z, v, impact time) | Pure-kinematics baseline; the key yardstick for the effect-vs-cause confound |
| 5.4 | Random-weight V-JEPA of the same architecture | Architecture-only baseline |
| 5.5 | Shuffled intervention labels; unchanged duplicate renders; unrelated-parameter interventions | Negative controls |
| 5.6 | Handcrafted state-vector Jacobian for composition | Is additivity trivial in simple features? |

Acceptance: every baseline run through the identical code path as V-JEPA features, with identical bootstrap. Deliverable: baseline tables.

## Stage 6 - Primary analysis (held-out worlds)

| Step | Action |
|---|---|
| 6.1 | Unblind held-out worlds for the first and only time; run the frozen primary endpoint |
| 6.2 | Report layerwise: effective rank (observable parameters only), induced metric G with standardized columns, principal angles to nuisance, cross-context gravity alignment vs shuffled null and baselines, composition error vs baselines |
| 6.3 | Base-world bootstrap 95% intervals and effect sizes; layers treated as repeated measures; secondary metrics labeled exploratory |
| 6.4 | Sensitivity: raw vs L2-normalized features, global vs object-window vs full-grid pooling, delta/2 and 2delta on a subset |
| 6.5 | Fill the claims table with observed, unsupported, not-yet-tested |

Decision rule (Gate 3): the pilot supports a "mechanism-like geometry" claim only if the primary contrast beats the trajectory baseline with the lower bound > 0 and negative controls are null. If it matches the trajectory baseline, report "kinematic-consistent, not distinguished from effect-based representation". If nothing exceeds noise, report null.

## Stage 7 - Cause-vs-effect extension (only if budget remains)

| Step | Action | Acceptance |
|---|---|---|
| 7.1 | Construct gravity and initial-velocity interventions with matched visible trajectories; verify the matching numerically | Trajectory difference between the two interventions below a stated tolerance |
| 7.2 | Extract features, compute the angle between v_g and v_v at each layer | Compared with the angle expected under identical effects |
| 7.3 | Same-cause-different-effect: g change across free fall, bounce, incline, pendulum (add pendulum) | Cross-context alignment vs baseline |

## Stage 8 - Report and decision

| Step | Action |
|---|---|
| 8.1 | `reports/pilot_results.md` with observed / unsupported / not yet tested; no simulated or illustrative figures presented as results |
| 8.2 | Release manifest: code commit, config hash, checkpoint hash, data hashes, cost and throughput log |
| 8.3 | Go/no-go for the expanded study (more scenarios, token-level Jacobians, second-order terms with 2delta/3delta steps, other models) based on measured evidence and cost |

---

## Critical path and parallelism

Stage 0 -> 1 (can start before 2) -> 2 -> 3 -> 4 -> 6 -> 8. Stage 5 starts as soon as Stage 1 data exists. Stage 7 follows 6 only if budget remains. Stage 1 steps 1.1-1.4 and Stage 2 can proceed in parallel on CPU and GPU respectively, but 2.7 requires 2.2.

## Budget and time (estimates, not quotes)

| Stage | Compute | Rough time | Rough cost |
|---|---|---|---|
| 0, 1, 5 | CPU | 2-4 days of work | none |
| 2, 3 | 24-48 GB GPU | 4-9 h | about $2-5 |
| 4 | same | 2-8 h | about $1-5 |
| 7 | same | 2-4 h | about $1-2 |

The connected RunPod balance (about $9.42 at last check) is below the $25 protocol ceiling. Stages 2-4 likely fit if throughput is reasonable; Stage 7 may not.

## Decisions needed from you before Stage 0 can freeze

1. Primary hypothesis/endpoint (proposed: H2 vs trajectory baseline).
2. Primary representation (proposed: object-window pooled tokens).
3. Budget: top up or accept dropping Stage 7.
4. Approval to rent the Stage 2 pod.
5. Where this lives: the new repo you create, or stay in this repo's `causal_geometry/` folder.

## Repository mapping

| Stage | Files |
|---|---|
| 0 | `configs/pilot.json`, `reports/claims.md`, `reports/power.md` |
| 1 | `cg/simulate.py` (MuJoCo backend), `scripts/build_manifest.py`, `data/manifests/` |
| 2-3 | `cg/extract.py`, `scripts/gate0.py`, `scripts/gate1.py`, `reports/gate0.md`, `reports/gate1.md` |
| 4 | `scripts/extract_all.py` |
| 5-6 | `cg/analysis.py`, `cg/baselines.py`, `cg/controls.py`, `scripts/run_primary.py` |
| 7-8 | `scripts/cause_vs_effect.py`, `reports/pilot_results.md` |
