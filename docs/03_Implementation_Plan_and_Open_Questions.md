# Implementation Plan and Open Questions (Protocol A)

Status: plan, not results. Items marked **[VERIFY]** are facts about V-JEPA 2/2.1 I have not confirmed against the official repo and must be checked at Gate 0.

## 1. What exists vs. what is missing

| Component | State |
|---|---|
| Geometry metrics (`cg/geometry.py`) | Done, unit-tested on synthetic encoders |
| Layer report (`cg/analysis.py`) | Done for the pooled-token case; no baselines yet |
| Simulator | Numpy stand-in only. Not MuJoCo/PyBullet, no 3D rendering, low resolution |
| Extraction (`cg/extract.py`) | Hook wrapper works; real V-JEPA 2.1 loader unverified |
| Baselines (pixel, optical flow, trajectory state) | Not written |
| Controls (duplicate, shuffled labels, cause-vs-effect) | Only duplicate-render determinism |
| Manifest, storage, resume | Not written |

## 2. Work packages (dependency order)

**WP0 - Freeze decisions (no GPU).** Resolve section 3 items D1-D8 and write `configs/pilot.json` with the primary endpoint, primary layer summary and perturbation sizes *before* looking at real-model features. Commit as the preregistration.

**WP1 - Real simulator + renderer.** MuJoCo (preferred: contact, friction, restitution built in) with headless EGL rendering. Implement `trajectory(world, theta)` plus `render(...)` returning frames at the model's native input size. Keep the numpy scenarios as an analytic cross-check (bounce and free fall have closed forms). Acceptance: deterministic repeat renders are bit-identical; every intervention changes the relevant event (impact time, post-impact speed, sliding distance).

**WP2 - Gate 0 on GPU.** Load the model, record the checkpoint SHA256 and repo commit, print the module tree, confirm input shape and token order, confirm hooks give one tensor per block, check repeat-inference determinism and bf16/fp32 differences. Output: `reports/gate0.md`. Stop here if anything is uncertain.

**WP3 - Dataset builder.** 50 worlds, 200 clips, manifest CSV (world id, kind, theta, nuisance, seed, impact frames, trajectory file, video hash). 5 development + 5 held-out worlds per primary scenario. Acceptance: all clips valid or failures logged.

**WP4 - Feature extraction.** Stream per-block features to disk (float16), pooled and region-pooled variants, resumable by clip hash. Measure clips/sec on the first 20 before approving the rest.

**WP5 - Baselines and controls.** Same Jacobian/alignment/composition analysis in (a) pixel space, (b) optical-flow features, (c) tracked state/trajectory vector, (d) random-weight V-JEPA with the same architecture. Plus shuffled intervention labels and duplicate-render noise floor.

**WP6 - Analysis and report.** Layerwise tables with base-world bootstrap CIs, figures, and a report separating observed / unsupported / not yet tested. Go/no-go per protocol section 7.

**WP7 (extension if budget allows).** Cause-vs-effect: gravity vs initial-velocity interventions with matched visible trajectories.

## 3. Details still to be figured out

**D1. Input format of the real model [VERIFY].** V-JEPA 2 takes roughly 64 frames at 256x256 with 2-frame tubelets and 16x16 patches (about 8k tokens); 2.1 may differ. My simulator currently makes 16 frames at 64 px. Decide frame count, fps and resolution to match the checkpoint, and whether to resample or re-render.

**D2. Which representation is "z_l".** Mean over all tokens will largely be dominated by background for a small sphere and may erase the effect. Options: global mean, spatial-temporal pooled grid, tokens in a window around the object (needs tracked position), or the full token tensor with a per-token Jacobian. Pick one *primary* before looking at data; report the rest as sensitivity.

**D3. Which layers.** Block outputs vs. pre-norm vs. final normalized output. V-JEPA 2.1 reportedly adds intermediate-layer supervision **[VERIFY]**, which changes how layer trends should be read.

**D4. Perturbation size and units.** delta must exceed render/numeric noise yet stay local. g, e and mu have different units, so G entries are not comparable unless columns are standardized. Candidate rule: choose delta per parameter so the induced trajectory displacement is equal across parameters (and report the alternative of a prior-sd rule). Needs a pilot sweep at delta/2, delta, 2delta, with a fixed rule frozen before the held-out worlds.

**D5. Observability per scenario.** Free fall in a short clip may barely distinguish g by eye; bounce needs an impact in the window under both e- and e+, and g also moves impact time (a timing confound for e); incline needs sliding under both mu- and mu+. Need a rejection rule and a count of rejected worlds per scenario.

**D6. The central confound: effect vs. cause.** Gravity changes motion in every scenario, so cross-scenario alignment of v_g could simply be "things move faster/slower". The same analysis on the tracked-trajectory baseline (WP5) is the yardstick. The result is only interesting if V-JEPA's alignment exceeds it, or the cause-vs-effect extension (WP7) separates them. Decide the exact comparison statistic and what margin counts.

**D7. Rank statistic.** Three columns in thousands of dimensions: effective rank of three vectors is sensitive to column norms and noise. Define the noise floor from duplicate renders and nuisance replicates, report singular values, and avoid claiming rank for scenarios with fewer than three observable parameters (free fall gives one).

**D8. Nuisance comparison.** Camera shift and colour columns are 2 vectors vs. up to 3 physical ones; the minimum principal angle is optimistic. Decide on the full angle set or projection overlap, and calibrate nuisance magnitude so latent displacement is comparable to the physical perturbation.

**D9. Statistics.** Only 10 worlds per family, 5 held-out. The primary endpoint, layer summary (e.g., a mean over a predeclared layer band) and multiplicity policy must be fixed up front. Layers are repeated measures, not independent samples.

**D10. Composition test.** One-sided (+,+) cannot show curvature; adding (-,-) or a 2x step costs 10 to 20 more clips. Decide whether to spend them. Also decide the epsilon in the denominator.

**D11. Scenario design.** Sphere vs. cube; rolling vs. sliding on the incline (rolling couples inertia to the dynamics); camera and lighting fixed within pairs; contact parameters shared across engines.

**D12. Budget.** Protocol ceiling is $25; the connected RunPod balance is about $9.42. An A40/A6000 (48 GB) is roughly $0.35-0.59/hr, so Gate 0 through 2 likely fit if throughput is reasonable, but this is unmeasured. Decide whether to top up or cut scope (drop WP7 first).

**D13. Storage and compute.** All-layer all-token storage can dwarf the videos. Plan pooled + region-pooled by default and store full tokens only for a small subset.

## 4. Risks

| Risk | Mitigation |
|---|---|
| Object occupies few tokens, global pooling hides the signal | D2 region pooling; sanity-check with a pixel-space positive control |
| Interventions change trajectory, so latent change is just trajectory change | D6 baselines and WP7 |
| Loader/hook mapping wrong | Gate 0 stop condition |
| Over-reading small-sample layer trends | Predeclared summary, base-world bootstrap, held-out split |
| Simulator nondeterminism or EGL issues on the pod | Pin versions, duplicate-render test in WP1 |
| Budget overrun | Throughput gate after first 20 clips, hard stop at ceiling |

## 5. Gates and exit criteria

- **Gate 0:** model loads, hashes recorded, hooks verified, repeat inference matches. 
- **Gate 1 (20 clips):** deterministic, interventions alter events, finite differences above noise floor.
- **Gate 2 (200 clips):** all valid or documented, throughput and cost recorded.
- **Gate 3:** metrics with CIs, baselines and negative controls reported regardless of direction.
- **Gate 4:** go/no-go for an expanded study, a null result is a valid outcome.

## 6. Immediate next steps

1. Decide D1-D4 and D12 (needs your input on budget and primary representation).
2. Run Gate 0 on a small GPU pod.
3. Build the MuJoCo simulator (WP1) in parallel.
