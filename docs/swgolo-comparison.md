# SWGOLO vs TambOpt: why one layout optimizer finds structure and the other spreads uniformly

Session of 2026-09-28. Compares the SWGO layout optimizer (T. Dorigo,
https://github.com/tdorigo/SWGOLO, cloned to `~/tambo/SWGOLO`) with TambOpt's
Stage 4, then tests the leading explanation. All numbers below were measured in
this session; job IDs are given so the logs can be re-read in `slurm_logs/`.

## Findings (current state)

**Verdict so far.** TambOpt settles on a uniform fill because its objective is
nearly flat around one: against the kernel, a grid, a uniform fill and a
rim-heavy layout all score within ~0.7 of each other (run 12, §8). The leading
explanation, that TambOpt lacks SWGO's acceptance term, is **not supported at
σ = 50**: adding the term reshapes the layout but does not improve the true
utility, because acceptance is already ~89% saturated. It is **still open at
σ = 10**, where acceptance is far from saturated but the result has not been
checked against the kernel. The other structural differences from SWGOLO (how
the score is built, whether the reconstruction adapts to the layout, step size)
have not been tested.

**Update (§9): σ = 10 is less flat.** Scored against the kernel at σ = 10, five
layouts span 7.5% of the utility (vs 1.6% at σ = 50), and the acceptance-driven
spine optimized at σ = 10 scores best (+3.1% over the grid). So a working σ = 10
pipeline is likely to find more structure than σ = 50. The gains are small and
not yet bootstrapped.

What is the same in both codes: TambOpt's utility functions were ported
directly from SWGOLO's notebook (`SWGOLO7.ipynb`). `reconstructability`,
`U_PR`, `U_E`, `U_angle` and `LearnableXY` are the same code, so the difference
is not the per-event score. SWGOLO's dense-core / sparse-halo layouts come from
a **trade-off between resolution (reward packing) and acceptance (reward
catching more showers)**, over shower cores thrown uniformly over an area larger
than the array (§1).

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | SWGO's structure needs cores to land unevenly relative to the array; TambOpt's taus don't | **Confirmed absent in TambOpt** | 100% of tau axes hit the mesh; interior illumination 0.92–1.15 of uniform, same in every energy bin (§2) |
| H2 | TambOpt lacks the acceptance term (`U_PR` is left out of the composite) | **Not supported at σ = 50; weakly supported at σ = 10** | σ = 50: λ = 640 moves detectors to the rim but scores 42.99 vs the control's 43.03 against the kernel (§8). σ = 10: the λ = 200 spine scores best against the kernel, 23.95 vs 23.75 for its control (§9) |
| H3 | The objective is nearly flat around a uniform fill | **Supported at σ = 50; much less so at σ = 10** | σ = 50: three layouts within 1.6% (§8). σ = 10: five layouts span 7.5% (§9); no error bars yet |
| H4 | Acceptance has room to move | **Depends on σ** | σ = 10: kernel mean r 0.57, 42% of events r < 0.5 (§4). σ = 50: mean r 0.89, bimodal, 80% fully lit / 11% dark (§8) |
| H5 | The surrogate misjudges acceptance | **Confirmed** | Optimistic in both runs: median lit 9.8 vs 6.2 (σ = 10); dark events 2% vs 11% (σ = 50) |
| H6 | The score type matters: frozen-recon point error vs SWGOLO's likelihood-based expected resolution | **Untested** | §1 |
| H7 | A reconstruction that doesn't adapt to the layout flattens the objective | **Untested** | SWGOLO refits (C++) or fine-tunes (notebook); TambOpt's recon is frozen (§1) |
| H8 | Kernel-in-the-loop optimization fails | **Untested fairly** | Run 12's attempt started from a ~30 m blob and moved 12–34 m (§3) |

Acceptance effect in detail: at σ = 50 the λ term only lifted the worst-lit
events (lit detectors, worst 10%: 7.8 → 10.4; r < 0.5: 10.5% → 10.0%), at no
cost to resolution. Run 14's spine and run 12's rim go in opposite directions,
so neither is a robust "SWGO-like" structure yet.

## 1. Architectural comparison

| | SWGOLO (`swgolo148.cpp`, `SWGOLO7.ipynb`) | TambOpt v6 Stage 4 |
|---|---|---|
| Forward model | Exact analytic particle density vs distance from core, `exp(-p1·R^p2)`; differentiable in detector position | Learned DeepSets surrogate predicting the **conditional mean** response (aleatoric floor R² ≈ 0.5–0.7); kernel variant uses 64 fixed events |
| Showers per step | Fresh showers at the current layout every step | Surrogate: resampled primaries. Kernel variant: the same 64 events every step |
| Where cores land | Uniform over ±3 km (notebook) / Rslack = 2 km (C++), array ~1 km | Tau axes selected to hit the mesh; illumination ~uniform (§2) |
| Score | C++: expected resolution from the Hessian of an exact likelihood (`inv_rms_E`), `dÊ/dR` via implicit function theorem, `U_GF = f_s/σ_fs`. Notebook: `U_θ + U_E + U_PR/sqrt(density)` | `(100·U_θ + 100·U_φ + 250·U_E)/1000`, each `r`-weighted 1/(err²+ε) from a frozen neural recon; `U_PR` excluded |
| Reconstruction vs layout | C++ refits every event by likelihood; notebook fine-tunes the recon every 5 steps on showers from the current layout | Recon frozen, trained on 7 fixed layout strategies; also takes (x, y) as direct inputs |
| Step size | Each detector's step normalised to ~0.25 × maxDispl (up to 400 m) with a floor; per-detector adaptive LR from the angle between consecutive moves | Adam, lr = 1 m/step |
| Starting layout | Rings to 1 km, or SWGO benchmark layouts | `center` = all 100 detectors inside ~30 m, or `grid` |

Evidence that SWGOLO's notebook layout really moved: rings (median radius
750 m, nearest-neighbour CV 0.17) → final median radius 1,800 m, NN CV 0.93;
mean displacement 1,477 m; utility 7,743 → 12,876 (first vs last 100 steps).
The outward push comes from its `U_PR/sqrt(density)` acceptance term.

## 2. Where tau shower axes hit the mesh

Job 49013148 (`serial_requeue`). Every tau in `tau_wholesky_n750k.h5` within
the 1e5–1e7 GeV band (602,278 events) ray-cast onto the 266 detector-region
faces (1.377 km²).

- **100% of axes hit the mesh.** `tau_wholesky.jl` samples directions
  isotropically in zenith 60–100°, azimuth 0–360°, then keeps only taus whose
  forward ray crosses the detector mesh (`before_obs_mesh`) and that decay
  ≥ 1 km from it. Direction is not computed from the array location; the cut
  selects it.
- **Interior illumination is flat.** Density relative to uniform, by equal-area
  distance rings from the mesh centre: 0.93 / 1.01 / 0.92 / 0.92 / 1.15. Same
  in every energy bin (inner-half / outer-half density 0.91–0.95).
- Face-to-face CV 0.32 against a Poisson expectation of 0.03, but the excess
  comes from a few small steep faces at the NW and S tips (up to 3.4×): a
  surface-tilt effect, not a core.
- Decay → mesh distance p10/p50/p90 = 1.2 / 3.1 / 7.2 km.
  |cos(axis, face normal)| median 0.30, so footprints are stretched ~3.3× along
  the slope.

Conclusion: SWGO's first ingredient (cores landing unevenly relative to the
array) does not exist here.

Output: `detector_optimization_v6/figures/eda_outputs/axis_impact_density/`
(`axis_impact_density.png`, top-left panel is the per-face heatmap;
`axis_impact_density.npz` holds every hit point, per-face counts and areas,
per-event log E, distance and incidence).

## 3. Run 12 revisited (`12_10pct_mean_analytic`)

- **Kernel-in-the-loop** (`04_optimize_lbfgs_kernel.py`, recon and analytic
  utilities): started from the `center` blob (median radius ~20 m), moved
  12–34 m on average in 200 Adam steps at lr = 1, and ended with 13–14
  detectors stacked. The analytic plane-fit direction error of ~82° (random
  ≈ 90°) and log-E correlation of 0.01–0.04 were measured **on that blob**, so
  they say nothing yet about whether kernel observables carry information.
- **Surrogate ensembles**: from `center` they spread into a near-uniform fill
  (median NN ~80 m, NN CV ~0.5), with true U roughly matching the grid
  baseline; from `grid` they gained nothing (ΔU_true +0.8).

## 4. Is acceptance saturated? (`U_PR` as a counterweight) — run 14, σ = 10

Jobs 49014005 (surrogate, 20k primaries) and 49014220 (kernel, held-out
events). `r` is the production reconstructability gate:
`lit = Σ sigmoid(5·(counts − 1))`, `r = sigmoid(0.2·(lit − 5))`.

| Layout | Source | Median lit detectors | Mean r | r < 0.5 | r > 0.95 |
|---|---|---|---|---|---|
| grid | surrogate | 9.8 | 0.67 | 28% | 11% |
| grid | kernel (1,024 events) | 6.2 | 0.57 | 42% | 3% |
| run-12 optimized | kernel | 6.1 | 0.56 | 42% | 2% |
| center blob | kernel | 0.7 | 0.39 | 86% | 11% |

- Not saturated: `U_PR` has a real layout signal. The surrogate is optimistic
  about acceptance (~10 vs ~6 lit detectors).
- Gradient scale: at the current `W_PR`, `|∇u_pr|` is ~200–1000× the combined
  resolution terms'. `λ·mean(r)` reaches gradient parity at λ ≈ 200 (surrogate)
  / ≈ 240 (kernel, 48 events).

## 5. The acceptance test — run 14, σ = 10

New utility (wrapper only; `modules/optimize/objective.py` is unchanged):

```
U' = U_res + λ · mean(r),   λ = 200
U_res = (100·U_θ + 100·U_φ + 250·U_E) / 1000      (production composite)
```

`mean(r)` replaces `sqrt(Σr)` so the term's strength does not change with batch
size (5k Adam / 6k L-BFGS).

Jobs 49014281 (λ = 0, control) and 49014282 (λ = 200): one chain each, grid
init (identical starts), run-14 models, 5,000 Adam steps + 500 L-BFGS
iterations, ~20 min each.

| | Grid start | λ = 0 | λ = 200 |
|---|---|---|---|
| Median NN spacing | 55 m | 64 m | **27 m** |
| NN spacing CV | 0.54 | 0.73 | **1.01** |
| Neighbours within 100 m, inner / outer half | 2.2 / 1.6 | 3.4 / 1.5 | **6.8 / 3.2** |
| Median radius from mesh centre | 448 m | 446 m | 394 m |
| Mean displacement from start | — | 99 m | 144 m |
| Mean r (Adam, start → end) | — | 0.671 → 0.685 | 0.680 → 0.699 |
| Surrogate U_res (Adam, end) | — | 37.65 | ≈37.6 (θ 28.9, φ 2.9, E 5.7) |

With λ = 200 the detectors form a dense spine along the mesh's NW–SE long axis
with sparse flanks: a line-shaped analogue of SWGO's dense core and sparse halo.
Resolution terms held level while acceptance rose, so the structure is bought by
the acceptance term, not at the cost of resolution.

Outputs:
`14_sigma10_10pct/test_v6_run_04_optimize_lbfgs_ensemble_upr{0,200}_grid/`
(layouts, logs, trajectories, standard plots) and
`14_sigma10_10pct/upr_layouts.npz` (start, Adam-end and best layouts for both).

### Not verified against the kernel

The kernel scoring for these run-14 layouts (job 49018463) was cancelled before
it ran, when the work moved to run 12 (§8). **The λ = 200 spine is a surrogate
result.** THEORY.md §11.3 documents surrogate gains that did not survive the
kernel, and §8 shows run 12's reshaping did not.

## 6. Caveats found along the way

- **Run-14 checkpoints do not load on branch `stage2-mean-only`.** They were
  trained with the old heteroscedastic head (decoder outputs μ_E, μ_T,
  logvar_E, logvar_T); commit 31633e0 made the DeepSets 2-output. The
  diagnostics here slice the final layer to its first two rows, which
  reproduces the old mean forward exactly. Any other Stage 4 script pointed at
  run 14 will fail at load until Stage 2 is retrained mean-only or the
  checkpoints are converted.
- λ was set by gradient parity on one layout; it is a starting point, not a
  tuned value.
- One chain per λ, one init scheme (grid). No ensemble spread yet, and no
  error bars on the kernel scores (§8). Differences of a few tenths of a unit on
  2,048 events should be treated as noise until measured.

## 7. Suggested next steps

1. Put error bars on the kernel scores (bootstrap over the held-out events) to
   tell whether λ = 0's +0.67 over the grid start in run 12 is real.
2. Score run 14's λ = 0 / λ = 200 layouts against the kernel (on a σ = 10
   checkout). If the spine also leaves the true utility unchanged, the
   acceptance trade-off is not what TambOpt is missing.
3. If the objective is genuinely flat around a uniform fill (§8), structure
   will not emerge from any weighting of these terms. The candidates are a
   resolution-based score (e.g. the expected direction uncertainty of a
   plane-front fit, as SWGOLO's C++ uses) and a recon refit or fine-tuned as the
   layout moves.
4. Rerun kernel-in-the-loop fairly: grid init, SWGOLO-style normalised
   per-detector steps (~100 m, decaying), fresh held-out events each step.

## 8. Run 12 repeat (σ = 50, NLL-head surrogates)

Branch `swgolo-comparison-run12`, based on `run12-nll-sigma50` (commit 5bd292b,
the last commit before the run-14 config, plus `RUN_LOCATION` set to
`12_10pct_mean_analytic`). This checkout has `SIGMA_SPATIAL = 50`, the NLL-head
DeepSets that loads run 12's 4-output checkpoints natively, and
`RECONSTRUCT_THRESHOLD = 10` (r = 0.5 at 10 lit detectors). The surrogate is
scored with its mean (`sample=False`), matching the ensemble's
`SAMPLE_SURROGATE = False`.

**Axis impact (job 49019687):** identical to §2. It depends only on the tau file
and the mesh, which the two runs share.

**Acceptance saturation (job 49019688):**

| Grid layout | Run 14 (σ = 10) | Run 12 (σ = 50) |
|---|---|---|
| Median lit detectors (kernel) | 6.2 | 41.7 |
| Mean r (kernel) | 0.57 | 0.89 |
| r < 0.5 (kernel) | 42% | 11% |
| r > 0.95 (kernel) | 3% | 80% |
| Mean r (surrogate) | 0.67 | 0.97 |
| Gradient-parity λ | ≈ 200 | ≈ 640 |

At σ = 50 acceptance is bimodal: an event lights dozens of detectors or almost
none. The surrogate undercounts the dark fraction (2% vs 11% in the kernel).

**Chains (jobs 49020922 λ = 0, 49020923 λ = 640):** one chain each, grid init
(identical starts), 5,000 Adam + 500 L-BFGS.

| | Grid start | λ = 0 | λ = 640 |
|---|---|---|---|
| Median NN spacing | 55 m | 84 m | 73 m |
| NN spacing CV | 0.54 | 0.44 | 0.70 |
| Neighbours within 100 m, inner / outer half | 2.2 / 1.6 | 1.2 / 1.4 | 1.6 / 3.0 |
| Median radius from mesh centre | 448 m | 498 m | 566 m |
| Mean displacement from start | — | 83 m | 142 m |
| Surrogate mean r (Adam, start → end) | — | 0.966 → 0.970 | 0.968 → 0.972 |
| Surrogate U_res (Adam, end) | — | 62.63 | 62.48 |

The control relaxes into an even, lattice-like fill: the "uniform" behaviour.
With λ = 640 the centre empties and detectors move to the rim, with a clump in
the NW corner: the opposite of run 14's spine.

**Kernel scoring (job 49022246), 2,048 held-out events,** plain U_res (the λ
term is not added; see "How the kernel score is computed" below):

| Layout | U_res | u_θ | u_φ | u_E | Mean r | r < 0.5 | Lit p10 / p50 / p90 |
|---|---|---|---|---|---|---|---|
| grid start | 42.36 | 33.01 | 3.57 | 5.78 | 0.892 | 10.5% | 7.8 / 41.9 / 70.1 |
| λ = 0 | 43.03 | 33.54 | 3.66 | 5.82 | 0.895 | 10.6% | 8.7 / 39.2 / 65.3 |
| λ = 640 | 42.99 | 33.43 | 3.67 | 5.89 | 0.897 | 10.0% | 10.4 / 37.3 / 62.7 |

Detector elevation spread: 266 / 283 / 293 m; shower-depth (z_cont) spread:
0.60 / 0.64 / 0.72 layers.

Reading:

- **The rim layout is not better than the uniform one against the truth.**
  λ = 640 and λ = 0 differ by 0.04 in U_res, well inside noise at this sample
  size. Both sit about 0.65 (≈1.5%) above the grid start, which is itself
  unconfirmed without error bars.
- The acceptance term did its narrow job: it lifted the worst-lit events
  (p10 lit detectors 7.8 → 10.4, r < 0.5 from 10.5% to 10.0%), at a small cost
  in median lit count (41.9 → 37.3) and no cost in resolution.
- At σ = 50, very different layouts score the same. The objective is nearly flat
  around a uniform fill, which is why the unconstrained optimizer lands there:
  nothing in U_res prefers a structure. Adding acceptance does not change that,
  because acceptance is already ~89% saturated.

Outputs:
`12_10pct_mean_analytic/test_v6_run_04_optimize_lbfgs_ensemble_upr{0,640}_grid/`,
`12_10pct_mean_analytic/upr_layouts.npz`,
`12_10pct_mean_analytic/diagnostics/upr_layouts.png` (side-by-side layouts) and
`12_10pct_mean_analytic/diagnostics/axis_impact_density/`.

### How the kernel score is computed

`scripts/diagnostics/eval_layouts_kernel.py` uses the optimizer's formula with
kernel readings in place of the surrogate's:

1. The first 2,048 held-out events (never seen by Stages 1–4): real AllShowers
   clouds per species, placed at the tau's decay vertex along its direction,
   with the true (E, θ, φ).
2. Per detector, `weight = exp(−d²/2σ²) · max(0, 1 − |layer − z_cont|)` with
   σ = `SIGMA_SPATIAL`; E = Σ energy · weight, T = earliest detected time. Species
   combined (counts add), log1p-encoded.
3. The run's frozen `recon.pt` gives (Ê, θ̂, φ̂).
4. `r = sigmoid(0.2·(lit − 10))`, `lit = Σ sigmoid(5·(counts − 1))`;
   U_θ, U_φ, U_E are r-weighted capped `1/(err² + ε)` (caps 500 / 50 / 80);
   `U_res = (100·U_θ + 100·U_φ + 250·U_E)/1000`. No gradients.

## 9. Is σ = 10 less flat? Kernel scoring at σ = 10

Job 49025182: `eval_layouts_kernel.py --run-location 14_sigma10_10pct --sigma 10
--reconstruct-threshold 5`. Run 14's first 2,048 held-out events and run 14's
recon; kernel only, no surrogate. The same five layouts are scored on the same
events, so the comparison is paired.

| Layout | U_res | u_θ | u_φ | u_E | Mean r | r < 0.5 | Median lit | z_cont spread |
|---|---|---|---|---|---|---|---|---|
| run-14 spine (λ = 200) | **23.95** | 19.44 | 1.56 | 2.95 | 0.595 | 38.4% | 7.6 | 0.49 |
| run-14 λ = 0 | 23.75 | 19.15 | 1.68 | 2.92 | 0.584 | 39.5% | 7.1 | 0.57 |
| grid start | 23.22 | 18.75 | 1.54 | 2.93 | 0.575 | 40.6% | 6.6 | 0.60 |
| run-12 uniform (λ = 0) | 22.94 | 18.50 | 1.60 | 2.83 | 0.562 | 40.6% | 6.5 | 0.64 |
| run-12 rim (λ = 640) | 22.20 | 17.86 | 1.56 | 2.79 | 0.550 | 45.3% | 5.6 | 0.72 |

Reading:

- **σ = 10 is less flat than σ = 50.** The five layouts span 1.75 (7.5% of the
  mean) against 0.67 (1.6%) for the three layouts scored at σ = 50 (§8).
- **The layouts optimized at σ = 10 beat the grid; the ones optimized at σ = 50
  lose to it.** The spine is best (+0.73 over the grid, +3.1%), ahead of run 14's
  control (+0.53). Run 12's rim, neutral at σ = 50, is worst here (−1.02).
- The ordering in U_res matches the ordering in acceptance (mean r, fraction
  under-lit, median lit), so the gains come through lighting more detectors per
  event.
- The spine concentrates detectors in shower depth: z_cont spread 0.49 layers
  vs 0.60 for the grid.
- This supports H2 at σ = 10 (the acceptance term found a layout that is
  better against the truth) and suggests the flatness in H3 is partly a σ = 50
  effect. Effect sizes are small and there are no error bars yet; the paired
  design helps, but a bootstrap over events should confirm the ordering.

## Files added this session

| File | Purpose |
|---|---|
| `scripts/diagnostics/axis_impact_density.py` + `slurm/axis_impact_density.sh` | §2 axis-impact heatmap |
| `scripts/diagnostics/upr_saturation.py` + `slurm/upr_saturation.sh` | §4 r distribution and per-term gradient norms (surrogate and kernel) |
| `scripts/diagnostics/ensemble_with_upr.py` + `slurm/ensemble_with_upr.sh` | §5 runs the production ensemble with `λ·mean(r)` added; `sbatch slurm/ensemble_with_upr.sh <λ> <suffix>` |
| `scripts/diagnostics/eval_layouts_kernel.py` + `slurm/eval_layouts_kernel.sh` | §8 kernel scoring of saved layouts |

All committed on branch `swgolo-comparison-run12`. They run against whatever
`modules/constants.py` points at; on this branch that is run 12 at σ = 50.
