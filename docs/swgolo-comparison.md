# SWGOLO vs TambOpt: why one layout optimizer finds structure and the other spreads uniformly

Session of 2026-09-28. Compares the SWGO layout optimizer (T. Dorigo,
https://github.com/tdorigo/SWGOLO, cloned to `~/tambo/SWGOLO`) with TambOpt's
Stage 4, then tests the leading explanation. All numbers below were measured in
this session; job IDs are given so the logs can be re-read in `slurm_logs/`.

## TL;DR

- TambOpt's utility functions were ported directly from SWGOLO's notebook
  (`SWGOLO7.ipynb`): `reconstructability`, `U_PR`, `U_E`, `U_angle` and
  `LearnableXY` are the same code. The difference is not the per-event score.
- SWGOLO's dense-core / sparse-halo layouts come from a **trade-off between
  resolution (reward packing) and acceptance (reward catching more showers)**,
  over shower cores thrown uniformly over an area larger than the array.
- In TambOpt both of those are weak or missing:
  - every tau axis hits the mesh by construction, and the mesh is lit almost
    uniformly (measured below), so no region is geometrically preferred;
  - `U_PR` is computed but left out of the composite, so there is no
    acceptance term at all.
- Acceptance is **not** saturated in TambOpt (median event lights ~6 detectors
  in the kernel, 42% have r < 0.5), so an acceptance term has a gradient to
  work with.
- Adding `λ·mean(r)` at gradient parity (λ = 200) turned the grid-start optimum
  from a near-uniform spread into a **dense spine along the mesh's long axis
  with sparse flanks**. This is a surrogate result; the kernel check is pending.
- Run 12's kernel-in-the-loop runs never tested the idea fairly: they started
  from a ~30 m "center" blob and moved only 12–34 m.

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

## 4. Is acceptance saturated? (`U_PR` as a counterweight)

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

## 5. The acceptance test

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

### Pending

Job 49018463 (`eval_layouts_kernel.sh`) scores the start, λ = 0 and λ = 200
layouts against the kernel on 2,048 held-out events, and reports detector
elevation and shower-depth (z_cont) spread to show what the spine lines up with.
It was still queued when this was written. **Until it reports, the λ = 200
structure is a surrogate result** (THEORY.md §11.3 documents surrogate gains
that did not survive the kernel).

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
- One chain per λ, one init scheme (grid). No ensemble spread yet.

## 7. Suggested next steps

1. Read job 49018463. If the kernel confirms λ = 200 beats λ = 0 on resolution
   and acceptance, the missing trade-off was the cause.
2. Sweep λ (e.g. 50 / 200 / 800) and add the `center` init, with ensembles, to
   see whether the spine is robust.
3. Rerun kernel-in-the-loop fairly: grid init, SWGOLO-style normalised
   per-detector steps (~100 m, decaying), fresh held-out events each step.
4. Longer term, replace the frozen-recon score with a resolution-based one
   (e.g. the expected direction uncertainty of a plane-front fit), as SWGOLO's
   C++ does, and fine-tune or refit the recon as the layout moves.

## Files added this session (uncommitted)

| File | Purpose |
|---|---|
| `scripts/diagnostics/axis_impact_density.py` + `slurm/axis_impact_density.sh` | §2 axis-impact heatmap |
| `scripts/diagnostics/upr_saturation.py` + `slurm/upr_saturation.sh` | §4 r distribution and per-term gradient norms (surrogate and kernel) |
| `scripts/diagnostics/ensemble_with_upr.py` + `slurm/ensemble_with_upr.sh` | §5 runs the production ensemble with `λ·mean(r)` added; `sbatch slurm/ensemble_with_upr.sh <λ> <suffix>` |
| `scripts/diagnostics/eval_layouts_kernel.py` + `slurm/eval_layouts_kernel.sh` | §5 kernel scoring of saved layouts |
