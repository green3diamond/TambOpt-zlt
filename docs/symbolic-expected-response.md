# Expected-response test with Jay's symbolic shower model

Status: code written on branch `symbolic-expected` (2026-11-18), not run yet. Model: Jay's frozen folder
`/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/symbolic_model` (branch `symbolic-run20`, commit
3f4d390; the `symbolic/` package is copied from that commit). Job scripts, not submitted:
`tmp_rings/symbolic/check_expected.sh` (step 3, `N=64` first) and `tmp_rings/symbolic/grad_cos_expected.sh` (step 5).

## Why

Optimizing on CORSIKA 8 showers with the analytical reconstruction (SWGO trigger at 1 deg, `TAMBOPT_TIME_MODEL=mean`)
does not improve the layout. `scripts/paper2/grad_cos.py` (uniform grid, Malata) shows that the gradients of two
independent 5,120-shower batches are uncorrelated: cosine 0.07 with the time noise, -0.005 without it. Each step
follows its own batch. The causes considered:

1. batch-specific gradients (shower-to-shower scatter);
2. kernel locality: the 10-50 m kernel only sees showers whose points pass within ~50 m of a detector;
3. hard on/off switches (mu >= 1 hits, the >= 3-hit trigger);
4. the uniform grid may already be close to the optimum.

A smooth expected response evaluated on very large batches would remove 1 and 2 and test 4.

## The symbolic model

Branches `symbolic-run20` and `symbolic-surrogate` (Jay Agarwal), package `symbolic/`. Fitted laws, 3,574 numbers:

- energy per plane: Gaisser-Hillas profiles;
- each plane's energy over 23 rings around the axis, 1 m to 9 km;
- mean energy per particle vs radius;
- arrival times: t = A r^p k^-q (r: radius, k: plane).

Detector-level agreement with CORSIKA 8: same lit detectors within 200 m of the axis, median time shift 0 ns;
hadronic showers light about 20 % too many detectors.

Drawing particles from it is too slow for large batches (about 12 min per 1,000 events on 8 cores), so the test
uses the laws directly as a mean response, without draws.

## Plan

1. **Expected-response function (new code).** For each shower (energy, zenith, type, axis) and the detector
   positions, return per detector and species
   - the expected particle count mu, from the ring energy laws and the mean energy per particle (2 m^2, the
     detection efficiencies of the counting model);
   - the expected arrival time, from the time law, carried to the detector's depth along the axis.

   Differentiable in the detector position (Jay's `gradients.py` differentiates in energy and zenith only).
   Check against the kernel on CORSIKA 8 showers: mu and t vs distance from the axis.
2. **Plug into the utility** in place of `BatchKernel`: SWGO trigger at 1 deg, analytical reconstruction
   (`AnalyticRecon`), uniform grid start.
3. **Gradient agreement.** Rerun `grad_cos.py` with two independent batches of about 300k showers: cosine, sign
   agreement and per-detector cosine, with and without time noise; compare with the kernel (0.07).
4. **Decide.**
   - Cosine well above 0: optimize from the uniform grid and from the centre; score the layouts with the kernel on
     CORSIKA 8 (`score_all.py`, `SRC=c8`, `RECO=v1`) as for the other runs.
   - Cosine still near 0: the scatter is not a batch-size problem.

Cost: a few hours of code and one gpu_test job for steps 1-3.

## What it does not fix

- The hard switches (hits at mu >= 1, the 3-hit trigger) stay unless the trigger is softened (e.g. `TRIG_WIDTH=1`).
- If the grid is near the optimum, the result is still the grid; consistent gradients would confirm it.
- It optimizes for an average shower without fluctuations, so the reconstruction looks cleaner than in reality.
- Hadronic showers light about 20 % too many detectors in the symbolic model.

## As built (2026-11-18)

`modules/optimize/expected_response.py`:
- `tables()` (numpy, per shower and species): energy per plane from ONE part-1 draw of the sampler (seeded), ring
  shares from the rings law stretched by the MEAN width law, particles per ring = ring energy / mean particle energy
  (energy law), divided by the ring area -> log density [1/m^2] on (24 planes, 23 rings); delay constants (A, p, q).
- `ExpectedResponse(logd, tconst, frames, surface)`: the kernel's call signature and `.mu`. Per detector: distance
  from the axis and depth from `detectors_in_frames`; log density linear in ln r between ring centres, density
  linear between the two planes around the depth (zero past the last plane); mu_s = density x 2 m^2 x efficiency
  (photons 0.02); T = depth / c + mu-weighted mean delay exp(A) r^p (k+1)^-q. The E channel carries mu (not a kernel
  energy). No noise anywhere.

`scripts/symbolic/check_expected.py` (step 3): CORSIKA 8 test showers placed as `score_all.py SRC=c8`, kernel vs
expected at the uniform grid: hits per shower, trigger share, mean mu vs distance from the axis, delay behind the
plane front of detectors hit in both, analytical recon v1 within 1 deg and median.

`scripts/symbolic/grad_cos_expected.py` (step 5): primaries from the run-20 training corpus (2 x max B events, seed
4200), tables in parallel on CPU, gradients of U (SWGO trigger 1 deg, analytical recon) at the uniform grid on
[0, B) and [B, 2B) for each B (default 5,120 and 51,200), cosine / sign agreement / per-detector cosine.

`scripts/paper2/score_all.py` on this branch derives the repository path from its own location (it was the main
checkout's), so the worktree's modules load.
