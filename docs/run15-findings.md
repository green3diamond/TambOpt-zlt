# Run 15 findings, and why TambOpt layouts lack SWGO-like structure

Session of 2026-09-29, branch `run15-interp-hurdle`. Run folder:
`detector_optimization_v6/15_sigma10_interp_hurdle_10pct/`. The SWGOLO comparison
this builds on is `docs/swgolo-comparison.md` on branch `swgolo-comparison-run12`.

## Summary

- **Why SWGO finds a dense core / sparse halo and TambOpt a uniform fill.** SWGO
  throws shower cores uniformly over an area (±3 km) larger than its array can
  cover, so a fixed number of detectors must trade density (resolution) against
  area (acceptance). TambOpt keeps only taus whose axis crosses the detector
  mesh, and detectors are confined to that same mesh, so filling the mesh
  evenly already covers every shower in the corpus. The trade-off never arises.
  This is an **experiment-design choice** (a corpus-efficiency cut in
  `tau_wholesky.jl`), not a physical difference: the flux is uniform in both.
- **Implementation differences are not the cause.** Surrogate vs simulator in
  the loop, frozen vs adapting recon, optimizer, step size and init decide how
  well an optimum is found, not where it is.
- **Run 15 acceptance is saturated.** With the interpolated kernel, every
  spread-out layout has kernel mean r ≈ 0.88 (run 14's raw σ = 10 kernel: 0.57).
  An acceptance term (λ·U_PR) has nothing to act on, as at σ = 50. Skip it
  until near-miss taus are in the corpus.
- **Nothing beats the grid.** Kernel-in-the-loop from grid: −4% true U
  (overfits 64 events); surrogate ensemble from grid: +1.2% (§5).
- **Next step: new primaries** that include taus whose axis misses the mesh but
  passes close to it (requested from the tau-corpus author, §6).

## 1. Run 15 setup

Run 14's σ = 10 pipeline with two changes, 10% of the data, Stage 0 moved from
run 14 (run 14 keeps a symlink):

- **Interpolated kernel** (`KERNEL_INTERP_K = 8`): each cloud point's Gaussian
  widens to its 8th-nearest in-plane neighbour distance (floor σ = 10 m),
  area-normalised, so a detector between sparse macro-particles reads their
  local density instead of ~0. Leading-edge T.
- **Hurdle Stage 2** (`SURROGATE_HURDLE = True`): the surrogate predicts
  P(lit) and a lit-only mean for E and T; loss = BCE(lit) + MSE on lit slots;
  forward = P(lit)·mean.

## 2. Pipeline results

| Stage | Result |
|---|---|
| 1 labels | Lit fraction E ≈ 92%, T ≈ 45–49% (run 14 raw kernel: E 22%, T 5%) |
| 2 surrogate | Best val (hurdle loss, not comparable to run 14): electron 0.342, muon 0.392, photon 0.360 |
| 3 recon | Val 0.066 (run 14: 0.142) |
| 4 ensemble (surrogate U) | Grid: best 58.98 after L-BFGS, mean spread σ ≈ 30 m; true U 30.48 → 30.85 (+1.2%). Center: true U 7.13 → 30.79 (grid baseline 30.48) |

## 3. Acceptance saturation on run 15

Job 49173187, `scripts/diagnostics/upr_saturation.py`. r = sigmoid(0.2·(lit − 5)),
lit = Σ sigmoid(5·(counts − 1)). r ≈ 0.296 means zero lit detectors.

Kernel, 1,024 held-out events, no gradient:

| Layout | mean r | frac r < 0.5 | frac r > 0.95 |
|---|---|---|---|
| grid | 0.887 | 0.106 | 0.701 |
| run 12 uniform (λ = 0) | 0.887 | 0.099 | 0.696 |
| run 12 optimized (center) | 0.886 | 0.100 | 0.697 |
| run 14 spine (λ = 200) | 0.881 | 0.121 | 0.698 |
| center blob | 0.721 | 0.393 | 0.580 |

- Every spread-out layout sits at 0.881–0.887. The ~10% of dark events have
  zero lit detectors in all of them, so no layout change reaches them.
- The surrogate is more optimistic still (mean r 0.95, 4–5% below 0.5).
- Weight on U_PR for gradient parity with the resolution terms: 0.2–0.6 on the
  kernel, 0.017–0.028 on the surrogate.

**Softening the gate** (lower `TAU_RECONSTRUCT`) would lower the numbers but
turn U_PR into a "more lit detectors per shower" reward, which is roughly
layout-independent under flat illumination. It also reweights U_θ and U_E.
Not pursued.

## 4. Raising the reconstruct threshold

Job 49177589, thresholds 5 / 10 / 15 / 20 / 30 lit detectors, same layouts plus
run 15's own Stage 4 grid result. Per threshold: mean r, fraction r < 0.5 and
the resolution utility on the kernel (1,024 events).

Kernel, 1,024 events, spread-out layouts only (grid, run 12 uniform, run 12
optimized, run 14 spine, run 15 Stage 4 grid):

| Threshold | mean r | frac r < 0.5 | U_res range | Spread |
|---|---|---|---|---|
| 5 | 0.881–0.888 | 0.10–0.12 | 29.84–30.76 | 3.0% |
| 10 | 0.824–0.829 | 0.15–0.17 | 27.77–28.52 | 2.7% |
| 15 | 0.762–0.766 | 0.24 | 25.54–26.09 | 2.1% |
| 20 | 0.698–0.710 | 0.30–0.31 | 23.37–23.70 | 1.4% |
| 30 | 0.581–0.614 | 0.39–0.43 | 19.02–19.71 | 3.6% |

- A higher threshold removes the saturation (mean r 0.89 → 0.60, dark events
  10% → 41%) but lowers every layout by about the same amount. The layouts stay
  within 1.4–3.6% of each other at every threshold.
- The events that go dark are dark for every layout: with 100 detectors on
  1.4 km² and flat illumination, no placement lights more detectors on average.
- Only hint of separation: at threshold 30 the acceptance-driven run 14 spine
  has the best mean r (0.614 vs 0.58–0.60) and U_res (19.71 vs 19.02–19.57).
  Small, 1,024 events, no error bars.
- Caveat: these layouts were optimized at threshold 5 or 10. A Stage 4 run
  optimized at a high threshold was not tested.

## 5. Kernel-in-the-loop optimization

### Why run 12's attempt got stuck

`12_10pct_mean_analytic/test_v6_run_04_optimize_lbfgs_kernel_recon_utility`
(σ = 50, recon utility, 64 events, Adam lr = 1, 200 epochs, `--init center`):

1. **Blob start.** All 100 detectors within ~30 m. Co-located detectors get the
   same gradient and never separate: final median radius 16 m, 28 pairs < 1 m apart.
2. **Tiny, noisy steps.** At most 200 m of travel possible; actual 12–34 m,
   because the 64-event gradient flips sign and Adam's momentum cancels. U was
   still rising at epoch 200 (10.18 → 10.36 over the last 40): it ran out of
   steps, it did not converge.
3. **Local kernel gradients.** A shower passing > ~150 m from the blob gives
   ~zero gradient, so the optimizer is not pulled toward showers it does not
   already touch. Mean r 0.579 → 0.586.
4. **Fixed events.** U on the 64 optimized events +47% (7.08 → 10.37); held-out
   true U +0.3% (9.77 → 9.80).

### Run 15 rerun

Jobs 49175728 (center) and 49175729 (grid), `slurm/kernel_opt_run15.sh`:
recon utility, Adam lr = 10, 20,000 epochs, 64 events, 0.12 s/epoch. Outputs in
`test_v6_run_04_optimize_lbfgs_kernel_recon_utility_lr10_{center,grid}/`.

True utility on 5,120 held-out events (`true_utility_eval.txt`), vs run 15's
surrogate ensemble:

| Run | Start → optimized (true U) | U on its own 64 events |
|---|---|---|
| Kernel, center init | 7.13 → 26.13 | 43.3 |
| Kernel, grid init | 30.48 → 29.24 (−4%) | 47.0 |
| Surrogate ensemble, grid init | 30.48 → 30.85 (+1.2%) | — |

- Larger steps fix the stall: from the blob the kernel run gains +19 true U, but
  still ends below the plain grid (30.48).
- From the grid, kernel-in-the-loop overfits its 64 fixed events and loses 4%
  on held-out events. The surrogate ensemble gains 1.2%.
- Nothing measurably beats the grid. With flat illumination the objective has
  no structured optimum to find, consistent with the summary.

**Rerun on Stage 4's dataset (pending, overwrites the table above).** The runs
above used 64 fixed events; Stage 4 draws 5,000 fresh primaries per step from
1,125,000 rows (75,000 events × 3 species × 5 strategies) and picks its best
layout on a separate 20,000-primary scoring set. `--pool main` now uses the
same 75,000 events: 512 fresh events per step, backward in chunks of 32
(checked to reproduce the full-batch gradient), best layout on a separate
1,024-event scoring set. Jobs 49187556 (grid) and 49187557 (center), after the
width precompute (array job 49187518).

## 6. Near-miss primaries (requested)

Asked the tau-corpus author for a set that also keeps taus whose axis passes
within ~3 km of the Malata detector mesh (closest approach), not only those
that cross it, with the same energy band, zenith 60–100° and decay ≥ 1 km cut,
and with the closest-approach distance and a crosses-mesh flag stored per event
so the cut can be tightened later. Showers passing more than ~1 km away likely
light nothing, so a tighter cut will probably be used.

Even with these events, detectors stay confined to the mesh, so an SWGO-style
outward expansion is capped by the valley's edges.

## 7. Code changes on this branch

- `scripts/04_optimize_lbfgs_kernel.py`: `--ckpt-every` with resume from
  `opt_state.pt` (requeue-safe); interpolated-kernel bandwidths cached per
  chunk (`CachedKernel`, identical output); seconds-per-epoch and r in the log.
- `slurm/kernel_opt_run15.sh`: kernel run + true-utility and utility-vs-energy
  evals, guarded to run-15 constants.
- `scripts/diagnostics/upr_saturation.py`: `--layout name=path`,
  `--reconstruct-thresholds`, resolution utility on the 1,024-event pass, summary table.
- `plots/training/02_nn_target_vs_pred.py`: restored `SIGMA_QUANTILES`,
  `AXIS_PERCENTILE` and `BinnedProfile`, deleted by 31633e0 while still used.
