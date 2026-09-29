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
| 4 ensemble (surrogate U) | Grid: best 58.98 after L-BFGS, mean spread σ ≈ 30 m. Center: Adam chains 59.5–59.7; L-BFGS and true-utility eval still running |

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

**Pending.** The question is whether a stricter definition of "reconstructable"
separates the layouts (today 0.881–0.887).

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

**Pending.** Held-out true utility vs the init layout.

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
