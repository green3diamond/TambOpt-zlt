# Symbolic shower generator

A replacement for the neural shower chain (PointCountFM + AllShowers) built from a small number of fitted laws, each
of which can be read and checked. It is fitted to CORSIKA (`h5_files_v3`) and judged at detector level through the
pipeline kernel.

Two parts:

1. **Energy per plane.** A Gaisser-Hillas curve per species (`families.gh4lr_signed`) with a start depth. The curve's
   numbers have Johnson SU margins linear in log10 E and cos zenith, one Gaussian copula per class across species, and
   a per-plane correction with AR(1) noise (`latent`, `copula`, `sampler`).
2. **Cells within each plane.** Per annulus of in-plane radius about the shower axis: a multinomial-logit law for how
   the cells split over the annuli, a law for the energy per cell, a law for the cell count, and a median arrival
   delay. One residual pattern per species varies the concentration plane to plane, AR(1) along the shower
   (`generator`). Cells are placed at a random angle and radius within their annulus, so every shower is centred on
   its axis.

The CORSIKA files carry a preprocessing defect: a cell-key overflow moved 17-25% of shower-planes 0.7-4 km South.
`unwrap` repairs it on read, everywhere.

## Pipeline

Every stage writes into one output folder `OUT`. Steps marked CPU run on `test` or `shared`; arrays can use
`serial_requeue` (parts are written atomically and skipped when already on disk).

```bash
P="python -u symbolic"
# phase 1: energy per plane
$P/m1.py --out $OUT --stage fit                                   # CPU, ~30 min: profile fits on the training slice and test file
$P/m2.py --out $OUT --stage part --n-parts 26 --part $i           # CPU array, i = 0..25
$P/m2.py --out $OUT --stage merge --n-parts 26
$P/m2.py --out $OUT --stage latent
$P/score.py --out $OUT                                            # sampler vs CORSIKA and the neural generator, energy score
# phase 2: cells within each plane
$P/annuli.py --out $OUT --stage hist --n-parts 10 --part $i       # CPU array, i = 0..9
$P/annuli.py --out $OUT --stage merge --n-parts 10
$P/generator.py --out $OUT --stage fit                            # CPU, ~20 min -> OUT/generator/model.npz
$P/generator.py --out $OUT --stage draw                           # one cloud per scored test primary
$P/generator.py --out $OUT --stage draw --oracle                  # CORSIKA's own plane energies (tests the in-plane laws alone)
# detector check (one GPU, ~5 min)
$P/detector.py --root $OUT --clouds generator/clouds.npz --chain chain/clouds_draw00.npz --out generator/check \
    --heldout <run folder>/v6_run_00/cashed_showers_tau_dual_heldout.pt
```

The scored test primaries are the rows of `OUT/chain/clouds_draw00.npz` (the neural chain's clouds at 1,000 test
primaries, 500 per class). `--heldout` must name a run whose held-out corpus has its `_positions.pt` file next to it;
run 12 (`12_10pct_mean_analytic`) has one, run 13 does not. Self-tests: `python tests/analysis/test_symbolic.py`.

A complete set of outputs, including the fitted model, is in
`/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/symbolic` (`generator/model_clean.npz`).

## Where it stands

1,000 test primaries against repaired CORSIKA. "Oracle" uses CORSIKA's own plane energies.

| | symbolic | symbolic, oracle | neural chain |
|---|---|---|---|
| energy per plane, energy score vs the neural generator (EM / hadronic) | -0.24 / -0.23 (better) | | 0 |
| shower centred on its axis | yes | yes | 150-900 m off |
| deposited energy, median per-event ratio (EM / hadronic) | 0.97 / 0.62 | 1.00 / 0.96 | 0.51 / 0.32 |
| leading-edge time shift, median | 0 ns | 0 ns | -37 / -70 ns |
| lit detectors, relative (EM / hadronic) | +7.8% / +20.6% | +8.3% / +13.6% | +1.5% / +24.2% |

Within 200 m of the axis the symbolic model lights the same detectors as CORSIKA; its excess sits beyond 200 m, where
detectors are near threshold and the model spreads the same energy more evenly than real planes do. The neural chain's
small EM total is a cancellation: it misses detectors near the axis (its showers are off axis) and adds detectors far
out.

Open: the hadronic excess has two parts, the sampled plane energies (about 8 points) and the in-plane laws (about 7);
plane-to-plane variety of the lateral profile is not fully captured by one pattern per species.
