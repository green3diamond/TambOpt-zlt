# Symbolic shower generator

A replacement for the neural shower chain (PointCountFM + AllShowers) built from a few fitted laws that can each be
read and checked. It is fitted to CORSIKA (`h5_files_v3`) and judged at detector level through the pipeline kernel.

Two parts:

1. **Energy per plane** (`longitudinal`, `copula`, `sampler`). Each species gets a Gaisser-Hillas curve in width and
   asymmetry plus a start depth. These numbers get Johnson SU distributions that depend linearly on log10 E and
   cos zenith, joined by one Gaussian copula per class. A per-plane correction with AR(1) noise is added on top.
2. **Cells within each plane** (`annuli`, `generator`). Rings of radius about the shower axis, with laws for the cell
   count, its split over the rings, the energy per cell, one random concentration number per plane (AR(1) along the
   shower) and an arrival delay per ring. Cells sit at a random angle within their ring, so every shower is centred
   on its axis.

The CORSIKA files carry a preprocessing defect: a cell-key overflow moved 17-25% of shower-planes 0.7-4 km South.
`unwrap` repairs it on read.

## Files

| file | what it does |
|---|---|
| `atmosphere.py`, `lateral.py`, `planes.py` | slant depth of the planes, in-plane geometry, energy per plane of CORSIKA files and clouds |
| `sources.py` | paths of the CORSIKA files and the pinned neural chain |
| `unwrap.py` | the repair of the cell-key overflow |
| `longitudinal.py`, `fit_profiles.py` | the profile curve and its fit to every shower |
| `copula.py`, `sampler.py` | the law of the profile numbers and the energy-per-plane sampler |
| `annuli.py`, `generator.py` | ring profiles of CORSIKA, the in-plane laws and the cloud draw |
| `chain.py` | the neural chain at the same test primaries, for comparison (GPU) |
| `score.py`, `detector.py` | energy per plane scored against CORSIKA; the detector-level check (GPU) |

## Pipeline

Every stage writes into one output folder `OUT`. CPU stages run on `test` or `shared`; parts already on disk are
skipped, so arrays can use `serial_requeue`.

```bash
P="python -u symbolic"
# energy per plane
$P/fit_profiles.py --out $OUT --stage slice                        # CPU: 10,000 training showers + the test file
$P/fit_profiles.py --out $OUT --stage part --n-parts 26 --part $i  # CPU array, i = 0..25
$P/fit_profiles.py --out $OUT --stage merge --n-parts 26
# cells within each plane
$P/annuli.py --out $OUT --stage hist --n-parts 10 --part $i        # CPU array, i = 0..9
$P/annuli.py --out $OUT --stage merge --n-parts 10
$P/generator.py --out $OUT --stage fit                             # CPU, ~20 min -> OUT/generator/model.npz
$P/generator.py --out $OUT --stage draw                            # one cloud per scored test primary
$P/generator.py --out $OUT --stage draw --oracle                   # CORSIKA's own plane energies (in-plane laws alone)
# comparison with the neural chain and CORSIKA
$P/chain.py --out $OUT --stage draws --first 0 --last 7            # GPU
$P/chain.py --out $OUT --stage clouds                              # GPU
$P/score.py --out $OUT                                             # CPU
$P/detector.py --root $OUT --clouds generator/clouds.npz --chain chain/clouds_draw00.npz --out generator/check \
    --heldout <run folder>/v6_run_00/cashed_showers_tau_dual_heldout.pt   # GPU, ~5 min
```

The scored test primaries are 500 per class (`chain.primaries`). `--heldout` must name a run whose held-out corpus
has its `_positions.pt` file next to it; run 12 (`12_10pct_mean_analytic`) has one, run 13 does not. Self-tests:
`python tests/analysis/test_symbolic.py`.

## Where it stands

1,000 test primaries against repaired CORSIKA. "Oracle" uses CORSIKA's own plane energies. The numbers are from
`/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/symbolic/generator/model_clean.npz`; the code
here fits the same model on 0.02% more planes (planes are no longer dropped when an NKG fit fails to converge).

| | symbolic | symbolic, oracle | neural chain |
|---|---|---|---|
| energy per plane, energy score vs the neural chain (EM / hadronic) | -0.24 / -0.23 (better) | | 0 |
| shower centred on its axis | yes | yes | 150-900 m off |
| deposited energy, median per-event ratio (EM / hadronic) | 0.97 / 0.62 | 1.00 / 0.96 | 0.51 / 0.32 |
| leading-edge time shift, median | 0 ns | 0 ns | -37 / -70 ns |
| lit detectors, relative (EM / hadronic) | +7.8% / +20.6% | +8.3% / +13.6% | +1.5% / +24.2% |

Within 200 m of the axis the symbolic model lights the same detectors as CORSIKA; its excess sits beyond 200 m, where
detectors are near threshold and the model spreads the same energy more evenly than real planes do. The neural
chain's small EM total is a cancellation: it misses detectors near the axis (its showers are off axis) and adds
detectors far out.

Open: the hadronic excess has two parts, the sampled plane energies (about 8 points) and the in-plane laws (about 7).
