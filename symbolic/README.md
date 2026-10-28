# Symbolic shower generator

A drop-in alternative to the neural shower chain (PointCountFM + AllShowers). Instead of two trained networks it uses
a handful of fitted formulas, each of which can be printed, read and checked against physics. It is fitted to the
CORSIKA training showers (`h5_files_v3`) and judged the way the pipeline uses showers: through the detector kernel.

In one sentence: each shower gets a longitudinal curve that says how much energy reaches each of the 24 planes, and
within each plane the energy is spread over rings around the shower axis by a few more fitted laws.

## How it works

The generator draws a shower for a primary (energy, direction, EM or hadronic class) in two steps.

1. Energy per plane (`longitudinal.py`, `copula.py`, `sampler.py`). Each species (electron, muon, photon) follows a
   Gaisser-Hillas curve written in width L and asymmetry R, plus a start depth before which the planes are empty. Its
   five numbers are drawn from Johnson SU distributions whose parameters are linear in log10 E and cos zenith, and the
   15 numbers of the three species are correlated through one Gaussian copula per class. A fixed correction and
   AR(1) noise per plane add what the smooth curve misses.
2. Cells within each plane (`annuli.py`, `generator.py`). The plane's energy is split over 20 rings of radius around
   the shower axis (10 m wide at the core, widening to 9 km, plus one open outer ring). Fitted laws give the number of cells, how they split over the rings, the energy per
   cell in each ring, one random "concentration" number per plane (correlated plane to plane along the shower) and an
   arrival delay per ring. Cells are placed at a random angle within their ring, so every shower is centred on its
   axis.

All laws are fitted separately per species and class, with the same code for every one of them.

The CORSIKA files have a preprocessing defect: a cell-key overflow moved 17-25% of shower-planes 0.7-4 km South of
where they belong. `unwrap.py` repairs every shower when it is read (exact for e/gamma, approximate for muons, with a
detector-level effect under 0.3%), so all fits and comparisons use repaired CORSIKA.

## Running it

Run from the repository root with the `tambo` environment, one script per stage. Every stage writes into one output
folder `OUT`; array stages skip parts already on disk, so they can run on `serial_requeue` and be requeued safely.

```bash
P="python -u symbolic"

# 1. energy per plane: fit the curve to every shower
$P/fit_profiles.py --out $OUT --stage slice                        # CPU, 16 cores ~12 min: 10,000 training showers + test file
$P/fit_profiles.py --out $OUT --stage part --n-parts 26 --part $i  # CPU array, i = 0..25, ~10 min each
$P/fit_profiles.py --out $OUT --stage merge --n-parts 26

# 2. cells within each plane: ring profiles of CORSIKA, then the laws
$P/annuli.py --out $OUT --stage hist --n-parts 10 --part $i        # CPU array, i = 0..9, ~5 min each
$P/annuli.py --out $OUT --stage merge --n-parts 10
$P/generator.py --out $OUT --stage fit                             # CPU ~20 min -> OUT/generator/model.npz

# 3. draw showers at the 1,000 scored test primaries
$P/generator.py --out $OUT --stage draw                            # -> OUT/generator/clouds.npz
$P/generator.py --out $OUT --stage draw --oracle                   # same, with CORSIKA's own energy per plane

# 4. compare with CORSIKA and the neural chain
$P/chain.py --out $OUT --stage draws --first 0 --last 7            # GPU, neural chain energy per plane, 8 draws
$P/chain.py --out $OUT --stage clouds                              # GPU, neural chain full clouds of draw 0
$P/score.py --out $OUT                                             # CPU, energy per plane against CORSIKA
$P/detector.py --root $OUT --clouds generator/clouds.npz --chain chain/clouds_draw00.npz --out generator/check \
    --heldout <run folder>/v6_run_00/cashed_showers_tau_dual_heldout.pt   # GPU ~5 min
```

`--heldout` places the showers at real decay vertices and must name a run whose held-out corpus has its
`_positions.pt` file next to it: run 12 (`12_10pct_mean_analytic`) has one, run 13 does not. The scored primaries are
500 per class drawn once with a fixed seed (`chain.primaries`), the same for every model.

Self-tests (CPU, about two minutes): `python tests/analysis/test_symbolic.py`.

## Outputs

| folder in `OUT` | contents |
|---|---|
| `profiles/` | `fits_slice.npz`, `fits_test.npz`, `fits_train.npz`: energy per plane, plane depths and the fitted curve of every shower |
| `annuli/` | `annuli.npz`: energy, cell count and delay per ring for every shower and plane of the slice |
| `generator/` | `model.npz` (all fitted laws), `clouds*.npz` (drawn showers), `check/` (detector results) |
| `chain/` | the neural chain's draws at the same primaries |
| `score/` | energy score of the energy per plane, and a figure of median profiles |

Each stage also writes a `results.json` with the git commit, host and job id, and all but `detector.py` a `run.log`
(under `<folder>/logs/<stage>/`, or `score/run/`).

A drawn cloud file holds, per species, `<species>.points` with shape (showers, cells, 5) and columns
[x, y, plane, energy, time]: x and y are ground East and North offsets [m] from the injection point, plane is 0-23,
energy is in GeV and time is in seconds from injection. This is the same layout as the CORSIKA files and the neural
chain, padded with zero rows. `<species>.E` holds the energy per plane and `rows` the CORSIKA test rows drawn.

## Results

1,000 test primaries (500 EM, 500 hadronic), each compared with its own repaired CORSIKA shower. "Oracle" feeds the
generator CORSIKA's real energy per plane, which isolates the in-plane laws.

| | symbolic | symbolic, oracle | neural chain |
|---|---|---|---|
| energy per plane: energy score minus the neural chain's (EM / hadronic; negative is better) | -0.24 / -0.23 | | 0 |
| shower centred on its axis | yes | yes | 150-900 m off |
| deposited energy: median per-event ratio to CORSIKA (EM / hadronic) | 0.97 / 0.62 | 1.00 / 0.96 | 0.51 / 0.32 |
| arrival time at the detectors: median shift from CORSIKA | 0 ns | 0 ns | -37 / -70 ns |
| detectors lit: relative to CORSIKA (EM / hadronic) | +7.8% / +20.6% | +8.3% / +13.6% | +1.5% / +24.2% |

Near the axis (within 200 m) the symbolic model lights the same detectors as CORSIKA. Its excess sits farther out,
where detectors are close to threshold and the model spreads energy more evenly than real planes do. The neural
chain's small EM number is two errors cancelling: its showers are off axis, so it misses detectors near the axis and
lights extra ones far out.

These numbers come from `model_clean.npz` in `/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/symbolic`.
The code on this branch reproduces its inputs exactly and fits the same laws on 0.02% more planes (earlier code
dropped planes whose NKG fit did not converge).

## Known limits

- Hadronic showers light too many detectors (+20.6%) and deposit too little energy (0.62). The oracle row shows most of
  this comes from the sampled energy per plane rather than from the in-plane laws.
- One concentration pattern per species does not capture all the plane-to-plane variety of real lateral profiles;
  this is most of the remaining EM excess far from the axis.
- The energy-per-plane sampler is fitted on showers with log10 E < 6.125 [GeV]: the training file thins out above
  that, so higher primaries are an extrapolation.
- Showers can only be drawn at rows of the CORSIKA test file (`generator.py --stage draw`); there is no function yet
  that takes an arbitrary list of primaries.
