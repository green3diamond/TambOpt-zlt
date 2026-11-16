# Symbolic shower model

A stochastic model of TAMBO tau-decay showers built from fitted formulas and a few tables, in place of the neural
chain (PointCountFM + AllShowers). It draws what the chain draws, the stored ground cells of each species (electron,
muon, photon) in the same file format, so the detector pipeline reads its output unchanged.

## The model

Part 1, energy along the shower (`sampler.py`, `longitudinal.py`, `copula.py`). Per species a Gaisser-Hillas curve in
width and asymmetry plus a start depth: 15 numbers per shower, each with a Johnson SU margin linear in log10 E,
cos zenith and their product, joined by one Gaussian copula per class. On the curve, plane-to-plane noise read from
quantile tables of the real leftovers, per position relative to the peak, with AR(1) memory. Output: the total
energy per plane of each species.

Part 2, inside each plane (`draw.py`):

| Step | Law | File |
|---|---|---|
| ring energies | multinomial logit of the shares of 23 rings about the axis (a coefficient table) | `rings.py` |
| plane-to-plane width | the ring profile stretched by A tanh(B (ln k - C)) + D + E l + F age, AR(1) | `width.py` |
| particles per ring | ring energy / mean particle energy, a 3-term tanh sum | `energy.py` |
| particle energies | Burr XII above the tracking threshold, the ring total exact | `spread.py` |
| times | lead and mean delay per ring, t = exp(A) r^p k^-q | `times.py` |
| stored cells | the per-simulation preprocessing (10 m cells, radius cut, caps) | `render.py` |

Every law has one form for every species and class (EM, hadronic). The fitted model is a folder
(`model.py`): `part1.npz`, `rings.npz`, `width.npz`, `times.npz`, `energy.npz`, `spread.npz`, each with a `.json`
of what it was fitted on. The frozen model is
`/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/symbolic_model`.

## Fitting (`fit.py`)

Data: per-simulation CORSIKA folders (an h5 file per species with `plane_stats` and stored cells; the raw runs also
keep every particle in parquet). Part 1 and every in-plane law but the spread come from the 150k summary tables, the
spread from the raw v2 showers (it needs every particle's energy).

```bash
python symbolic/fit.py tables --list sims.txt --out TABLES            # --raw for the raw showers; Slurm array
python symbolic/fit.py profiles --tables TABLES --list train.txt --top 7 --out fits.npz
python symbolic/fit.py part1 --fits fits.npz --model MODEL
python symbolic/fit.py rows --tables TABLES --list train.txt --out rows.npz
python symbolic/fit.py rings  --rows rows.npz --model MODEL
python symbolic/fit.py width  --rows rows.npz --model MODEL           # after rings
python symbolic/fit.py times  --rows rows.npz --model MODEL
python symbolic/fit.py energy --rows rows.npz --model MODEL           # about 2 h on 32 cores
python symbolic/fit.py spread --rows rows_v2.npz --model MODEL
```

## Drawing (`draw.py`)

```bash
python symbolic/draw.py energies --model MODEL --primaries primaries.npz --draws 8 --out DRAW
sbatch --array=0-19 -c 8 --mem=32g ... python symbolic/draw.py draw --model MODEL --out DRAW
python symbolic/draw.py merge --model MODEL --out DRAW                  # DRAW/clouds.npz
```

One seed per entry, so the split into tasks does not change the result. About 16 GB per worker: showers up to
10^7 GeV make millions of particles before the render.

## Checking

- `judge.py`: W1 to CORSIKA's stored cells at held-out primaries, per species, class and zenith band (steep /
  grazing at |cos z| = 0.1) for across-track offset, cell energy, cells per plane, energy centre, arrival times and
  energy per plane, against an optional second source, with floors and resampled intervals. It refuses a model
  fitted on a judged shower.
- `detector.py`: the draws and CORSIKA through the run's kernel on the grid layout (one GPU, `source
  slurm/run20_guard.sh` first): lit detectors, deposits and arrival time, with the draw-to-draw noise floor.

## A symbolic corpus for a pipeline run (`corpus.py`)

```bash
SRC=<run>/v6_run_00/cashed_showers_tau_dual.pt
DST=<new run>/v6_run_00
python -u symbolic/corpus.py primaries --src $SRC --dst $DST --name cashed_showers_tau_dual      # --limit N to test
sbatch -p shared -c 8 --mem=128g -t 8:00:00 --array=0-<tasks - 1> \
    --wrap "python -u symbolic/corpus.py draw --dst $DST --name cashed_showers_tau_dual"
python -u symbolic/corpus.py merge --src $SRC --dst $DST --name cashed_showers_tau_dual
cp $SRC.filter.json $DST/cashed_showers_tau_dual.pt.filter.json                                   # run 20's guard needs it
```

`merge` checks the written file against the source row by row. Events outside the zeniths the model was fitted on
(`ZENITH_MIN`-`ZENITH_MAX`) are drawn with the zenith clamped for the laws and placed along their own direction;
their indices are in `<name>_zenith_clamped.npy`.

## Tests

`tests/symbolic/`: known-answer tests of every law, the sampler round trip through the model file, the draw, the
judge and the render against the preprocessing script itself. Run each file with python.
