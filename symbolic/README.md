# Symbolic shower generator

An alternative to the neural shower chain (PointCountFM + AllShowers) built from fitted formulas instead of two
trained networks. It draws the same thing the chain draws, the stored ground cells of a tau-decay shower for each
species (electron, muon, photon), in the same file format, so the detector pipeline (Stages 1-4) reads its output
unchanged.

The whole model is 3,574 numbers. Every law can be printed and read. On correct CORSIKA showers it lands closer than
the neural chain on most comparisons (see Results), and drawing the energy per plane is about 500x faster than
PointCountFM on one core.

## Quick start: a symbolic corpus for a pipeline run

This is the common use. You have a run whose Step-0 corpus (`v6_run_00/cashed_showers_tau_dual.pt`) was made by the
neural chain, and you want the same events with symbolic showers.

```bash
cd tambo_symbolic20                     # this branch, symbolic-run20
conda activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/envs/tambo
SRC=<run>/v6_run_00/cashed_showers_tau_dual.pt
DST=<new run>/v6_run_00
NAME=cashed_showers_tau_dual

# 1. read the source corpus' events (add --limit 1000 for a smoke test)
python -u symbolic/corpus.py primaries --src $SRC --dst $DST --name $NAME

# 2. draw: a Slurm array, each task takes a contiguous slice of events
sbatch -p shared -c 8 --mem=128g -t 8:00:00 --array=0-<tasks - 1> \
    --wrap "python -u symbolic/corpus.py draw --dst $DST --name $NAME"

# 3. merge the parts into the corpus file, copy the sidecars, check the result
python -u symbolic/corpus.py merge --src $SRC --dst $DST --name $NAME
```

Then point the pipeline at the new run folder (`TAMBOPT_RUN_LOCATION=<new run>`, with `inputs/` linked to the source
run's) and run Stage 1 as usual.

Things to know:

- Memory: about 16 GB per worker. Showers up to 10^7 GeV make millions of particles before they are binned into
  cells. If a worker is killed for memory the whole task hangs until its time limit, so give each task
  `--mem` = 16 GB x `-c`.
- Speed: the 1,000-event smoke test (3,000 rows) took 12 minutes on 8 cores, draw and merge together.
- The draw stage reads `--task` and `--tasks` from the Slurm array variables. Outside an array pass them yourself,
  and use the same `--tasks` for every task.
- `merge` checks the written file against the source row by row: energies, directions, pdg, shower ids, positions
  and species must all be equal, or it stops with an error. Look for the `check:` line in its output.
- Run 20's guard (`slurm/run20_guard.sh`) requires a `.filter.json` next to the corpus. `merge` does not write one:
  copy the source's (`cp $SRC.filter.json $DST/$NAME.pt.filter.json`) before Stage 1.
- Zenith: the model was fitted on zeniths 60-100 deg (`ZENITH_MIN`, `ZENITH_MAX` in `modules/constants.py`).
  Events outside that range are drawn with the zenith clamped into it for the depths and law inputs, and placed
  along their real direction. Their indices are in `<name>_zenith_clamped.npy` (7.4% of run 20 lies above 100 deg).
- The fitted laws are read from `corpus.ROOT`
  (`/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/symbolic_trim`), readable by the lab group.
- `merge` also writes `<name>.pt.symbolic.json` (source, event count, seed, parts) so a symbolic corpus can be told
  apart from a neural one.

## How a shower is drawn

Per primary (energy, direction, EM or hadronic class) and species, in `draw.shower`:

1. Energy per plane (`sampler.py`, `longitudinal.py`, `copula.py`). Each species follows a Gaisser-Hillas curve in
   width and asymmetry, plus a start depth before which planes are empty. Its numbers are drawn from distributions
   whose parameters are straight lines in log10 E and cos zenith, and the three species are correlated through one
   Gaussian copula per class. A fixed correction and AR(1) noise per plane add what the smooth curve misses.
   Fitted on the 130,000 CORSIKA training showers (`profiles/fits_train.npz`).
2. Energy before the radius cut: the plane's stored energy divided by the stored share (stored law).
3. Ring energies: the plane's energy split over 23 rings of radius around the axis (1 m at the core out to 9 km).
   The ring shares come from one law per species and class, stretched in radius by a width that follows a tanh in
   ln(plane number) along the shower, with AR(1) noise from plane to plane.
4. Particles per ring: Poisson(ring energy / mean particle energy). The mean particle energy is a sum of three tanh
   terms in the plane's inputs (28 numbers per species and class).
5. Particle energies: CORSIKA's tracking threshold plus the ring's remaining energy split by Burr XII draws, so the
   ring energy is exact.
6. Positions: uniform in area within the ring, uniform azimuth.
7. Times: light time along the axis plus a lead and an exponential delay, from a power law
   t = A r^p k^-q (r radius, k plane).
8. Render (`render.py`): the same preprocessing the CORSIKA files went through (plane frame to ground, 90th
   percentile radius cut per plane, binning into ground cells, the brightest cells kept up to the species cap). Each
   plane's stored energy is then rescaled to the energy drawn in step 1.

Every law has one form for all three species, refitted per species and class. Steps 2-7 were fitted on Hamza's raw
CORSIKA showers with every particle stored (`tambo_simulations_test_raw_v2`, 100 showers, 74 of them in the training
energy range); step 1 on the h5 training files.

## Files the draw uses

All under `v6_runs/symbolic_trim` (the `--out` / `ROOT` folder); defaults in `draw.py`.

| file | what |
|---|---|
| `profiles/fits_train.npz` | profile fits of the training showers; the energy-per-plane sampler is fitted from it at draw time |
| `particles/fit_v2/particle_laws.npz` | stored share, ring shares, Burr spread, ring-by-ring laws (`draw.LAWS`) |
| `particles/sr_width/width_law.npz` | width along the shower (`draw.WIDTH_LAW`) |
| `particles/sr_times/time_law.npz` | arrival time power law (`draw.TIME_LAW`) |
| `particles/judge/sr_energy_tanh/energy_law.npz` | mean particle energy, tanh sum (`draw.ENERGY_LAW`) |

`draw.py --laws` with a different laws file also requires `--width-law`, `--time-law` and `--energy-law`, so a
held-out fold cannot silently pick up laws fitted on its own showers. `--energy-law none` and `--time-law none` use
the ring-by-ring tables instead of the formulas.

## Other entry points

Draw at the 1,000 scored test primaries (the comparison set), in the suite's cloud format:

```bash
python -u symbolic/draw.py energies --out $ROOT --name run
sbatch -p shared -c 36 --mem=290g --array=0-2 --wrap "python -u symbolic/draw.py draw --out $ROOT --name run"
python -u symbolic/draw.py merge --out $ROOT --name run      # -> particles/draw/run/clouds.npz
```

`draw.py energies --primaries <file> --draws N` draws at the primaries of a `judge.py` file instead.

A cloud file holds, per species, `<species>.points` of shape (showers, cells, 5) with columns
[x, y, plane, energy, time]: ground East and North offsets [m] from the injection point, plane 0-23, energy [GeV],
time [s] from injection, padded with zero rows.

Comparisons (each script's docstring has its rule and outputs):

| script | compares | against |
|---|---|---|
| `score.py` | energy per plane (energy score) | CORSIKA test file and the neural chain |
| `suite.py` | cells within planes: across-track spread, centre, cell count, cell energy, times | CORSIKA test file and the chain |
| `judge.py` | the same rows, on correct particles (raw showers through `render`), laws fitted without the judged showers | raw v2 showers and the chain |
| `detector.py` | lit detectors, deposited energy, arrival time through the run's kernel (GPU; `source slurm/run20_guard.sh` first) | CORSIKA and the chain |
| `chain.py` | draws the neural chain at the same primaries (GPU) | |
| `gradients.py` | gradients of observables through the sampler | slopes on CORSIKA |
| `extrapolate.py` | sampler fitted below 10^5.9 GeV, scored above | the full fit and the chain |

## Refitting on new CORSIKA data

The order, each step in its own script (stages and flags in the docstrings):

1. `fit_profiles.py --stage slice | part | merge`: profile fit of every training shower (CPU array).
2. `particles.py tables <sim>` per raw shower (needs pyarrow), then `particles.py fit --name <name>`: the particle
   laws.
3. `width_sr.py table | fit | common | law`: the width law (PySR in `envs/symbolic`).
4. `times_sr.py table | fit | law --form b`: the time law.
5. `tiny_net.py --terms 3 --law`: the mean-energy tanh sum on every shower.
6. Draw, then `judge.py` and `detector.py` to compare with the previous fit and the chain.

Steps 3-5 depend on step 2's laws. The particle laws need raw showers with every particle stored; laws fitted from
per-ring summaries (rings, width, times, mean energy) only need the h5 `plane_stats`.

## Environments

- `envs/tambo` (`/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/envs/tambo`): everything except the
  formula searches.
- `envs/symbolic`: PySR and torch-symbolic, for the `fit` and `search` stages of `*_sr.py` and `struct_symtorch.py`.
- `particles.py tables` reads parquet: use an env with pyarrow (Hamza's `tamboOpt_env`).

## Tests

```bash
for t in tests/analysis/test_*.py; do python $t; done      # CPU, about 2 minutes, 96 checks
```

## Results

The fair comparison (`judge.py`): 74 raw CORSIKA showers in the training energy range, rendered through the same
preprocessing, laws fitted without the showers being judged (two folds), 8 draws per primary for both models.
12 groups per row (3 species x EM/hadronic x steep/grazing). Symbolic closer than the neural chain, per plane:

| row | groups |
|---|---|
| across-track spread, shower centre | 12 of 12 |
| first and mean arrival time | 12 of 12 |
| energy per plane | 9 of 12 |
| cell energy | 7 of 12 |
| cells per plane | 6 of 12 |

Detector row on the same showers (run-20 kernel), lit detectors per event, EM / hadronic: CORSIKA 3.95 / 7.51,
symbolic 4.53 / 7.52, neural chain 27.6 / 29.0.

Energy per plane on the full CORSIKA test file (1,000 primaries): energy score 0.25 (EM) and 0.19 (hadronic) better
than PointCountFM, with no measurable cost above the energies it was fitted on.

## Known limits

- Cells per plane on steep EM showers: the chain is closer (its learned cell count per plane carries over).
- Muons in EM showers fit worst of all groups, on cell counts and arrival times.
- A few very late single cells (microseconds) in CORSIKA are not drawn.
- The particle laws come from 100 raw showers. They will be refitted when the 150k regenerated set has raw
  particles; laws from ring summaries can be refitted on the 150k set now.
- The energy-per-plane sampler was fitted below 10^6.125 GeV; above that it extrapolates its straight-line laws.
