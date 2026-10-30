# AllShowers flow-matching surrogate

The shower generator used by TambOpt. It is a conditional flow-matching model
with a transformer backbone that generates calorimeter shower point clouds.
It is a fork of [FLC-QU-hep/AllShowers](https://github.com/FLC-QU-hep/AllShowers),
modified for TAMBO electron, photon and muon showers. It was copied into
TambOpt from the standalone fork at
[hamzahanif2210/AllShowers](https://github.com/hamzahanif2210/AllShowers),
without its git history or training outputs.

The companion model that predicts the number of points per layer lives in
[`PointCountFM/`](PointCountFM/README.md). Its output is the `<condition>` file
the generator needs (see step 5).

## Layout

```
allshowers/            # the package: model, training, generation, OT matching
  plotting_scripts/    # standalone comparison plots (paths set at the top of each file)
conf/                  # training configs: electrons, photons, muons, geant4
mkresultdir.py         # creates a results/ run folder + Slurm scripts, optionally submits
logs/                  # Slurm logs from the OT scripts (contents gitignored)
results/               # created by training runs (gitignored)
PointCountFM/          # points-per-layer model, see its own README
```

## Setup

Run every command below from this folder (`TambOpt-zlt/AllShowers-flow-matching-surrogate/`).

### Environment
On the cluster, use the shared lab environment:
```bash
module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/python313
```
`mkresultdir.py` activates the same environment inside Slurm jobs. Pass
`--mamba-env <path>` to use a different one.

Elsewhere, create an environment from `environment.yaml` (or install the
dependencies in `pyproject.toml`). The code has been tested with Python 3.13
and torch 2.8.

### No `pip install` needed
The entry scripts (`allshowers/train.py`, `generator.py`, `OT_match.py`, …)
put this folder at the front of `sys.path` themselves. `from allshowers import …`
therefore always loads this copy, even in an environment that still has an
editable install of the old standalone `AllShowers` repo. Library code imported
from outside (a notebook, TambOpt modules) must add this folder to `sys.path`
first, or you can run `pip install -e .` here.

## Usage

### 1. Data
Training data paths are set in `conf/*.yaml` (`data.path`). On the cluster
they point at the lab store
`/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/`.
`conf/geant4.yaml` points at a file in a personal home directory; change it
before using that config.

### 2. Optimal-transport matching
`return_noise: True` in the configs means training uses OT-matched latent
points, which must first be computed and written into the data file:
```bash
python allshowers/OT_match.py conf/photons.yaml --with-time
```
On Slurm, submit it from this folder so the relative log paths resolve:
```bash
sbatch allshowers/ot_match.sh               # edit the config at the bottom of the script
```
For large files, `allshowers/ot_match_big.sh` splits the job into three
chained Slurm jobs: fit the preprocessing, match chunks as an array job, then
merge the chunks. It submits the jobs itself, so run it with `bash`:
```bash
CONFIG=conf/muons.yaml TOTAL_SAMPLES=130000 bash allshowers/ot_match_big.sh
```
Re-run the matching whenever the preprocessing or the data file changes. Both
scripts activate the `tamboOpt_env` environment; edit them if you use a
different one.

On some filesystems (for example Apple File System) you need
`export HDF5_USE_FILE_LOCKING=FALSE` first.

### 3. Training
Interactively (one GPU):
```bash
python allshowers/train.py conf/electrons.yaml
python allshowers/train.py --fast-dev-run conf/electrons.yaml   # quick smoke test
```
On Slurm, use `mkresultdir.py`. It creates `results/<timestamp>_<run_name>/`,
writes `run.sh`/`script.sh` into it, and submits with `-r`:
```bash
python mkresultdir.py conf/electrons.yaml -p gpu_requeue -g 1 -n 1 --mem 64G  --time 1:00:00 -r
python mkresultdir.py conf/photons.yaml   -p gpu_requeue -g 1 -n 1 --mem 128G --time 5:00:00 -r
python mkresultdir.py conf/muons.yaml     -p gpu         -g 1 -n 1 --mem 500G --time 12:00:00 -r
```
To resume a run, pass `--resume-training --resume-ckpt-file <ckpt>` with the
run's `conf.yaml`. The whole dataset is loaded into memory, so size `--mem`
accordingly.

### 4. Generating samples
```bash
python allshowers/generator.py \
  --run-dir results/<run_name> \
  --cond_file <condition.h5> \
  --num-samples 6141 --num-timesteps 16 --solver midpoint \
  --device cuda:0 --pdgs 0 1 --max-points 4096
```
`<condition.h5>` is a `showerdata` file with the incident particles and the
number of points per layer. Generate one with [`PointCountFM`](PointCountFM/README.md),
or take it from simulation after running
`showerdata add-observables <file> --num-layers <n>`. More complete examples
for each particle type are in the docstring at the top of
`allshowers/generator.py`. Samples are written to `results/<run_name>/samplesNN.h5`.

Checkpoints and samples from runs made before the copy are **not** in this
repo. They are still in `/n/home04/hhanif/AllShowers/results/` and
`.../tambo_simulations_for_training/best_ckpts/` on the lab store, which the
examples in `generator.py` and `plotting_scripts/` point at.

### 5. Evaluating samples
```bash
showerdata add-observables <samples.h5>
```
```python
import showerdata

path = "<samples.h5>"
showers = showerdata.load(path)
observables = showerdata.observables.read_observables_from_file(path)
print(showers.points.shape, showers.energies.shape, showers.directions.shape)
```
The comparison plots in `allshowers/plotting_scripts/` (`plot_electron.py`,
`plot_photons.py`, …) take the generated and simulated file paths from
variables at the top of each script.

### Your own configurations
Copy a config into `conf-test/`, which is gitignored:
```bash
mkdir -p conf-test && cp conf/electrons.yaml conf-test/my_run.yaml
```

## Configuration
The configuration files are written in YAML format. See `conf/electrons.yaml` for an example.

### Global
- `run_name`: Name of the training run, will be part of the result folder name and set as the job-name in SLURM when using `mkresultdir.py`

### Data

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Path to the HDF5 data file |
| `samples_energy_trafo` | list | List of transformations applied to point energies |
| `samples_coordinate_trafo` | list | List of transformations applied to point coordinates |
| `cond_trafo` | list | List of transformations applied to incident energies |
| `return_noise` | boolean | Whether to use OT matched latent space points (has to be stored in the data file) |
| `val_len` | integer | Validation set size |
| `stop` | integer | Optional stop index when only a subset of the data should be used for training |

**Transformation types** can include:
- `Affine`: Linear transformation with scale and shift parameters. example: `[Affine, {scale: 2, shift: 0.0}]`
- `Log`: Logarithmic transformation with alpha (for numerical stability) and base (default `math.e`). example: `[Log, {alpha: 0.0, base: 10}]`
- `LogIt`: Logit transformation with alpha for numerical stability. example: `[LogIt, {alpha: 0.001}]`
- `StandardScaler`: Standardization with specified shape. The Shape parameter is a list of integers with as many entries as the number of dimensions of the data to be transformed. Each entry is either 1 (take mean and standard deviation along this dimension) or the size of the dimension (do not take mean and standard deviation along this dimension). The mean and standard deviation values will be computed from the first 100k samples in the training data. example: `[StandardScaler, {shape: [1, 1, 2]}]`

### Model

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_layers` | integer | Number of calorimeter layers |
| `dim_inputs` | list | Dimensions of input features [point features, 2x Fourier frequencies, kinematic features] |
| `dim_embedding` | integer | Dimension of the embedding space |
| `num_head` | integer | Number of attention heads in multi-head attention |
| `num_blocks` | integer | Number of transformer blocks |
| `dim_feedforward` | integer | Dimension of feedforward network |
| `num_points_cond` | integer | Hidden layer size for num points conditioning |
| `activation` | string | Activation function (e.g., GELU, ReLU) |
| `num_layer_cond` | integer | Number of calorimeter layers points can attend to in addition to their own layer |
| `num_particles` | integer | Number of incident particle types |
| `flow_config.frequencies` | integer | Number of frequencies for Fourier feature encoding |

### Training

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `num_epochs` | integer | Number of training epochs | |
| `learning_rate` | float | Initial learning rate | |
| `batch_size` | integer | Batch size for training | |
| `optimizer` | string | Optimizer type (SGD, Adam, AdamW or Ranger) | AdamW |
| `scheduler` | string | Learning rate scheduler type (Step, Exponential, OneCycle, Cosine, CosineWarmup) | None |
| `weight_decay` | float | Weight decay for optimizer | 0.0 |
| `grad_clip` | float | Euclidean norm for gradient clipping | None |
| `grad_accum` | integer | Number of gradient accumulation steps | 1 |
| `momentum` | float | Momentum for SGD optimizer. Ignored for other optimizers. | 0.0 |


