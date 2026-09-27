# PointCountFM

A conditional flow-matching model that generates the number of points per
layer in a particle shower, given the incident particle type and kinematics.
Its output is the condition file that the AllShowers generator in the parent
folder ([`../README.md`](../README.md)) needs.

It is a fork of [FLC-QU-hep/PointCountFM](https://github.com/FLC-QU-hep/PointCountFM)
modified for TAMBO showers. It was copied into TambOpt from the standalone fork at
[hamzahanif2210/PointCountFM](https://github.com/hamzahanif2210/PointCountFM),
without its git history, `results/` or `plots/`.

## Layout

```
pointcountfm/      # the package: models, flow matching, trainer, data/inference scripts
config/            # training configs: electrons, photons, muons
mkresultdir.py     # creates a results/ run folder + Slurm scripts, optionally submits
results/           # created by training runs (gitignored)
plots/             # created by the plotting scripts (gitignored)
```

## Setup

Run every command below from this folder
(`TambOpt-zlt/AllShowers-flow-matching-surrogate/PointCountFM/`).

On the cluster, use the shared lab environment:
```bash
module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/python313
```
Elsewhere, create an environment from `environment.yaml` (or install the
dependencies in `pyproject.toml`).

No `pip install` is needed. `pointcountfm/trainer.py` puts this folder on
`sys.path` itself, and `python -m pointcountfm.trainer` works from here too.
You need PyTorch, NumPy, h5py, Matplotlib, PyYAML and `showerdata`.

## Data

Training data paths are set in `config/*.yaml` (`data.data_file`). On the
cluster they point at `*_for_layers.h5` files in
`/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/`.
To build such a file from a full shower file:
```bash
python pointcountfm/create_dataset.py --input <showers.h5> --output <showers_for_layers.h5> ...
```
(See the docstring at the top of `pointcountfm/create_dataset.py` for the exact
per-particle commands.) The file must contain these keys:

| key        | shape  | dtype   | description                                       |
|------------|--------|---------|---------------------------------------------------|
| directions | (n, 3) | float32 | incident particle directions                      |
| energies   | (n, 1) | float32 | incident particle energies (in any consistent unit) |
| labels     | (n)    | int32   | incident particle labels                          |
| num_points | (n, m) | int32   | number of points per layer                        |

n is the data set size and m is the number of calorimeter layers. m must
match `dim_input` in the configuration file.

## Usage

### Training
Interactively:
```bash
python pointcountfm/trainer.py config/electrons.yaml
python pointcountfm/trainer.py --fast-dev-run config/electrons.yaml   # quick smoke test
```
On Slurm, use `mkresultdir.py`. It creates `results/<timestamp>_<run_name>/`,
writes `run.sh`/`script.sh` into it (they run `python -m pointcountfm.trainer`
from this folder), and submits with `-r`:
```bash
python mkresultdir.py config/electrons.yaml -p gpu_requeue -g 1 -n 1 --mem 10G --cpus-per-task 1 --time 0:30:00 -r
```
A run saves the model to `results/<run_name>/` and exports a TorchScript
sampler, `compiled.pt`. It also generates 50,000 samples into
`results/<run_name>/new_samples.h5`.

| Option          | Short | Description                                             |
|-----------------|-------|---------------------------------------------------------|
| `--help`        | `-h`  | Show the help message                                   |
| `--device`      | `-d`  | Device to run on (`cpu`, `mps`, `cuda`); auto-selected if omitted |
| `--time`        | `-t`  | Run a timing test on the model                          |
| `--distill`     |       | Distill the model                                       |
| `--fast-dev-run`|       | Run a fast development run for testing                  |

### Making the AllShowers condition file
Add the generated number of points per layer to a file of incident particles:
```bash
python pointcountfm/inference_cond_file.py \
  results/<run_name>/compiled.pt <particles.h5> \
  --output <particles_with_points.h5> --pdg-codes 11 211
```
Without `--output`, the points are added to the input file.
`pointcountfm/inference_cond_file_plot.py` does the same and also writes
comparison plots. Pass the output to `allshowers/generator.py --cond_file`.

Models trained before the copy are still in `/n/home04/hhanif/PointCountFM/results/`
on the cluster, which the docstring examples point at.

## configuration
The configuration file is a YAML with the following keys:

- `model`: specifies the model architecture and hyper-parameters
- `data`: specifies the training data and preprocessing steps
- `training`: specifies the training hyper-parameters
- `run_name`: a descriptive name for the run (part of the result folder name)

### model
| Key             | Type   | Description                                             |
|-----------------|--------|---------------------------------------------------------|
| `name`          | string | The model class (`FullyConnected` or `ConcatSquash`)    |
| `dim_input`     | int    | The dimension of the input data                         |
| `dim_condition` | int    | The dimension of the condition (number of particle labels + 3 (directions) + 1 (energies))                      |
| `dim_time`      | int    | The dimension of the time embedding                     |
| `hidden_dims`   | list   | A list of hidden dimensions for the model               |

### data
| Key             | Type   | Description                                             |
|-----------------|--------|---------------------------------------------------------|
| `data_file`     | string | The path to the training data                           |
| `batch_size`    | int    | The batch size for training                             |
| `batch_size_val`| int    | The batch size for validation                           |
|`transform_num_points`| list | A list of the preprocessing steps for the number of points per layer (optional) |
| `transform_inc` | list   | A list of the preprocessing steps for the incident energy (optional) |

### training
| Key             | Type   | Description                                             |
|-----------------|--------|---------------------------------------------------------|
| `epochs`        | int    | The number of epochs to train the model                 |
| `optimizer`     | dict   | The optimizer (`name` key) and its hyper-parameters     |
| `scheduler`     | dict   | The learning rate scheduler (`name` key) and its hyper-parameters (optional) |

If you use OneCycleLR or CosineAnnealing as a scheduler, the maximum number of iterations is calculated automatically.

For an example configuration file, see `config/electrons.yaml`.

## pre-commit
This repository uses [pre-commit](https://pre-commit.com) to run checks on the code before committing. To install pre-commit, run:
```bash
pre-commit install
```
This will install pre-commit and set up the checks. If you want to run the checks manually, you can run:
```bash
pre-commit run --all-files
```
This will run all checks on all files.
