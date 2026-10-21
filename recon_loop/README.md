# Reconstruction refresh during detector optimisation

The optimiser moves detectors to improve reconstruction of shower direction and
energy. It can periodically retrain the reconstruction network on the current
layout, accepting an update only when its validation error improves.
`run.py` drives the optimisation; `refresh.py` handles data and network training.
Existing pipeline files are unchanged.

## Pull request base

This code uses the kernel-pool APIs on the `run18-agnostic-recon` branch.
Open a pull request against that branch. Its parent changes are not part of
this feature, and `master` does not yet contain the APIs this driver uses.

## Inputs

Use the repository's Python environment with PyTorch and `showerdata`. The shower
corpus, geometry and detector count come from `modules/constants.py`; the kernel
bandwidth caches must already exist.

Provide a kernel-trained `recon.pt` and its labelled dataset. The dataset needs
`xy.pt`, `E.pt`, `T.pt`, `primary.pt`, `species_ids.pt`, `strategy_ids.pt` and
`blob_ids.pt`, with the same shower ordering as the optimisation corpus and the
same training/validation split as the starting network.

## Run

Activate the environment and submit from the repository root:

```bash
export RECON=/absolute/path/to/recon.pt
export DATASET=/absolute/path/to/labelled_dataset
export RESULTS=/absolute/path/to/results
```

Prepare a bank of earlier training examples and validation examples:

```bash
sbatch -p test --cpus-per-task=8 --mem=60G --time=00:30:00 \
  --wrap='python recon_loop/refresh.py --dataset "$DATASET" --recon "$RECON" --out "$RESULTS/bank.pt"'
```

After that job finishes, launch matched fixed-network and retraining runs:

```bash
for start in grid center; do
  sbatch slurm/recon_loop.sh --out "$RESULTS/fixed_$start" \
    --recon "$RECON" --init "$start" --refresh-every 0

  sbatch slurm/recon_loop.sh --out "$RESULTS/refresh_$start" \
    --recon "$RECON" --init "$start" --bank "$RESULTS/bank.pt" \
    --refresh-every 100
done
```

The launcher inherits the active environment and requests one GPU, eight CPUs,
96 GB host RAM and six hours. Override these settings for your cluster.
`RECON_LOOP_PYTHON` can select an interpreter. `--refresh-every 0` is the default
and does not require a bank. Run either Python file with `--help` for options.

## Method

Defaults are 4,000 detector updates with 512 showers per update. Every 100
updates, the retraining arm labels 8,192 selected showers at the current layout,
excludes unusable showers, and reserves 10% for validation. The remaining rows
are mixed with equally many earlier training examples to reduce forgetting.

A copy of the network trains for five passes with Adam at learning rate `1e-5`.
The best pass replaces the live network only if validation error decreases.
Normalisation stays fixed; a separate validation sample reports forgetting.
Detector Adam state is retained, and gradients do not pass through retraining.

## Outputs and resume

`layout_best.pt` and `recon_best.pt` hold the selected layout and matching network.
`recon_final.pt` holds the last network. Position snapshots are in `layouts/`,
accepted networks in `recons/`, and progress and losses in the log/result files.

Resubmit the same command with the same `--out` to resume from `opt_state.pt`.
Use a separate output directory for each arm. You may extend `--epochs` or change
`--log-every`; other settings must stay the same.

Evaluate selected layout/network pairs on independent showers. Cross-score the
layouts with the same network to distinguish better layouts from better
reconstruction. Refreshed checkpoints retain the starting `val_total` metadata;
new training losses are recorded in `refresh_log`.
