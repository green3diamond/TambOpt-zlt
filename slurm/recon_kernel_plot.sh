#!/bin/bash
#SBATCH -p test
#SBATCH --mem=150g
#SBATCH --time=4:00:00
#SBATCH -c 16
#SBATCH -J recon_kernel_plot
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

# Back-fill the recon-on-kernel figures for whichever run RUN_LOCATION points
# at (currently 13_full_750k_no_sampling).
#
# Step 3 (03_train_recon_deepsets.py) auto-renders only plot_recon_only(), the
# surrogate-fed recon_target_vs_pred.png -- run 12 got its *_kernel pair from a
# separate pass over the standalone plotter, run 13 never had one. --kernel-only
# skips every surrogate-dependent figure, so this never loads fnn_*.pt and
# cannot overwrite the scatters Step 3 already wrote.
#
# CPU, no --gres: 02_nn_target_vs_pred.py resolves DEVICE itself ("cuda" if
# available else "cpu") and this is one forward pass of recon.pt over the val
# split plus two hexbin renders -- no training, no backward. A GPU only makes it
# finish sooner, and asking for one costs far more: gpu_test caps at 64 GB per
# GPU, which this does not fit under (_load_corpus reads primary/xy/E/T ~14 GB,
# then load_kernel_responses reads E/T again and the species-combine holds the
# per-species stack, the broadcast copy and both contiguous outputs at once).
# -p test has no such cap and usually starts immediately.

source slurm/env.sh

python -u plots/training/02_nn_target_vs_pred.py --kernel-only
