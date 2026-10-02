#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=100g
#SBATCH --time=3-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J recon_kernel_run17
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 17 Stage 3 recon trained on 100% kernel (Step-1) responses instead of the 30% kernel /
# 70% surrogate mix, written to its own folder (RECON_FOLDER + "_deepsets_kernel100") so the
# production recon.pt is untouched. Same settings otherwise; resumes after preemption.
# Then the recon conditional plots (surrogate and kernel input) into that folder.
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate

python -c 'import sys; sys.path.insert(0, "."); from modules.constants import RUN_LOCATION as R; sys.exit(0 if "17_plane45_interp_hurdle_100k" in R else 1)' \
    || { echo "constants.py does not point to run 17"; exit 1; }
python -u scripts/03_train_recon_deepsets.py --recon-input kernel --out-suffix kernel100 || exit 1
MPLCONFIGDIR=/tmp/mplcache_$USER python -u - <<'PY'
import importlib.util, os, sys
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("p02", "plots/training/02_nn_target_vs_pred.py")
p02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p02)
out = p02.RECON_DEEPSETS_FOLDER + "_kernel100"
p02.plot_recon_dual(recon_folder=out)
p02.plot_recon_on_kernel(recon_folder=out)
PY
