#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=200g
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH -J lbfgs_finish
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# L-BFGS phase only, on top of an Adam-only recon run (e.g. the weight-decay sweep, --lbfgs-iters 0).
# Marks the run "Adam done" (03 then loads its recon.pt and runs L-BFGS), keeps the Adam log/curves as
# *_adam.*.   sbatch slurm/lbfgs_finish_run18.sh <out-suffix> <weight-decay>   e.g. zoo_reltime_wd0.03 0.03
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
SUF=$1; WD=$2
export TAMBOPT_DATASET_TAG=zoo MPLCONFIGDIR=/tmp/mplcache_$USER
D=$(python -c 'import sys; sys.path.insert(0, "."); import modules.constants as C; print(C.RECON_FOLDER + "_deepsets")')_$SUF
[ -f "$D/recon.pt" ] || { echo "missing $D/recon.pt"; exit 1; }
[ -f "$D/recon_train_log_adam.json" ] || cp "$D/recon_train_log.json" "$D/recon_train_log_adam.json"
[ -f "$D/recon_train_curves_adam.png" ] || cp "$D/recon_train_curves.png" "$D/recon_train_curves_adam.png"
# First start only: mark Adam done (a requeue keeps the L-BFGS resume state the script writes itself).
[ -f "$D/recon_resume.pt" ] || python -c "import torch; torch.save({'adam_done': True}, '$D/recon_resume.pt')"
python -u scripts/03_train_recon_deepsets.py --recon-input kernel --input-mode reltime --weight-decay "$WD" \
    --epochs 80 --out-suffix "$SUF"
