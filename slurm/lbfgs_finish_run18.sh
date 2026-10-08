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
D=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k/test_v6_run_03_recentered_deepsets_$SUF
cp -n $D/recon_train_log.json $D/recon_train_log_adam.json      # keep the Adam log (L-BFGS rewrites it)
cp -n $D/recon_train_curves.png $D/recon_train_curves_adam.png
# Tell 03 that Adam is done, so it loads recon.pt and runs only L-BFGS (skipped on a requeue: 03's own resume file is kept)
[ -f $D/recon_resume.pt ] || python -c "import torch; torch.save({'adam_done': True}, '$D/recon_resume.pt')"
python -u scripts/03_train_recon_deepsets.py --recon-input kernel --input-mode reltime --weight-decay $WD \
    --epochs 80 --out-suffix $SUF
