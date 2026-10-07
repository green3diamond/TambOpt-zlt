#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=200g
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --array=0-3
#SBATCH -J wd_sweep
#SBATCH -o slurm_logs/slurm-%A_%a-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Weight-decay sweep for the relative-time recon on run 18's zoo data (run from the run 19 branch via
# TAMBOPT_RUN_LOCATION). AdamW, 80 epochs, patience 30, no L-BFGS. Baseline (wd 0): the run 18 reltime
# recon, best val 0.0788 at epoch 43 (then overfits). Output: <run18>/test_v6_run_03_recentered_deepsets_zoo_reltime_wd<wd>.
source slurm/env.sh
WDS=(0.01 0.03 0.1 0.3)
WD=${WDS[$SLURM_ARRAY_TASK_ID]}
export TAMBOPT_RUN_LOCATION=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k
export TAMBOPT_DATASET_TAG=zoo
export MPLCONFIGDIR=/tmp/mplcache_$USER
python -u scripts/03_train_recon_deepsets.py --recon-input kernel --input-mode reltime --weight-decay $WD \
    --epochs 80 --adam-patience 30 --lbfgs-iters 0 --out-suffix zoo_reltime_wd$WD
