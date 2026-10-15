#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -c 4
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH -J r19_noise_ft
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k/diagnostics/noise/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Score all test layouts + optimised designs with background, using the noise-fine-tuned recon (noise_scores_ft.json),
# and the clean recon again through the shared background module (noise_scores_check.json: must equal noise_scores.json).
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
export MPLCONFIGDIR=/tmp/mplcache_$USER
N=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k/diagnostics/noise
RD=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k/test_v6_run_03_recentered_deepsets
python -u plots/analysis/run19/noise/score_noise.py --recon-dir ${RD}_noise --out $N/noise_scores_ft.json
python -u plots/analysis/run19/noise/score_noise.py --out $N/noise_scores_check.json
