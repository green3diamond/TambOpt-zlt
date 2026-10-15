#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -c 4
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH -J r19_tri_ft
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k/diagnostics/triplets/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 19 triplet tests with the noise-fine-tuned recon: separate readout, averaged stations (+ 2-of-3 gate), local 2-of-3.
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
export MPLCONFIGDIR=/tmp/mplcache_$USER
R=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k
T=$R/diagnostics/triplets; FT=$R/test_v6_run_03_recentered_deepsets_noise
python -u plots/analysis/run19/noise/score_noise.py --layouts $T/layouts.txt --no-opt --recon-dir $FT --out $T/triplet_scores_ft.json
python -u plots/analysis/run19/triplets/score_avg.py --recon-dir $FT --out $T/triplet_avg_scores_ft.json
python -u plots/analysis/run19/triplets/score_local.py --recon-dir $FT --out $T/triplet_local_scores_ft.json
