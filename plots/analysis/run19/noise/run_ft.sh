#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -c 4
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH -J r19_noise_ft
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k/diagnostics/noise/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Noise scoring with the readout window centred on the earliest hit (modules/optimize/background.py), with the clean
# recon and the noise fine-tune: test layouts + optimised designs, triplets (separate and averaged stations).
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
export MPLCONFIGDIR=/tmp/mplcache_$USER
R=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k
N=$R/diagnostics/noise; T=$R/diagnostics/triplets; FT=$R/test_v6_run_03_recentered_deepsets_noise
S=plots/analysis/run19/noise/score_noise.py; A=plots/analysis/run19/triplets/score_avg.py
python -u $S --out $N/noise_scores.json
python -u $S --recon-dir $FT --out $N/noise_scores_ft.json
python -u $S --layouts $T/layouts.txt --no-opt --out $T/triplet_scores.json
python -u $S --layouts $T/layouts.txt --no-opt --recon-dir $FT --out $T/triplet_scores_ft.json
python -u $A --out $T/triplet_avg_scores.json
python -u $A --recon-dir $FT --out $T/triplet_avg_scores_ft.json
