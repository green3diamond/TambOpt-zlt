#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J r19_optnoise_score
#SBATCH -o /n/home05/zdimitrov/tambo/TambOpt/slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
R=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k
python -u plots/analysis/run19/opt_noise/score_opt_noise.py 2>&1 | grep -v Warning
