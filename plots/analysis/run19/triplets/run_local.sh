#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -c 4
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH -J r19_tri_local
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k/diagnostics/triplets/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 19 triplet test, local 2-of-3 trigger per triangle (triangles with < 2 hits are zeroed).
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
export MPLCONFIGDIR=/tmp/mplcache_$USER
T=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k/diagnostics/triplets
python -u plots/analysis/run19/triplets/score_local.py
