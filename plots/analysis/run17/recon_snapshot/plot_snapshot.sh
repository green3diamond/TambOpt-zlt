#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH -J r17_recon_snap
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/recon_snapshot/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
source slurm/env.sh
python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/recon_snapshot/plot_snapshot.py
