#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=recon_loop
#SBATCH --output=slurm-%x-%j.out

# Submit from the repository root with the Python environment already activated.
# All arguments go to run.py; RECON_LOOP_PYTHON can select a specific interpreter.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${PWD}/.pycache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${RECON_LOOP_PYTHON:-python}" -u recon_loop/run.py "$@"
