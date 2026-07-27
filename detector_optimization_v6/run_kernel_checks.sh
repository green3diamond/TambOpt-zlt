#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g
#SBATCH --time=4:00:00
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --open-mode=append
#SBATCH -J kernel_checks
#SBATCH -o slurm_logs/slurm-%j-%x.out

module load python

conda deactivate
conda deactivate

conda activate multiproc_env

export PYTHONUNBUFFERED=1

python -u plots/check_compaction_fidelity.py --n-events 512 --deltas 125,62.5,31.25
python -u plots/check_kernel_grad_timing.py --n-events 20000 --recon-events 4096
