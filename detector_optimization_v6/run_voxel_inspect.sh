#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g
#SBATCH --time=0:30:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -J voxel_inspect
#SBATCH -o slurm_logs/slurm-%j-%x.out

module load python

conda deactivate
conda deactivate

conda activate multiproc_env

export PYTHONUNBUFFERED=1

python -u plots/inspect_voxel_corpus.py --delta 31.25
