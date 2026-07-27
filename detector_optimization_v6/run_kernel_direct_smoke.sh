#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g
#SBATCH --time=1:30:00
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --open-mode=append
#SBATCH -J kernel_direct_smoke
#SBATCH -o slurm_logs/slurm-%j-%x.out

module load python

conda deactivate
conda deactivate

conda activate multiproc_env

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== rerunning validate_limitations.py (fixed check3 + check5) ==="
python -u plots/validate_limitations.py

echo
echo "=== rebuilding voxel corpus with the corruption-point filter ==="
python -u 04_optimize_kernel_direct.py --schemes grid --chains 2 --adam-epochs 300 --lbfgs-iters 25 --rebuild-voxels --overwrite
