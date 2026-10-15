#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=200g
#SBATCH --time=2-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH -J r19_ft_noise
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 19: fine-tune the recon on inputs with background (scripts/03c_finetune_recon_noise.py); new checkpoint in
# <RECON_FOLDER>_deepsets_noise/recon.pt, the clean recon is untouched. Resumes after a preemption.
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
python -u scripts/03c_finetune_recon_noise.py "$@"
