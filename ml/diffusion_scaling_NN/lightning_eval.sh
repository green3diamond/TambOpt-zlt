#!/bin/bash
#SBATCH --job-name=lightning_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/home04/hhanif/tam/logs/eval/lightning_unet_%j.out
#SBATCH --error=/n/home04/hhanif/tam/logs/eval/lightning_unet_%j.err

set -euo pipefail

mkdir -p /n/home04/hhanif/tam/logs/eval

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/

echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
nvidia-smi || true
which python
python -V

# Helpful defaults for multi-GPU runs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python /n/home04/hhanif/tam/unet/lightining_eval.py     --data_dir  /n/holylfs05/LABS/arguelles_delglado_lab/Everyone/hhanif/tambo_simulations/pre_processed_3rd_step/       --ckpt /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/checkpoints/tam_unet/epoch_epoch=839-val_loss_val_loss=0.0340.ckpt     --out_dir /n/home04/hhanif/tam/plots/lighting_eval_plots_w_2      --num_samples 10     --ddim_steps 1000     --eta 0.0     --guidance_w 2

