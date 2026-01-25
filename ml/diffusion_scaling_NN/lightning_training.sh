#!/bin/bash
#SBATCH --job-name=lightning_train
#SBATCH --partition=arguelles_delgado_gpu_mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:2 
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/home05/zdimitrov/tambo/logs/training/lightning_unet_%j.out
#SBATCH --error=/n/home05/zdimitrov/tambo/logs/training/lightning_unet_%j.err 


set -euo pipefail

mkdir -p /n/home05/zdimitrov/tambo/logs/training

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

# TODO: Update DATA_DIR to point to your preprocessed data
python /n/home05/zdimitrov/tambo/TambOpt/ml/diffusion_scaling_NN/lightning_training.py \
  --data_dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/pre_processed_3rd_step_min_50/ \
  --save_weight_dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/checkpoints/tam_unet/ \
  --log_dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/checkpoints/tam_unet/logs \
  --batch_size 128 \
  --epoch 2000 \
  --lr 1e-4 \
  --channel 64 \
  --channel_mult 1 2 2 2 \
  --use_cfg \
  --precision 32 \
  --use_ema \
  --num_gpus 2 \
  --num_workers 8 \
  --cache_size 500 \
  --num_res_blocks 3

