#!/bin/bash
#SBATCH --job-name=fnn_train
#SBATCH --partition=arguelles_delgado_gpu_mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/home05/zdimitrov/tambo/logs/training/fnn_%j.out
#SBATCH --error=/n/home05/zdimitrov/tambo/logs/training/fnn_%j.err


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

# Train FNN model - generally faster to train than diffusion
python /n/home05/zdimitrov/tambo/TambOpt/ml/scaling_NN/FNN/lightning_training_fnn.py \
  --data_dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/pre_processed_3rd_step_min_50/ \
  --save_weight_dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/checkpoints/tam_fnn/ \
  --log_dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/checkpoints/tam_fnn/logs \
  --batch_size 256 \
  --epoch 500 \
  --lr 1e-3 \
  --channel 64 \
  --loss_type mse \
  --use_ema \
  --precision 32 \
  --num_gpus 2 \
  --num_workers 8 \
  --cache_size 500 \
  --num_res_blocks 4
