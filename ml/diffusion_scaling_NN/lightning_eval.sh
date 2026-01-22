#!/bin/bash
#SBATCH --job-name=lightning_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/home05/zdimitrov/tambo/logs/eval/lightning_eval_%j.out
#SBATCH --error=/n/home05/zdimitrov/tambo/logs/eval/lightning_eval_%j.err

set -euo pipefail

mkdir -p /n/home05/zdimitrov/tambo/logs/eval

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/conda_env/

echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
nvidia-smi || true
which python
python -V

# Helpful defaults
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# TODO: Update these paths to your data and checkpoint
DATA_DIR="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/pre_processed_3rd_step/"
CHECKPOINT="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/checkpoints/tam_unet/last.ckpt"
OUT_DIR="/n/home05/zdimitrov/tambo/eval_results"

python /n/home05/zdimitrov/tambo/TambOpt/ml/diffusion_scaling_NN/lightning_eval.py \
  --data_dir ${DATA_DIR} \
  --ckpt ${CHECKPOINT} \
  --out_dir ${OUT_DIR} \
  --num_samples 100 \
  --num_visualize 10 \
  --ddim_steps 50 \
  --eta 0.0 \
  --guidance_w 0.0 \
  --batch_size 16 \
  --num_workers 4 \
  --cache_size 10
