#!/bin/bash
#SBATCH --job-name=lightning_train
#SBATCH --partition=arguelles_delgado_gpu_mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:2 
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/home05/zdimitrov/tambo/logs/eval/lightning_unet_%j.out
#SBATCH --error=/n/home05/zdimitrov/tambo/logs/eval/lightning_unet_%j.err 

set -euo pipefail

mkdir -p /n/home05/zdimitrov/tambo/logs/eval

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/

echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
nvidia-smi || true
which python
python -V

# Helpful defaults
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python /n/home05/zdimitrov/tambo/TambOpt/ml/scaling_NN/diffusion_model/lightning_eval.py \
  --data_dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/pre_processed_3rd_step_min_50/ \
  --ckpt /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/checkpoints/tam_unet/epoch_epoch=1649-val_loss_val_loss=0.0590.ckpt \
  --out_dir /n/home05/zdimitrov/tambo/eval_results \
  --num_samples 100 \
  --num_visualize 10 \
  --ddim_steps 50 \
  --eta 0.0 \
  --guidance_w 0.0 \
  --batch_size 16 \
  --num_workers 4 \
  --cache_size 10 \
  --no_ddim
