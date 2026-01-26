#!/bin/bash
#SBATCH --job-name=fnn_eval
#SBATCH --partition=arguelles_delgado_gpu_mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=/n/home05/zdimitrov/tambo/logs/eval/fnn_%j.out
#SBATCH --error=/n/home05/zdimitrov/tambo/logs/eval/fnn_%j.err

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

# Evaluate FNN model - much faster than diffusion since no iterative sampling
python /n/home05/zdimitrov/tambo/TambOpt/ml/scaling_NN/FNN/lightning_eval.py \
  --data_dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/pre_processed_3rd_step_min_50/ \
  --ckpt /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/checkpoints/tam_fnn/last.ckpt \
  --out_dir /n/home05/zdimitrov/tambo/eval_results_fnn \
  --num_samples 1000 \
  --num_visualize 20 \
  --batch_size 64 \
  --num_workers 4 \
  --cache_size 10
