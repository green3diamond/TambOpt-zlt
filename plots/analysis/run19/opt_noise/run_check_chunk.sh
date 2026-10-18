#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=96g
#SBATCH --time=01:30:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J r19_chunkcheck
#SBATCH -o /n/home05/zdimitrov/tambo/TambOpt/slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Stage 4 sub-batch (--chunk) check: same showers and layouts, chunk 32 ... 512 (U, gradient, time, GPU memory).
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
R=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k
for L in $R/diagnostics/opt/layouts/start_grid.pt $R/test_v6_run_04_optimize_lbfgs_kernel_twotier_l0p3_lr10_center_b4096_bg1us_knn23/layout_best.pt; do
    python -u plots/analysis/run19/opt_noise/check_chunk.py $L 32 64 128 256 512 2>&1 | grep -v Warning
done
