#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=200g
#SBATCH --time=2-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J r19_stage2
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 19 Stage 2: the hurdle surrogate (P(lit) + lit value, SURROGATE_HURDLE) for one species on the run 19 corpus,
# split as the recon (SPLIT_SEED = 1). Output <RECON-run>/test_v6_run_02_recentered/fnn_<species>.pt; resumes after
# a preemption. Usage: sbatch slurm/run19_stage2.sh <electron|muon|photon>. No mail on preemption.
source slurm/env.sh
source slurm/notify.sh
source slurm/gpu_gate.sh
gpu_health_gate
python -u scripts/02_train_fnn_deepsets.py --species "${1:?species}"
rc=$?
notify_end $rc
exit $rc
