#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=200g
#SBATCH --time=15:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=holygpu8a19102
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J r20_stage2
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 20 (Malata) Stage 2: the hurdle surrogate for one species on the run 20 Stage 1 dataset (as run19_stage2.sh), split
# as the recon. Output <run 20>/test_v6_run_02_recentered/fnn_<species>.pt; resumes after a preemption (run 19 took
# 6-8 h per species on an A100). PREV_JOB=<id> for a continuation (sbatch -d afternotok:<id>): runs only if <id> hit its
# time limit. No mail on preemption.
#   sbatch -J r20_stage2_<species> slurm/run20_stage2.sh <electron|muon|photon>
if [ -n "$PREV_JOB" ]; then
    PREV_STATE=$(sacct -j "$PREV_JOB" -X -n -o State | awk 'NR==1 {print $1}')
    case "$PREV_STATE" in
        TIMEOUT|NODE_FAIL) echo "[continue] job $PREV_JOB ended $PREV_STATE -> resuming" ;;
        *) echo "[continue] job $PREV_JOB ended ${PREV_STATE:-?}, not a time limit -> nothing to do"; exit 0 ;;
    esac
fi
source slurm/env.sh
source slurm/notify.sh
source slurm/gpu_gate.sh
gpu_health_gate
source slurm/run20_guard.sh
python -u scripts/02_train_fnn_deepsets.py --species "${1:?species}"
rc=$?
notify_end $rc
exit $rc
