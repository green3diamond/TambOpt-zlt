# One mail when a job really ends (not when it is preempted): call `notify_end <rc>` as the script's last step.
# Preemption SIGTERMs the script before it gets here, so requeues stay silent; the mail itself comes from a
# 1-minute serial_requeue job with --mail-type=END, i.e. through Slurm's normal mail path.
notify_end() {
    local rc=${1:-0} what
    [ "${NOTIFY:-1}" = 0 ] && return 0               # e.g. smoke runs
    [ "$rc" = 0 ] && what=done || what="FAILED_rc$rc"
    sbatch -p serial_requeue -t 1 --mem=100m -c 1 --mail-type=END -o /dev/null \
        -J "${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID:-}_${what}" --wrap true >/dev/null 2>&1 || true
}
