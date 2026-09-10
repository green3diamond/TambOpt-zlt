# GPU health gate, shared by every gpu_requeue wrapper that starts a training
# job. A requeued job can land back on the SAME node it was preempted from,
# still pinned by the killed process's CUDA context, so the first alloc fails
# with "CUDA-capable device(s) is/are busy or unavailable" (jobs 42556695,
# 43192939, 43844284, 43857690, 44186645). Waiting it out is the wrong trade
# (a stuck writer survived 210s of waiting in job 44186645, vs a ~3 min
# requeue turnaround that also re-rolls the node -- the only thing that
# actually clears it), so this probes briefly then requeues rather than
# waiting long.
#
# Source it, then call the function -- `exit`/`scontrol requeue` need to run
# in the caller's shell, not a subshell, to actually stop the sbatch job:
#
#     source slurm/gpu_gate.sh
#     gpu_health_gate

gpu_health_gate() {
    local this_node gate_fails n
    this_node="$(hostname -s)"
    gate_fails=".gpu_gate_fails"     # consecutive gate-triggered requeues

    # Poll inside ONE python process: `import torch` alone costs ~30s, so a
    # shell retry loop would burn its budget on imports.
    python - <<'PY'
import sys, time, torch
DEADLINE, STEP = 90, 30
t0 = time.time()
while True:
    try:
        torch.zeros(1, device="cuda"); torch.cuda.synchronize(); sys.exit(0)
    except RuntimeError as e:
        if time.time() - t0 > DEADLINE:
            print(f"[gpu-gate] still busy after {DEADLINE}s: {e}", flush=True)
            sys.exit(1)
        print(f"[gpu-gate] busy at t+{time.time()-t0:.0f}s, retrying", flush=True)
        time.sleep(STEP)
PY
    if [ $? -ne 0 ]; then
        # Consecutive failures, not SLURM_RESTART_COUNT: that also counts
        # ordinary preemptions (6 in 10.5h on this partition), which would
        # conflate "unlucky node once" with "GPU broken everywhere".
        n=$(( $(cat "$gate_fails" 2>/dev/null || echo 0) + 1 ))
        echo "$n" > "$gate_fails"
        if [ "$n" -ge 5 ]; then
            echo ">>> GPU busy on $n consecutive nodes -- real fault, giving up"
            rm -f "$gate_fails"
            exit 1
        fi
        # Cannot blacklist the node (scontrol update ExcNodeList is refused on
        # a RUNNING job, job 44336590) -- a bare requeue re-rolls it anyway.
        echo ">>> GPU busy on ${this_node} (strike $n/5) -- requeueing to re-roll the node"
        scontrol requeue "$SLURM_JOB_ID"
        sleep 120        # let the controller act before the script would exit
        exit 0
    fi
    rm -f "$gate_fails"
    echo ">>> GPU OK on ${this_node} (restart #${SLURM_RESTART_COUNT:-0})"
}
