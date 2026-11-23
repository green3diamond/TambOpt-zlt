#!/bin/bash
# Rescore all layouts of the paper 2 background families at the cosmic-muon rate (bgrate_rescore.py): one gpu_requeue
# job per family, as m5k_rescore.sh, with TAMBOPT_BG_RATE set. A last job (after all of them) builds <out>
# (bgrate_rescore.py build) and mails when done; then make_paper2.sh <overleaf clone> <out>.
#   RATE=4e-7 OUT=.../res4 bash scripts/paper2/bgrate_rescore.sh
set -e
W=/n/home05/zdimitrov/tambo/TambOpt
T=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings
D=$T/res3; O=${OUT:-$T/res4}; RATE=${RATE:-4e-7}
cd $W
mkdir -p $O/bgrate
IDS=""
while IFS='|' read -r fam site e n; do
    id=$(sbatch --parsable -p gpu_requeue --gres=gpu:1 --constraint='a100|h100|h200' --exclude=holygpu8a19102 -c 4 \
        --mem=64g -t 03:00:00 --requeue --open-mode=append -J bgr_$fam -o $O/bgrate/$fam.out --chdir=$W --wrap "
        source slurm/env.sh; source slurm/gpu_gate.sh; gpu_health_gate
        export MPLCONFIGDIR=/tmp/mplcache_\$USER PYTHONDONTWRITEBYTECODE=1 TAMBOPT_BG_RATE=$RATE; unset TAMBOPT_TIME_MODEL TAMBOPT_TIME_NOISE
        source slurm/run${site}_guard.sh
        env $e python -u scripts/paper2/score_all.py $O/bgrate/$fam.json 5000 \$(cat $O/bgrate/$fam.txt)")
    echo "  $fam ($n layouts): job $id"; IDS="$IDS:$id"
done < <(python3 scripts/paper2/bgrate_rescore.py lists $D $O)
sbatch -p serial_requeue -c 1 --mem=4g -t 00:30:00 --dependency=afterany$IDS --mail-type=END -J bgrate_rescore_built \
    -o $O/bgrate/build.out --chdir=$W --wrap "python3 scripts/paper2/bgrate_rescore.py build $D $O"
