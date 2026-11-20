#!/bin/bash
# Score the rerun optimizations (_m5k / _m5kc folders) for paper 2: one gpu_requeue job per score family, same scorer
# and settings as the res2 files (score_all.py, 5,000 flow-matching showers or all CORSIKA 8 test showers, default
# time model), then build the score folder res3 (m5k_rescore.py build) for make_paper2.sh.
#   bash scripts/paper2/m5k_rescore.sh            submit
#   python scripts/paper2/m5k_rescore.py build <res2> <res3>   after the jobs
set -e
W=/n/home05/zdimitrov/tambo/TambOpt
T=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings
D=$T/res2; O=$T/res3
cd $W
mkdir -p $O/m5k
python3 scripts/paper2/m5k_rescore.py lists $D $O | while IFS='|' read -r fam site e n; do
    t=01:00:00; [ $fam = scoresc8ana_20 ] && t=02:00:00   # analytical v1 on CORSIKA 8 is the slowest
    sbatch --parsable -p gpu_requeue --gres=gpu:1 --constraint='a100|h100|h200' --exclude=holygpu8a19102 -c 4 \
        --mem=64g -t $t --requeue --open-mode=append -J m5ks_$fam -o $O/m5k/$fam.out --chdir=$W --wrap "
        source slurm/env.sh; source slurm/gpu_gate.sh; gpu_health_gate
        export MPLCONFIGDIR=/tmp/mplcache_\$USER PYTHONDONTWRITEBYTECODE=1; unset TAMBOPT_TIME_MODEL TAMBOPT_TIME_NOISE
        source slurm/run${site}_guard.sh
        env $e python -u scripts/paper2/score_all.py $O/m5k/$fam.json 5000 \$(cat $O/m5k/$fam.txt)" \
        | sed "s/^/  $fam ($n layouts): job /"
done
