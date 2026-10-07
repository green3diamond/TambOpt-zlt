#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH -J r18_zoo_compare
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k/diagnostics/zoo_compare/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Old recon (5 strategies, archive/) vs zoo recon (9 strategies) on the same held-out showers and layouts:
# core_test_ndet (cores, hybrids, disks at 20/50 det.), clusters, angleonly (100-det. set + grid_nN + optimized),
# and the detector-count grid. Same scorer for both (score_pad: unused slots zero), within 1 / 2.5 / 5 deg.
source slurm/env.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER
R=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k
D=$R/diagnostics; Z=$D/zoo_compare
declare -A RD=([old]=$R/archive/test_v6_run_03_recentered_deepsets [zoo]=$R/test_v6_run_03_recentered_deepsets_zoo)
for tag in old zoo; do
  for set in core_test_ndet clusters angleonly; do
    L=$D/$set/layouts.txt; [ $set = angleonly ] && L=$Z/layouts_angleonly.txt
    O=$Z/$tag/$set; mkdir -p $O
    [ -s $O/scores_pad_1deg.json ] && continue
    python -u $D/core_test_ndet/score_pad.py --layouts $L --thetas 1 2.5 5 --recon-dir ${RD[$tag]} --out-dir $O || exit 1
  done
  O=$Z/$tag/detcount; mkdir -p $O
  [ -s $O/detector_count_grid.json ] || python -u scripts/diagnostics/detector_count_grid.py --recon-dir ${RD[$tag]} --out $O || exit 1
done
python -u $Z/plot_compare.py
