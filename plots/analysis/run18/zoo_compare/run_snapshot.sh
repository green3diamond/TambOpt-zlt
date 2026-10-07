#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH -J r18_score_snap
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k/diagnostics/zoo_compare/slurm-snap-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Score a recon snapshot (zoo_compare/snapshots/<tag>/recon.pt) on every test set, then compare with the zoo recon.
#   sbatch run_snapshot.sh <tag>
source slurm/env.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER
R=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k
D=$R/diagnostics; Z=$D/zoo_compare; TAG=$1; RD=$Z/snapshots/$TAG
for set in core_test_ndet clusters angleonly disks_n10; do
  L=$D/$set/layouts.txt; [ $set = angleonly ] && L=$Z/layouts_angleonly.txt; [ $set = disks_n10 ] && L=$Z/layouts_disks_n10.txt
  O=$Z/$TAG/$set; mkdir -p $O
  python -u $D/core_test_ndet/score_pad.py --layouts $L --thetas 1 2.5 5 --recon-dir $RD --out-dir $O || exit 1
done
mkdir -p $Z/$TAG/detcount
python -u scripts/diagnostics/detector_count_grid.py --recon-dir $RD --out $Z/$TAG/detcount || exit 1
python -u $Z/lit_vs_angle.py --recon-dir $RD --tag $TAG || exit 1
python -u $Z/lit_resolution.py $TAG || exit 1
python -u $Z/compare_tags.py zoo $TAG
