#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH -J r18_disks_n10
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k/diagnostics/zoo_compare/slurm-disks_n10-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# 10-detector disks scored by the old and zoo recons (same scorer as run.sh), then the disk plot and the PDF.
source slurm/env.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER
R=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k
Z=$R/diagnostics/zoo_compare
python -u $Z/make_disks_n10.py || exit 1
declare -A RD=([old]=$R/archive/test_v6_run_03_recentered_deepsets [zoo]=$R/test_v6_run_03_recentered_deepsets_zoo)
for tag in old zoo; do
  O=$Z/$tag/disks_n10; mkdir -p $O
  python -u $R/diagnostics/core_test_ndet/score_pad.py --layouts $Z/layouts_disks_n10.txt --thetas 1 2.5 5 \
      --recon-dir ${RD[$tag]} --out-dir $O || exit 1
done
python -u $Z/plot_disks.py || exit 1
python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/figures/make_zoo_recon_pdf.py
