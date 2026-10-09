#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH -J r18_opt1deg_score
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k/diagnostics/zoo_compare/opt1deg/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Score the 1 deg optimisations and their starts at 1 / 2.5 / 5 deg on the 1,536 held-out showers, with the
# recon they were optimised with (relative-time) and with the wd 0.03 recon.
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
export MPLCONFIGDIR=/tmp/mplcache_$USER
R=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k
O=$R/diagnostics/zoo_compare/opt1deg
python -u $O/make_layouts.py || exit 1
for t in reltime:test_v6_run_03_recentered_deepsets_zoo_reltime wd0.03:test_v6_run_03_recentered_deepsets_zoo_reltime_wd0.03; do
  mkdir -p $O/${t%%:*}
  python -u $R/diagnostics/core_test_ndet/score_pad.py --layouts $O/layouts.txt --thetas 1 2.5 5 \
      --recon-dir $R/${t#*:} --out-dir $O/${t%%:*} || exit 1
done
python -u $O/plot.py
