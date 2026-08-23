#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=60g
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -J upcmp
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

# Stage 1 of the upstream comparison: do the local AllShowers generator and
# pristine upstream produce the same showers from identical inputs?
#
#   sbatch -J upcmp_electron slurm/run_upstream_compare.sh electron 32 8
#   sbatch -J upcmp_photon   slurm/run_upstream_compare.sh photon   32 4
#   sbatch -J upcmp_muon     slurm/run_upstream_compare.sh muon     32 1
#
# One job per species so a slow muon does not hold up the fast results.
# `batch` is the GPU generation batch: flex_attention's block mask is
# O(batch x cap^2) and the caps differ 6x, so it is set per species by the
# caller rather than guessed here.

source slurm/env.sh

SPECIES="${1:?usage: $0 <species> [n] [batch] [impls]}"
N="${2:-32}"
BATCH="${3:-4}"
IMPLS="${4:-local local-shim hamza}"

# Above a few thousand showers the sample tensors run to gigabytes (12k muons is
# ~5 GB per run), so keep per-shower stats + digests instead. Equal digests means
# equal bytes, so bit-identity is still decided.
EXTRA=""
[ "$N" -gt 2000 ] && EXTRA="--stats-only"

OUT=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/tests/upstream_comparison
mkdir -p "$OUT"
T="tests/compare_upstream_generator.py"
PREP="$OUT/prep_${SPECIES}_n${N}.pt"

set -e
echo "===== prep ====="
python -u $T prep --species "$SPECIES" --n "$N" --out "$PREP"

SUF=""
[ "$N" -gt 2000 ] && SUF="_n${N}"

for IMPL in $IMPLS; do
    echo "===== gen $IMPL ====="
    python -u $T gen --impl "$IMPL" --prep "$PREP" --batch "$BATCH" $EXTRA \
        --out "$OUT/gen_${SPECIES}_${IMPL}${SUF}.pt"
done

# local vs local-shim validates the shim adds nothing beyond time support;
# then local (or local-shim) vs hamza is a clean read on the code difference.
for PAIR in "local local-shim" "local-shim hamza" "local hamza"; do
    set -- $PAIR
    A="$OUT/gen_${SPECIES}_${1}${SUF}.pt"; B="$OUT/gen_${SPECIES}_${2}${SUF}.pt"
    [ -f "$A" ] && [ -f "$B" ] || continue
    echo "===== compare: $1 vs $2 ====="
    python -u $T compare "$A" "$B"
done
