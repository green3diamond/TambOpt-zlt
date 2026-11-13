#!/bin/bash
# Paper 2 (TAMBO layout study): every results table and result figure, generated from the score JSONs and written
# into the paper folder (tables_2/*.tex complete tables, figures/results_2/*.pdf). The chapters only \input them.
#
#   scripts/paper2/make_paper2.sh <overleaf clone> [score dir]
#
# Run on a compute node (e.g. srun -p test -c 2 --mem=8g -t 15 ...). Scoring: score_all.py (GPU jobs);
# CORSIKA 8 fine-tune: ft_c8.py. Compile check: check_paper2.sh.
set -euo pipefail
OUT=$(cd "${1:?usage: make_paper2.sh <overleaf clone> [score dir]}" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../slurm/env.sh" >/dev/null 2>&1 || true
RES=()
[ $# -ge 2 ] && RES=(--res "$2")
python "$HERE/make_tables.py" --out "$OUT" "${RES[@]}"
python "$HERE/plot_exp.py" --out "$OUT" "${RES[@]}"
python "$HERE/plot_c8ana.py" --out "$OUT"
