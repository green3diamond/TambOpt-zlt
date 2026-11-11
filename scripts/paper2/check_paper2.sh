#!/bin/bash
# Compile check of paper 2's chapters at the journal page size (192 x 262 mm, 13.7 mm side margins). The cluster TeX
# lacks the cas-sc class packages, so the chapters are compiled inside a plain article with the paper's table packages.
#
#   scripts/paper2/check_paper2.sh <overleaf clone> [build dir]     (run on a compute node)
set -uo pipefail
SRC=$(cd "${1:?usage: check_paper2.sh <overleaf clone> [build dir]}" && pwd)
B=${2:-$HOME/.cache/paper2_check}
mkdir -p "$B"; rsync -a --delete --exclude .git "$SRC/" "$B/"
cat > "$B/check.tex" <<'TEX'
\documentclass{article}
\usepackage[paperwidth=192mm,paperheight=262mm,vmargin={19mm,19mm},hmargin={13.7mm,13.7mm}]{geometry}
\usepackage[authoryear]{natbib}
\usepackage{amsmath,amssymb,graphicx,longtable,tikz,float,array,pdflscape}
\usepackage[normalem]{ulem}
\newcommand{\hm}[2]{{\setlength{\fboxsep}{0.8pt}\colorbox{#1}{#2}}}
\begin{document}
\input{chapters_2/01_introduction}
\input{chapters_2/02_methodology}
\input{chapters_2/03_results}
\input{chapters_2/03b_comparisons}
\appendix
\input{chapters_2/appendix}
\end{document}
TEX
cd "$B"
for i in 1 2; do pdflatex -interaction=nonstopmode check.tex > check.out 2>&1; done
echo "errors: $(grep -c '^! ' check.out)  overfull hbox: $(grep -c 'Overfull \\hbox' check.log)"
grep -A3 '^! ' check.out | head -20
grep -o 'Overfull \\hbox ([0-9.]*pt' check.log | sort | uniq -c
grep -o "Reference \`[^']*' on page [0-9]* undefined" check.log | sort -u
echo "pdf: $B/check.pdf"
