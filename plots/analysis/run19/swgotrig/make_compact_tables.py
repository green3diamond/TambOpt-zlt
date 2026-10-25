"""Run 19 recon comparison in one short file: default run 19 recon vs dropout and drawn-count fine-tunes, each cell
"default / dropout / count" (bold = best of the three). Drawn particle counts only. Two tables:
  1. per condition: 1 / 2.5 / 5 deg of triggered, mean and median angle error of triggered, 5 deg of all, Old U and
     New U (drawn trigger) over all showers; shower and ghost trigger rate
  2. robustness to background: 5 deg of triggered at 1 / 10 us as a share of the no-background value
  3. triggered ghosts (background only): median zenith and log10 E per recon, mean over layouts
Reads the score JSONs of make_triggered_tables.py and make_ghost_tables.py; writes
validation_compact.md (the long tables stay where they are). --only <recon>: that recon alone, one value per cell,
bold = best layout in the column; writes validation_compact_<recon>.md. --notion: the best values in a red box
(Notion LaTeX colorbox) instead of bold; adds _notion to the file name.

    python plots/analysis/run19/swgotrig/make_compact_tables.py [--only default|dropout|count] [--notion]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout_names import layouts  # noqa: E402

D = os.path.join(C.RUN_LOCATION, "diagnostics", "swgotrig")
LAYOUTS = layouts(D, short=True)                                 # validate_count_layouts.txt order
CONDS = [("No background, all read out", "none", "0"), ("No background, kNN gate", "knn", "0"),
         ("1 µs background, all read out", "none", "1000"), ("1 µs background, kNN gate", "knn", "1000"),
         ("10 µs background, all read out", "none", "10000"), ("10 µs background, kNN gate", "knn", "10000")]
ap = argparse.ArgumentParser()
ap.add_argument("--only", choices=("default", "dropout", "count"), default=None)
ap.add_argument("--notion", action="store_true")
A = ap.parse_args()
ONLY = A.only
mark = ((lambda x: r"$\colorbox{#f8c4c4}{\textbf{" + x.replace("%", r"\%") + "}}$") if A.notion
        else (lambda x: f"**{x}**"))
RECONS = [("default", "validate_count", "default"), ("dropout", "valdrop_count", "dropout fine-tune"),
          ("count", "valcnt_count", "drawn-count fine-tune")]
SEL = [i for i, r in enumerate(RECONS) if ONLY in (None, r[0])                 # recons with score files
       and glob.glob(os.path.join(D, f"{r[1]}_s*_*.json"))]
pct = lambda v: f"{100 * v:.1f}"                                               # noqa: E731
U3 = lambda v: f"{v:.3f}"                                                      # noqa: E731


def chunks(files):
    """Shower-weighted merge of held-out chunk score files -> {layout: {readout: {window: {metric: value}}}}."""
    ch = [json.load(open(f))["scores"] for f in files]
    out = {}
    for k in ch[0]:
        for rd, by_w in ch[0][k].items():
            if isinstance(by_w, dict):
                for w in by_w:
                    rows = [c[k][rd][w] for c in ch]
                    n = np.array([r["n_showers"] for r in rows], float)
                    nt = n * np.array([r["triggered"] for r in rows])          # triggered showers per chunk
                    out.setdefault(k, {}).setdefault(rd, {})[w] = {           # *_trig: over triggered showers
                        m: float(np.nansum(np.array([r[m] for r in rows]) * (nt if m.endswith("_trig") else n))
                                 / (nt if m.endswith("_trig") else n)[~np.isnan([r[m] for r in rows])].sum())
                        for m, v in rows[0].items() if isinstance(v, (int, float))}
    return out


def seeds(prefix):
    """Per complete count seed, the merged chunks."""
    by = {}
    for f in glob.glob(os.path.join(D, f"{prefix}_s*_*.json")):
        by.setdefault(int(re.search(r"_s(\d+)_\d+\.json$", f).group(1)), []).append(f)
    return [chunks(fs) for fs in by.values() if len(fs) == 5]


def trig_vals(S, k, rd, w):
    """Seed means of [1, 2.5, 5 deg of triggered, 5 deg of all, Old U, New U (drawn), trigger rate, mean and median
    angle error of triggered (nan in score files from before that metric)]."""
    v = [[s[k][rd][w][f"all_{a}"] / s[k][rd][w]["triggered"] for a in ("1", "2.5", "5")]
         + [s[k][rd][w].get(m, float("nan")) for m in ("notrig_5", "u_two_tier", "u_swgo_drawn", "triggered",
                                                       "ang_mean_trig", "ang_med_trig")] for s in S]
    return np.mean(v, 0)


def cell(vals, fmt=pct, low=False):
    """'a / b / c' with the best bold (ties at the shown precision; low: smallest is best); a single recon: the plain
    value."""
    s = [fmt(v) for v in vals]
    if len(s) == 1:
        return s[0]
    best = (min if low else max)(float(x) for x in s)
    return " / ".join(mark(x) if float(x) == best else x for x in s)


def emit(rows, first, low=()):
    """Table rows from [name, cells]; a single recon: bold the best layout in each column from index first (columns in
    low: smallest is best)."""
    if len(SEL) == 1:
        for j in range(first, len(rows[0])):
            best = (min if j in low else max)(float(r[j]) for r in rows)
            for r in rows:
                r[j] = mark(r[j]) if float(r[j]) == best else r[j]
    return ["| " + " | ".join(r) + " |" for r in rows]


cnt = [seeds(RECONS[i][1]) for i in SEL]
WHO = " / ".join(RECONS[i][0] for i in SEL)
ghost = json.load(open(os.path.join(D, "ghost_scores.json")))
G = ghost["scores"]
L = ["## Run 19 recon comparison — " + " / ".join(RECONS[i][2] for i in SEL), "",
     "21,060 held-out showers. " + (f"Every cell is **{WHO}** ({'red' if A.notion else 'bold'} = best). " if len(SEL) > 1 else
                                    f"{'Red' if A.notion else 'Bold'} = best layout in the column. ") + "Triggered = ≥ 3 read-out "
     "detectors with a drawn particle (background muons count too). Old U = mean[g + 0.3(1−g)q] (angle and energy, "
     "no trigger); New U = mean[trig · (g_a + 0.3(1−g_a)q_a)] (angle only, drawn trigger); both over all showers. "
     "Mean / median err = angle error [°] over triggered showers (lower is better).",
     "", "### 1. Drawn counts — angles [%] of triggered showers, utilities over all showers", "",
     "Trig = share of showers that trigger; Ghost (background conditions) = share of background-only windows that "
     "trigger (same trigger, "
     f"{ghost['n_windows']:,} windows). Both are recon independent. Mean of {len(cnt[0])} count seeds."]
for title, rd, w in CONDS:
    bg = w != "0"                                                              # no ghosts without background
    L.append(f"\n**{title}**\n\n| Layout | Trig | " + ("Ghost | " if bg else "")
             + "1° of trig | 2.5° of trig | 5° of trig | Mean err | Median err | 5° of all | Old U | New U |\n|"
             + "---|" * (11 if bg else 10))
    rows = []
    for k, name in LAYOUTS:
        v = [trig_vals(S, k, rd, w) for S in cnt]
        rows.append([name, pct(v[0][6])] + ([pct(G[k][rd][w]["ghost_trig"])] if bg else [])
                    + [cell([x[i] for x in v]) for i in range(3)]
                    + [cell([x[i] for x in v], lambda e: f"{e:.1f}", low=True) for i in (7, 8)]
                    + [cell([x[3] for x in v])] + [cell([x[i] for x in v], U3) for i in (4, 5)])
    c0 = 3 if bg else 2
    L += emit(rows, c0, low=(c0 + 3, c0 + 4))

L += ["", "### 2. Robustness to background — share of the no-background value kept [%]", "",
      "5° of triggered with background / without, same readout; 100 = unaffected. "
      + ("A share of a low baseline: the dropout recon keeps more, but its absolute value stays below the others "
         "(table 1). " if len(SEL) > 1 else "") + "New U is left out: with "
      "background it rises because background-only triggers add showers (see the Ghost column), not because the "
      "reconstruction improves."]
for title, rd in (("All read out", "none"), ("kNN gate", "knn")):
    L.append(f"\n**{title}**\n\n| Layout | 1 µs | 10 µs |\n|---|---|---|")
    rows = []
    for k, name in LAYOUTS:
        v = {w: [trig_vals(S, k, rd, w) for S in cnt] for w in ("0", "1000", "10000")}
        keep = lambda w: [x[2] / max(y[2], 1e-12) for x, y in zip(v[w], v["0"])]  # noqa: E731
        rows.append([name] + [cell(keep(w), lambda r: f"{100 * r:.0f}") for w in ("1000", "10000")])
    L += emit(rows, 1)

names = [n for n in ghost["recon_dirs"]
         if any(n.rstrip("/").endswith(sfx) for sfx in [("_deepsets", "_deepsets_dropout", "_deepsets_count")[i]
                                                        for i in SEL])]
L += ["", "### 3. What the recons report for triggered ghosts (mean over layouts)", "",
      "| Condition | Median zenith [°] | Median log10 E | In E range 5–7 |", "|---|---|---|---|"]
for title, rd, w in (("10 µs, all read out", "none", "10000"), ("10 µs, kNN gate", "knn", "10000")):
    per = [[G[k][rd][w]["recons"][os.path.basename(n.rstrip("/"))] for k, _ in LAYOUTS] for n in names]
    f = lambda key, i=None: " / ".join(f"{np.nanmean([x[key][i] if i is not None else x[key] for x in p]):.{0 if i is not None else 2}f}"  # noqa: E731
                                       for p in per)
    L.append(f"| {title} | {f('zen_p10_50_90', 1)} | {f('log10E_med')} | "
             + " / ".join(f"{100 * np.nanmean([x['in_e_range'] for x in p]):.0f}%" for p in per) + " |")
txt = "\n".join(L) + "\n"
OUT = os.path.join(D, ("validation_compact" + (f"_{ONLY}" if ONLY else "") + ("_notion" if A.notion else "") + ".md"))
open(OUT, "w").write(txt)
print(txt)
print("[write]", OUT)
