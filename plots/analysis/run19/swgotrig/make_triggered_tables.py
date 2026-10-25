"""Run 19 drawn-count validation on triggered showers only, one Notion table file per recon: angle columns are
(triggered and within X deg) / triggered, triggered = >= 3 read-out detectors with a drawn particle (background muons
count too). Mean over the count seeds of each recon's held-out chunks (shower-weighted within a seed). Red \\colorbox =
best in the column (ties at table precision). Recons whose score files are missing are skipped. Also writes, per
fine-tuned recon, validation_count_triggered_default_vs_<name>.md: default → fine-tuned, each angle / U cell
"of triggered (of all)".

    python plots/analysis/run19/swgotrig/make_triggered_tables.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402

D = os.path.join(C.RUN_LOCATION, "diagnostics", "swgotrig")
LAYOUTS = [("disk_r6000_n99", "Uniform disk 6 km"), ("grid_n99", "Grid"),
           ("tri33_r10_n99", "Triplets r = 10 m on grid"), ("tri33_r25_n99", "Triplets r = 25 m on grid"),
           ("opt_a2_grid", "Opt. Poisson 2 m², grid start"), ("opt_a2_center", "Opt. Poisson 2 m², centre start"),
           ("opt_a20_grid", "Opt. Poisson 20 m², grid start"),
           ("opt_a2_step_grid_5000", "Opt. step 2 m², grid start (5,000 steps)"),
           ("opt_a2_step_grid_8000", "Opt. step 2 m², grid start (8,000 steps)"),
           ("opt_a2_step_center", "Opt. step 2 m², centre start"), ("disk_r1500_n99", "Disk 1.5 km"),
           ("tri33disk_R1000_r25_n99", "Triplets on 1 km disk")]
CONDS = [("No background, all read out", "none", "0"), ("1 µs background, all read out", "none", "1000"),
         ("10 µs background, all read out", "none", "10000"), ("No background, kNN gate", "knn", "0"),
         ("1 µs background, kNN gate", "knn", "1000"), ("10 µs background, kNN gate", "knn", "10000")]
RECONS = [("validate_count_s*_*.json", "original run 19 recon", "validation_count_triggered_notion.md"),
          ("valdrop_count_s*_*.json", "dropout fine-tuned recon", "validation_count_triggered_dropout_notion.md"),
          ("valcnt_count_s*_*.json", "drawn-count fine-tuned recon", "validation_count_triggered_count_notion.md")]
M = ["all_1", "all_2.5", "all_5", "u_swgo_drawn", "triggered"]
FMT = [lambda v: f"{100 * v:.1f}%"] * 3 + [lambda v: f"{v:.3f}", lambda v: f"{100 * v:.1f}%"]
box = lambda s: r"$\colorbox{#f8c4c4}{\textbf{" + s.replace("%", r"\%") + "}}$"  # noqa: E731


def load(pattern):
    """[per complete count seed: [chunk scores]]."""
    by = {}
    for f in glob.glob(os.path.join(D, pattern)):
        sd = int(re.search(r"_s(\d+)_\d+\.json$", f).group(1))
        by.setdefault(sd, []).append(json.load(open(f))["scores"])
    return [chs for chs in by.values() if len(chs) == 5]


def values(by, k, rd, w):
    """Seed means of [1, 2.5, 5 deg, New U] of triggered, the same of all showers, and the trigger rate."""
    per = []
    for chs in by:
        n = np.array([c[k][rd][w]["n_showers"] for c in chs], float)
        s = {m: np.dot([c[k][rd][w][m] for c in chs], n) / n.sum() for m in M + ["notrig_1", "notrig_2.5", "notrig_5"]}
        t = s["triggered"]
        per.append([s["all_1"] / t, s["all_2.5"] / t, s["all_5"] / t, s["u_swgo_drawn"] / t,
                    s["notrig_1"], s["notrig_2.5"], s["notrig_5"], s["u_swgo_drawn"], t])
    v = np.mean(per, 0)
    return v[:4], v[4:8], v[8]


def build(pattern, title, out):
    by = load(pattern)
    if not by:
        print(f"[skip] {pattern}: no complete seed")
        return

    def val(k, rd, w):
        trig, _, t = values(by, k, rd, w)
        return np.append(trig, t)

    L = [f"## Drawn particle counts, triggered showers only — {title}", "",
         "Angle columns are computed only on showers recognised as triggered: (triggered and within X°) / triggered, "
         "triggered = ≥ 3 read-out detectors with a drawn particle (background muons count too). New U \\| trig = mean "
         "reward over triggered showers; Trig = share of all showers that trigger. Mean of " + str(len(by))
         + " count seeds, 21,060 held-out showers. Red = best in the column (ties marked)."]
    for t, rd, w in CONDS:
        rows = [(name, [f(v) for f, v in zip(FMT, val(k, rd, w))]) for k, name in LAYOUTS]
        best = [max(float(r[1][i].rstrip("%")) for r in rows) for i in range(len(M))]
        L.append(f"\n### {t}\n\n| Layout | 1° (of triggered) | 2.5° (of triggered) | 5° (of triggered) | New U \\| trig "
                 "| Trig (drawn, of all) |\n| --- | --- | --- | --- | --- | --- |")
        for name, v in rows:
            L.append("| " + name + " | " + " | ".join(box(x) if float(x.rstrip("%")) == best[i] else x
                                                      for i, x in enumerate(v)) + " |")
    open(os.path.join(D, out), "w").write("\n".join(L) + "\n")
    print("[write]", os.path.join(D, out))




def compare(pattern, title, out):
    """Default → fine-tuned, cells "of triggered (of all)"."""
    old, new = load(RECONS[0][0]), load(pattern)
    if not old or not new:
        print(f"[skip] {pattern}: no complete seed")
        return
    L = [f"## Drawn particle counts — default run 19 recon → {title}", "",
         "Cells: default → fine-tuned, each as of triggered (of all showers); bold = fine-tuned better on triggered. "
         "Of triggered = (triggered and within X°) / triggered, triggered = ≥ 3 read-out detectors with a drawn particle "
         "(background muons count too); of all = within X° / all showers. New U \\| trig = mean reward over triggered "
         "showers (of all: mean[trig · reward] over all showers). Trig = share of all showers that trigger (independent "
         f"of the recon). Mean of {len(old)} / {len(new)} count seeds, 21,060 held-out showers."]
    for t, rd, w in CONDS:
        L.append(f"\n### {t}\n\n| Layout | 1° | 2.5° | 5° | New U | Trig |\n| --- | --- | --- | --- | --- | --- |")
        for k, name in LAYOUTS:
            (at, aa, tr), (bt, ba, _) = values(old, k, rd, w), values(new, k, rd, w)
            cells = []
            for i, f in enumerate(FMT[:4]):
                right = f"{f(bt[i])} ({f(ba[i])})"
                cells.append(f"{f(at[i])} ({f(aa[i])}) → " + (f"**{right}**" if f(bt[i]) != f(at[i]) and bt[i] > at[i]
                                                                else right))
            L.append(f"| {name} | " + " | ".join(cells) + f" | {FMT[4](tr)} |")
    open(os.path.join(D, out), "w").write("\n".join(L) + "\n")
    print("[write]", os.path.join(D, out))


for pattern, title, out in RECONS:
    build(pattern, title, out)
for pattern, title, out in RECONS[1:]:
    compare(pattern, title, out.replace("validation_count_triggered_", "validation_count_triggered_default_vs_")
                                .replace("_notion", ""))
