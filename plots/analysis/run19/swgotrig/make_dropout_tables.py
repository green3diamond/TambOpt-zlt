"""Run 19: the run 19 recon vs a fine-tuned copy on the 21,060 held-out showers. --variant dropout (default): 03c
--dropout 0.9 --min-hit 3, <RECON_FOLDER>_deepsets_dropout, score prefixes valdrop / robust_new; --variant count: 03c
--counting poisson --min-hit 3, <RECON_FOLDER>_deepsets_count, prefixes valcnt / robust_cnt. Three comparisons, each cell "original → fine-tuned":
  expected-value inputs   validate_full.json (original) vs <prefix>_exp_<start>.json
  drawn particle counts   validate_count_s<seed>_<start>.json vs <prefix>_count_s<seed>_<start>.json, mean over seeds
  missing detectors       robust_old_p{30,60} vs <robust prefix>_p{30,60} (30 / 60 % of detectors removed, no background)
Chunks are merged as shower-weighted means. Writes validation_dropout_tables.md (validation_countft_tables.md for count) next to the scores and prints it.

    python plots/analysis/run19/swgotrig/make_dropout_tables.py [--variant dropout|count]
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

D = os.path.join(C.RUN_LOCATION, "diagnostics", "swgotrig")
ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=("dropout", "count"), default="dropout")
VARIANT = ap.parse_args().variant
PRE, ROB = {"dropout": ("valdrop", "robust_new"), "count": ("valcnt", "robust_cnt")}[VARIANT]
FT_DESC = {"dropout": "03c --dropout 0.9 (per row a fraction ~ U(0, 0.9) of detectors missing) --min-hit 3 (train on "
                      "rows with >= 3 detectors left), noise-free expected inputs",
           "count": "03c --counting poisson (Poisson-drawn particle counts per species, 2 m², ε_γ = 0.02) --min-hit 3 "
                    "(train on rows with >= 3 detectors with a drawn particle)"}[VARIANT]
LAYOUTS = [("disk_r6000_n99", "Uniform disk 6 km"), ("grid_n99", "Grid"),
           ("tri33_r10_n99", "Triplets r = 10 m on grid"), ("tri33_r25_n99", "Triplets r = 25 m on grid"),
           ("opt_a2_grid", "Opt. Poisson 2 m², grid start"), ("opt_a2_center", "Opt. Poisson 2 m², centre start"),
           ("opt_a20_grid", "Opt. Poisson 20 m², grid start"),
           ("opt_a2_step_grid_5000", "Opt. step 2 m², grid start (5,000 steps)"),
           ("opt_a2_step_grid_8000", "Opt. step 2 m², grid start (8,000 steps)"),
           ("opt_a2_step_center", "Opt. step 2 m², centre start"), ("disk_r1500_n99", "Disk 1.5 km"),
           ("tri33disk_R1000_r25_n99", "Triplets on 1 km disk")]
PCT = lambda v: f"{100 * v:.1f}%"                                             # noqa: E731
U3 = lambda v: f"{v:.3f}"                                                      # noqa: E731
CONDS = [("No background, all read out", "none", "0"), ("No background, kNN gate", "knn", "0"),
         ("1 µs background, all read out", "none", "1000"), ("1 µs background, kNN gate", "knn", "1000"),
         ("10 µs background, all read out", "none", "10000"), ("10 µs background, kNN gate", "knn", "10000")]


def merge_chunks(files):
    """{layout: {readout: {window: {metric: shower-weighted mean}}}} over held-out chunk files."""
    ch = [json.load(open(f))["scores"] for f in files]
    out = {}
    for k in ch[0]:
        for rd, by_w in ch[0][k].items():
            if not isinstance(by_w, dict):
                continue
            for w in by_w:
                rows = [c[k][rd][w] for c in ch]
                n = np.array([r["n_showers"] for r in rows], float)
                out.setdefault(k, {}).setdefault(rd, {})[w] = {
                    m: float(np.dot([r[m] for r in rows], n) / n.sum())
                    for m, v in rows[0].items() if isinstance(v, (int, float)) and m in rows[-1]}
    return out


def seed_mean(pattern):
    by_seed = {}
    for f in glob.glob(os.path.join(D, pattern)):
        sd = int(re.search(r"_s(\d+)_\d+\.json$", f).group(1))
        by_seed.setdefault(sd, []).append(f)
    merged = [merge_chunks(fs) for fs in by_seed.values() if len(fs) == 5]
    if not merged:
        return None, 0
    out = {}
    for k in merged[0]:
        for rd in merged[0][k]:
            for w in merged[0][k][rd]:
                out.setdefault(k, {}).setdefault(rd, {})[w] = {
                    m: float(np.mean([s[k][rd][w][m] for s in merged])) for m in merged[0][k][rd][w]}
    return out, len(merged)


def chunks(prefix):
    fs = sorted(glob.glob(os.path.join(D, f"{prefix}_[0-9]*.json")))
    return merge_chunks(fs) if len(fs) == 5 else None


def table(title, old, new, conds, cols):
    lines = [f"\n### {title}"]
    for ctitle, rd, w in conds:
        lines.append(f"\n**{ctitle}**\n\n| Layout | " + " | ".join(lab for _, lab, _ in cols) + " |\n|"
                     + "---|" * (len(cols) + 1))
        for k, name in LAYOUTS:
            if k not in old or k not in new:
                continue
            cells = []
            for m, _, f in cols:
                a, b = old[k][rd][w][m], new[k][rd][w][m]
                cells.append(f"{f(a)} → **{f(b)}**" if b > a else f"{f(a)} → {f(b)}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


FT_LOG = json.load(open(os.path.join(C.RECON_FOLDER + f"_deepsets_{VARIANT}", "finetune_log.json")))
out = [f"## Run 19 recon vs {VARIANT} fine-tuned recon — 21,060 held-out showers",
       "",
       f"Fine-tune: {FT_DESC} + 0 / 1 / 10 µs background, {FT_LOG['log'][-1]['epoch']} epochs (best epoch "
       f"{FT_LOG['best_epoch']}, val {FT_LOG['log'][0]['val_mean']:.3f} -> {FT_LOG['best_val_mean']:.3f}). "
       "Cells: original → fine-tuned (bold = fine-tuned better)."]
COLS_EXP = [("notrig_1", "1°", PCT), ("notrig_2.5", "2.5°", PCT), ("notrig_5", "5°", PCT),
            ("u_two_tier", "Old U", U3), ("u_swgo", "New U (Poisson)", U3)]
COLS_CNT = [("notrig_1", "1°", PCT), ("notrig_2.5", "2.5°", PCT), ("notrig_5", "5°", PCT),
            ("u_two_tier", "Old U", U3), ("u_swgo_drawn", "New U (drawn trigger)", U3)]
old_exp = json.load(open(os.path.join(D, "validate_full.json")))["scores"]
new_exp = chunks(f"{PRE}_exp")
if new_exp:
    out += table("Expected-value inputs (noise-free mean signal)", old_exp, new_exp, CONDS, COLS_EXP)
old_cnt, n_old = seed_mean("validate_count_s*_*.json")
new_cnt, n_new = seed_mean(f"{PRE}_count_s*_*.json")
if old_cnt and new_cnt:
    out += table(f"Drawn particle counts (mean of {n_old} / {n_new} seeds)", old_cnt, new_cnt, CONDS, COLS_CNT)
rob = [(p, chunks(f"robust_old_p{p}"), chunks(f"{ROB}_p{p}")) for p in (30, 60)]
for p, o, n in rob:
    if o and n:
        out += table(f"Missing detectors: {p} % removed at random (expected inputs, no background)", o, n,
                     [("All read out", "none", "0"), ("kNN gate", "knn", "0")], COLS_EXP[:4])
txt = "\n".join(out) + "\n"
OUT = os.path.join(D, {"dropout": "validation_dropout_tables.md", "count": "validation_countft_tables.md"}[VARIANT])
open(OUT, "w").write(txt)
print(txt)
print("[write]", OUT)
