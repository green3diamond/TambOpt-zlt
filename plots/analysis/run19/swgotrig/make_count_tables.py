"""Run 19 swgo_trigger validation with Poisson-drawn particle counts (score_knn.py --counting poisson), averaged over
count seeds: per seed the held-out chunks (validate_count_s<seed>_<start>.json) are merged as shower-weighted means,
then every metric is averaged over the seeds (std over seeds kept). Writes validate_count_mean.json and
validation_count_tables.md (red bold = best per column, ties at table precision) next to the scores.

Columns: within 1 / 2.5 / 5 deg (recon on counted inputs); Old U (two-tier); New U drawn = mean[trig_drawn (g_a + 0.3
(1 - g_a) q_a)], trig_drawn = >= 3 read-out detectors with a drawn particle (background muons count too); Trig = its
rate.

    python plots/analysis/run19/swgotrig/make_count_tables.py
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
COLS = [("notrig_1", "1°", lambda v: f"{100 * v:.1f}%"), ("notrig_2.5", "2.5°", lambda v: f"{100 * v:.1f}%"),
        ("notrig_5", "5°", lambda v: f"{100 * v:.1f}%"), ("u_two_tier", "Old U", lambda v: f"{v:.3f}"),
        ("u_swgo_drawn", "New U (drawn trigger)", lambda v: f"{v:.3f}"),
        ("triggered", "Trig (drawn)", lambda v: f"{100 * v:.1f}%")]
CONDS = [("No background, all read out", "none", "0"), ("No background, kNN gate", "knn", "0"),
         ("1 µs background, all read out", "none", "1000"), ("1 µs background, kNN gate", "knn", "1000"),
         ("10 µs background, all read out", "none", "10000"), ("10 µs background, kNN gate", "knn", "10000")]

files = glob.glob(os.path.join(D, "validate_count_s*_*.json"))
by_seed = {}
for f in files:
    seed, start = map(int, re.search(r"validate_count_s(\d+)_(\d+)\.json", f).groups())
    by_seed.setdefault(seed, {})[start] = json.load(open(f))["scores"]
seeds = sorted(s for s, ch in by_seed.items() if len(ch) == 5)                # complete seeds only
if not seeds:
    sys.exit(f"no complete seed yet ({ {s: len(c) for s, c in by_seed.items()} } chunks per seed)")


def merged(seed, k, rd, w):
    rows = [by_seed[seed][st][k][rd][w] for st in sorted(by_seed[seed])]
    n = np.array([r["n_showers"] for r in rows], float)
    return {m: float(np.dot([r[m] for r in rows], n) / n.sum()) for m, _, _ in COLS}


mean, std = {}, {}
for k, _ in LAYOUTS:
    for _, rd, w in CONDS:
        per = [merged(sd, k, rd, w) for sd in seeds]
        for m, _, _ in COLS:
            v = np.array([p[m] for p in per])
            mean.setdefault(k, {}).setdefault(rd, {}).setdefault(w, {})[m] = float(v.mean())
            std.setdefault(k, {}).setdefault(rd, {}).setdefault(w, {})[m] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
json.dump(dict(seeds=seeds, mean=mean, std=std), open(os.path.join(D, "validate_count_mean.json"), "w"), indent=1)

max_sd = {lab: max(std[k][rd][w][m] for k, _ in LAYOUTS for _, rd, w in CONDS) for m, lab, _ in COLS}
red = lambda v: f'<span style="color:red"><b>{v}</b></span>'                 # noqa: E731
out = [f"""## swgo_trigger validation with drawn particle counts — run 19, 21,060 held-out showers, mean of {len(seeds)} count seeds

Particle counts per species drawn from Poisson(μ) (2 m², ε_γ = 0.02) and fed to the run 19 recon (trained on noise-free
inputs, so accuracies are lower bounds). Trig (drawn) = share of showers with ≥ 3 read-out detectors that got a drawn
particle (background muons count too); New U (drawn trigger) = mean[trig · (g_a + 0.3(1−g_a)q_a)], g_a = soft(a < 5°),
q_a = 1/(1 + (a/20)²); Old U = two-tier. Optimised layouts: swgo_trigger Stage 4, 2,048 showers per step, λ = 0.3, lr 10.
Seeds {seeds}; largest seed-to-seed std: """ + ", ".join(
    f"{lab} {100 * s:.2f} pts" if lab in ("1°", "2.5°", "5°", "Trig (drawn)") else f"{lab} {s:.4f}"
    for lab, s in max_sd.items()) + ". Red bold = best in the column (ties marked)."]
for title, rd, w in CONDS:
    rows = [(name, [f(mean[k][rd][w][m]) for m, _, f in COLS]) for k, name in LAYOUTS]
    best = [max(float(r[1][i].rstrip("%")) for r in rows) for i in range(len(COLS))]
    out.append(f"\n### {title}\n| Layout | " + " | ".join(lab for _, lab, _ in COLS) + " |\n|" + "---|" * (len(COLS) + 1))
    for name, vals in rows:
        out.append("| " + name + " | " + " | ".join(red(v) if float(v.rstrip("%")) == best[i] else v
                                                    for i, v in enumerate(vals)) + " |")
txt = "\n".join(out) + "\n"
open(os.path.join(D, "validation_count_tables.md"), "w").write(txt)
print(txt)
print("[write]", os.path.join(D, "validation_count_tables.md"), os.path.join(D, "validate_count_mean.json"))
