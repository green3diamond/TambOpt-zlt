"""Run 19 swgo_trigger validation tables (markdown) from validate_full.json: 6 conditions (no / 1 µs / 10 µs background x
all read out / kNN gate), 11 layouts (LAYOUTS in plot_validation_layouts.py), best value per column marked (ties at the
table precision all marked). Writes validation_tables.md (bold) and validation_tables_red.md (red bold, HTML spans)
next to the scores and prints the red version.

    python plots/analysis/run19/swgotrig/make_validation_tables.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402

D = os.path.join(C.RUN_LOCATION, "diagnostics", "swgotrig")
LAYOUTS = [("disk_r6000_n99", "Uniform disk 6 km"), ("grid_n99", "Grid"),
           ("tri33_r10_n99", "Triplets r = 10 m on grid"), ("tri33_r25_n99", "Triplets r = 25 m on grid"),
           ("opt_a2_grid", "Opt. Poisson 2 m², grid start"), ("opt_a2_center", "Opt. Poisson 2 m², centre start"),
           ("opt_a20_grid", "Opt. Poisson 20 m², grid start"),
           ("opt_a2_step_grid_8000", "Opt. step 2 m², grid start (8,000 steps)"),
           ("opt_a2_step_center", "Opt. step 2 m², centre start"), ("disk_r1500_n99", "Disk 1.5 km"),
           ("tri33disk_R1000_r25_n99", "Triplets on 1 km disk")]
COLS = [("notrig_1", "1°", lambda v: f"{100 * v:.1f}%"), ("notrig_2.5", "2.5°", lambda v: f"{100 * v:.1f}%"),
        ("notrig_5", "5°", lambda v: f"{100 * v:.1f}%"), ("u_two_tier", "Old U", lambda v: f"{v:.3f}"),
        ("u_swgo", "New U (Poisson)", lambda v: f"{v:.3f}"), ("u_swgo_step", "New U (step)", lambda v: f"{v:.3f}"),
        ("trig_poisson", "Trig", lambda v: f"{100 * v:.1f}%")]
CONDS = [("No background, all read out", "none", "0"), ("No background, kNN gate", "knn", "0"),
         ("1 µs background, all read out", "none", "1000"), ("1 µs background, kNN gate", "knn", "1000"),
         ("10 µs background, all read out", "none", "10000"), ("10 µs background, kNN gate", "knn", "10000")]
HEAD = """## swgo_trigger validation — run 19, 21,060 held-out showers (whole plane), run 19 recon (trained without background)

Metrics: share of all showers within 1° / 2.5° / 5°; Old U = two-tier, mean[g + 0.3(1−g)q], g = soft(a < 5°)·soft(|Δlog10 E| < 0.2);
New U = mean[r_b · (g_a + 0.3(1−g_a)q_a)], angle only, r_b = P(≥ 3 detectors with ≥ 1 particle) (Poisson) or the notebook step;
Trig = Poisson trigger rate. kNN gate = a detector is read out only if ≥ 2 of {itself, its 2 nearest} have E > 10⁻³.
Optimised layouts: swgo_trigger Stage 4, 2,048 showers per step, λ = 0.3, lr 10 (5,000 steps; the step grid start 8,000).
{mark} = best in the column (ties marked)."""


def tables(mark):
    out = [HEAD.replace("{mark}", mark[1])]
    s = json.load(open(os.path.join(D, "validate_full.json")))["scores"]
    for title, rd, w in CONDS:
        rows = [(name, [f(s[k][rd][w][c]) for c, _, f in COLS]) for k, name in LAYOUTS]
        best = [max(float(r[1][i].rstrip("%")) for r in rows) for i in range(len(COLS))]
        out.append(f"\n### {title}\n| Layout | " + " | ".join(lab for _, lab, _ in COLS) + " |\n|"
                   + "---|" * (len(COLS) + 1))
        for name, vals in rows:
            cells = [mark[0](v) if float(v.rstrip("%")) == best[i] else v for i, v in enumerate(vals)]
            out.append("| " + name + " | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


bold = (lambda v: f"**{v}**", "Bold")
red = (lambda v: f'<span style="color:red"><b>{v}</b></span>', "Red bold")
for fname, mark in (("validation_tables.md", bold), ("validation_tables_red.md", red)):
    txt = tables(mark)
    open(os.path.join(D, fname), "w").write(txt)
    print("[write]", os.path.join(D, fname))
print(txt)
