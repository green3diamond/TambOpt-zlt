"""Run 19 ghost-event tables from ghost_scores.json (score_ghost.py): per readout and window, each layout's false-trigger
rate on background-only windows next to the shower trigger rate in the same window (validate_count_mean.json, drawn
counts, mean of the count seeds), and what each recon reports for the triggered ghosts (median zenith with the
10-90 % range, median log10 E, share inside the training energy range 5-7). Writes validation_ghost_tables.md.

    python plots/analysis/run19/swgotrig/make_ghost_tables.py
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
           ("opt_a2_step_grid_5000", "Opt. step 2 m², grid start (5,000 steps)"),
           ("opt_a2_step_grid_8000", "Opt. step 2 m², grid start (8,000 steps)"),
           ("opt_a2_step_center", "Opt. step 2 m², centre start"), ("disk_r1500_n99", "Disk 1.5 km"),
           ("tri33disk_R1000_r25_n99", "Triplets on 1 km disk")]
RECON_NAME = {"_deepsets": "default", "_deepsets_dropout": "dropout ft", "_deepsets_count": "drawn-count ft"}
CONDS = [("1 µs, all read out", "none", "1000"), ("1 µs, kNN gate", "knn", "1000"),
         ("10 µs, all read out", "none", "10000"), ("10 µs, kNN gate", "knn", "10000")]

g = json.load(open(os.path.join(D, "ghost_scores.json")))
S = g["scores"]
shower = json.load(open(os.path.join(D, "validate_count_mean.json")))["mean"]
recons = list(next(iter(S.values()))["none"][f"{g['windows_ns'][0]:g}"]["recons"])
label = lambda r: next((v for k, v in sorted(RECON_NAME.items(), key=lambda kv: -len(kv[0])) if r.endswith(k)), r)  # noqa: E731
pct = lambda v: "–" if v != v else f"{100 * v:.1f}%"                          # noqa: E731

L = ["## Ghost events — background only, no shower (run 19)", "",
     f"{g['n_windows']:,} readout windows per layout with cosmic muons only (random KE per muon), window centred at "
     f"{g['t_centre_us']:.1f} µs, the same draws for every recon. Ghost trig = share of ghost windows with ≥ "
     f"{g['min_hit']} read-out detectors with E′ > 1, the same trigger as the drawn-count validation; Shower trig = "
     "share of held-out showers that trigger in the same window (drawn counts, background included). Per recon, over "
     "triggered ghosts: median predicted zenith [10–90 %] and median predicted log10 E; In E range = share predicted "
     "inside the training range log10 E 5–7, i.e. ghosts the recon passes off as showers. – = no ghost triggered."]
for title, rd, w in CONDS:
    if w not in next(iter(S.values()))[rd]:
        continue
    head = "| Layout | Ghost trig | Shower trig | " + " | ".join(
        f"{label(r)}: zenith | {label(r)}: log10 E | {label(r)}: in E range" for r in recons) + " |"
    L.append(f"\n### {title}\n\n{head}\n|" + " --- |" * (3 + 3 * len(recons)))
    for k, name in LAYOUTS:
        if k not in S:
            continue
        r = S[k][rd][w]
        cells = [pct(r["ghost_trig"]), pct(shower[k][rd][w]["triggered"]) if k in shower else "–"]
        for rc in recons:
            x = r["recons"][rc]
            z = x["zen_p10_50_90"]
            cells += ["–" if z[1] != z[1] else f"{z[1]:.0f}° [{z[0]:.0f}–{z[2]:.0f}]",
                      "–" if x["log10E_med"] != x["log10E_med"] else f"{x['log10E_med']:.2f}", pct(x["in_e_range"])]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
txt = "\n".join(L) + "\n"
open(os.path.join(D, "validation_ghost_tables.md"), "w").write(txt)
print(txt)
print("[write]", os.path.join(D, "validation_ghost_tables.md"))
