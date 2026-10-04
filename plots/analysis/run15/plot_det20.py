"""All finished 20-detector runs of the detector-count scan: layouts (top) and held-out
kernel metrics (bottom). Grid start: current, two-tier lambda 0.3 and 0. Center start:
current and two-tier lambda 0.3 (metrics read from that job's log while its JSON is pending)."""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, sample_initial_layout_ne, project_to_mountain_ne  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
INK, GREY = "#1f2328", "#52514e"
COL = {"start": "#b9b7af", "current": "#2a78d6", "tt03": "#eb6834", "tt0": "#1baf7a"}
LAB = {"start": "start layout", "current": "current utility", "tt03": "two-tier λ = 0.3", "tt0": "two-tier λ = 0"}

grid = json.load(open(os.path.join(D, "detector_count_scan.json")))
lam0 = json.load(open(os.path.join(D, "det_count_lam0", "detector_count_scan.json")))
runs = {  # (start, variant) -> (metrics_start, metrics_opt, layout path)
    ("grid", "current"): (grid["n20_grid_current"]["start"], grid["n20_grid_current"]["optimized"],
                          os.path.join(D, "det_count_layout_n20_grid_current.pt")),
    ("grid", "tt03"): (grid["n20_grid_two_tier"]["start"], grid["n20_grid_two_tier"]["optimized"],
                       os.path.join(D, "det_count_layout_n20_grid_two_tier.pt")),
    ("grid", "tt0"): (lam0["n20_grid_two_tier"]["start"], lam0["n20_grid_two_tier"]["optimized"],
                      os.path.join(D, "det_count_lam0", "det_count_layout_n20_grid_two_tier.pt")),
}
# center start: parse the job log (JSON is written only when the whole job ends)
log = open(os.path.join(REPO, "slurm_logs", "slurm-49417342-det_count_center.out")).read()
pat = re.compile(r"\[n20_center_(\w+)\s*\] U_kernel\s+([\d.]+) ->\s+([\d.]+).*?angle\s+([\d.]+) ->\s+([\d.]+)\s+"
                 r"good ([\d.]+) -> ([\d.]+)")
for m in pat.finditer(log):
    u = {"current": "current", "two_tier": "tt03"}[m.group(1)]
    s = dict(U_kernel=float(m.group(2)), ang_med=float(m.group(4)), frac_good_5=float(m.group(6)))
    o = dict(U_kernel=float(m.group(3)), ang_med=float(m.group(5)), frac_good_5=float(m.group(7)))
    runs[("center", u)] = (s, o, os.path.join(D, "det_count_center", f"det_count_layout_n20_center_{m.group(1)}.pt"))

# job 49417342 was requeued and its log rewritten; its first attempt's two-tier line (read
# before the rewrite) is kept here so the figure stays reproducible
if ("center", "tt03") not in runs:
    runs[("center", "tt03")] = (dict(U_kernel=5.80, ang_med=41.87, frac_good_5=0.001),
                                dict(U_kernel=19.21, ang_med=10.30, frac_good_5=0.106),
                                os.path.join(D, "det_count_center", "det_count_layout_n20_center_two_tier.pt"))

# center start, two-tier lambda 0 (job 49420642): its JSON if finished, else its log
cl0 = os.path.join(D, "det_count_center_lam0")
if os.path.exists(os.path.join(cl0, "detector_count_scan.json")):
    r = json.load(open(os.path.join(cl0, "detector_count_scan.json")))["n20_center_two_tier"]
    runs[("center", "tt0")] = (r["start"], r["optimized"], os.path.join(cl0, "det_count_layout_n20_center_two_tier.pt"))
else:
    lp = os.path.join(REPO, "slurm_logs", "slurm-49421767-det20_center_lam0.out")
    m = pat.search(open(lp).read()) if os.path.exists(lp) else None
    if m:
        s_ = dict(U_kernel=float(m.group(2)), ang_med=float(m.group(4)), frac_good_5=float(m.group(6)))
        o_ = dict(U_kernel=float(m.group(3)), ang_med=float(m.group(5)), frac_good_5=float(m.group(7)))
        runs[("center", "tt0")] = (s_, o_, os.path.join(cl0, "det_count_layout_n20_center_two_tier.pt"))

mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                            layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(mountain.centroids_ENU)
starts = {}
for scheme in ("grid", "center"):
    torch.manual_seed(0)
    e0, n0 = sample_initial_layout_ne(mountain, n_units=20, scheme=scheme)
    starts[scheme] = project_to_mountain_ne(mountain, torch.as_tensor(e0).float(), torch.as_tensor(n0).float())

keys = [k for k in [("grid", "current"), ("grid", "tt03"), ("grid", "tt0"), ("center", "current"), ("center", "tt03"),
                   ("center", "tt0")]
        if k in runs]
fig = plt.figure(figsize=(17, 8.2))
gs = fig.add_gridspec(2, len(keys), height_ratios=[1.25, 1], hspace=0.28, wspace=0.05)
for i, k in enumerate(keys):
    a = fig.add_subplot(gs[0, i])
    L = torch.load(runs[k][2], map_location="cpu")
    e0, n0 = starts[k[0]]
    a.scatter(mesh[:, 0], mesh[:, 1], s=3, color="#e8e6df", lw=0)
    a.scatter(e0.numpy(), n0.numpy(), s=30, facecolor="none", edgecolor=COL["start"], lw=1.0)
    a.scatter(L["x"].numpy(), L["y"].numpy(), s=34, color=COL[k[1]], lw=0)
    o = runs[k][1]
    a.set_title(f"{k[0]} start · {LAB[k[1]]}\ngood {o['frac_good_5']:.1%} · angle {o['ang_med']:.1f}°",
                fontsize=10, color=INK)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)

metrics = [("frac_good_5", "showers reconstructed well (5°, 0.2)", 100, "%"),
           ("ang_med", "median angle error [deg]", 1, "°"),
           ("U_kernel", "current utility on the kernel", 1, "")]
sub = gs[1, :].subgridspec(1, len(metrics), wspace=0.3)
groups = ["grid", "center"]
variants = ["start", "current", "tt03", "tt0"]
w = 0.2
for m, (key, title, scale, unit) in enumerate(metrics):
    a = fig.add_subplot(sub[0, m])
    for gi, g in enumerate(groups):
        present = [v for v in variants if v == "start" or (g, v) in runs]
        for vi, v in enumerate(present):
            val = (runs[(g, "current")][0] if v == "start" else runs[(g, v)][1])[key] * scale
            xx = gi + (vi - (len(present) - 1) / 2) * w
            a.bar(xx, val, width=w - 0.03, color=COL[v], label=LAB[v] if gi == 0 else None)
            a.text(xx, val, f"{val:.1f}{unit}", ha="center", va="bottom", fontsize=7.5, color=GREY)
    a.set_xticks(range(len(groups))); a.set_xticklabels([f"{g} start" for g in groups])
    a.set_title(title, fontsize=10, color=INK)
    a.grid(axis="y", alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    if m == 0:
        handles, labels = a.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("20 detectors (run 15): 3,000 Adam steps through the surrogate, judged on held-out kernel showers",
             fontsize=12, color=INK)
out = os.path.join(D, "det20_summary.png")
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(f"[write] {out}")
