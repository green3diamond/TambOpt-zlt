"""Paper 2: optimization with CORSIKA 8 showers and the analytical reconstruction (Malata).

Adam at lr 1 (scripts/04_optimize_lbfgs_kernel.py --pool c8 --recon-type analytic, SWGO trigger utility at 1 deg,
512 showers per epoch) from the uniform grid and from the centre; both stopped by the 3 h limit after the epoch-800
checkpoint. Panels per start: the start layout and the layout at the last checkpoint, with each detector's move.
Titles: utility on the scoring set (1,024 showers) and the move distances.
-> <out>/figures/results_2/c8ana_lr.pdf
"""
import argparse, os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/20_malata_showerframe"
SITES = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings/sites"
RUN = V6 + "/test_v6_run_04_optimize_lbfgs_kernel_c8ana_trig1_lr1_{}/opt_state.pt"
RUNS = [("uniform grid", "ugrid", f"{V6}/diagnostics/paper1/ugrid.pt"),
        ("centre", "center", f"{V6}/diagnostics/paper1/center.pt")]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--res", default=None)         # accepted for make_paper2.sh; unused
args = ap.parse_args()
FIG = os.path.join(args.out, "figures", "results_2")

tris = np.load(f"{SITES}/malata.npz")["tris"]
cE, cN = tris[:, :, 0].mean(1), tris[:, :, 1].mean(1)
runs = [(t, torch.load(RUN.format(s), map_location="cpu", weights_only=False), torch.load(p, map_location="cpu"))
        for t, s, p in RUNS if os.path.exists(RUN.format(s)) and os.path.exists(p)]
if runs:
    fig, ax = plt.subplots(1, 2 * len(runs), figsize=(10, 3.0), constrained_layout=True)
    for i, (t, st, g) in enumerate(runs):
        gx, gy = np.asarray(g["x"]), np.asarray(g["y"])
        fx, fy = st["xy"]["x"].detach().numpy(), st["xy"]["y"].detach().numpy()
        d = np.hypot(fx - gx, fy - gy)
        u0, u1 = (float(st["score_log"][k]["U_score"]) for k in (0, -1))
        a0, a1 = ax[2 * i], ax[2 * i + 1]
        for a in (a0, a1):
            a.scatter(cE, cN, s=3, c="0.85", lw=0, rasterized=True)
        a0.scatter(gx, gy, s=6, c="0.15", lw=0)
        a0.set_title(f"start: {t}\nU = {u0:.2f}", fontsize=7)
        a1.scatter(gx, gy, s=6, facecolors="none", edgecolors="0.45", lw=0.4)
        for j in range(len(gx)):
            a1.plot([gx[j], fx[j]], [gy[j], fy[j]], c="tab:red", lw=0.4)
        a1.scatter(fx, fy, s=6, c="tab:blue", lw=0)
        a1.set_title(f"{t}, epoch {st['epoch']}: U = {u1:.2f} (best {st['best_u']:.2f})\n"
                     f"moved: median {np.median(d):.0f} m, max {d.max():.0f} m", fontsize=7)
    for a in ax:
        a.set_aspect("equal"); a.tick_params(labelsize=6); a.set_xlabel("East [m]", fontsize=7)
    ax[0].set_ylabel("North [m]", fontsize=7)
    fig.savefig(f"{FIG}/c8ana_lr.pdf", dpi=300)
    print("c8ana_lr.pdf:", ", ".join(t for t, *_ in runs))
