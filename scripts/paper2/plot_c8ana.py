"""Paper 2: step-size test of the optimization with CORSIKA 8 showers and the analytical reconstruction (Malata).

100 Adam epochs from the uniform grid at lr 1, 3 and 10 (scripts/04_optimize_lbfgs_kernel.py --pool c8 --recon-type
analytic, SWGO trigger utility at 1 deg, 512 showers per epoch); panels: the start and each run's layout at epoch 100,
with each detector's move. Titles: utility on the scoring set (1,024 showers) and the move distances.
-> <out>/figures/results_2/c8ana_lr.pdf
"""
import argparse, os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/20_malata_showerframe"
SITES = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings/sites"
RUN = V6 + "/test_v6_run_04_optimize_lbfgs_kernel_c8ana_trig1_ugrid_test100{}/opt_state.pt"
RUNS = [("lr 1", "_lr1"), ("lr 3", "_lr3"), ("lr 10", "")]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--res", default=None)         # accepted for make_paper2.sh; unused
args = ap.parse_args()
FIG = os.path.join(args.out, "figures", "results_2")

tris = np.load(f"{SITES}/malata.npz")["tris"]
cE, cN = tris[:, :, 0].mean(1), tris[:, :, 1].mean(1)
g = torch.load(f"{V6}/diagnostics/paper1/ugrid.pt", map_location="cpu", weights_only=False)
gx, gy = np.asarray(g["x"]), np.asarray(g["y"])
states = [(t, torch.load(RUN.format(s), map_location="cpu", weights_only=False)) for t, s in RUNS
          if os.path.exists(RUN.format(s))]
if states:
    fig, ax = plt.subplots(1, 1 + len(states), figsize=(10, 3.3), constrained_layout=True)
    ax[0].scatter(cE, cN, s=3, c="0.85", lw=0, rasterized=True)
    ax[0].scatter(gx, gy, s=6, c="0.15", lw=0)
    init_u = float(states[0][1]["score_log"][0]["U_score"])          # epoch 0: the start layout
    ax[0].set_title(f"start: uniform grid\nU = {init_u:.2f}", fontsize=7)
    for a, (t, st) in zip(ax[1:], states):
        fx, fy = st["xy"]["x"].detach().numpy(), st["xy"]["y"].detach().numpy()
        d = np.hypot(fx - gx, fy - gy)
        u = float(st["score_log"][-1]["U_score"])
        a.scatter(cE, cN, s=3, c="0.85", lw=0, rasterized=True)
        a.scatter(gx, gy, s=6, facecolors="none", edgecolors="0.45", lw=0.4)
        for i in range(len(gx)):
            a.plot([gx[i], fx[i]], [gy[i], fy[i]], c="tab:red", lw=0.4)
        a.scatter(fx, fy, s=6, c="tab:blue", lw=0)
        a.set_title(f"{t}, epoch 100: U = {u:.2f}\nmoved: median {np.median(d):.0f} m, max {d.max():.0f} m", fontsize=7)
    for a in ax:
        a.set_aspect("equal"); a.tick_params(labelsize=6); a.set_xlabel("East [m]", fontsize=7)
    ax[0].set_ylabel("North [m]", fontsize=7)
    fig.savefig(f"{FIG}/c8ana_lr.pdf", dpi=300)
    print("c8ana_lr.pdf:", ", ".join(t for t, _ in states))
