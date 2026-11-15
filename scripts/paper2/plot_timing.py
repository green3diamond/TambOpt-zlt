"""Paper 2: optimization with realistic detector arrival times (branch timing-model, TAMBOPT_TIME_MODEL=mean).

Runs: scripts/04_optimize_lbfgs_kernel.py --pool c8 --recon-type analytic, SWGO trigger utility at 1 deg, Adam lr 3,
5,120 CORSIKA 8 showers per step, 100 steps, from the uniform grid and from the centre. Gradient check:
scripts/paper2/grad_check.py (64 showers, uniform grid, fixed time draws).

  <out>/figures/results_2/timing_layouts.pdf  start and final layout per start (no arrows)
  <out>/figures/results_2/timing_opt.pdf      (a) scoring-set utility vs step, relative to the start;
                                              (b) change of U on the gradient's own 64 showers when every detector
                                                  moves delta along the sign of the gradient (Adam-like)
"""
import argparse, json, os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/20_malata_showerframe"
TR = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings"
RUN = V6 + "/test_v6_run_04_optimize_lbfgs_kernel_c8mean_ana_{}_t100_lr3_b5120/opt_state.pt"
RUNS = [("uniform grid", "ugrid", f"{V6}/diagnostics/paper1/ugrid.pt"),
        ("centre", "center", f"{V6}/diagnostics/paper1/center.pt")]
GC = {"mean": f"{TR}/timing/gradcheck2_mean.json", "edge": f"{TR}/timing/gradcheck2_edge.json"}

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--res", default=None)         # accepted for make_paper2.sh; unused
args = ap.parse_args()
FIG = os.path.join(args.out, "figures", "results_2")

tris = np.load(f"{TR}/sites/malata.npz")["tris"]
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
        a1.scatter(fx, fy, s=6, c="0.15", lw=0)
        a1.set_title(f"{t}, step {st['epoch']}: U = {u1:.2f} (best {st['best_u']:.2f})\n"
                     f"moved: median {np.median(d):.0f} m, max {d.max():.0f} m", fontsize=7)
    for a in ax:
        a.set_aspect("equal"); a.tick_params(labelsize=6); a.set_xlabel("East [m]", fontsize=7)
    ax[0].set_ylabel("North [m]", fontsize=7)
    fig.savefig(f"{FIG}/timing_layouts.pdf", dpi=300)
    print("timing_layouts.pdf:", ", ".join(t for t, *_ in runs))

    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.6), constrained_layout=True)
    for (t, st, _), c in zip(runs, ("C0", "C1")):
        sl = st["score_log"]; u0 = float(sl[0]["U_score"])
        a.plot([s["epoch"] for s in sl], [float(s["U_score"]) / u0 for s in sl], "o-", c=c, ms=3, label=f"{t} (start {u0:.2f})")
    a.axhline(1, c="0.6", lw=0.6, ls=":")
    a.set_xlabel("Adam step", fontsize=7); a.set_ylabel("U / U(start), scoring set", fontsize=7)
    a.set_title("(a) held-out showers (1,024)", fontsize=7); a.legend(fontsize=6, frameon=False)
    for (m, f), c in zip(GC.items(), ("C2", "C3")):
        if os.path.exists(f):
            gc = json.load(open(f)); rows = [r for r in gc["directional"] if r["kind"] == "sign"]
            b.plot([r["delta_m"] for r in rows], [100 * r["dU"] / gc["U"] for r in rows], "s-", c=c, ms=3,
                   label=f"{m} time model (U = {gc['U']:.2f})")
    b.axhline(0, c="0.6", lw=0.6, ls=":")
    b.set_xlabel(r"move per detector and axis $\delta$ [m]", fontsize=7); b.set_ylabel(r"$\Delta U / U$ [%]", fontsize=7)
    b.set_title("(b) the gradient's own showers (64), sign step", fontsize=7); b.legend(fontsize=6, frameon=False)
    for x in (a, b):
        x.tick_params(labelsize=6)
    fig.savefig(f"{FIG}/timing_opt.pdf")
    print("timing_opt.pdf")
