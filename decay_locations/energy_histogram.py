"""Histogram of tau_wholesky training-sample energies, 100 log-spaced bins.

Reads TAU_WHOLESKY_PATH's `energy` dataset (GeV) and plots a histogram of
sample counts per energy bin, over the full range and zoomed to 1e5-1e7 GeV.
Energies span ~1e1-1e9 GeV (8 decades) overall, so bins are log-spaced.
"""
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H5_PATH = os.path.join(os.path.dirname(__file__), "tau_wholesky_n750k.h5")
N_BINS = 100
ZOOM_LO_GEV, ZOOM_HI_GEV = 1e5, 1e7

with h5py.File(H5_PATH, "r") as f:
    energy = f["energy"][:]

log_e = np.log10(energy)
bin_edges = np.linspace(log_e.min(), log_e.max(), N_BINS + 1)
counts, _ = np.histogram(log_e, bins=bin_edges)
bin_edges_gev = 10 ** bin_edges

out_dir = os.path.dirname(__file__)

fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))

for ax, title in ((ax_all, "all data"),
                   (ax_zoom, f"{ZOOM_LO_GEV:.0e}-{ZOOM_HI_GEV:.0e} GeV")):
    ax.bar(bin_edges_gev[:-1], counts, width=np.diff(bin_edges_gev),
           align="edge", color="#4C78A8", edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel("tau energy [GeV]")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")

ax_zoom.set_xlim(ZOOM_LO_GEV, ZOOM_HI_GEV)
in_zoom = (bin_edges_gev[:-1] >= ZOOM_LO_GEV) & (bin_edges_gev[:-1] < ZOOM_HI_GEV)
ax_zoom.set_ylim(0, counts[in_zoom].max() * 1.05)

fig.suptitle(f"tau_wholesky training samples per energy bin "
             f"({N_BINS} log-spaced bins, n={energy.size:,})")
fig.tight_layout()
png_path = os.path.join(out_dir, "energy_histogram.png")
fig.savefig(png_path, dpi=130)

print(f"n samples   : {energy.size}")
print(f"energy range: {energy.min():.3g} - {energy.max():.3g} GeV")
print(f"[save] {png_path}")
