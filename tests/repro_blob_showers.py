"""Standalone reproduction: AllShowers muon model emits rare degenerate showers.

Needs only numpy + matplotlib. No TambOpt, no showerdata, no GPU, no checkpoints.
The three showers in blob_showers.npz were taken straight out of the production
corpus (see README.md for provenance).

    python repro_blob_showers.py                # prints the diagnostic table
    python repro_blob_showers.py --plot out.png # + the 3D comparison figure
"""
import argparse
import numpy as np

# The 1.1 MB shower pack lives beside the run outputs, not in git -- it is data.
# Copy it next to this script and pass --npz blob_showers.npz if you are off-cluster.
DEFAULT_NPZ = ("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/"
               "detector_optimization_v6/tests/blob_showers.npz")

N_PLANES = 24                      # CORSIKA observation planes, 500 m apart
TAGS = ("good_rod", "blob_at_cap", "blob_below_cap")
CAP_MUON = 25088                   # the muon model's point cap


def stats(points, e_prim):
    """Everything that distinguishes a good shower from a degenerate one."""
    real = points[points[:, 3] > 0]
    e, lay = real[:, 3], np.clip(np.rint(real[:, 2]).astype(int), 0, N_PLANES - 1)
    plane_e = np.bincount(lay, weights=e, minlength=N_PLANES)
    srt = np.sort(e)[::-1]
    return dict(
        n_points=len(real),
        e_prim=float(e_prim),
        median_point=float(np.median(e)),
        total=float(e.sum()),
        total_over_prim=float(e.sum() / e_prim),
        max_plane_over_prim=float(plane_e.max() / e_prim),
        top1_share=float(srt[0] / e.sum()),
        top100_share=float(srt[:100].sum() / e.sum()),
        x_span=float(real[:, 0].max() - real[:, 0].min()),
        y_span=float(real[:, 1].max() - real[:, 1].min()),
        at_cap=len(real) >= CAP_MUON,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=DEFAULT_NPZ,
                    help="shower pack; defaults to the shared copy on holylfs05, "
                         "override with a local path if you were sent the file")
    ap.add_argument("--plot", default=None, help="write the 3D comparison here")
    args = ap.parse_args()

    z = np.load(args.npz)
    rows = {t: stats(z[f"{t}/points"], z[f"{t}/e_prim"]) for t in TAGS}

    w = 22
    print(f"{'':<22}" + "".join(f"{t:>{w}}" for t in TAGS))
    for k in ("n_points", "at_cap", "e_prim", "median_point", "total",
              "total_over_prim", "max_plane_over_prim", "top1_share",
              "top100_share", "x_span", "y_span"):
        cells = ""
        for t in TAGS:
            v = rows[t][k]
            cells += f"{v:>{w}}" if isinstance(v, (bool, np.bool_)) else \
                     f"{v:>{w},}" if isinstance(v, int) else f"{v:>{w}.4g}"
        print(f"{k:<22}{cells}")

    print("""
WHAT TO LOOK AT
  median_point   a normal muon cluster carries ~3.3; the blobs carry 1e8 and 4e9.
                 Every point is inflated, uniformly -- this is a global scale
                 error on the shower, not one corrupt value.
  top1_share     a NORMAL shower is spiky: its top 100 points hold ~40% of the
                 energy. The blobs hold ~7%, i.e. MORE uniform than normal. If a
                 single number had gone bad, top1_share would be ~1.0, not 0.002.
  x_span/y_span  the blobs are spread over tens of km and sit far off-axis; the
                 good shower is a compact rod along its axis.
  at_cap         blob_at_cap hit the 25088 point cap (truncation, the known
                 "rod -> blob" mode). blob_below_cap did NOT -- so truncation
                 cannot be the whole explanation, and the anti-clip re-roll
                 could not have prevented it.

  All values are finite -- np.isfinite is True everywhere -- so a non-finite
  guard does not catch these.""")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(5.2 * len(TAGS), 5.4))
        for k, t in enumerate(TAGS):
            p = z[f"{t}/points"]
            p = p[p[:, 3] > 0][::3]
            ax = fig.add_subplot(1, len(TAGS), k + 1, projection="3d")
            e = p[:, 3]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=e, s=2 + 14 * (e / e.max()),
                       cmap="inferno", norm=LogNorm(vmin=max(e[e > 0].min(), 1e-12),
                                                    vmax=e.max()),
                       alpha=0.55, edgecolors="none")
            ax.set_xlabel("x [m]", fontsize=8); ax.set_ylabel("y [m]", fontsize=8)
            ax.set_zlabel("layer", fontsize=8); ax.set_zlim(-0.5, N_PLANES - 0.5)
            ax.tick_params(labelsize=6); ax.view_init(elev=18, azim=-60)
            r = rows[t]
            ax.set_title(f"{t}  (row {int(z[f'{t}/row'])})\n"
                         f"n={r['n_points']:,}   median point {r['median_point']:.3g}",
                         fontsize=9)
        fig.suptitle("AllShowers muon model — one good shower, two degenerate "
                     "(colour = cluster energy, log, per panel)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(args.plot, dpi=140, bbox_inches="tight")
        print(f"\n[save] {args.plot}")


if __name__ == "__main__":
    main()
