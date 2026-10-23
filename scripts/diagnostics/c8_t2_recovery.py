"""Can the second transverse coordinate t2 be recovered from what the AllShowers cells keep (x, y, plane)?

CORSIKA 8 hit files have x, y, z per particle, so the true t2 is known. Frame per shower (as
modules/data/dataset_builder.shower_frames): n = propagation direction (event CSV), e1 = horizontal unit vector across
the heading, e2 = n x e1. Each plane is perpendicular to n, so a particle's offset from the axis crossing c is
d = t1 e1 + t2 e2. e1 has no component along the horizontal heading h, so
    d_xy . h = t2 (e2 . h) = +- t2 n_z    ->    t2_rec = (d_xy . h) / (e2 . h)
which needs only x, y and the axis crossing's horizontal position (in TambOpt: decay + (l + 1) * 500 m * n_h).
Here c = the plane's energy-weighted centre. Two versions: exact x, y, and x, y snapped to the AllShowers cell
centres (5 m e/gamma, 10 m mu; postprocess_showers.py). Real particles only.

    python scripts/diagnostics/c8_t2_recovery.py
"""
import csv
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

SRC = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training2/pdg_-211"
OUT = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/c8_calibration/c8_t2_recovery.png"
CELL = {11: 5.0, 22: 5.0, 13: 10.0}
rng = np.random.default_rng(0)
rows = []
for f in sorted(glob.glob(f"{SRC}/energy_*/*/*_hits.parquet")):
    ev = next(csv.DictReader(open(f.replace("_hits.parquet", "_event.csv"))))
    n = np.array([float(ev[k]) for k in ("direction_x", "direction_y", "direction_z")]); n /= np.linalg.norm(n)
    e1 = np.array([-n[1], n[0], 0.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    h = np.array([n[0], n[1], 0.0]); h /= np.linalg.norm(h)
    t = pq.read_table(f, columns=["x", "y", "z", "pdg", "kinetic_energy", "plane_index"])
    P = np.stack([t.column(c).to_numpy().astype(np.float64) for c in ("x", "y", "z")], 1)
    pdg, k, pl = np.abs(t.column("pdg").to_numpy()), t.column("kinetic_energy").to_numpy().astype(np.float64), t.column("plane_index").to_numpy()
    cs = np.vectorize(CELL.get)(pdg).astype(np.float64)
    t2, rec, recq = np.empty(len(k)), np.empty(len(k)), np.empty(len(k))
    for q in np.unique(pl):
        m = pl == q
        c = np.average(P[m], axis=0, weights=np.maximum(k[m], 1e-12))
        d = P[m] - c
        t2[m] = d @ e2
        rec[m] = (d[:, :2] @ h[:2]) / (e2 @ h)
        xq = (np.floor(P[m, :2] / cs[m, None]) + 0.5) * cs[m, None]                      # AllShowers cell centres
        recq[m] = ((xq - c[:2]) @ h[:2]) / (e2 @ h)
    sel = rng.choice(len(k), size=min(200000, len(k)), replace=False)
    zen = float(ev["incident_zenith"])
    rows.append(dict(f=f, zen=zen, nz=float(n[2]), t2=t2[sel], rec=rec[sel], recq=recq[sel],
                     rms_t2=float(np.sqrt(np.mean(t2 ** 2))), err=float(np.sqrt(np.mean((rec - t2) ** 2))),
                     errq=float(np.sqrt(np.mean((recq - t2) ** 2))),
                     errq_med=float(np.median(np.abs(recq - t2)))))
    r = rows[-1]
    print(f"zenith {zen:6.2f}  n_z {n[2]:+.3f}  rms t2 {r['rms_t2']:7.1f} m  rms error: exact x,y {r['err']:.3g} m, "
          f"cells {r['errq']:.3g} m (median |err| {r['errq_med']:.3g} m)", flush=True)

rows.sort(key=lambda r: r["zen"])
show = [rows[0], min(rows, key=lambda r: abs(r["zen"] - 75)), min(rows, key=lambda r: abs(r["zen"] - 85)), min(rows, key=lambda r: abs(r["zen"] - 90))]
fig, ax = plt.subplots(1, 5, figsize=(27, 5.6))
for a, r in zip(ax, show):
    L = np.percentile(np.abs(r["t2"]), 99)
    b = np.linspace(-L, L, 120)
    a.hist2d(r["t2"], np.clip(r["recq"], -L, L), bins=[b, b], cmap="Blues", norm=matplotlib.colors.LogNorm())
    a.plot([-L, L], [-L, L], "r--", lw=1)
    a.set(xlabel="true t2 [m]", ylabel="t2 from x, y, plane (cell centres) [m]", aspect="equal",
          title=f"zenith {r['zen']:.1f}° (n_z {r['nz']:+.3f})\nrms t2 {r['rms_t2']:.0f} m, rms error {r['errq']:.0f} m")
z = np.array([r["zen"] for r in rows])
ax[4].semilogy(z, [r["errq"] for r in rows], "o", color="#c0392b", label="rms error, AllShowers cell centres")
ax[4].semilogy(z, [max(r["err"], 1e-3) for r in rows], "s", color="#2a78d6", label="rms error, exact x, y")
ax[4].semilogy(z, [r["rms_t2"] for r in rows], "k^", label="rms of true t2 (the spread to recover)")
ax[4].set(xlabel="zenith [°]", ylabel="[m]", title="Recovery error vs zenith, all 16 showers"); ax[4].grid(alpha=0.25, lw=0.5)
ax[4].legend(fontsize=9, frameon=False)
fig.suptitle("Recovering the dropped transverse coordinate t2 from x, y and the plane number · CORSIKA 8 π⁻ showers, real particles", fontsize=14)
fig.tight_layout(); fig.savefig(OUT, dpi=120, facecolor="white")
print("[write]", OUT)
