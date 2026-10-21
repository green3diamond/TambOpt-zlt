"""Schematic of the coordinates on a real CORSIKA 8 shower: global x (East), y (North), z (Up); the shower axis n;
the observation planes (perpendicular to n, every 500 m along it); and the transverse frame in each plane:
    e1 = horizontal unit vector across the heading       (t1 = offset along e1)
    e2 = n x e1, the other in-plane direction (~vertical for near-horizontal showers)   (t2 = offset along e2)
    r  = sqrt(t1^2 + t2^2) = true distance from the axis
Geometry and particles are real (shower 5184217_1, zenith 67 deg); plane squares span 2.5 x each plane's rms
transverse spread.

    python scripts/diagnostics/c8_frame_schematic.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

F = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training2/pdg_-211/energy_1.07e+05_to_1.15e+05/5184217_1/5184217_1_hits.parquet"
OUT = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/c8_calibration/c8_frame_schematic.png"
ev = next(csv.DictReader(open(F.replace("_hits.parquet", "_event.csv"))))
n = np.array([float(ev[k]) for k in ("direction_x", "direction_y", "direction_z")]); n /= np.linalg.norm(n)
e1 = np.array([-n[1], n[0], 0.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
t = pq.read_table(F, columns=["x", "y", "z", "pdg", "kinetic_energy", "plane_index"])
P = np.stack([t.column(c).to_numpy().astype(np.float64) for c in ("x", "y", "z")], 1) / 1000.0          # km
k, pl, pdg = t.column("kinetic_energy").to_numpy().astype(np.float64), t.column("plane_index").to_numpy(), np.abs(t.column("pdg").to_numpy())
PLANES = (8, 13, 18, 23)
cen = {q: np.average(P[pl == q], axis=0, weights=np.maximum(k[pl == q], 1e-12)) for q in PLANES}
rng = np.random.default_rng(0)

fig = plt.figure(figsize=(22, 9.5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
axis_pts = np.array([cen[q] for q in PLANES])
o = cen[PLANES[0]] - 2.5 * n
ax.plot(*np.stack([o, cen[PLANES[-1]] + 1.5 * n]).T, color="k", lw=2.5)
ax.text(*(cen[PLANES[-1]] + 1.6 * n), "shower axis n", fontsize=12)
for q in PLANES:
    m = np.flatnonzero(pl == q)
    d = P[m] - cen[q]
    L = 2.5 * np.sqrt(np.mean((d @ e1) ** 2 + (d @ e2) ** 2))
    sq = np.array([cen[q] + a * L * e1 + b * L * e2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))])
    ax.plot(*sq.T, color="#555", lw=1)
    j = rng.choice(m, size=min(3000, len(m)), replace=False)
    ax.scatter(*P[j].T, s=0.5, color="#2a78d6", alpha=0.3)
    ax.text(*(sq[2] + 0.1 * e2), f"plane {q} (s = {(q + 1) * 0.5:g} km)", fontsize=9, color="#555")
c = cen[PLANES[1]]; S = 1.2
for v, lab, col in ((e1, "e1 (horizontal, across heading)", "#c0392b"), (e2, "e2 = n × e1", "#1baf7a"), (n * 0.8, "n", "k")):
    ax.quiver(*c, *(S * v), color=col, lw=2.5, arrow_length_ratio=0.12)
    ax.text(*(c + S * v * 1.08), lab, color=col, fontsize=11)
g0 = axis_pts.min(0) - np.array([1.5, 1.5, 0.5])
for v, lab in ((np.array([1, 0, 0]), "x = East"), (np.array([0, 1, 0]), "y = North"), (np.array([0, 0, 1]), "z = Up")):
    ax.quiver(*g0, *(1.5 * v), color="#7b8794", lw=2, arrow_length_ratio=0.15); ax.text(*(g0 + 1.7 * v), lab, color="#7b8794", fontsize=11)
ax.set(xlabel="x East [km]", ylabel="y North [km]", zlabel="z Up [km]")
ax.set_title(f"Real shower 5184217_1, zenith {float(ev['incident_zenith']):.1f}°: planes perpendicular to the axis, every 500 m", fontsize=13)
ax.view_init(elev=18, azim=-60)
mid = axis_pts.mean(0); R = np.ptp(axis_pts, 0).max() / 2 + 2
ax.set_xlim(mid[0] - R, mid[0] + R); ax.set_ylim(mid[1] - R, mid[1] + R); ax.set_zlim(mid[2] - R, mid[2] + R)

b = fig.add_subplot(1, 2, 2)
q = PLANES[1]; m = np.flatnonzero(pl == q); d = (P[m] - cen[q]) * 1000
t1, t2 = d @ e1, d @ e2
j = rng.choice(len(m), size=min(40000, len(m)), replace=False)
for pid, col, lab in ((22, "#e8a33a", "γ"), (11, "#2a78d6", "e±"), (13, "#c0392b", "μ±")):
    jj = j[pdg[m][j] == pid]; b.scatter(t1[jj], t2[jj], s=1, color=col, alpha=0.4, lw=0, label=lab)
rr = np.hypot(t1, t2); ph = np.arctan2(t2, t1)
p = np.argmin(np.abs(rr - 350) + 400 * np.abs(ph - np.radians(55)))
b.plot([0, t1[p]], [0, t2[p]], "k-", lw=1.5); b.plot([0, t1[p]], [0, 0], color="#c0392b", lw=3); b.plot([t1[p], t1[p]], [0, t2[p]], color="#1baf7a", lw=3)
b.plot(t1[p], t2[p], "k*", ms=18)
b.text(t1[p] / 2, -35, f"t1 = {t1[p]:.0f} m", color="#c0392b", fontsize=12, ha="center")
b.text(t1[p] + 15, t2[p] / 2, f"t2 = {t2[p]:.0f} m", color="#1baf7a", fontsize=12)
b.text(t1[p] / 2 - 120, t2[p] / 2 + 20, f"r = √(t1² + t2²) = {rr[p]:.0f} m", fontsize=12)
th = np.linspace(0, 2 * np.pi, 300)
b.plot(rr[p] * np.cos(th), rr[p] * np.sin(th), "k-", lw=1, alpha=0.6)
b.plot(abs(t1[p]) * np.cos(th), abs(t1[p]) * np.sin(th), "k--", lw=1, alpha=0.6)
b.text(0, -rr[p] - 45, "solid circle: true r (option a) · dashed: ring of radius |t1| the kernel uses (option b)", ha="center", fontsize=10)
b.axhline(0, color="#999", lw=0.8); b.axvline(0, color="#999", lw=0.8)
Lb = 700; b.set(xlim=(-Lb, Lb), ylim=(-Lb, Lb), aspect="equal", xlabel="t1 along e1: horizontal, across the heading [m]",
                ylabel="t2 along e2 [m]  (AllShowers cells keep x, y and the plane number; z is dropped)")
b.set_title(f"Inside plane {q}: real particles in the transverse frame (t1, t2), axis at the centre", fontsize=13)
b.legend(markerscale=8, fontsize=11, loc="upper right")
fig.tight_layout(); fig.savefig(OUT, dpi=120, facecolor="white")
print("[write]", OUT)
