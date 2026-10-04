"""Schematic: how a recorded shower point is matched to detectors, now (stretched along global East)
vs updated (stretched along the shower's own axis). C8 records particles only on planes 500 m apart
along the axis; between planes the kernel stretches each point into a +-500 m segment with a Gaussian
cross-section. Top row: side view (East-Up) of a shower heading West and climbing 30 deg. Bottom row:
top view (East-North) of a horizontal shower heading 40 deg off West. Widths exaggerated.

    python kernel_schematic.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
INK, GREY, AXIS, PT = "#1f2328", "#8a887f", "#2a78d6", "#d9480f"
NOW, NEW = "#eb6834", "#2a78d6"
OK, BAD = "#1baf7a", "#c92a2a"
DX, H = 500.0, 70.0                       # plane spacing; drawn Gaussian half-width (exaggerated)
rng = np.random.default_rng(3)


def setup(a, title, xl, yl):
    a.set_aspect("equal")
    a.set_title(title, fontsize=11.5, color=INK, loc="left")
    a.set_xlabel(xl, color=GREY)
    a.set_ylabel(yl, color=GREY)
    a.tick_params(colors=GREY, labelsize=8)
    for sp in a.spines.values():
        sp.set_color("#d8d6cf")


def shower(a, p0, d, n_planes, spread, perp_label=None):
    """Axis, the recorded planes (perpendicular to the axis), points on them. Returns the points."""
    d = np.asarray(d) / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    L = (n_planes + 1.4) * DX
    a.annotate("", xy=p0 + L * d, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=1.6))
    a.plot(*p0, "o", color=AXIS, ms=7)
    pts = []
    for k in range(1, n_planes + 1):
        c = p0 + k * DX * d
        a.plot(*np.stack([c - 1.15 * spread * n, c + 1.15 * spread * n]).T, color=AXIS, lw=1, alpha=0.55)
        for o in rng.uniform(-spread, spread, 3) * np.array([1, 0.35, 0.8]):
            q = c + o * n
            pts.append(q)
    pts = np.array(pts)
    a.plot(pts[:, 0], pts[:, 1], "o", color=PT, ms=5, zorder=5)
    return pts, d


def stretch(a, pts, direc, color):
    ang = math.degrees(math.atan2(direc[1], direc[0]))
    for q in pts:
        a.add_patch(Ellipse(q, 2 * DX, 2 * H, angle=ang, fc=color, ec=color, alpha=0.18, lw=0.8))


def detector(a, xy, ok, label, dxy=(25, 25)):
    a.plot(*xy, marker="s", ms=10, mfc="white", mec=OK if ok else BAD, mew=2.2, zorder=7)
    a.annotate(label, xy, xytext=dxy, textcoords="offset points", fontsize=9, color=OK if ok else BAD,
               fontweight="bold", arrowprops=dict(arrowstyle="-", color=OK if ok else BAD, lw=0.8))


fig, ax = plt.subplots(2, 3, figsize=(20, 10.6))

# ---- top row: side view, heading West and climbing 30 deg --------------------------------------------
# C8 keeps a point's horizontal position (East from x) but not its height: placement puts every point at
# the axis height for its plane (Up = decay Up + depth x d_Up). So in this view the placed points of one
# plane lie on a horizontal line through the axis, not along the tilted plane.
el = math.radians(30)
d_side = np.array([-math.cos(el), math.sin(el)])
n_side = np.array([math.sin(el), math.cos(el)])                 # in-plane direction (side view)
p0 = np.array([1200.0, 0.0])
offs = (-110.0, 0.0, 90.0)                                       # in-plane offsets of three recorded particles
cols = (("1. What C8 records", None), ("2. Kernel now: each point stretched ±500 m along East", NOW),
        ("3. Kernel updated: stretched ±500 m along the shower axis", NEW))
for c, (ttl, col) in enumerate(cols):
    a = ax[0, c]
    L = 5.4 * DX
    a.annotate("", xy=p0 + L * d_side, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=1.6))
    a.plot(*p0, "o", color=AXIS, ms=7)
    true_pts, placed = [], []
    for k in range(1, 5):
        cc = p0 + k * DX * d_side
        a.plot(*np.stack([cc - 150 * n_side, cc + 150 * n_side]).T, color=AXIS, lw=1, alpha=0.55)
        for o in offs:
            q = cc + o * n_side
            true_pts.append(q)
            placed.append(np.array([q[0], cc[1]]))               # East kept, Up from the axis
    true_pts, placed = np.array(true_pts), np.array(placed)
    P = placed[4]                                                # the on-axis point of plane 2
    if c == 0:
        a.plot(true_pts[:, 0], true_pts[:, 1], "o", color=PT, ms=5, alpha=0.35, label="particle where it crossed the plane")
        a.plot(placed[:, 0], placed[:, 1], "o", color=PT, ms=5, label="as placed: East kept, height = axis height")
        a.legend(loc="upper right", fontsize=9, frameon=False)
        a.text(0.03, 0.04, "planes perpendicular to the axis, 500 m apart\nthe height within a plane is not recorded",
               transform=a.transAxes, fontsize=9.5, color=GREY)
    else:
        a.plot(placed[:, 0], placed[:, 1], "o", color=PT, ms=5, zorder=5)
        if col == NOW:
            for xe in np.arange(1500, -1600, -DX):
                a.axvline(xe, color="#c9c7bf", lw=0.8, zorder=0)
            stretch(a, placed, (1, 0), NOW)
        else:
            stretch(a, placed, d_side, NEW)
        on_path = P + 250 * d_side                               # 250 m further along the axis, between planes
        same_h = P + np.array([-250.0, 0.0])                     # 250 m further West at the same height
        detector(a, on_path, col == NEW, "on the particles' path:\n" + ("matched" if col == NEW else "missed"),
                 dxy=(-170, 45))
        detector(a, same_h, col == NOW, "same height, off the path:\n" +
                 ("matched (wrong)" if col == NOW else "not matched"), dxy=(-30, -75))
    a.set_xlim(-1700, 1500)
    a.set_ylim(-250, 1500)
    setup(a, ttl if c else ttl + "  (side view, climbing 30°)", "East [m]", "Up [m]")

# ---- bottom row: top view, horizontal, heading 40 deg off West (toward North-West) --------------------
hd = math.radians(180 - 40)
d_top = np.array([math.cos(hd), math.sin(hd)])
q0 = np.array([1000.0, -1300.0])
for c, (ttl, col) in enumerate(cols):
    a = ax[1, c]
    pts, dd = shower(a, q0, d_top, 4, 260)
    if col == NOW:
        for xe in np.arange(1500, -2000, -DX):
            a.axvline(xe, color="#c9c7bf", lw=0.8, zorder=0)
        stretch(a, pts, (1, 0), NOW)
        a.text(0.03, 0.04, "grey: the kernel's 500 m East slabs\neach point smeared along East,\n"
               "across the shower's own layers", transform=a.transAxes, fontsize=9.5, color=GREY)
    elif col == NEW:
        stretch(a, pts, dd, NEW)
        a.text(0.03, 0.04, "each point smeared along the axis,\nbetween its own layers",
               transform=a.transAxes, fontsize=9.5, color=GREY)
    else:
        a.text(0.03, 0.04, "planes perpendicular to the heading, 500 m apart\nsideways spread within a plane: recorded",
               transform=a.transAxes, fontsize=9.5, color=GREY)
    a.set_xlim(-2100, 1400)
    a.set_ylim(-1600, 1500)
    setup(a, ttl if c else ttl + "  (top view, 40° off West)", "East [m]", "North [m]")

fig.suptitle("How a recorded shower point is matched to detectors. C8 records particles only on planes 500 m apart along the "
             "shower axis;\nbetween planes the kernel stretches each point into a ±500 m segment with a Gaussian cross-section "
             "(width exaggerated here). Positions are the same in both; only the stretch direction changes.",
             fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.9))
out = os.path.join(D, "kernel_schematic.png")
fig.savefig(out, dpi=120, facecolor="white")
print(f"[write] {out}")
