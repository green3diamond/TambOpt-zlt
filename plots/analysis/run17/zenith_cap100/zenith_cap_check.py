"""Zenith cap test (run 17, 45° plane): do the issues found in ../array_size_single_shower/ persist
when showers are limited to zenith <= 100 deg, the AllShowers training range (run 17 allows 120)?
  1. axis tilt: the clouds' horizontal axis minus the labelled azimuth (energy-weighted principal axis
     of the raw East/North offsets), for West-going showers, by angle from due West and zenith band:
     C8 test showers (combined_electrons_test.h5) and run 17 generated held-out showers (3 species);
  2. reconstruction: the grid's per-shower angle / energy errors on 1,536 held-out kernel showers by
     zenith band, kernel widths uncapped and capped at 50 m (recon trained on 100% uncapped kernel
     responses, *_deepsets_kernel100);
  3. run 18 geometry (6 km plane, box scaled with it): azimuth spread and share of throws kept with the
     zenith cut at 100 vs 120.
Writes zenith_cap_check.{json,png} next to this file.

    python zenith_cap_check.py
"""
import importlib.util
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import showerdata
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"   # branch run17-bwcap50 (run 17 constants)
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
import modules.showers.kernel as KMOD  # noqa: E402
from modules.geometry import (load_tr_mountain, SurfaceUpMap, sample_initial_layout_ne,  # noqa: E402
                              project_to_mountain_ne)
from modules.surrogates import build_recon_from_ckpt  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

_spec = importlib.util.spec_from_file_location("dcs", os.path.join(REPO, "scripts/diagnostics/detector_count_scan.py"))
DCS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DCS)
GS = DCS.GS

D = os.path.dirname(os.path.abspath(__file__))
C8 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_electrons_test.h5"
RECON = C.RECON_FOLDER + "_deepsets_kernel100"
AZ_BINS = [0, 10, 20, 30, 45, 90]                      # |azimuth - due West| [deg]
ZE_BANDS = [(60, 80), (80, 90), (90, 100), (100, 110), (110, 120)]
INK, GREY = "#1f2328", "#52514e"
res = {}


# ---- 1. axis tilt ----------------------------------------------------------------------------------
def tilt(path, start, stop, chunk=1000):
    """(offset, |az - West|, zenith) per West-going shower in rows [start, stop)."""
    out = []
    for lo in range(start, stop, chunk):
        sh = showerdata.load(path, start=lo, stop=min(lo + chunk, stop))
        d = np.asarray(sh.directions, dtype=float)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pts = np.asarray(sh.points, dtype=np.float32)
        for i in range(len(d)):
            az = np.degrees(np.arctan2(d[i, 1], d[i, 0]))
            k = pts[i, :, 3] > 0
            if d[i, 0] >= 0 or k.sum() < 30 or np.hypot(d[i, 0], d[i, 1]) < np.sin(np.radians(20)):
                continue
            xy, w = pts[i, k, :2].astype(float), pts[i, k, 3].astype(float)
            v = np.linalg.eigh((xy * w[:, None]).T @ xy)[1][:, -1]
            v = v if v @ d[i, :2] > 0 else -v
            off = (np.degrees(np.arctan2(v[1], v[0])) - az + 180) % 360 - 180
            out.append((off, 180 - abs(az), np.degrees(np.arccos(d[i, 2]))))
    return np.array(out)


def tilt_table(name, t):
    rows = {}
    print(f"\n[tilt] {name}: {len(t)} West-going showers  (median offset deg, 16-84%, n)")
    for zlo, zhi, zname in ((60, 100, "zenith 60-100"), (100, 120.01, "zenith 100-120")):
        cells = []
        for a, b in zip(AZ_BINS[:-1], AZ_BINS[1:]):
            m = (t[:, 1] >= a) & (t[:, 1] < b) & (t[:, 2] >= zlo) & (t[:, 2] < zhi)
            if m.sum() >= 10:
                q = np.percentile(t[m, 0], [16, 50, 84])
                cells.append(dict(az=[a, b], n=int(m.sum()), p16=q[0], med=q[1], p84=q[2]))
        if cells:
            rows[zname] = cells
            print(f"  {zname}: " + "  ".join(f"[{c['az'][0]}-{c['az'][1]}] {c['med']:+.1f} ({c['p16']:+.0f}..{c['p84']:+.0f}, {c['n']})"
                                          for c in cells))
    return rows


res["tilt"] = {"C8 electrons (simulated)": tilt_table("C8 electrons", tilt(C8, 0, showerdata.get_file_length(C8)))}
n_pairs = showerdata.get_file_length(C.HELDOUT_SHOWER_CACHE_PATH) // len(C.SPECIES_NAMES)
for s, sp in enumerate(C.SPECIES_NAMES):
    res["tilt"][f"run 17 {sp}s (generated)"] = tilt_table(
        f"run 17 generated {sp}s", tilt(C.HELDOUT_SHOWER_CACHE_PATH, s * n_pairs, (s + 1) * n_pairs))

# ---- 2. grid reconstruction by zenith band ---------------------------------------------------------
dev = DCS.DEVICE
mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                            layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(dev)
ckpt = torch.load(os.path.join(RECON, "recon.pt"), map_location=dev, weights_only=False)
recon = build_recon_from_ckpt(ckpt, C.N_DETECTORS, dev).eval()
clouds, _, _, prim_k = load_events(1536, dev, mountain)
torch.manual_seed(0)
ge, gn = sample_initial_layout_ne(mountain, n_units=C.N_DETECTORS, scheme="grid")
ge, gn = project_to_mountain_ne(mountain, torch.as_tensor(ge).float(), torch.as_tensor(gn).float())
ge, gn = ge.float().to(dev), gn.float().to(dev)
res["recon"] = {}
for cap in (float("inf"), 50.0):
    tag = "uncapped" if math.isinf(cap) else f"cap {cap:g} m"
    KMOD.KERNEL_INTERP_MAX = cap                 # adaptive_bandwidths reads it per call
    kern = GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim_k)
    with torch.no_grad():
        counts, parts = GS.predict(ge, gn, prim_k, kern, recon)
    a, de = (x.cpu().numpy() for x in DCS.ang_err(parts))
    ze = np.degrees(parts[4].cpu().numpy())
    lit = (counts > 1.0).sum(1).cpu().numpy()
    rows = []
    for lo, hi in ZE_BANDS + [(60, 100), (100, 120)]:
        m = (ze >= lo) & (ze < hi + (0.01 if hi == 120 else 0))
        rows.append(dict(band=[lo, hi], n=int(m.sum()), ang_med=float(np.median(a[m])),
                         dE_med=float(np.median(de[m])), good5=float(np.mean((a[m] < 5) & (de[m] < 0.2))),
                         good25=float(np.mean((a[m] < 2.5) & (de[m] < 0.2))), lit_med=float(np.median(lit[m]))))
    res["recon"][tag] = rows
    print(f"\n[recon, grid, {tag}]")
    for r in rows:
        print(f"  zenith {r['band'][0]:3d}-{r['band'][1]:3d}: n {r['n']:4d}  angle median {r['ang_med']:5.1f}  "
              f"|dlog10E| median {r['dE_med']:.2f}  good@5 {r['good5']:.1%}  good@2.5 {r['good25']:.1%}  lit {r['lit_med']:.0f}")


# ---- 3. run 18 geometry with the zenith cut at 100 vs 120 ---------------------------------------------
def throws(side, zmax, n=400000, seed=0):
    rng = np.random.default_rng(seed)
    t = math.radians(45)
    dec = np.stack([rng.uniform(500, 500 + side, n), rng.uniform(-side / 2, side / 2, n),
                    rng.uniform(0, side, n)], 1)
    east = -side * math.cos(t) + rng.uniform(0, side, n) * math.cos(t)
    tgt = np.stack([east, rng.uniform(0, side, n) - side / 2, -east], 1)
    v = tgt - dec
    L = np.linalg.norm(v, axis=1)
    u = v / L[:, None]
    ze = np.degrees(np.arccos(u[:, 2]))
    az_w = 180 - np.abs(np.degrees(np.arctan2(u[:, 1], u[:, 0])))
    k = (L >= 1000) & (ze >= 60) & (ze <= zmax)
    h = np.histogram(az_w[k], bins=AZ_BINS + [180])[0] / k.sum()
    return dict(kept=float(k.mean()), az_from_west=dict(zip([f"{a}-{b}" for a, b in zip(AZ_BINS, AZ_BINS[1:] + [180])],
                                                            map(float, h))))


res["run18_geometry"] = {f"zenith 60-{z}": throws(6000, z) for z in (120, 100)}
print("\n[run 18 geometry]")
for k, v in res["run18_geometry"].items():
    print(f"  {k}: kept {v['kept']:.1%} of throws; |az - West| " +
          "  ".join(f"{b}: {f:.0%}" for b, f in v["az_from_west"].items()))

json.dump(res, open(os.path.join(D, "zenith_cap_check.json"), "w"), indent=1, default=float)

# ---- figure -------------------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
a = ax[0, 0]
styles = {"C8 electrons (simulated)": ("#1f2328", "o"), "run 17 electrons (generated)": ("#eb6834", "s"),
          "run 17 muons (generated)": ("#2a78d6", "^"), "run 17 photons (generated)": ("#1baf7a", "D")}
for name, rows in res["tilt"].items():
    col, mk = styles[name]
    for zname, ls, dx in (("zenith 60-100", "-", 0), ("zenith 100-120", ":", 1.2)):
        if zname not in rows:
            continue
        c = rows[zname]
        x = np.array([(r["az"][0] + r["az"][1]) / 2 for r in c]) + dx
        a.errorbar(x, [r["med"] for r in c], yerr=[[r["med"] - r["p16"] for r in c], [r["p84"] - r["med"] for r in c]],
                   color=col, marker=mk, ls=ls, ms=5, lw=1.2, capsize=2, label=f"{name}, {zname}")
a.axhline(0, color=GREY, lw=0.6)
a.set_xlabel("|azimuth − due West| [deg]  (run 17/18 showers: 80% within 30°)")
a.set_ylabel("cloud axis − labelled azimuth [deg]")
a.set_title("1. axis tilt of West-going showers (solid: zenith ≤ 100°, dotted: > 100°)", loc="left", fontsize=10.5)
a.legend(fontsize=7, ncol=2)
for k, key, ylab in ((1, "ang_med", "median angle error [deg]"), (2, "good5", "good at 5°, |Δlog10 E| < 0.2 [%]")):
    a = ax.flat[k]
    labels = [f"{r['band'][0]}–{r['band'][1]}" for r in res["recon"]["uncapped"]]
    x = np.arange(len(labels))
    for j, (tag, col) in enumerate((("uncapped", "#b9b7af"), ("cap 50 m", "#2a78d6"))):
        vals = [r[key] * (100 if key == "good5" else 1) for r in res["recon"][tag]]
        a.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=col, label=f"kernel widths {tag}")
        for xx, v in zip(x + (j - 0.5) * 0.38, vals):
            a.text(xx, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=GREY)
    a.axvline(len(ZE_BANDS) - 0.5, color=GREY, lw=0.6, ls="--")
    a.set_xticks(x)
    a.set_xticklabels([f"{l}\n(n={r['n']})" for l, r in zip(labels, res["recon"]["uncapped"])], fontsize=8)
    a.set_xlabel("true zenith band [deg]")
    a.set_ylabel(ylab)
    a.set_title(f"{k + 1}. grid reconstruction by zenith, 1,536 held-out kernel showers", loc="left", fontsize=10.5)
    a.legend(fontsize=8)
a = ax[1, 1]
bins = list(res["run18_geometry"]["zenith 60-120"]["az_from_west"])
x = np.arange(len(bins))
for j, (k, col) in enumerate((("zenith 60-120", "#b9b7af"), ("zenith 60-100", "#eb6834"))):
    v = res["run18_geometry"][k]
    a.bar(x + (j - 0.5) * 0.38, [100 * v["az_from_west"][b] for b in bins], 0.38, color=col,
          label=f"{k}: {v['kept']:.0%} of throws kept")
a.set_xticks(x)
a.set_xticklabels(bins)
a.set_xlabel("|azimuth − due West| [deg]")
a.set_ylabel("share of showers [%]")
a.set_title("4. run 18 geometry (6 km plane): azimuth spread with the zenith cut", loc="left", fontsize=10.5)
a.legend(fontsize=8)
for a in ax.flat:
    a.grid(axis="y", alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle("Zenith capped at 100° (AllShowers training range) vs run 17's 120°: axis tilt, grid reconstruction, run 18 "
             "geometry\nrecon trained on 100% uncapped kernel responses, all zeniths", fontsize=12, color=INK)
fig.tight_layout()
out = os.path.join(D, "zenith_cap_check.png")
fig.savefig(out, dpi=125, facecolor="white")
print(f"[write] {out}")
