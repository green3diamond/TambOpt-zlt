"""Single-shower array-size test (run 17, 45° plane). One held-out shower whose core lands near the
plane centre (--where center) or ~600 m from it (--where off) is seen by 100 detectors in sunflower
disks of growing radius at the plane centre (the core test's layouts) and by the grid. Per array: what each detector captures (kernel deposit, top
row), its arrival time (second row), then the recon's angle and energy errors and the total deposit.
Two figures with shared colour scales: adaptive kernel widths uncapped, and capped at --cap metres.
Both use the recon trained on 100% (uncapped) kernel responses, the one the kernel optimizer uses.
Options: --kernel fixed computes the detector signals with the kernel in the shower's own frame and the
C8 layer depth fixed ((l + 1) x 500 m; see ../kernel_frame/); --centre-on core puts the disks on the
shower's core instead of the plane centre; --min-angle-b keeps only showers at least that far from the
C8 magnetic field (+x), outside the range where C8 shifts the clouds sideways.

    python array_size_shower.py --where center|off [--event i] [--cap 50] [--max-zenith 100] [--out-dir DIR]
                                [--kernel used|fixed] [--centre-on plane|core] [--min-angle-b 30]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import showerdata
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"   # branch run17-bwcap50 (run 17 constants)
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.data import compute_labels_batch, place_clouds_enu  # noqa: E402
from modules.geometry import (load_tr_mountain, SurfaceUpMap, sample_initial_layout_ne,  # noqa: E402
                              project_to_mountain_ne)
from modules.optimize.objective import primary_to_physical_labels  # noqa: E402
from modules.showers.kernel import adaptive_bandwidths, GetCounts_planeaware  # noqa: E402
from modules.surrogates import build_recon_from_ckpt, combine_species_outputs, encode_primary  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
RECON = C.RECON_FOLDER + "_deepsets_kernel100"
RADII = (50, 100, 200, 300, 500, 750, 1000)
N = C.N_DETECTORS
LIT = 1.0                                     # counts for a detector to count as lit (detector_count_scan)
INK, GREY = "#1f2328", "#52514e"
dev = "cuda" if torch.cuda.is_available() else "cpu"

ap = argparse.ArgumentParser()
ap.add_argument("--event", type=int, default=None, help="held-out event index (default: auto-pick)")
ap.add_argument("--cap", type=float, default=50.0)
ap.add_argument("--where", choices=("center", "off"), default="center",
                help="core within 150 m of the plane centre, or 500-700 m from it")
ap.add_argument("--max-zenith", type=float, default=None, help="auto-pick only showers with zenith <= this [deg]")
ap.add_argument("--out-dir", default=None, help="default: this script's folder")
ap.add_argument("--kernel", choices=("used", "fixed"), default="used",
                help="used: global frame as in the pipeline; fixed: shower frame + layer depth (l + 1) x 500 m")
ap.add_argument("--centre-on", choices=("plane", "core"), default="plane", help="where the disks are centred")
ap.add_argument("--time-interp", action="store_true",
                help="with --kernel fixed: point time + (detector depth - plane depth) / c instead of the plane's time")
ap.add_argument("--min-angle-b", type=float, default=0.0,
                help="auto-pick only showers at least this far [deg] from the C8 field direction (+x)")
args = ap.parse_args()
OUT = args.out_dir or D
TAG = (args.where + (f"_zen{args.max_zenith:g}" if args.max_zenith else "") + ("_fixed" if args.kernel == "fixed" else "")
       + ("_tinterp" if args.time_interp else "")
       + ("_oncore" if args.centre_on == "core" else "") + (f"_b{args.min_angle_b:g}" if args.min_angle_b else ""))

mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                            layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(dev)
mesh = np.asarray(mountain.centroids_ENU)
centre = mesh[:, :2].mean(0)

# --- held-out events: decay point, direction, core = where the axis meets the plane (Up = -East)
n_sp = len(C.SPECIES_NAMES)
pos_all = torch.load(C.HELDOUT_POSITIONS_PATH).float()
n_pairs = pos_all.shape[0] // n_sp
pos = pos_all[:n_pairs]
ip = showerdata.load_inc_particles(C.HELDOUT_SHOWER_CACHE_PATH, 0, n_pairs)
dirs = torch.as_tensor(ip.directions, dtype=torch.float32)
dirs = dirs / dirs.norm(dim=1, keepdim=True)
energies = torch.as_tensor(ip.energies, dtype=torch.float32)
pdg = torch.as_tensor(ip.pdg, dtype=torch.long)
t_hit = -(pos[:, 0] + pos[:, 2]) / (dirs[:, 0] + dirs[:, 2])
core = (pos + t_hit[:, None] * dirs).numpy()
dist_c = np.hypot(core[:, 0] - centre[0], core[:, 1] - centre[1])
logE = np.log10(energies.numpy().reshape(-1))
if args.event is None:   # energy in the upper middle of the 5-7 range
    lo_d, hi_d = (0, 150) if args.where == "center" else (500, 700)
    ze_all = np.degrees(np.arccos(dirs[:, 2].clamp(-1, 1).numpy()))
    ang_b = np.degrees(np.arccos(np.abs(dirs[:, 0].clamp(-1, 1).numpy())))       # angle to the C8 field (+x)
    cand = np.nonzero((dist_c >= lo_d) & (dist_c < hi_d) & (ze_all <= (args.max_zenith or 180))
                      & (ang_b >= args.min_angle_b))[0]
    ev = int(cand[np.argmin(np.abs(logE[cand] - 6.3))])
    print(f"[pick] {len(cand)} events with core {lo_d}-{hi_d} m from the centre; event {ev}")
else:
    ev = args.event
zen = float(np.degrees(np.arccos(float(dirs[ev, 2]))))
track = float(t_hit[ev])
angB = float(np.degrees(np.arccos(abs(float(dirs[ev, 0])))))
print(f"[event {ev}] log10E {logE[ev]:.2f}  zenith {zen:.1f}  decay->core {track:.0f} m  "
      f"core E {core[ev, 0]:.0f} N {core[ev, 1]:.0f} (centre E {centre[0]:.0f} N {centre[1]:.0f})  angle to B {angB:.0f}")

# Kernel input per species. "used": placed as in the pipeline (global North-Up + East slabs). "fixed":
# shower frame with the C8 depth (l + 1) x 500 m: points [t1, t2 = 0, l + 1, e, t], t1 the horizontal
# offset across the axis (exact from x/y), t2 the vertical-ish one (not recorded by C8).
p_ev, d_ev = pos[ev].double().numpy(), dirs[ev].double().numpy()
e1 = np.array([-d_ev[1], d_ev[0], 0.0]); e1 /= np.linalg.norm(e1)
e2 = np.cross(d_ev, e1)
clouds = []
for s in range(n_sp):
    sh = showerdata.load(C.HELDOUT_SHOWER_CACHE_PATH, start=s * n_pairs + ev, stop=s * n_pairs + ev + 1)
    c = torch.as_tensor(sh.points, dtype=torch.float32)
    if args.kernel == "used":
        place_clouds_enu(c, pos[ev:ev + 1], dirs[ev:ev + 1], east_entry=C.EAST_ENTRY, layer_east_dx=C.LAYER_EAST_DX)
    else:
        real = c[..., 3] > 0
        t1 = c[..., 0] * float(e1[0]) + c[..., 1] * float(e1[1])
        c[..., 2] = torch.where(real, c[..., 2] + 1.0, c[..., 2])
        c[..., 0], c[..., 1] = t1, 0.0
    clouds.append(c.to(dev))
prim = encode_primary(dirs[ev:ev + 1], energies[ev:ev + 1], pdg[ev:ev + 1], pos[ev:ev + 1],
                      torch.as_tensor(mesh, dtype=torch.float32).mean(0))
E_true, th_t, ph_t = (x.to(dev) for x in primary_to_physical_labels(prim))

ckpt = torch.load(os.path.join(RECON, "recon.pt"), map_location=dev, weights_only=False)
recon = build_recon_from_ckpt(ckpt, N, dev).eval()


# --- layouts: the core test's sunflower disks at the plane centre, and the grid
def snap(e, n):
    e, n = project_to_mountain_ne(mountain, torch.as_tensor(e).float(), torch.as_tensor(n).float())
    return e.float().to(dev), n.float().to(dev)


def disk(R):
    i = np.arange(N) + 0.5
    r, th = R * np.sqrt(i / N), i * np.pi * (3 - np.sqrt(5))
    return snap(DC[0] + r * np.cos(th), DC[1] + r * np.sin(th))


DC = core[ev, :2] if args.centre_on == "core" else centre      # disk centre
C_LIGHT = 299_792_458.0


def counts_interp(samples, x_det, y_det, z_cont, bandwidth):
    """GetCounts_planeaware (adaptive widths, leading edge) with each point's time moved along the axis to the
    detector at c: t = t_p + (s_d - s_p) / c, so a detector between two planes no longer gets a plane's time."""
    px, py, pl, pe, pt = (samples[..., i] for i in range(5))
    dx = px.unsqueeze(2) - x_det.view(1, 1, -1)
    dy = py.unsqueeze(2) - y_det.view(1, 1, -1)
    h2 = bandwidth.unsqueeze(2) ** 2
    spatial = (C.SIGMA_SPATIAL ** 2 / h2) * torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * h2))
    delta_l = pl.unsqueeze(2) - z_cont.view(1, 1, -1)                 # (s_p - s_d) / 500 m
    ek = pe.unsqueeze(2) * spatial * torch.relu(1.0 - delta_l.abs())
    t_eff = (pt.unsqueeze(2) - delta_l * C.LAYER_EAST_DX / C_LIGHT).clamp(min=1e-12)
    t = torch.where(ek > C.HIT_DEPOSIT_MIN, t_eff, torch.full_like(ek, float("inf"))).amin(dim=1)
    return ek.sum(dim=1), torch.where(torch.isinf(t), torch.zeros_like(t), t)
torch.manual_seed(0)
layouts = {f"r = {R} m": disk(R) for R in RADII}
layouts["grid"] = snap(*sample_initial_layout_ne(mountain, n_units=N, scheme="grid"))


def sph(th, ph):
    return torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)], -1)


@torch.no_grad()
def run(cap):
    bws = [adaptive_bandwidths(c, C.KERNEL_INTERP_K, C.SIGMA_SPATIAL, sigma_max=cap) for c in clouds]
    out = {}
    for name, (e, n) in layouts.items():
        labs = []
        if args.kernel == "fixed":                                  # detectors in the shower frame
            up = surface(n, e)
            r3 = torch.stack([e, n, up], -1).double() - torch.as_tensor(p_ev, device=dev)
            s_d, t1_d, t2_d = ((r3 @ torch.as_tensor(v, device=dev)).float() for v in (d_ev, e1, e2))
        for c, bw in zip(clouds, bws):
            if args.kernel == "fixed" and args.time_interp:
                E, T = counts_interp(c, t1_d, t2_d, s_d / C.LAYER_EAST_DX, bw)
            elif args.kernel == "fixed":
                E, T = GetCounts_planeaware(c, t1_d, t2_d, s_d / C.LAYER_EAST_DX, None, torch.zeros(1, device=dev),
                                            None, sigma=C.SIGMA_SPATIAL, bandwidth=bw)
            else:
                E, T = compute_labels_batch(c, e, n, surface, bandwidth=bw)
            labs.append(torch.stack([torch.log1p(E), torch.log1p(T * C.T_LOG_SCALE)], -1))
        lab = combine_species_outputs(*labs)                                  # (1, N, 2)
        feats = torch.stack([e[None], n[None], lab[..., 0], lab[..., 1]], -1)
        E_p, th_p, ph_p = primary_to_physical_labels(recon(feats))
        E_p = E_p.clamp(min=1.0)
        u, v = sph(th_p, ph_p), sph(th_t, ph_t)
        ang = float(torch.rad2deg(2 * torch.atan2((u - v).norm(dim=-1), (u + v).norm(dim=-1)))[0])
        dE = float((torch.log10(E_p) - torch.log10(E_true))[0])
        dep = torch.expm1(lab[0, :, 0]).cpu().numpy()
        t = (torch.expm1(lab[0, :, 1]) / C.T_LOG_SCALE).cpu().numpy()          # s, 0 = no hit
        out[name] = dict(e=e.cpu().numpy(), n=n.cpu().numpy(), dep=dep, t=t, ang=ang, dE=dE,
                         dep_sum=float(dep.sum()), n_lit=int((dep > LIT).sum()))
        print(f"  [{'uncapped' if cap is None or np.isinf(cap) else f'cap {cap:g} m'}] {name:>10}: "
              f"angle {ang:6.2f} deg  dlog10E {dE:+.3f}  deposit {dep.sum():10.1f}  lit {out[name]['n_lit']}")
    return out, float(np.concatenate([b[b > 0].cpu().numpy() for b in bws]).max())


cases = {"uncapped": run(float("inf")), f"cap{args.cap:g}": run(args.cap)}

# shared colour scales over both cases and every array
all_dep = np.concatenate([r["dep"] for res, _ in cases.values() for r in res.values()])
all_t = np.concatenate([r["t"][r["dep"] > LIT] for res, _ in cases.values() for r in res.values()])
dnorm = LogNorm(vmin=LIT, vmax=max(all_dep.max(), LIT * 10))
t0 = all_t.min()
tnorm = Normalize(vmin=0, vmax=(all_t.max() - t0) * 1e9)
COL = [plt.cm.Oranges(0.35 + 0.6 * i / (len(RADII) - 1)) for i in range(len(RADII))] + ["#b9b7af"]


def draw(tag, res, bw_max):
    names = list(res)
    fig = plt.figure(figsize=(2.3 * len(names) + 1.2, 13))
    gs = fig.add_gridspec(5, len(names) + 1, height_ratios=[1.25, 1.25, 1, 1, 1],
                          width_ratios=[1] * len(names) + [0.06], hspace=0.42, wspace=0.08)
    for row, (key, cmap, norm, lab) in enumerate((("dep", "YlOrRd", dnorm, "deposit [counts]"),
                                                 ("t", "viridis", tnorm, "arrival time [ns]"))):
        for i, name in enumerate(names):
            r = res[name]
            a = fig.add_subplot(gs[row, i])
            if name == "grid":
                lo, hi = mesh[:, :2].min(0), mesh[:, :2].max(0)
            else:
                R = float(name.split()[2]) * 1.15        # the array, and the core if it lies outside
                lo = np.minimum(DC - R, core[ev, :2] - 60)
                hi = np.maximum(DC + R, core[ev, :2] + 60)
            a.scatter(mesh[:, 0], mesh[:, 1], s=1.5, color="#e8e6df", lw=0)
            lit = r["dep"] > LIT
            a.scatter(r["e"][~lit], r["n"][~lit], s=10, facecolors="none", edgecolors="#c9c7bf", lw=0.6)
            val = r["dep"][lit] if key == "dep" else (r["t"][lit] - t0) * 1e9
            sc = a.scatter(r["e"][lit], r["n"][lit], s=16, c=val, cmap=cmap, norm=norm, lw=0.3, edgecolors=INK)
            a.plot(*core[ev, :2], marker="*", ms=10, color="#2a78d6", mec="white", mew=0.6)
            a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1])
            a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color("#d8d6cf")
            if row == 0:
                a.set_title(f"{name}\n{r['n_lit']} lit", fontsize=10, color=INK)
            if i == 0:
                a.set_ylabel(lab, fontsize=10, color=INK)
        fig.colorbar(sc, cax=fig.add_subplot(gs[row, -1]), label=lab)
    bars = (("ang", "angle error [deg]", "{:.1f}°"), ("dE", "Δlog10 E (recon − true)", "{:+.2f}"),
            ("dep_sum", "total deposit over the 100 detectors [counts]", "{:.0f}"))
    x = np.arange(len(names))
    for k, (key, title, fmt) in enumerate(bars):
        a = fig.add_subplot(gs[2 + k, :-1])
        vals = np.array([res[nm][key] for nm in names])
        a.bar(x, vals, color=COL, width=0.6)
        for xx, v in zip(x, vals):
            a.text(xx, v, fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=GREY)
        a.axhline(res["grid"][key], color="#b9b7af", ls="--", lw=1)
        if key == "dE":
            a.axhline(0, color=GREY, lw=0.6)
            m = max(np.abs(vals).max(), 0.2) * 1.3
            a.set_ylim(-m, m)
            for s in (-0.2, 0.2):
                a.axhline(s, color="#1baf7a", ls=":", lw=1)
        elif key == "dep_sum":
            a.set_yscale("log"); a.set_ylim(max(vals.min(), 1) / 2, vals.max() * 3)
        else:
            a.set_ylim(0, max(vals.max(), 5) * 1.25)
            a.axhline(5, color="#1baf7a", ls=":", lw=1)
        a.set_xticks(x); a.set_xticklabels(names if k == 2 else [], fontsize=9)
        a.set_title(title + "   (dashed: grid" + (", dotted: 'good' threshold)" if key != "dep_sum" else ")"),
                    fontsize=10, color=INK, loc="left")
        a.grid(axis="y", alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    capt = "uncapped" if tag == "uncapped" else f"capped at {args.cap:g} m"
    fig.suptitle(f"One held-out shower (event {ev}: log10 E/GeV = {logE[ev]:.2f}, zenith {zen:.0f}°, "
                 f"{track:.0f} m from decay to core, core ★ {dist_c[ev]:.0f} m from the plane centre) seen by 100 "
                 f"detectors in disks of growing radius {'centred on its core' if args.centre_on == 'core' else 'at the plane centre'}"
                 f" and by the grid; {angB:.0f}° from the C8 field direction\n"
                 f"kernel: {'shower frame, layer depth fixed' if args.kernel == 'fixed' else 'as used (global frame)'}"
                 f"{', times interpolated between planes' if args.time_interp else ''} · adaptive widths "
                 f"{capt} (largest used: {bw_max:.0f} m) · recon trained on the old kernel (uncapped, global frame) · hollow: < 1 count",
                 fontsize=11.5, color=INK)
    out = os.path.join(OUT, f"array_size_shower_{TAG}_{tag}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[write] {out}")


summary = dict(event=ev, log10E=float(logE[ev]), zenith=zen, decay_to_core_m=track,
               core_EN=[float(core[ev, 0]), float(core[ev, 1])], core_to_centre_m=float(dist_c[ev]), recon=RECON,
               kernel=args.kernel, centre_on=args.centre_on, angle_to_B=angB)
for tag, (res, bw_max) in cases.items():
    draw(tag, res, bw_max)
    summary[tag] = dict(bw_max=bw_max, arrays={nm: {k: r[k] for k in ("ang", "dE", "dep_sum", "n_lit")}
                                                for nm, r in res.items()})
js = os.path.join(OUT, f"array_size_shower_{TAG}.json")
json.dump(summary, open(js, "w"), indent=1)
print(f"[write] {js}")
