"""Event displays: chosen held-out showers' energy deposits with the layout's kernel and surrogate response and reconstructed directions."""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))

import numpy as np
import torch
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from analysis_core import events, logs, plots
from analysis_core.run import LAYOUTS, Run, add_run_args
from analysis_core.scoring import LABELS, Scorer
from modules.constants import EAST_ENTRY, LAYER_EAST_DX, LOG_E_MAX, LOG_E_MIN, SIGMA_SPATIAL
from modules.optimize import LAYOUT_THRESHOLD
from recon_resolution import SPAN, angular_deg, reconstruct

TRUE, RECON = "k", "C3"
MAX_POINTS = 60000


def pick(mask, key):
    """Index of the event in `mask` whose `key` is closest to the median over `mask`."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        raise SystemExit("no candidate events")
    return int(idx[np.argmin(np.abs(key[idx] - np.median(key[idx])))]), len(idx)


def choose(log_e, dist, ang, n_lit, n_lit_s, heading, far_km):
    """The showers on display, from the primary layout's kernel-input errors; the first four also get the overview."""
    p10, p90 = np.percentile(ang, [10, 90])
    q40, q60 = np.percentile(ang, [40, 60])
    excess = n_lit_s - n_lit
    x10, x90 = np.percentile(excess, [10, 90])
    top = log_e >= LOG_E_MAX - SPAN / 6
    low = log_e <= LOG_E_MIN + SPAN / 6
    mid = np.abs(log_e - np.median(log_e)) < SPAN / 6
    seen = n_lit >= 10
    near = np.abs(dist - far_km * 1e3) < 200
    toward = lambda deg: np.abs((heading - deg + 180) % 360 - 180) < 30
    rules = {"well reconstructed": ((ang <= p10) & mid, dist),
             "high-energy failure": (top & (ang >= p90), ang),
             "near-field failure": ((dist < 1000) & (ang >= p90), ang),
             f"near {far_km} km, undetected": (near & ~seen, log_e),
             "typical": (mid & (ang >= q40) & (ang <= q60), dist),
             "low-energy failure": (low & (ang >= p90), ang),
             "mid-energy, undetected": (mid & ~seen, dist),
             "decay beyond 6 km, detected": ((dist > 6000) & seen, ang),
             "surrogate over-fires": (excess >= x90, ang),
             "surrogate under-fires": (excess <= x10, ang),
             "10 to 15 lit": ((n_lit >= 10) & (n_lit <= 15), ang),
             "many lit": (n_lit >= np.percentile(n_lit, 95), ang),
             **{f"heading {d}": (mid & toward(deg), ang)
                for d, deg in (("north", 90), ("east", 0), ("south", 270), ("west", 180))}}
    return {name: pick(m, key) for name, (m, key) in rules.items()}


def cloud_enu(ev, i):
    """(M, 4) East, North, Up, deposited energy of one shower, every species."""
    c = torch.cat([s[i] for s in ev.clouds]).numpy()
    c = c[c[:, 3] > 0]
    return np.stack([EAST_ENTRY - c[:, 2] * LAYER_EAST_DX, c[:, 0], c[:, 1], c[:, 3]], 1)


def kernel_from_enu(pts, det):
    """(points, detectors) kernel contributions recomputed from ENU points and (East, North, Up) detectors."""
    g = np.exp(-((pts[:, None, 1] - det[None, :, 1]) ** 2 + (pts[:, None, 2] - det[None, :, 2]) ** 2)
               / (2 * SIGMA_SPATIAL ** 2))
    w = np.clip(1 - np.abs(pts[:, None, 0] - det[None, :, 0]) / LAYER_EAST_DX, 0, None)
    return pts[:, 3:4] * g * w


def sizes(count):
    return 6 + 18 * np.log1p(count)


def arrows(ax, shower, pts, a, b, lo, hi, show_vertex):
    anchor = np.average(pts[:, [a, b]], axis=0, weights=pts[:, 3]) if len(pts) else (lo + hi) / 2
    length = 0.25 * (hi - lo).max()
    for d, color in ((shower["dir_true"], TRUE), (shower["dir_recon"], RECON)):
        d = np.asarray(d) / np.linalg.norm(d)
        ax.annotate("", anchor + length * d[[a, b]], anchor, zorder=6,
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2})
    v = shower["vertex"][[a, b]]
    if show_vertex and np.all((v > lo) & (v < hi)):
        ax.plot(*v, "x", color=TRUE, ms=8, mew=2, zorder=6)


def detectors(ax, det, shower, a, b):
    k, s = shower["count"]["kernel"], shower["count"]["surrogate"]
    lit_k, lit_s = k > LAYOUT_THRESHOLD, s > LAYOUT_THRESHOLD
    ax.scatter(det[~lit_k, a], det[~lit_k, b], s=sizes(k[~lit_k]), c=plots.KERNEL, alpha=0.35, lw=0, zorder=2)
    ax.scatter(det[lit_s, a], det[lit_s, b], s=sizes(s[lit_s]) + 40, fc="none", ec=plots.SURROGATE, lw=1.3, zorder=3)
    ax.scatter(det[lit_k, a], det[lit_k, b], s=sizes(k[lit_k]), c=plots.KERNEL, ec="k", lw=1, zorder=4)


def draw(ax, run, pts, det, shower, view, norm, rng):
    """One panel: the whole cloud in plan, the array in plan, or the array side-on (North, Up)."""
    a, b = (1, 2) if view == "side" else (0, 1)
    pad = np.array([300.0, 300.0])
    lo, hi = det[:, [a, b]].min(0) - pad, det[:, [a, b]].max(0) + pad
    if view == "side":
        c = run.mountain.centroids_ENU[::max(1, len(run.mountain.centroids_ENU) // 20000)]
        ax.scatter(c[:, 1], c[:, 2], s=0.3, c="0.8", zorder=0, rasterized=True)
        ax.set_xlabel("North [m]")
        ax.set_ylabel("Up [m]")
        pts = pts[(pts[:, 0] > det[:, 0].min() - LAYER_EAST_DX) & (pts[:, 0] < det[:, 0].max() + LAYER_EAST_DX)
                  & (pts[:, 1] > lo[0]) & (pts[:, 1] < hi[0])]
        if len(pts):
            lo[1], hi[1] = min(lo[1], np.percentile(pts[:, 2], 1) - 100), max(hi[1], np.percentile(pts[:, 2], 99) + 100)
    else:
        plots.footprint(ax, run.mountain)
        if view == "wide":
            lo = np.minimum(lo, np.percentile(pts[:, :2], 1, axis=0)) - 200
            hi = np.maximum(hi, np.percentile(pts[:, :2], 99, axis=0)) + 200
        pts = pts[np.all((pts[:, :2] > lo) & (pts[:, :2] < hi), axis=1)]
    if len(pts) > MAX_POINTS:
        pts = pts[rng.choice(len(pts), MAX_POINTS, replace=False)]
    pts = pts[np.argsort(pts[:, 3])]
    sc = ax.scatter(pts[:, a], pts[:, b], c=pts[:, 3], s=2 if view == "wide" else 4, cmap="Greys", norm=norm,
                    rasterized=True, zorder=1)
    if view == "wide":
        box = det[:, :2].min(0) - pad, det[:, :2].max(0) + pad
        ax.add_patch(plots.plt.Rectangle(box[0], *(box[1] - box[0]), fill=False, ec=plots.KERNEL, lw=1.2, zorder=5))
    else:
        detectors(ax, det, shower, a, b)
    if view != "plan":
        in_reach = det[:, 0].min() - LAYER_EAST_DX < shower["vertex"][0] < det[:, 0].max() + LAYER_EAST_DX
        arrows(ax, shower, pts, a, b, lo, hi, view == "wide" or in_reach)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal" if view != "side" else "auto")
    return sc


def caption(name, sh):
    return (f"{name}, event {sh['index']}: E 10^{sh['log_e']:.2f} GeV (recon 10^{sh['log_e_recon']:.2f}), "
            f"decay {sh['distance_m'] / 1e3:.2f} km\nlit kernel {sh['n_lit']['kernel']}, surrogate "
            f"{sh['n_lit']['surrogate']}; angle error {sh['angle_deg']['kernel']:.1f} deg "
            f"(surrogate input {sh['angle_deg']['surrogate']:.1f})")


VIEWS = {"wide": "whole cloud (box: array)", "plan": "array, plan view",
         "side": "array side-on; points within 500 m East of a detector"}


def row(fig, axes, run, name, sh, det, layout, rng):
    e = sh["pts"][:, 3]
    norm = LogNorm(max(np.percentile(e, 5), 1e-6), e.max())
    for ax, view in zip(axes, VIEWS):
        sc = draw(ax, run, sh["pts"], det, sh[layout], view, norm, rng)
        ax.set_title(f"{layout}: {VIEWS[view]}", fontsize=9)
    axes[1].set_title(caption(name, sh[layout]) + f"\n{layout}: {VIEWS['plan']}", fontsize=9)
    fig.colorbar(sc, ax=list(axes), shrink=0.8, pad=0.01, label="deposited energy per point")


def legend(fig):
    h = [Line2D([], [], ls="", marker="o", color=plots.KERNEL, mec="k", ms=8, label="kernel count, lit (size ~ log count)"),
         Line2D([], [], ls="", marker="o", color=plots.KERNEL, alpha=0.35, ms=5, label="kernel count, dark"),
         Line2D([], [], ls="", marker="o", mfc="none", mec=plots.SURROGATE, mew=1.3, ms=11, label="surrogate mean count, lit"),
         Line2D([], [], color=TRUE, lw=2, label="true direction (projected)"),
         Line2D([], [], color=RECON, lw=2, label="recon on kernel input (projected)"),
         Line2D([], [], ls="", marker="x", color=TRUE, mew=2, label="decay vertex")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01),
               title=f"lit: kernel or surrogate mean count > {LAYOUT_THRESHOLD:g}")


def figure(run, rows, rng):
    """One row per (shower, layout): whole cloud, array in plan, array side-on."""
    fig, axes = plots.plt.subplots(len(rows), 3, figsize=(19, 6 * len(rows)), squeeze=False,
                                   gridspec_kw={"width_ratios": [1, 1, 1.2], "hspace": 0.4})
    for ax_row, (name, sh, det, layout) in zip(axes, rows):
        row(fig, ax_row, run, name, sh, det, layout, rng)
    legend(fig)
    return fig


def square(ax):
    ax.set_box_aspect(1)


def scatter_deposits(ax, pts, a, b, norm, rng, size):
    if len(pts) > MAX_POINTS:
        pts = pts[rng.choice(len(pts), MAX_POINTS, replace=False)]
    pts = pts[np.argsort(pts[:, 3])]
    return ax.scatter(pts[:, a], pts[:, b], c=pts[:, 3], s=size, cmap="Greys", norm=norm, rasterized=True, zorder=1)


def top_view(ax, run, pts, det, norm, rng, vertex):
    plots.footprint(ax, run.mountain)
    lo = np.minimum.reduce([det[:, :2].min(0), np.percentile(pts[:, :2], 1, axis=0), vertex[:2]]) - 300
    hi = np.maximum.reduce([det[:, :2].max(0), np.percentile(pts[:, :2], 99, axis=0), vertex[:2]]) + 300
    pts = pts[np.all((pts[:, :2] > lo) & (pts[:, :2] < hi), axis=1)]
    sc = scatter_deposits(ax, pts, 0, 1, norm, rng, 2)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Energy deposits, top view", fontsize=10)
    return sc


def side_view(ax, run, pts, det, norm, rng):
    c = run.mountain.centroids_ENU
    ax.scatter(c[:, 1], c[:, 2], s=1, c="0.8", zorder=0)
    lo, hi = det[:, 1:].min(0) - 300, det[:, 1:].max(0) + 300
    pts = pts[(pts[:, 0] > det[:, 0].min() - LAYER_EAST_DX) & (pts[:, 0] < det[:, 0].max() + LAYER_EAST_DX)
              & (pts[:, 1] > lo[0]) & (pts[:, 1] < hi[0])]
    if len(pts):
        lo[1], hi[1] = min(lo[1], np.percentile(pts[:, 2], 1) - 100), max(hi[1], np.percentile(pts[:, 2], 99) + 100)
    sc = scatter_deposits(ax, pts, 1, 2, norm, rng, 4)
    ax.scatter(det[:, 1], det[:, 2], s=10, c=plots.KERNEL, lw=0, zorder=2, label="detectors")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("North [m]")
    ax.set_ylabel("Up [m]")
    ax.set_title("Energy deposits, side view", fontsize=10)
    return sc


def count_map(ax, run, det, count, norm, source):
    plots.footprint(ax, run.mountain)
    lit = count > LAYOUT_THRESHOLD
    ax.scatter(det[~lit, 0], det[~lit, 1], s=14, c="0.55", lw=0, zorder=2)
    sc = ax.scatter(det[lit, 0], det[lit, 1], s=30, c=count[lit], cmap="viridis", norm=norm, ec="k", lw=0.4, zorder=3)
    ax.set_xlim(det[:, 0].min() - 150, det[:, 0].max() + 150)
    ax.set_ylim(det[:, 1].min() - 150, det[:, 1].max() + 150)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Detector counts, {source}", fontsize=10)
    return sc


def colorbar(fig, cax, sc, label):
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", label=label)
    cb.ax.xaxis.set_minor_formatter(NullFormatter())


def mark_array(ax, det, vertex, legend):
    ax.scatter(det[:, 0], det[:, 1], s=3, c=plots.KERNEL, lw=0, zorder=2, label="detectors")
    ax.plot(*vertex[:2], "x", color="C3", ms=6, mew=1.5, zorder=3, label="decay vertex")
    if legend:
        ax.legend(loc="upper left", fontsize=8)


def combined_figure(run, sh, dets, layouts, rng):
    """One shower: a row per layout with deposits from above and side-on, kernel and surrogate counts on the array."""
    e = sh["pts"][:, 3]
    dep = LogNorm(max(np.percentile(e, 5), 1e-6), e.max())
    cnt = LogNorm(LAYOUT_THRESHOLD, max(10.0, max(sh[l]["count"][src].max() for l in layouts for src in LABELS)))
    fig = plots.plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(len(layouts) + 1, 4, height_ratios=[1] * len(layouts) + [0.04], wspace=0.35, hspace=0.35)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(len(layouts))])
    for r, (row_axes, layout) in enumerate(zip(axes, layouts)):
        s, det = sh[layout], dets[layout]
        top_view(row_axes[0], run, sh["pts"], det, dep, rng, s["vertex"])
        mark_array(row_axes[0], det, s["vertex"], legend=r == 0)
        sd = side_view(row_axes[1], run, sh["pts"], det, dep, rng)
        count_map(row_axes[2], run, det, s["count"]["kernel"], cnt, "kernel")
        sc = count_map(row_axes[3], run, det, s["count"]["surrogate"], cnt, "surrogate mean")
        if r == 0:
            row_axes[1].legend(loc="upper left", fontsize=8)
        else:
            for ax in row_axes:
                ax.set_title("")
        row_axes[0].annotate(f"{layout}\nangular error {s['angle_deg']['kernel']:.1f} deg", (-0.42, 0.5),
                             xycoords="axes fraction", rotation=90, ha="center", va="center", fontsize=11)
        for ax in row_axes:
            square(ax)
    colorbar(fig, fig.add_subplot(gs[-1, :2]), sd, "deposited energy per point")
    colorbar(fig, fig.add_subplot(gs[-1, 2:]), sc, f"detector count (grey: at most {LAYOUT_THRESHOLD:g})")
    s = sh[layouts[0]]
    fig.suptitle(f"Event {s['index']}: log10(E / GeV) = {s['log_e']:.2f}, decay distance {s['distance_m'] / 1e3:.2f} km",
                 fontsize=12, y=0.95)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_run_args(ap)
    ap.add_argument("--layouts", nargs="+", default=["opt_grid", "grid"], choices=LAYOUTS,
                    help="the first picks the showers")
    ap.add_argument("--far-km", type=float, default=3.8)
    ap.add_argument("--errors", default="recon_resolution/errors.npz", help="relative to --out")
    ap.add_argument("--lit", default="detection_stats/results.json", help="relative to --out")
    args = ap.parse_args()

    out, log = logs.start("event_display", args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run(args.run_dir, device)
    ev = events.load(run, args.n_events, args.seed, args.chunk)
    prim = ev.primary.cpu().numpy()
    log_e = prim[:, 3] * SPAN + LOG_E_MIN
    dist = np.hypot(prim[:, 5], prim[:, 6])
    centre = run.mountain.centroids_ENU.mean(0)
    scorers = {src: Scorer(run, ev, src, args.chunk, args.kernel_chunk) for src in LABELS}
    saved = np.load(os.path.join(args.out, args.errors))
    with open(os.path.join(args.out, args.lit)) as f:
        saved_lit = json.load(f)["results"]

    per, dets, checks = {}, {}, {}
    for layout in args.layouts:
        e, n = run.layout(layout)
        up = run.surface(n.to(device), e.to(device)).cpu()
        dets[layout] = torch.stack([e, n, up], 1).numpy()
        per[layout] = {}
        for src, sc in scorers.items():
            count = torch.expm1(sc.labels(e, n)[..., 0]).clamp(min=0).cpu().numpy()
            pred = reconstruct(run, sc, e, n, args.chunk)
            ang = angular_deg(pred[:, :3], prim[:, :3])
            per[layout][src] = {"count": count, "pred": pred, "ang": ang}
            n_lit = (count > LAYOUT_THRESHOLD).sum(1)
            checks[f"{layout}.{src}"] = {
                "max_angle_diff_vs_recon_resolution": float(np.abs(ang - saved[f"{layout}.{src}.angle_deg"]).max()),
                "n_lit_mismatches_vs_detection_stats": int((n_lit != np.array(saved_lit[layout]["n_lit"][src])).sum())}
            log.info("%-9s %-9s check vs saved: %s", layout, src, checks[f"{layout}.{src}"])
        near = np.abs(dets[layout][:, 2] - np.array([run.mountain.centroids_ENU[
            np.argmin(((run.mountain.centroids_ENU[:, :2] - d[:2]) ** 2).sum(1)), 2] for d in dets[layout]]))
        log.info("%-9s detectors East [%.0f, %.0f] North [%.0f, %.0f] Up [%.0f, %.0f]; |Up - nearest mesh Up| max %.1f m",
                 layout, *np.ravel(np.stack([dets[layout].min(0), dets[layout].max(0)], 1)), near.max())
        checks[f"{layout}.detector_up_vs_mesh_max_m"] = float(near.max())

    first = per[args.layouts[0]]
    n_lit0 = (first["kernel"]["count"] > LAYOUT_THRESHOLD).sum(1)
    n_lit_s0 = (first["surrogate"]["count"] > LAYOUT_THRESHOLD).sum(1)
    heading = np.degrees(np.arctan2(prim[:, 1], prim[:, 0]))
    chosen = choose(log_e, dist, first["kernel"]["ang"], n_lit0, n_lit_s0, heading, args.far_km)
    rng = np.random.default_rng(args.seed)
    shows, results = {}, {}
    for name, (i, n_cand) in chosen.items():
        pts = cloud_enu(ev, i)
        sh = {"pts": pts}
        log.info("%s: event %d of %d candidates; cloud %d points, East [%.0f, %.0f] North [%.0f, %.0f] Up [%.0f, %.0f]",
                 name, i, n_cand, len(pts), *np.ravel(np.stack([pts[:, :3].min(0), pts[:, :3].max(0)], 1)))
        for layout in args.layouts:
            p = per[layout]
            count = {src: p[src]["count"][i] for src in LABELS}
            contrib = kernel_from_enu(pts, dets[layout])
            lit_k = count["kernel"] > LAYOUT_THRESHOLD
            rel = float(np.abs(contrib.sum(0) - count["kernel"]).max() / max(count["kernel"].max(), 1e-9))
            top = pts[contrib.argmax(0), :3][lit_k] - dets[layout][lit_k]
            sh[layout] = {
                "index": i, "log_e": float(log_e[i]), "log_e_recon": float(p["kernel"]["pred"][i, 3] * SPAN + LOG_E_MIN),
                "distance_m": float(dist[i]), "vertex": prim[i, 5:8] + centre, "dir_true": prim[i, :3],
                "dir_recon": p["kernel"]["pred"][i, :3], "count": count,
                "n_lit": {src: int((count[src] > LAYOUT_THRESHOLD).sum()) for src in LABELS},
                "angle_deg": {src: float(p[src]["ang"][i]) for src in LABELS},
                "saved_angle_deg": float(saved[f"{layout}.kernel.angle_deg"][i]),
                "saved_n_lit": {src: saved_lit[layout]["n_lit"][src][i] for src in LABELS},
                "kernel_from_enu_max_rel_diff": rel,
                "lit_dominant_point_north_up_m": np.hypot(top[:, 1], top[:, 2]).round(0).tolist(),
                "lit_dominant_point_east_m": np.abs(top[:, 0]).round(0).tolist(),
                "heading_deg": float(heading[i]),
                "cloud_bearing_deg": float(np.degrees(np.arctan2(
                    *(np.average(pts[:, :2], axis=0, weights=pts[:, 3]) - (prim[i, 5:7] + centre[:2]))[::-1]))),
                "surrogate_only_lit": int(((count["surrogate"] > LAYOUT_THRESHOLD) & ~lit_k).sum()),
                "kernel_only_lit": int((lit_k & ~(count["surrogate"] > LAYOUT_THRESHOLD)).sum())}
            r = {k: v for k, v in sh[layout].items() if k not in ("count", "vertex", "dir_true", "dir_recon")}
            r.update(candidates=n_cand, vertex=sh[layout]["vertex"].tolist(),
                     dir_true=prim[i, :3].tolist(), dir_recon=p["kernel"]["pred"][i, :3].tolist())
            results.setdefault(name, {})[layout] = r
            log.info("  %-9s %s", layout, json.dumps({k: v for k, v in r.items() if k not in ("vertex", "dir_true", "dir_recon")}))
        shows[name] = sh

    main_layout = args.layouts[0]
    overview = dict(list(shows.items())[:4])
    plots.save(figure(run, [(name, sh, dets[main_layout], main_layout) for name, sh in overview.items()], rng),
               out, f"overview_{main_layout}", log)
    for k, (name, sh) in enumerate(overview.items()):
        plots.save(figure(run, [(name, sh, dets[l], l) for l in args.layouts], rng),
                   out, f"event_{k}_{sh[main_layout]['index']}", log)
    for k, (name, sh) in enumerate(shows.items()):
        slug = name.replace(" ", "_").replace(",", "").replace(".", "p")
        plots.save(combined_figure(run, sh, dets, args.layouts[::-1], rng), out,
                   f"shower_{k:02d}_{slug}_{sh[main_layout]['index']}", log)
    logs.save(out, {"n_events": len(ev), "threshold": LAYOUT_THRESHOLD, "checks": checks, "showers": results}, args, log)


if __name__ == "__main__":
    main()
