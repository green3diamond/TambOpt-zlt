"""Per-event composite utility vs. primary tau energy, for a saved optimized layout.

`objective.utility_of_xy` returns a single batch-averaged U — that's what the
optimizer selects on, but it hides whether the layout helps uniformly across the
energy range or is buying its average from one end. This script re-derives the
SAME composite (identical weights, caps, thresholds — `modules/optimize/objective.py`'s
W_THETA/W_PHI/W_E/W_DIV, CAP_*, LAYOUT_THRESHOLD, RECONSTRUCT_THRESHOLD,
TAU_LAYOUT, TAU_RECONSTRUCT) but stops short of the final `torch.mean` inside
`U_E`/`U_angle`, keeping one U value per event instead. Since mean is linear,
`mean(per_event_U) == utility_of_xy(...)`'s scalar U exactly — checked at
runtime against the unmodified function as a correctness guard, not just an
assertion in a comment.

Reuses `plots/layouts/true_utility.py`'s event loader (held-out corpus, unseen by
Steps 1-4) and model loader, so the numbers here are on the same footing as
that script's baseline/optimized comparison — just broken out per event and
plotted against each event's true tau energy instead of collapsed to one number.

Output: `utility_vs_energy.png` next to each `layout_best.pt` passed via
`--run-dir` (mirrors `plots/layouts/replot_optimize_curves.py`'s convention of writing
back into the run's own directory).

    cd TambOpt
    python plots/layouts/utility_vs_energy.py --run-dir "RUN_DIR_A" "RUN_DIR_B"
    python plots/layouts/utility_vs_energy.py --run-dir "RUN_DIR" --n-events 20000
"""
import argparse
import os
import sys

# `_HERE` is this file's own directory; `_V6` the repo root, found by walking up
# to the _pathfix.py marker instead of counting parents. Counting is what broke
# the old plots/single_species/ scripts: written for plots/*.py, they resolved
# the "repo root" to plots/ and could not import `modules` at all.
_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = _HERE
while _V6 != os.path.dirname(_V6) and not os.path.exists(os.path.join(_V6, "_pathfix.py")):
    _V6 = os.path.dirname(_V6)
for _p in (_V6, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modules  # noqa: F401 — package import; keeps modules on the path
from modules.geometry import project_to_mountain_ne
from modules.geometry import load_tr_mountain
from modules.geometry import SurfaceUpMap
from modules.optimize import (
    primary_to_physical_labels, load_models,
    W_THETA, W_PHI, W_E, W_DIV, CAP_THETA, CAP_PHI, CAP_E,
    LAYOUT_THRESHOLD, RECONSTRUCT_THRESHOLD, TAU_LAYOUT, TAU_RECONSTRUCT,
)
from modules.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
)
# modules/__init__ injected the v3 (`modules`) path on package import.
from modules.optimize import reconstructability
from modules.optimize.utility import _soft_cap
from true_utility import (
    load_events, utility_of_xy, KernelDualLabels, grid_layout, center_layout,
)


def _snap(mountain, e, n):
    e, n = project_to_mountain_ne(mountain, e.float().reshape(-1), n.float().reshape(-1))
    return e.float(), n.float()


def load_layout(path, mountain):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    e, n = (raw["x"], raw["y"]) if isinstance(raw, dict) else (raw[:, 0], raw[:, 1])
    return _snap(mountain, e, n)


@torch.no_grad()
def utility_per_event(x_det, y_det, primary_batch, fnn, recon):
    """(U_i, E_true_gev, r) — per-event decomposition of `utility_of_xy`.

    Mirrors that function line for line up to the point where `U_E`/`U_angle`
    collapse their `r * inv_err` term with `torch.mean`; here that reduction is
    simply omitted, so every return is (B,) instead of a scalar. Same weights,
    same soft caps, same reconstructability thresholds — nothing about the
    objective changes, only whether the batch dimension survives.
    """
    B = primary_batch.shape[0]
    xy_per_det = torch.stack([x_det, y_det], dim=-1)
    xy_batch = xy_per_det.unsqueeze(0).expand(B, -1, -1)

    pred_ET = fnn(primary_batch, xy_batch)
    E_pred_det = pred_ET[..., 0]
    T_pred_det = pred_ET[..., 1]
    recon_feats = torch.stack(
        [xy_batch[..., 0], xy_batch[..., 1], E_pred_det, T_pred_det], dim=-1)
    pred = recon(recon_feats)
    E_pred_phys, theta_pred, phi_pred = primary_to_physical_labels(pred)
    E_pred_phys = E_pred_phys.clamp(min=1.0)
    E_true, theta_true, phi_true = primary_to_physical_labels(primary_batch)

    r = reconstructability(
        torch.expm1(E_pred_det), layout_threshold=LAYOUT_THRESHOLD,
        tau_layout=TAU_LAYOUT, reconstruct_threshold=RECONSTRUCT_THRESHOLD,
        tau_reconstruct=TAU_RECONSTRUCT)

    inv_theta = _soft_cap(1.0 / ((theta_pred - theta_true) ** 2 + .001), CAP_THETA)
    inv_phi   = _soft_cap(1.0 / ((phi_pred   - phi_true)   ** 2 + .001), CAP_PHI)
    inv_e     = _soft_cap(1.0 / ((torch.log10(E_pred_phys)
                                  - torch.log10(E_true)) ** 2 + .01), CAP_E)

    U_i = (W_THETA * r * inv_theta + W_PHI * r * inv_phi
          + W_E * r * inv_e) / W_DIV
    return U_i, E_true, r


def _binned(log_e, U, edges, n_bins, min_count):
    """(centre, mean, std) per energy bin, dropping bins under `min_count`."""
    idx = np.clip(np.digitize(log_e, edges[1:-1]), 0, n_bins - 1)
    ctr, mean, sd = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < min_count:
            continue
        ctr.append(0.5 * (edges[b] + edges[b + 1]))
        mean.append(float(U[m].mean()))
        sd.append(float(U[m].std()))
    return map(np.asarray, (ctr, mean, sd))


def _plot(ax, log_e, U_opt, U_base, title, base_label="baseline",
          n_bins=40, min_count=20):
    """Hexbin density of the OPTIMIZED per-event U over log10(E_tau), linear
    colour clipped at p95 (the convention `plots/training/02_nn_target_vs_pred.py
    ::_scatter` uses for the same reason: counts span orders of magnitude, so a
    plain linear scale lets the single densest cell own the whole ramp), with the
    binned mean +/- std curve for BOTH the optimized and the baseline layout on
    top. The overlay is the point: it answers whether optimization moved
    per-event utility at all, and where in energy it did, rather than showing one
    curve with nothing to compare it against.

    Only the optimized layout gets a hexbin — two overlapping density maps are
    unreadable, and the baseline's job here is to be a reference line.

    The two curves are distinguished by linestyle and marker as well as colour,
    so they stay separable in greyscale and for colour-vision deficiency."""
    lo, hi = float(log_e.min()), float(log_e.max())
    y_hi = float(np.percentile(np.concatenate([U_opt, U_base]), 99.5))
    hb = ax.hexbin(log_e, np.clip(U_opt, None, y_hi), gridsize=70, cmap="viridis",
                   mincnt=1, extent=(lo, hi, 0.0, y_hi))
    counts = np.asarray(hb.get_array())
    if counts.size:
        hb.set_clim(0.0, float(np.percentile(counts, 95.0)))
    plt.colorbar(hb, ax=ax, label="count  (linear, clipped at p95)",
                 pad=0.02, fraction=0.046)

    edges = np.linspace(lo, hi, n_bins + 1)

    c_b, m_b, s_b = _binned(log_e, U_base, edges, n_bins, min_count)
    ax.fill_between(c_b, m_b - s_b, m_b + s_b, color="#00e5ff", alpha=0.15)
    ax.plot(c_b, m_b, color="#00e5ff", marker="s", ms=3.5, lw=2.0, ls="--",
            label=f"{base_label}  (±1 std)")

    c_o, m_o, s_o = _binned(log_e, U_opt, edges, n_bins, min_count)
    ax.fill_between(c_o, m_o - s_o, m_o + s_o, color="#ff2d95", alpha=0.18)
    ax.plot(c_o, m_o, color="#ff2d95", marker="o", ms=3.5, lw=2.0, ls="-",
            label="optimized  (±1 std)")

    ax.set_xlim(lo, hi); ax.set_ylim(0.0, y_hi)
    # `log_e` is log10 of the caller's GeV energies with no unit shift applied,
    # so these tick numbers ARE log10(E/GeV) -- the label states what the numbers
    # already are, not a conversion still owed.
    ax.set_xlabel(r"$\log_{10}(E_\tau \,/\, \mathrm{GeV})$")
    ax.set_ylabel("per-event utility U")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def score_run(run_dir, prim, labeller, recon, mountain, device,
              baseline_xy, base_label, label_src, formats=("png",)):
    """Per-event U vs energy for `run_dir`'s optimized layout AND the baseline,
    both scored through the SAME `labeller` — pass the kernel one and both curves
    are on real physics, which is the only way the two are comparable to the
    numbers plots/layouts/true_utility.py reports."""
    layout_path = os.path.join(run_dir, "layout_best.pt")
    e_o, n_o = load_layout(layout_path, mountain)
    x, y = e_o.to(device), n_o.to(device)
    xb, yb = baseline_xy

    U_i, E_true, r = utility_per_event(x, y, prim, labeller, recon)
    U_b, _, r_b    = utility_per_event(xb, yb, prim, labeller, recon)
    U_i_np, U_b_np = U_i.cpu().numpy(), U_b.cpu().numpy()
    # E_true is in GeV (objective.primary_to_physical_labels' own docstring names
    # its return E_GeV) and is plotted in GeV unshifted, matching the units the
    # rest of the pipeline states its energy band in (constants.LOG_E_MIN/MAX are
    # log10(E/GeV), and this corpus spans ~1e5-1e8 GeV).
    log_e = torch.log10(E_true).cpu().numpy()

    # Correctness guard: mean is linear, so this must reproduce utility_of_xy's
    # own scalar U to floating-point precision. A mismatch means the per-event
    # decomposition above drifted from the composite it claims to mirror.
    # sample=False: utility_per_event above always uses the labeller's MEAN
    # (line ~92, plain call, no forward_sample) -- utility_of_xy defaults to a
    # stochastic draw for production/optimizer use, which would make this guard
    # compare against a different random realization every call instead of the
    # same deterministic reference this function computes. (With the kernel
    # labeller there is no sampling either way; the flag is then a no-op.)
    U_ref, _, _ = utility_of_xy(x, y, prim, labeller, recon, sample=False)
    U_ref = float(U_ref.item())
    U_mean, U_base = float(U_i_np.mean()), float(U_b_np.mean())
    print(f"[{os.path.basename(run_dir)}] labels={label_src}  "
          f"mean(per-event U) = {U_mean:.4f}  utility_of_xy U = {U_ref:.4f}  "
          f"(diff {abs(U_mean - U_ref):.2e}, should be ~0)")
    print(f"    {base_label:<22s} mean U = {U_base:.4f}  (mean r={float(r_b.mean()):.3f})")
    print(f"    {'optimized':<22s} mean U = {U_mean:.4f}  (mean r={float(r.mean()):.3f})"
          f"   ΔU = {U_mean - U_base:+.4f}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _plot(ax, log_e, U_i_np, U_b_np,
          f"{os.path.basename(run_dir)}  —  labels: {label_src}\n"
          f"N={U_i_np.size:,}   optimized U={U_mean:.3f}   "
          f"{base_label} U={U_base:.3f}   ΔU={U_mean - U_base:+.3f}",
          base_label=base_label)
    fig.tight_layout()
    out = os.path.join(run_dir, "utility_vs_energy.png")
    for fmt in formats:
        path = out if fmt == "png" else os.path.splitext(out)[0] + f".{fmt}"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"[save] {path}")
    plt.close(fig)
    return U_mean


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", nargs="+", required=True,
                    help="one or more optimizer run dirs, each containing layout_best.pt; "
                         "utility_vs_energy.png is written back into each")
    ap.add_argument("--n-events", type=int, default=5120,
                    help="fixed primary/cloud batch size, shared across all --run-dir "
                         "(same held-out events score every layout)")
    ap.add_argument("--labels", choices=("kernel", "surrogate"), default="kernel",
                    help="response source both curves are scored through. "
                         "'kernel' (default) = the real plane-aware kernel on the "
                         "held-out clouds, so the numbers are comparable to "
                         "true_utility.py's true-U; 'surrogate' = the stage-2 net, "
                         "which reproduces the optimizer's own (inflated) view.")
    ap.add_argument("--grid-layout", action="store_true",
                    help="use the grid layout as the baseline curve (default: "
                         "center), matching true_utility.py's flag of the same name")
    ap.add_argument("--kernel-chunk", type=int, default=128,
                    help="events per kernel call; memory only, result is identical")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("utility vs tau energy — per-event composite, held-out corpus")
    print("=" * 72)
    print(f"device : {device}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    clouds, B, n_pairs, prim = load_events(args.n_events, device, mountain)
    print(f"events : {B} of {n_pairs} pairs")

    fnn, recon = load_models(device)

    if args.labels == "kernel":
        surface = SurfaceUpMap.from_mountain(mountain).to(device)
        labeller = KernelDualLabels(clouds, surface, device, chunk=args.kernel_chunk,
                                    expected_primary=prim)
        print(f"labels : kernel  ({labeller.chunk} events/call)")
    else:
        labeller = fnn
        print("labels : surrogate")

    if args.grid_layout:
        e_b, n_b = grid_layout(mountain)
        base_label = "baseline grid"
    else:
        e_b, n_b = center_layout(mountain)
        base_label = "baseline center"
    baseline_xy = (e_b.to(device), n_b.to(device))
    print(f"baseline: {base_label}")

    for run_dir in args.run_dir:
        if not os.path.exists(os.path.join(run_dir, "layout_best.pt")):
            print(f"[skip] no layout_best.pt in {run_dir}")
            continue
        score_run(run_dir, prim, labeller, recon, mountain, device,
                  baseline_xy, base_label, args.labels)


if __name__ == "__main__":
    main()
