"""Alternative Stage 4: optimize the layout against the plane-aware KERNEL
directly — the raw shower physics — instead of through the frozen FNN
surrogate `04_optimize_lbfgs_ensemble.py` backprops through.

docs/THEORY.md SS11.3 ("The surrogate artifact gap") measured the mechanism this
script exists to sidestep: optimizing through the surrogate manufactures an
artifact gap of +134 U on one layout (surrogate-U ~170 vs true-U ~36), because
the recon was trained to invert a smooth learned mean-field the surrogate
produces, not the sparse, genuinely stochastic response the kernel does. Four
facts from that section make this script cheap to try:

  1. A full-corpus kernel FORWARD pass is only ~6-10x the surrogate's own cost
     (~5 TFLOP, seconds on an A100) — it manufactures the +134 artifact at
     ~1/6 the kernel's price. A backward pass costs more, but the multiplier
     stays modest.
  2. The kernel (`modules/showers/kernel.py`'s `GetCounts_planeaware`) is
     ALREADY pure, differentiable torch — `compute_labels_batch`'s blanket
     `@torch.no_grad()` was THE ONLY thing blocking exact
     d(U)/d(East, North). It now takes `differentiable=True`
     (`modules/data/dataset_builder.py`) and every other caller (Step 1's
     label build, `aleatoric_floor.py`, `KernelDualLabels`'s scoring use in
     `true_utility.py`) is unaffected — default False keeps them no_grad.
  3. The adapter already existed: `KernelDualLabels` in `plots/layouts/
     true_utility.py` has the frozen-surrogate's exact
     `(primary_batch, xy_batch) -> (B, n_det, 2)` call signature, so it drops
     straight into the UNMODIFIED `objective.utility_of_xy` recon stays
     frozen; only the label source changes.
  4. Recon was trained on kernel labels in `kernel`/`mix` mode
     (`--recon-input`), so the OOD direction THEORY SS11.3 originally worried
     about ("recon fed surrogate mean-fields it wasn't trained on") is not
     automatic here if the checkpoint used one of those modes — check
     `recon.pt`'s own `config["recon_input"]` before trusting the numbers.

WHAT THIS IS NOT: `04_optimize_lbfgs_ensemble.py`'s full multi-restart,
Adam-then-L-BFGS, Hungarian-aligned ensemble over the ENTIRE main corpus,
resampled fresh every step. That is not reachable here for a structural
reason, not a laziness one: the kernel needs materialized shower POINT CLOUDS
(`compute_labels_batch(clouds, ...)`), not a `(primary, xy)` row from
`primary.pt` — so this script is bounded to a FIXED batch of pre-generated,
already-placed events from the HELD-OUT corpus (`true_utility.py`'s
`load_events`), reused every step, and a single Adam chain rather than an
L-BFGS sweep (a strong-Wolfe line search calls the closure several times PER
step, multiplying an already several-x-costlier-than-the-surrogate forward+
backward that many times over). Treat this as the "does kernel-in-the-loop
optimization even move the needle" experiment THEORY SS11.3 calls for, not a
drop-in replacement for the production optimizer.

MEMORY: `KernelDualLabels`'s own docstring measures ~4.79 GiB per 512 events
at ~25k points/event for a forward pass alone; a backward pass roughly
doubles that (autograd keeps the forward activations live). Defaults here
are deliberately small (`--n-events 64 --chunk 8`) — raise them only after
`--time-one-step` (below) says the memory/time cost is acceptable on your
node.

Usage:
    # THEORY SS11.3's "cheap experiment #1": time one fwd+bwd minibatch and
    # exit, no optimization, before committing to a full run.
    python scripts/04_optimize_lbfgs_kernel.py --time-one-step

    # A real (but short, single-chain) run:
    python scripts/04_optimize_lbfgs_kernel.py --n-events 64 --chunk 8 --epochs 200
"""
import argparse
import importlib.util
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from _pathfix import V6_ROOT  # noqa: F401 — idempotent, registers v6 root

import torch

import modules  # noqa: F401 — package import; keeps modules on the path
from modules.constants import (
    N_DETECTORS, GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, RECON_FOLDER, OPT_FOLDER,
    LOG_E_MIN, LOG_E_MAX, T_LOG_SCALE, KERNEL_INTERP_K, SIGMA_SPATIAL,
)
from modules.data.dataset_builder import compute_labels_batch
from modules.showers.kernel import adaptive_bandwidths
from modules.layouts import LearnableXY
from modules.geometry import load_tr_mountain, SurfaceUpMap
from modules.geometry import project_to_mountain_ne, sample_initial_layout_ne
from modules.optimize import utility_of_xy
from modules.surrogates import build_recon_from_ckpt

# plots/layouts/true_utility.py has no digit prefix, so it imports normally as
# a namespace-package module (no plots/layouts/__init__.py needed in Py3).
from plots.layouts.true_utility import load_events, KernelDualLabels

# plots/training/04_analytic_recon.py starts with a digit, so it cannot be
# `import`ed by name -- loaded by path, the same trick 03_recon_diagnostics.py
# uses for 02_nn_target_vs_pred.py. Only its pure math (weighted_plane_fit,
# perp_distance, weighted_centroid, fit_ldf_poisson) is reused here, NOT its
# main()/aggregate reporting (that file's A0 R2 aggregation has a known,
# separate, not-yet-fixed bug with near-zero-variance outlier events --
# unrelated to the per-event math below, which is unit-tested and used as-is).
_spec = importlib.util.spec_from_file_location(
    "_analytic_recon", os.path.join(V6_ROOT, "plots", "training", "04_analytic_recon.py"))
_analytic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_analytic)

# plots/lib/opt_plotting.py, loaded by path the same way -- reused here only
# for plot_density_heatmap, the same function 04_optimize_lbfgs_ensemble.py
# calls for its layout_density.png.
_plt_spec = importlib.util.spec_from_file_location(
    "opt_plotting", os.path.join(V6_ROOT, "plots", "lib", "opt_plotting.py"))
_opt_plotting = importlib.util.module_from_spec(_plt_spec)
_plt_spec.loader.exec_module(_opt_plotting)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 4200


class CachedKernel(KernelDualLabels):
    """KernelDualLabels that searches neighbours once. The interpolated kernel's
    bandwidths depend only on the (fixed) clouds, so each species/chunk slice's
    widths are computed on first use and reused every epoch. Output is identical
    to KernelDualLabels; only the per-step neighbour search is skipped."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._bw = {}

    def _labels(self, clouds, e_det, n_det):
        key = (clouds.data_ptr(), tuple(clouds.shape))
        if key not in self._bw:
            with torch.no_grad():
                self._bw[key] = (adaptive_bandwidths(clouds, KERNEL_INTERP_K, SIGMA_SPATIAL)
                                 if KERNEL_INTERP_K else None)
        E, T = compute_labels_batch(clouds, e_det, n_det, self.surface,
                                    differentiable=self.differentiable,
                                    bandwidth=self._bw[key])
        return torch.stack([torch.log1p(E), torch.log1p(T * T_LOG_SCALE)], dim=-1)


def plot_layout_density(best_x, best_y, mountain, out_dir):
    """layout_density.png for a SINGLE layout (K=1) -- this script optimizes
    one chain, not an ensemble, so there is no Hungarian-aligned set of runs
    to build a real cross-run density from. `plot_density_heatmap` accepts
    that degenerately (aligned_xy of shape (1, n_det, 2)): the heatmap
    collapses to per-detector Gaussian blobs around this one layout's
    positions, still useful as the same visual format the ensemble
    optimizer's runs use, not as a measure of run-to-run variance.

    Builds its own CPU SurfaceUpMap rather than reusing main()'s (possibly
    CUDA) `surface` -- plotting is the last, read-only step, and
    04_optimize_lbfgs_ensemble.py's own density call does the same for the
    same reason (grid_sample needs the surface and the (x, y) it's called
    with on the same device)."""
    import numpy as np
    surface = SurfaceUpMap.from_mountain(mountain).to("cpu")
    x_np = best_x.cpu().numpy()
    y_np = best_y.cpu().numpy()
    aligned = np.stack([x_np, y_np], axis=-1)[None, :, :]
    _opt_plotting.plot_density_heatmap(
        aligned, x_np, y_np, mountain,
        os.path.join(out_dir, "layout_density.png"), surface=surface,
        count_word="K", count_suffix=" run (single chain, not an ensemble)")


def load_recon(recon_dir, device):
    """Frozen DeepSets recon only — this script never loads the FNN surrogate
    at all (unlike `04_optimize_lbfgs_ensemble.py`'s `load_models`, which
    loads both), since the whole point is not depending on it."""
    ckpt = torch.load(os.path.join(recon_dir, "recon.pt"),
                      map_location=device, weights_only=False)
    recon = build_recon_from_ckpt(ckpt, N_DETECTORS, device)
    cfg = ckpt.get("config", {})
    print(f"[load] recon.pt  model={cfg.get('model_type', 'mlp')}  "
          f"epoch={ckpt.get('epoch', '?')}  val={ckpt.get('val_total', '?')}  "
          f"recon_input={cfg.get('recon_input', '?')}  <- {recon_dir}")
    if cfg.get("recon_input") == "sample":
        print("    [warn] this recon was trained on --recon-input sample "
              "(surrogate draws only) -- see module docstring point 4: it "
              "was never shown kernel-ground-truth inputs, so numbers from "
              "this script may reflect that mismatch rather than the "
              "kernel-in-the-loop idea being tested.")
    return recon


def init_layout(mountain, scheme, device):
    e, n = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme=scheme)
    e = torch.as_tensor(e, dtype=torch.float32)
    n = torch.as_tensor(n, dtype=torch.float32)
    e, n = project_to_mountain_ne(mountain, e, n)
    return e.to(device), n.to(device)


def time_one_step(utility, kernel_fnn, recon, primary_batch, xy_module,
                  analytic_min_fired):
    """THEORY SS11.3's cheap experiment #1: one fwd+bwd, timed, no optimizer
    step. Answers "is kernel-in-the-loop even tractable here" before anyone
    commits to a multi-epoch run."""
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    x, y = xy_module()
    if utility == "analytic":
        U, r, parts = analytic_utility_of_xy(x, y, primary_batch, kernel_fnn,
                                             min_fired=analytic_min_fired)
    else:
        U, r, parts = utility_of_xy(x, y, primary_batch, kernel_fnn, recon)
    (-U).backward()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[time-one-step] U={float(U):.3f}  fwd+bwd wall time={dt:.2f}s")
    if DEVICE.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[time-one-step] peak CUDA memory={peak_gb:.2f} GB")
    print(f"[time-one-step] extrapolated: {dt * 1000:.0f}s for 1000 epochs "
          f"at this (n_events, chunk) -- scale --n-events/--chunk from here, "
          f"not from a hunch")


@torch.no_grad()
def analytic_eval(kernel_fnn, primary_batch, e_det, n_det, surface, device,
                  min_fired=4, ldf_steps=200, ldf_r_ref=100.0):
    """Score a layout with the ANALYTIC (plane-front + LDF) reconstruction
    instead of the frozen neural recon -- a second, reconstruction-
    independent read on whether this layout is actually better, following
    the same logic docs/stage3-analytic-validation.md's Part A applies to
    Stage 3: the frozen recon sitting in this script's own optimization loop
    could itself be exploited the way THEORY SS11.3 documents for the
    surrogate, so a metric that owes it nothing is worth having alongside U.

    Reuses `plots/training/04_analytic_recon.py`'s per-event math verbatim
    (weighted_plane_fit / perp_distance / weighted_centroid / fit_ldf_poisson
    -- unit-tested there against synthetic data), NOT that file's aggregate
    R2 reporting, which has a separate, known, not-yet-fixed bug with
    near-zero-variance outlier events. Reports plain angular error (degrees)
    and an UNCALIBRATED energy correlation instead: with only `B = n_events`
    events available (default 64, the same fixed batch the optimizer itself
    uses) there usually is not enough left for a proper train/val calibration
    split the way the standalone script does it, so log_A's correlation with
    true log_e is reported directly rather than mapped through a fitted
    (m, b) -- a real number, just not a calibrated dex-RMSE.

    No gradient here (@torch.no_grad()): this is a REPORTING pass on a given
    layout, not part of the optimized objective.
    """
    A = _analytic
    B = primary_batch.shape[0]
    xy_batch = torch.stack([e_det, n_det], dim=-1).unsqueeze(0).expand(B, -1, -1)
    pred = kernel_fnn(primary_batch, xy_batch)                       # (B, n_det, 2), log-encoded
    E_raw = torch.expm1(pred[..., 0]).clamp(min=0)
    T_raw = torch.expm1(pred[..., 1]).clamp(min=0) / T_LOG_SCALE

    up = surface(n_det, e_det)                                        # (n_det,)
    pos = torch.stack([e_det, n_det, up], dim=-1).unsqueeze(0).expand(B, -1, -1)

    fired = T_raw > 0
    n_fired = fired.sum(dim=-1)
    ok = n_fired >= min_fired
    w = torch.where(fired, E_raw, torch.zeros_like(E_raw))

    t0, s, resid, cond = A.weighted_plane_fit(pos, T_raw, w)
    s_norm = s.norm(dim=-1).clamp(min=1e-30)
    n_hat = s / s_norm.unsqueeze(-1)

    true_dir = primary_batch[:, :3]
    true_dir = true_dir / true_dir.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    if not ok.any():
        return dict(qualifying_frac=0.0, mean_angular_error_deg=None,
                    log_e_pearson_r=None, n_events=B)
    sign = torch.sign((n_hat[ok] * true_dir[ok]).sum(-1).mean())
    if sign == 0:
        sign = torch.tensor(1.0, device=device)
    n_hat = n_hat * sign
    cos = (n_hat * true_dir).sum(-1).clamp(-1.0, 1.0)
    ang_err_deg = torch.rad2deg(torch.arccos(cos))

    core = A.weighted_centroid(pos, w)
    r_perp = A.perp_distance(pos, core, n_hat)
    log_A, beta = A.fit_ldf_poisson(r_perp, E_raw, steps=ldf_steps, r_ref=ldf_r_ref)
    true_log_e = primary_batch[:, 3] * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN

    ok_e = ok & torch.isfinite(log_A)
    if ok_e.sum() >= 2:
        r_e = torch.corrcoef(torch.stack([log_A[ok_e], true_log_e[ok_e]]))[0, 1]
        r_e = float(r_e)
    else:
        r_e = None

    return dict(
        qualifying_frac=float(ok.float().mean()),
        mean_angular_error_deg=float(ang_err_deg[ok].mean()),
        median_angular_error_deg=float(ang_err_deg[ok].median()),
        log_e_pearson_r=r_e,
        n_events=B,
    )


def analytic_utility_of_xy(x_det, y_det, primary_batch, kernel_fnn,
                           min_fired=4):
    """The `--utility analytic` objective: DIRECTION ONLY, from
    `weighted_plane_fit` run on the kernel's differentiable output, in place
    of `utility_of_xy`'s frozen-recon-based U.

    Deliberately does NOT include `04_analytic_recon.py`'s LDF energy term
    (`fit_ldf_poisson`). That function runs its OWN internal Adam loop to fit
    (log_A, beta) per call (see its docstring) -- differentiating an OUTER
    optimizer's gradient through that INNER optimizer means either unrolling
    every one of its steps (multiplies this already several-x-costlier-than-
    the-surrogate forward+backward by `ldf_steps`, ~200) or an implicit-
    function-theorem shortcut (a real derivation, not a small change). This
    is scope for a follow-up, not an oversight -- `--utility analytic` is
    direction-only until that's done.

    Returns (U, ok_frac, parts) mirroring `utility_of_xy`'s (U, r, parts)
    contract so the epoch loop below needs no branching beyond which
    function it calls: `ok_frac` (fraction of the batch with >= min_fired
    fired detectors and a well-posed plane fit) stands in for `r`
    (reconstructability) -- a different quantity, not a substitute, but the
    loop only ever calls `.mean()` on it. `parts["u_theta"]` carries the real
    signal (mean cosine-similarity to the true direction, maximized directly
    rather than via `arccos`, whose gradient blows up near perfect alignment);
    `u_phi`/`u_e` are zero tensors so the loop's existing per-part print line
    doesn't need a separate code path -- they are NOT a phi or energy score.
    """
    A = _analytic
    B = primary_batch.shape[0]
    xy_batch = torch.stack([x_det, y_det], dim=-1).unsqueeze(0).expand(B, -1, -1)
    pred = kernel_fnn(primary_batch, xy_batch)                       # (B, n_det, 2), log-encoded
    E_raw = torch.expm1(pred[..., 0]).clamp(min=0)
    T_raw = torch.expm1(pred[..., 1]).clamp(min=0) / T_LOG_SCALE

    up = kernel_fnn.surface(y_det, x_det)                             # (n_det,)
    pos = torch.stack([x_det, y_det, up], dim=-1).unsqueeze(0).expand(B, -1, -1)

    fired = T_raw > 0
    n_fired = fired.sum(dim=-1)
    ok = n_fired >= min_fired
    w = torch.where(fired, E_raw, torch.zeros_like(E_raw))

    t0, s, resid, cond = A.weighted_plane_fit(pos, T_raw, w)
    n_hat = s / s.norm(dim=-1, keepdim=True).clamp(min=1e-30)

    true_dir = primary_batch[:, :3]
    true_dir = true_dir / true_dir.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    cos = (n_hat * true_dir).sum(-1).clamp(-1.0, 1.0)                 # sign-ambiguous, see below

    ok_f = ok.float()
    if ok_f.sum() < 1:
        # No qualifying event this batch -- a zero, differentiable-w.r.t.-
        # nothing loss rather than a NaN from dividing by an empty sum.
        u_theta = torch.zeros((), device=x_det.device)
    else:
        # weighted_plane_fit's normal has a sign ambiguity (a plane fit
        # cannot tell "up the front" from "down the front" on its own) --
        # `analytic_eval` resolves it with one batch-level majority-sign
        # vote against ground truth for REPORTING. Using |cos| here instead
        # keeps the optimized loss itself free of that non-differentiable,
        # data-dependent branch (the sign vote would have to be taken inside
        # the loss every step, on a mask that changes as the layout moves).
        u_theta = (cos.abs() * ok_f).sum() / ok_f.sum()

    U = u_theta
    parts = dict(u_theta=u_theta, u_phi=torch.zeros((), device=x_det.device),
                u_e=torch.zeros((), device=x_det.device))
    return U, ok_f, parts


def _print_analytic_eval(tag, ev):
    if ev["qualifying_frac"] == 0.0:
        print(f"[analytic-eval:{tag}] no qualifying events (need >= min-fired "
              f"detectors with a well-posed plane fit)")
        return
    r_e = f"{ev['log_e_pearson_r']:.3f}" if ev["log_e_pearson_r"] is not None else "n/a"
    print(f"[analytic-eval:{tag}] qualifying={ev['qualifying_frac']:.1%}  "
          f"angular error mean={ev['mean_angular_error_deg']:.1f} deg "
          f"median={ev['median_angular_error_deg']:.1f} deg  "
          f"log_A~log_e pearson_r={r_e}  (uncalibrated, see analytic_eval docstring)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", choices=("grid", "center"), default="center")
    ap.add_argument("--n-events", type=int, default=64,
                    help="held-out events the kernel scores every step "
                         "(FIXED for the whole run -- see module docstring)")
    ap.add_argument("--chunk", type=int, default=8,
                    help="KernelDualLabels chunk size -- bounds peak memory, "
                         "not accuracy (the split is exact, see its docstring)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=100.0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--recon-folder", default=RECON_FOLDER + "_deepsets")
    ap.add_argument("--utility", choices=("recon", "analytic"), default="recon",
                    help="'recon' (default): optimize against the frozen "
                         "DeepSets recon, as today. 'analytic': optimize "
                         "against the differentiable plane-front direction "
                         "fit instead (`analytic_utility_of_xy` -- direction "
                         "only, no LDF/energy term, no recon.pt dependency; "
                         "see that function's docstring for why energy is "
                         "out of scope here). The two are not comparable "
                         "numbers -- U's scale and meaning differ.")
    ap.add_argument("--out-suffix", type=str, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--time-one-step", action="store_true",
                    help="time one fwd+bwd at the given --n-events/--chunk "
                         "and exit -- no optimization, nothing written")
    ap.add_argument("--analytic-min-fired", type=int, default=4,
                    help="min fired detectors for the analytic plane fit -- "
                         "see analytic_eval's docstring")
    ap.add_argument("--analytic-ldf-steps", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=250,
                    help="save optimizer state every N epochs; a rerun with the "
                         "same --out-suffix resumes from it (requeue-safe)")
    ap.add_argument("--skip-analytic-eval", action="store_true",
                    help="skip the analytic (plane-front + LDF) evaluation "
                         "of the init/best layouts -- it never affects the "
                         "optimization, only what gets printed/saved")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    out_dir = OPT_FOLDER + "_lbfgs_kernel" + (f"_{args.out_suffix}" if args.out_suffix else "")

    print("=" * 72)
    print("v6/04_optimize_lbfgs_kernel.py — layout optimization against the "
          "KERNEL, not the surrogate")
    print("=" * 72)
    print(f"device={DEVICE}  n_events={args.n_events}  chunk={args.chunk}  "
          f"epochs={args.epochs}  utility={args.utility}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                               east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                               n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)

    # 'analytic' never touches the frozen recon -- that is the whole point of
    # this utility variant (a reconstruction-independent objective) -- so
    # skip the load rather than fetch a checkpoint main() then ignores.
    recon = load_recon(args.recon_folder, DEVICE) if args.utility == "recon" else None

    t0 = time.time()
    species_clouds, B, n_pairs, primary_batch = load_events(args.n_events, DEVICE, mountain)
    print(f"[load] {B} held-out events (of {n_pairs} available) in "
          f"{time.time() - t0:.1f}s")
    kernel_fnn = CachedKernel(species_clouds, surface, DEVICE,
                             chunk=args.chunk, differentiable=True,
                             expected_primary=primary_batch)

    e_init, n_init = init_layout(mountain, args.init, DEVICE)
    xy_module = LearnableXY(e_init, n_init, device=str(DEVICE)).to(DEVICE)

    if args.time_one_step:
        time_one_step(args.utility, kernel_fnn, recon, primary_batch,
                     xy_module, args.analytic_min_fired)
        return

    analytic_init = None
    if not args.skip_analytic_eval:
        analytic_init = analytic_eval(
            kernel_fnn, primary_batch, e_init, n_init, surface, DEVICE,
            min_fired=args.analytic_min_fired, ldf_steps=args.analytic_ldf_steps)
        _print_analytic_eval("init", analytic_init)

    optimizer = torch.optim.Adam(xy_module.parameters(), lr=args.lr)
    log = []
    best_u = -float("inf")
    best_x = e_init.detach().cpu().clone()
    best_y = n_init.detach().cpu().clone()
    start = 0

    # Resume after a requeue: same out_dir means same run.
    state_path = os.path.join(out_dir, "opt_state.pt")
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=DEVICE, weights_only=False)
        xy_module.load_state_dict(st["xy"])
        optimizer.load_state_dict(st["optimizer"])
        start, log, best_u = st["epoch"], st["log"], st["best_u"]
        best_x, best_y = st["best_x"], st["best_y"]
        print(f"[resume] {state_path} at epoch {start}, best U={best_u:+.3f}")
    t_loop = time.time()

    for epoch in range(start, args.epochs):
        x_det, y_det = xy_module()
        if args.utility == "analytic":
            U, r, parts = analytic_utility_of_xy(
                x_det, y_det, primary_batch, kernel_fnn,
                min_fired=args.analytic_min_fired)
        else:
            U, r, parts = utility_of_xy(x_det, y_det, primary_batch, kernel_fnn, recon)
        loss = -U

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(xy_module.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            e_cpu = xy_module.x.detach().cpu()
            n_cpu = xy_module.y.detach().cpu()
            e_new, n_new = project_to_mountain_ne(mountain, e_cpu, n_cpu)
            xy_module.x.data.copy_(e_new.to(DEVICE).to(xy_module.x.dtype))
            xy_module.y.data.copy_(n_new.to(DEVICE).to(xy_module.y.dtype))

        u_val = float(U.item())
        if u_val > best_u:
            best_u = u_val
            best_x = xy_module.x.detach().cpu().clone()
            best_y = xy_module.y.detach().cpu().clone()

        log.append(dict(epoch=epoch + 1, U=u_val, r_mean=float(r.mean().item()),
                        u_theta=float(parts["u_theta"].item()),
                        u_phi=float(parts["u_phi"].item()),
                        u_e=float(parts["u_e"].item())))
        if epoch == start or (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            print(f"  [{epoch+1:4d}/{args.epochs}] U={u_val:+.3f}  "
                  f"(u_theta={float(parts['u_theta']):.3f} "
                  f"u_phi={float(parts['u_phi']):.3f} "
                  f"u_e={float(parts['u_e']):.3f})  "
                  f"r={log[-1]['r_mean']:.3f}  "
                  f"{(time.time() - t_loop) / (epoch + 1 - start):.2f}s/epoch", flush=True)
        if (epoch + 1) % args.ckpt_every == 0:
            os.makedirs(out_dir, exist_ok=True)
            torch.save(dict(xy=xy_module.state_dict(), optimizer=optimizer.state_dict(),
                            epoch=epoch + 1, log=log, best_u=best_u,
                            best_x=best_x, best_y=best_y),
                       state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)

    print(f"[done] best U={best_u:+.3f}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(dict(x=best_x, y=best_y), os.path.join(out_dir, "layout_best.pt"))
    with open(os.path.join(out_dir, "optimize_log.json"), "w") as f:
        json.dump(dict(config=vars(args), log=log, best_u=best_u), f, indent=2)
    print(f"[write] {out_dir}/layout_best.pt, optimize_log.json")

    try:
        plot_layout_density(best_x, best_y, mountain, out_dir)
        print(f"[write] {out_dir}/layout_density.png")
    except Exception as exc:
        print(f"[plot] layout_density skipped ({exc!r})")

    if not args.skip_analytic_eval:
        analytic_best = analytic_eval(
            kernel_fnn, primary_batch, best_x.to(DEVICE), best_y.to(DEVICE),
            surface, DEVICE, min_fired=args.analytic_min_fired,
            ldf_steps=args.analytic_ldf_steps)
        _print_analytic_eval("best", analytic_best)

        # Own, clearly-named subfolder next to layout_best.pt/optimize_log.json --
        # this is a DIFFERENT metric family (reconstruction-independent, from
        # plots/training/04_analytic_recon.py's math) than the neural-recon-based
        # U this run optimized, so it does not belong folded into optimize_log.json.
        analytic_dir = os.path.join(out_dir, "analytic_eval")
        os.makedirs(analytic_dir, exist_ok=True)
        with open(os.path.join(analytic_dir, "analytic_eval.json"), "w") as f:
            json.dump(dict(config=vars(args), init=analytic_init, best=analytic_best),
                      f, indent=2)
        print(f"[write] {analytic_dir}/analytic_eval.json")


if __name__ == "__main__":
    main()
