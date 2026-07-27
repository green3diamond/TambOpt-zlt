"""Kernel-direct layout optimization — the surrogate is out of the loop.

Same Adam-warm-start + L-BFGS-refine ensemble as `04_optimize_lbfgs_ensemble.py`,
but the label source inside `utility_of_xy` is the plane-aware KERNEL itself,
evaluated on the voxel-compacted, GPU-resident corpus (`modules_v6/voxel_corpus.py`):

  * Adam phase   — minibatches of PRIMARIES_PER_STEP events, kernel labels,
                   mountain projection each step (structure unchanged).
  * L-BFGS phase — the closure sweeps the ENTIRE corpus in chunks and
                   accumulates the exact full-corpus gradient (U is a
                   per-event mean, so chunked backward is exact). The
                   objective is deterministic BY CONSTRUCTION — the
                   strong-Wolfe line search is finally legitimate, with no
                   fixed-batch overfitting possible.

Because only the label source changes, "surrogate-U" and "true-U" are the same
number here: the artifact gap is structurally zero. The frozen DeepSets recon
is still in the objective (retraining it at the final layout is a follow-up).

Feasibility numbers (slurm-35271796): voxel fidelity R^2 > 0.999 at
delta = 31.25 m with the full corpus ~7.4 GiB resident; one full-corpus kernel
fwd+bwd ~135 s on full clouds, several-fold cheaper on voxels.

Artifacts (NEW directory per scheme — never the surrogate runs' folders):
``<OPT_FOLDER>_kernel_direct_{scheme}/`` with the same files as the surrogate
ensemble (layout_best.pt, layout_mean.pt, layouts_all.pt, optimize_log.json,
plots). Existing non-empty output dirs are refused without --overwrite.

    cd TambOpt/detector_optimization_v6
    python 04_optimize_kernel_direct.py --schemes grid --chains 2 \
        --adam-epochs 300 --lbfgs-iters 25        # smoke test
"""
import argparse
import importlib.util
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import modules_v6   # sys.path injection for v3 + v4
from modules_v6.constants import (
    N_DETECTORS,
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, RECON_FOLDER, OPT_FOLDER,
)
from modules.layout_optimization import LearnableXY
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.tr_geometry_ne import project_to_mountain_ne, sample_initial_layout_ne
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v6.opt_core import (
    utility_of_xy, align_to_reference, consecutive_cos_distance,
    W_THETA, W_PHI, W_E, W_PR, W_DIV,
    LAYOUT_THRESHOLD, RECONSTRUCT_THRESHOLD,
)
from modules_v6.reconstruction import build_recon_from_ckpt
from modules_v6.voxel_corpus import (
    VoxelCorpus, KernelVoxelLabels, build_voxel_corpus, voxel_cache_path,
    DEFAULT_DELTA,
)

_plt_spec = importlib.util.spec_from_file_location(
    "opt_plotting", os.path.join(_HERE, "plots", "opt_plotting.py"))
_plt = importlib.util.module_from_spec(_plt_spec); _plt_spec.loader.exec_module(_plt)


# ── Config ───────────────────────────────────────────────────────────────────
INIT_SCHEMES     = ("grid",)
OPT_DIR_TEMPLATE = OPT_FOLDER + "_kernel_direct_{scheme}"
RECON_DIR        = RECON_FOLDER + "_deepsets"

N_CHAINS            = 5
INIT_OVERDISP_SIGMA = 1000.0   # metres — per-restart init spread

# Adam warm-start (kernel minibatches)
N_ADAM_EPOCHS      = 1_500
PRIMARIES_PER_STEP = 256
ADAM_LR            = 1.0
GRAD_CLIP          = 100.0
ADAM_LOG_EVERY     = 100
GRAD_COS_WINDOW    = 10

# L-BFGS refinement — FULL corpus per closure, chunked (exact gradient)
LBFGS_MAX_ITER     = 100
LBFGS_LR           = 1.0
LBFGS_HISTORY_SIZE = 20
SLOT_BUDGET        = 4_000_000  # pack_chunks cap: n_events*(Vmax_e+Vmax_m) per chunk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_recon(device):
    ckpt = torch.load(os.path.join(RECON_DIR, "recon.pt"),
                      map_location=device, weights_only=False)
    recon = build_recon_from_ckpt(ckpt, N_DETECTORS, device)
    print(f"[load] recon.pt  model={ckpt.get('config', {}).get('model_type', 'mlp')}  "
          f"epoch={ckpt.get('epoch', '?')}  val={ckpt.get('val_total', '?')}  <- {RECON_DIR}")
    return recon


def chunked_utility(x_det, y_det, chunks, corpus, surface, primary_all, recon,
                    backward: bool):
    """Exact U over the events in `chunks` (per-event mean, chunk-accumulated;
    chunks come from corpus.pack_chunks so padded graphs stay memory-bounded).

    With backward=True each chunk's weighted -U is backpropagated into the
    layout leaves (caller zeroes grads). Returns (U, r_mean, parts) as floats;
    U is NaN if any chunk went non-finite (with a one-shot diagnostic of which
    stage produced it)."""
    n = sum(int(c.numel()) for c in chunks)
    U_tot, r_tot = 0.0, 0.0
    parts_tot = dict(u_theta=0.0, u_phi=0.0, u_e=0.0, u_pr=0.0)
    for cidx in chunks:
        w = cidx.numel() / n
        src = KernelVoxelLabels(corpus, cidx, surface)
        U, r, parts = utility_of_xy(x_det, y_det, primary_all[cidx], src, recon)
        if not torch.isfinite(U):
            with torch.no_grad():
                xy_b = torch.stack([x_det, y_det], dim=-1
                                   ).unsqueeze(0).expand(cidx.numel(), -1, -1)
                pred = src(None, xy_b)
                rec = recon(torch.cat([xy_b, pred], dim=-1))
                print(f"[nonfinite] U bad on chunk of {cidx.numel()} events: "
                      f"labels finite={bool(torch.isfinite(pred).all())}  "
                      f"recon finite={bool(torch.isfinite(rec).all())}  "
                      f"layout finite={bool(torch.isfinite(x_det).all() and torch.isfinite(y_det).all())}")
            return float("nan"), r_tot, parts_tot
        if backward:
            (-(U * w)).backward()
        U_tot += float(U.item()) * w
        r_tot += float(r.mean().item()) * w
        for k in parts_tot:
            parts_tot[k] += float(parts[k].item()) * w
    return U_tot, r_tot, parts_tot


def adam_warm_start(scheme, mountain, corpus, surface, recon,
                    primary_all, init_override):
    """Kernel-minibatch Adam with mountain projection each step.
    Mirrors the surrogate ensemble's adam_warm_start; returns the same tuple."""
    e_init, n_init = init_override
    e_init, n_init = e_init.float(), n_init.float()
    print(f"[adam] init {scheme}  E in [{e_init.min():.1f}, {e_init.max():.1f}]  "
          f"N in [{n_init.min():.1f}, {n_init.max():.1f}]")

    xy_module = LearnableXY(e_init, n_init, device=str(DEVICE)).to(DEVICE)
    optimizer = torch.optim.Adam(xy_module.parameters(), lr=ADAM_LR)

    log, grad_hist = [], []
    best_u, best_x, best_y = -float("inf"), e_init.clone(), n_init.clone()

    n_skipped = 0
    for epoch in range(N_ADAM_EPOCHS):
        idx = torch.randint(0, corpus.n_pairs, (PRIMARIES_PER_STEP,))
        chunks = corpus.pack_chunks(idx, slot_budget=SLOT_BUDGET)

        x_det, y_det = xy_module()
        optimizer.zero_grad(set_to_none=True)
        u_val, r_mean, parts = chunked_utility(
            x_det, y_det, chunks, corpus, surface, primary_all, recon,
            backward=True)

        # A non-finite U means chunked_utility returned before calling
        # .backward() on the offending chunk, so .grad can still be None here
        # (nothing accumulated into it this step) — check U and None BEFORE
        # touching .grad, never after. Stepping on a non-finite loss/grad
        # would poison the Adam moments and the positions permanently.
        grad_ok = xy_module.x.grad is not None and xy_module.y.grad is not None
        if not (np.isfinite(u_val) and grad_ok):
            optimizer.zero_grad(set_to_none=True)
            n_skipped += 1
            if n_skipped <= 5 or n_skipped % 100 == 0:
                print(f"  [adam {epoch+1:4d}] non-finite step SKIPPED "
                      f"(U={u_val}, grad_ok={grad_ok}, total skipped={n_skipped})")
            continue
        gvec = torch.cat([xy_module.x.grad.detach().reshape(-1),
                          xy_module.y.grad.detach().reshape(-1)])
        if not torch.isfinite(gvec).all():
            optimizer.zero_grad(set_to_none=True)
            n_skipped += 1
            if n_skipped <= 5 or n_skipped % 100 == 0:
                print(f"  [adam {epoch+1:4d}] non-finite gradient SKIPPED "
                      f"(U={u_val}, total skipped={n_skipped})")
            continue
        grad_hist.append(gvec.cpu())
        torch.nn.utils.clip_grad_norm_(xy_module.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        with torch.no_grad():
            e_new, n_new = project_to_mountain_ne(
                mountain, xy_module.x.detach().cpu(), xy_module.y.detach().cpu())
            xy_module.x.data.copy_(e_new.to(DEVICE).to(xy_module.x.dtype))
            xy_module.y.data.copy_(n_new.to(DEVICE).to(xy_module.y.dtype))

        if u_val > best_u:
            best_u = u_val
            best_x = xy_module.x.detach().cpu().clone()
            best_y = xy_module.y.detach().cpu().clone()
        log.append(dict(
            epoch=epoch + 1, U=u_val, r_mean=r_mean,
            u_theta=parts["u_theta"], u_phi=parts["u_phi"],
            u_e=parts["u_e"], u_pr=parts["u_pr"],
        ))
        if epoch == 0 or (epoch + 1) % ADAM_LOG_EVERY == 0 or epoch == N_ADAM_EPOCHS - 1:
            print(f"  [adam {epoch+1:4d}/{N_ADAM_EPOCHS}] U={u_val:+.3f}")

    print(f"[adam] best U={best_u:+.3f}  (minibatch estimate; "
          f"{n_skipped} non-finite steps skipped)")
    grad_hist = torch.stack(grad_hist, dim=0) if grad_hist else torch.zeros(0)
    return best_x, best_y, e_init, n_init, log, grad_hist


def lbfgs_refine(init_x, init_y, mountain, corpus, surface, recon, primary_all,
                 chunks_full):
    """L-BFGS on the exact FULL-corpus objective (chunk-accumulated closure).
    Deterministic by construction — no fixed-batch selection anywhere.
    `chunks_full` is the precomputed pack_chunks partition of the whole corpus."""
    xy = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0).detach().clone()
    xy.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [xy], lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY_SIZE, line_search_fn="strong_wolfe",
        tolerance_grad=1e-11, tolerance_change=1e-13,
    )

    iter_log, grad_hist = [], []

    class _NonFiniteLoss(Exception):
        pass

    def closure():
        optimizer.zero_grad()
        t0 = time.perf_counter()
        U, r_mean, parts = chunked_utility(
            xy[:N_DETECTORS], xy[N_DETECTORS:], chunks_full,
            corpus, surface, primary_all, recon, backward=True)
        if not np.isfinite(U):
            raise _NonFiniteLoss
        grad_hist.append(xy.grad.detach().reshape(-1).cpu())
        iter_log.append(dict(iter=len(iter_log), U=U, r_mean=r_mean,
                             u_theta=parts["u_theta"], u_phi=parts["u_phi"],
                             u_e=parts["u_e"], u_pr=parts["u_pr"]))
        if len(iter_log) % 5 == 1:
            print(f"  [lbfgs closure {len(iter_log):4d}] U={U:+.4f}  "
                  f"({time.perf_counter() - t0:.1f} s/closure)")
        return torch.tensor(-U)

    try:
        optimizer.step(closure)
    except _NonFiniteLoss:
        print(f"  [lbfgs] non-finite loss after {len(iter_log)} closure calls — aborting step")

    with torch.no_grad():
        if not torch.isfinite(xy).all():
            print("  [lbfgs] non-finite iterate — falling back to the Adam-best init")
            xy.data = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0)

    # Project the optimum to the mountain, re-score exactly on the full corpus.
    with torch.no_grad():
        x_proj, y_proj = project_to_mountain_ne(
            mountain, xy[:N_DETECTORS].detach().cpu(), xy[N_DETECTORS:].detach().cpu())
        U_proj, _, _ = chunked_utility(
            x_proj.to(DEVICE), y_proj.to(DEVICE), chunks_full,
            corpus, surface, primary_all, recon, backward=False)

    grad_hist = torch.stack(grad_hist, dim=0) if grad_hist else torch.zeros(0)
    return x_proj.float(), y_proj.float(), float(U_proj), iter_log, grad_hist


def run_scheme(scheme, mountain, corpus, surface, recon, primary_all, overwrite):
    opt_dir = OPT_DIR_TEMPLATE.format(scheme=scheme)
    if os.path.isdir(opt_dir) and os.listdir(opt_dir) and not overwrite:
        raise SystemExit(f"[abort] output dir exists and is non-empty: {opt_dir} "
                         f"(pass --overwrite to allow)")
    os.makedirs(opt_dir, exist_ok=True)
    print("-" * 72)
    print(f"[run] scheme={scheme}  ->  {opt_dir}")

    # K perturbed inits around the scheme layout.
    g = torch.Generator()
    e_np, n_np = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme=scheme)
    base = torch.cat([torch.as_tensor(np.asarray(e_np), dtype=torch.float32).reshape(-1),
                      torch.as_tensor(np.asarray(n_np), dtype=torch.float32).reshape(-1)])
    perturb = torch.randn(N_CHAINS, base.numel(), generator=g) * INIT_OVERDISP_SIGMA
    chains_init = base.unsqueeze(0) + perturb

    adam_bests, adam_logs, adam_grads = [], [], []
    for k in range(N_CHAINS):
        xk, yk = project_to_mountain_ne(
            mountain, chains_init[k, :N_DETECTORS], chains_init[k, N_DETECTORS:])
        print(f"\n[perturb→adam] scheme={scheme}  chain {k+1}/{N_CHAINS}")
        bx, by, _, _, log, ghist = adam_warm_start(
            scheme, mountain, corpus, surface, recon, primary_all,
            init_override=(xk.float(), yk.float()))
        adam_bests.append((bx, by)); adam_logs.append(log); adam_grads.append(ghist)

    chunks_full = corpus.pack_chunks(torch.arange(corpus.n_pairs),
                                     slot_budget=SLOT_BUDGET)
    print(f"[lbfgs] full corpus packed into {len(chunks_full)} chunks "
          f"(slot budget {SLOT_BUDGET})")

    refined, refined_U, lbfgs_logs, lbfgs_grads = [], [], [], []
    for k, (bx, by) in enumerate(adam_bests):
        print(f"[lbfgs] refine {k+1}/{N_CHAINS}  (full-corpus exact closure)")
        xp, yp, Up, lg, gh = lbfgs_refine(
            bx, by, mountain, corpus, surface, recon, primary_all, chunks_full)
        refined.append((xp, yp)); refined_U.append(Up)
        lbfgs_logs.append(lg); lbfgs_grads.append(gh)
        print(f"  [lbfgs] refine {k} DONE  full-corpus U={Up:+.4f}  "
              f"({len(lg)} closure calls)")

    layouts_xy = np.stack(
        [np.stack([xp.numpy(), yp.numpy()], axis=-1) for xp, yp in refined], axis=0)
    ref_idx = int(np.argmax(refined_U))     # selection IS the full-corpus objective
    aligned, perms = align_to_reference(layouts_xy, ref_idx)
    mean_xy, std_xy = aligned.mean(axis=0), aligned.std(axis=0)
    best_x, best_y = refined[ref_idx]
    print(f"[ensemble] K={N_CHAINS}  best full-corpus U={refined_U[ref_idx]:+.4f} "
          f"(run {ref_idx})  mean σx={std_xy[:, 0].mean():.1f}m σy={std_xy[:, 1].mean():.1f}m")

    torch.save({"x": best_x, "y": best_y, "U": refined_U[ref_idx], "run": ref_idx,
                "source": scheme}, os.path.join(opt_dir, "layout_best.pt"))
    torch.save({"mean_x": torch.as_tensor(mean_xy[:, 0]),
                "mean_y": torch.as_tensor(mean_xy[:, 1]),
                "std_x": torch.as_tensor(std_xy[:, 0]),
                "std_y": torch.as_tensor(std_xy[:, 1])},
               os.path.join(opt_dir, "layout_mean.pt"))
    torch.save({"aligned": torch.as_tensor(aligned), "perms": torch.as_tensor(perms),
                "utilities": torch.as_tensor(refined_U),
                "source_per_run": [scheme] * N_CHAINS, "ref_idx": ref_idx},
               os.path.join(opt_dir, "layouts_all.pt"))
    with open(os.path.join(opt_dir, "optimize_log.json"), "w") as f:
        json.dump({
            "scheme": scheme, "label_source": "kernel_voxel",
            "voxel_delta": corpus.delta, "ref_idx": ref_idx,
            "refined_U": refined_U, "best_U": refined_U[ref_idx],
            "grad_cos_consecutive": dict(
                adam=[consecutive_cos_distance(g_, GRAD_COS_WINDOW).tolist()
                      for g_ in adam_grads],
                lbfgs=[consecutive_cos_distance(g_, GRAD_COS_WINDOW).tolist()
                       for g_ in lbfgs_grads]),
            "adam_logs": adam_logs, "lbfgs_logs": lbfgs_logs,
            "config": dict(
                n_chains=N_CHAINS, init_overdisp_sigma=INIT_OVERDISP_SIGMA,
                n_adam_epochs=N_ADAM_EPOCHS, primaries_per_step=PRIMARIES_PER_STEP,
                adam_lr=ADAM_LR, grad_clip=GRAD_CLIP,
                lbfgs_max_iter=LBFGS_MAX_ITER, lbfgs_lr=LBFGS_LR,
                lbfgs_history_size=LBFGS_HISTORY_SIZE, slot_budget=SLOT_BUDGET,
                w_theta=W_THETA, w_phi=W_PHI, w_e=W_E, w_pr=W_PR, w_div=W_DIV,
                layout_threshold=LAYOUT_THRESHOLD,
                reconstruct_threshold=RECONSTRUCT_THRESHOLD,
            ),
        }, f, indent=2)

    try:
        _plt.plot_curves_lbfgs(adam_logs, lbfgs_logs, adam_grads, lbfgs_grads,
                               os.path.join(opt_dir, "optimize_curves.png"),
                               grad_cos_window=GRAD_COS_WINDOW)
        _plt.plot_components_lbfgs(adam_logs, lbfgs_logs,
                                   os.path.join(opt_dir, "utility_components.png"))
        surf_cpu = SurfaceUpMap.from_mountain(mountain).to("cpu")
        _plt.plot_ensemble(aligned, mean_xy, std_xy, best_x, best_y,
                           mountain, os.path.join(opt_dir, "layout_ensemble.png"),
                           surface=surf_cpu, title_kind="kernel-direct ensemble")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")

    print(f"[done] scheme={scheme}  best FULL-CORPUS U={refined_U[ref_idx]:+.4f}  "
          f"({opt_dir})")
    return dict(scheme=scheme, best_U=refined_U[ref_idx], opt_dir=opt_dir)


def main():
    global N_CHAINS, N_ADAM_EPOCHS, LBFGS_MAX_ITER, SLOT_BUDGET, INIT_SCHEMES
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schemes", type=str, default=",".join(INIT_SCHEMES))
    ap.add_argument("--chains", type=int, default=N_CHAINS)
    ap.add_argument("--adam-epochs", type=int, default=N_ADAM_EPOCHS)
    ap.add_argument("--lbfgs-iters", type=int, default=LBFGS_MAX_ITER)
    ap.add_argument("--slot-budget", type=int, default=SLOT_BUDGET)
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA,
                    help="voxel size [m]; cache is built on first use")
    ap.add_argument("--rebuild-voxels", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into an existing non-empty output dir")
    args = ap.parse_args()
    N_CHAINS, N_ADAM_EPOCHS = int(args.chains), int(args.adam_epochs)
    LBFGS_MAX_ITER, SLOT_BUDGET = int(args.lbfgs_iters), int(args.slot_budget)
    INIT_SCHEMES = tuple(s.strip() for s in args.schemes.split(",") if s.strip())

    print("=" * 72)
    print("v6/04_optimize_kernel_direct.py — kernel-in-the-loop Adam + L-BFGS")
    print("=" * 72)
    print(f"device       : {DEVICE}")
    print(f"init schemes : {INIT_SCHEMES}   chains (K): {N_CHAINS}")
    print(f"Adam epochs  : {N_ADAM_EPOCHS}  (events/step={PRIMARIES_PER_STEP})")
    print(f"L-BFGS       : max_iter={LBFGS_MAX_ITER}  FULL corpus, slot budget={SLOT_BUDGET}")

    cache = voxel_cache_path(args.delta)
    if args.rebuild_voxels or not os.path.exists(cache):
        print(f"[voxel] building cache (delta={args.delta:g} m) -> {cache}")
        build_voxel_corpus(args.delta, device=str(DEVICE), out_path=cache)
    corpus = VoxelCorpus(cache, DEVICE)

    primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")
                             ).float()[:corpus.n_pairs].to(DEVICE)
    print(f"[load] {primary_all.shape[0]} primaries")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain).to(DEVICE)
    recon = load_recon(DEVICE)

    results = [run_scheme(s, mountain, corpus, surface, recon, primary_all,
                          args.overwrite)
               for s in INIT_SCHEMES]

    print()
    print("=" * 72)
    print("summary  (U here IS the full-corpus kernel ground truth)")
    print("=" * 72)
    for r in results:
        print(f"  {r['scheme']:<10}  best U={r['best_U']:+.4f}  ->  {r['opt_dir']}")


if __name__ == "__main__":
    main()
