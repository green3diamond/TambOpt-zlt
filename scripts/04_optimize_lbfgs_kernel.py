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
)
from modules.layouts import LearnableXY
from modules.geometry import load_tr_mountain, SurfaceUpMap
from modules.geometry import project_to_mountain_ne, sample_initial_layout_ne
from modules.optimize import utility_of_xy
from modules.surrogates import build_recon_from_ckpt

# plots/layouts/true_utility.py has no digit prefix, so it imports normally as
# a namespace-package module (no plots/layouts/__init__.py needed in Py3).
from plots.layouts.true_utility import load_events, KernelDualLabels

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 4200


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


def time_one_step(kernel_fnn, recon, primary_batch, xy_module):
    """THEORY SS11.3's cheap experiment #1: one fwd+bwd, timed, no optimizer
    step. Answers "is kernel-in-the-loop even tractable here" before anyone
    commits to a multi-epoch run."""
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    x, y = xy_module()
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
    ap.add_argument("--out-suffix", type=str, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--time-one-step", action="store_true",
                    help="time one fwd+bwd at the given --n-events/--chunk "
                         "and exit -- no optimization, nothing written")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    out_dir = OPT_FOLDER + "_lbfgs_kernel" + (f"_{args.out_suffix}" if args.out_suffix else "")

    print("=" * 72)
    print("v6/04_optimize_lbfgs_kernel.py — layout optimization against the "
          "KERNEL, not the surrogate")
    print("=" * 72)
    print(f"device={DEVICE}  n_events={args.n_events}  chunk={args.chunk}  "
          f"epochs={args.epochs}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                               east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                               n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)

    recon = load_recon(args.recon_folder, DEVICE)

    t0 = time.time()
    species_clouds, B, n_pairs, primary_batch = load_events(args.n_events, DEVICE, mountain)
    print(f"[load] {B} held-out events (of {n_pairs} available) in "
          f"{time.time() - t0:.1f}s")
    kernel_fnn = KernelDualLabels(species_clouds, surface, DEVICE,
                                 chunk=args.chunk, differentiable=True)

    e_init, n_init = init_layout(mountain, args.init, DEVICE)
    xy_module = LearnableXY(e_init, n_init, device=str(DEVICE)).to(DEVICE)

    if args.time_one_step:
        time_one_step(kernel_fnn, recon, primary_batch, xy_module)
        return

    optimizer = torch.optim.Adam(xy_module.parameters(), lr=args.lr)
    log = []
    best_u = -float("inf")
    best_x = e_init.detach().cpu().clone()
    best_y = n_init.detach().cpu().clone()

    for epoch in range(args.epochs):
        x_det, y_det = xy_module()
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
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            print(f"  [{epoch+1:4d}/{args.epochs}] U={u_val:+.3f}  "
                  f"(u_theta={float(parts['u_theta']):.3f} "
                  f"u_phi={float(parts['u_phi']):.3f} "
                  f"u_e={float(parts['u_e']):.3f})")

    print(f"[done] best U={best_u:+.3f}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(dict(x=best_x, y=best_y), os.path.join(out_dir, "layout_best.pt"))
    with open(os.path.join(out_dir, "optimize_log.json"), "w") as f:
        json.dump(dict(config=vars(args), log=log, best_u=best_u), f, indent=2)
    print(f"[write] {out_dir}/layout_best.pt, optimize_log.json")


if __name__ == "__main__":
    main()
