"""Optimise detector positions with optional reconstruction-network retraining.

A reconstruction network estimates a particle shower's direction and energy from
signals at the detectors. This driver moves the detectors to improve a score
based on those estimates. It can also retrain the network on signals recomputed
at the current detector positions, accepting an update only when its validation
error decreases. Detector updates and network training are separate operations.

Run the fixed-network control with::

    python recon_loop/run.py --recon /path/to/recon.pt --out /path/to/control

Enable periodic retraining by adding --bank /path/to/bank.pt and
--refresh-every 100. Build that bank first with recon_loop/refresh.py; it contains
previous training examples and a separate sample for monitoring forgetting.
The repository's modules/constants.py selects the shower corpus and geometry.

Each output directory contains the selected detector layout and matching model,
periodic snapshots, training/score logs, and an optimisation checkpoint. Reuse
--out to resume with the same settings; --epochs may extend the total step count.
With retraining, every snapshot is rescored with the final model at the end and the
best one is selected. Evaluate the selected layout and model on independent showers
before claiming an improvement.
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from recon_loop import refresh

# Use independent random streams for detector-update and refresh draws.
REFRESH_TAG = 7_303


def main():
    # Read layout-optimisation and optional retraining settings.
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="this run's output directory; reused to resume")
    ap.add_argument("--init", choices=("grid", "center"), default="grid")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=10.0)
    ap.add_argument("--grad-clip", type=float, default=100.0)
    ap.add_argument("--batch-events", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--n-score", type=int, default=1024)
    ap.add_argument("--score-every", type=int, default=250)
    ap.add_argument("--score-chunk", type=int, default=32)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=4200)
    ap.add_argument("--recon", required=True, help="kernel-trained recon.pt")
    ap.add_argument("--bank", help="bank.pt from refresh.py; required when refreshing")
    ap.add_argument("--refresh-every", type=int, default=0, help="0: frozen recon (control)")
    ap.add_argument("--refresh-showers", type=int, default=8192)
    for f in fields(refresh.Config):
        ap.add_argument(
            f"--refresh-{f.name.replace('_', '-')}", type=type(f.default), default=f.default
        )
    args = ap.parse_args()
    cfg = refresh.Config(
        **{f.name: getattr(args, f"refresh_{f.name}") for f in fields(refresh.Config)}
    )

    # Reject invalid settings before loading the large shower corpus.
    positive = (
        "epochs",
        "batch_events",
        "chunk",
        "n_score",
        "score_every",
        "score_chunk",
        "ckpt_every",
        "log_every",
    )
    if any(getattr(args, name) <= 0 for name in positive) or args.refresh_every < 0:
        ap.error("step counts and batch sizes must be positive; refresh-every may be 0")
    if args.refresh_every and (not args.bank or args.refresh_showers < 2):
        ap.error("refreshing requires --bank and at least two refresh showers")

    # Resolve paths and create this optimisation's output folders.
    args.out, args.recon = os.path.abspath(args.out), os.path.abspath(args.recon)
    if args.bank:
        args.bank = os.path.abspath(args.bank)
    out = args.out
    os.makedirs(out, exist_ok=True)

    # Write the same progress messages to the terminal and run.log.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(out, "run.log")),
        ],
    )
    log = logging.getLogger("recon_loop")
    log.info("args %s", json.dumps(vars(args)))
    for sub in ("layouts", "recons"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    kernel = refresh.import_script("04_optimize_lbfgs_kernel.py")
    device = kernel.DEVICE
    torch.manual_seed(args.seed)

    # Load the surface, starting model and simulation pool.
    from modules import constants as C

    mountain = kernel.load_tr_mountain(
        C.GEOMETRY_PATH_RESOLVED,
        C.GEOMETRY_GROUP,
        C.DET_KEY,
        east_entry=C.EAST_ENTRY,
        layer_east_dx=C.LAYER_EAST_DX,
        n_planes=C.N_PLANES,
    )
    surface = kernel.SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(device)
    initial_ckpt = torch.load(args.recon, map_location="cpu", weights_only=False)
    recon = kernel.build_recon_from_ckpt(initial_ckpt, kernel.N_DETECTORS, device).eval()
    pool = kernel.MainPool(mountain)

    # Only retraining needs previous examples and a forgetting monitor.
    bank, replay, anchor = None, None, None
    if args.refresh_every:
        bank = torch.load(args.bank, map_location="cpu", weights_only=False)
        if os.path.realpath(bank["recon"]) != os.path.realpath(args.recon):
            raise ValueError("bank was built for a different recon")
        if len(bank["pool_blob"]) != pool.n:
            raise ValueError("bank and optimisation pool have different event counts")
        replay = (bank["replay_feats"], bank["replay_target"])
        anchor = (bank["anchor_feats"], bank["anchor_target"])

    # Reserve a fixed scoring slice outside the layout/refresh draws.
    n_sweep, B, M = pool.n - args.n_score, args.batch_events, args.refresh_showers
    if n_sweep < max(B, M if args.refresh_every else B):
        ap.error("the pool minus the scoring set is too small for the requested draws")
    log.info(
        "pool %d events; sweep %d, scoring set %d; %d per step; refresh every %d steps on %d showers %s",
        pool.n,
        n_sweep,
        args.n_score,
        B,
        args.refresh_every,
        M,
        cfg,
    )
    t0 = time.time()
    score_set = pool.load(n_sweep, pool.n)
    log.info("scoring set loaded (%d events) in %.0f s", score_set[2].shape[0], time.time() - t0)

    # Detector coordinates are the only parameters updated by this Adam.
    e_init, n_init = kernel.init_layout(mountain, args.init, device)
    xy_module = kernel.LearnableXY(e_init, n_init, device=str(device)).to(device)
    optimizer = torch.optim.Adam(xy_module.parameters(), lr=args.lr)

    # Restore a saved optimisation, or evaluate the initial layout.
    state_path = os.path.join(out, "opt_state.pt")
    log_steps, score_log, refresh_log, start, ver = [], [], [], 0, 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        saved = st["config"]
        changed = [
            k for k, v in saved.items() if k not in ("epochs", "log_every") and vars(args)[k] != v
        ]
        if changed or args.epochs < st["epoch"]:
            raise ValueError(
                f"resume configuration changed: {changed}; epochs must be >= saved step"
            )
        xy_module.load_state_dict(st["xy"])
        optimizer.load_state_dict(st["optimizer"])
        start, log_steps, score_log, refresh_log = (
            st["epoch"],
            st["log"],
            st["score_log"],
            st["refresh_log"],
        )
        best = st["best"]
        ver = st["recon_version"]
        if ver:
            refresh.install(
                recon,
                torch.load(
                    os.path.join(out, "recons", f"v{ver:03d}.pt"),
                    map_location=device,
                    weights_only=False,
                )["state_dict"],
            )
        log.info(
            "resume at step %d, recon version %d, best U %.3f at step %d",
            start,
            ver,
            best["U"],
            best["epoch"],
        )
    else:
        u = kernel.pool_utility(
            xy_module, score_set, surface, recon, args.score_chunk, backward=False
        )["U"]
        best = dict(
            U=u,
            epoch=0,
            x=e_init.detach().cpu().clone(),
            y=n_init.detach().cpu().clone(),
            recon_version=0,
        )
        score_log.append(dict(epoch=0, U_score=u, recon_version=0))
        refresh.save_atomic(
            dict(x=best["x"], y=best["y"], epoch=0, U_score=u, recon_version=0),
            os.path.join(out, "layouts", "step_00000.pt"),
        )
        log.info("init U %.6f on the scoring set", u)

    # Choose each batch from its step number, independently of refresh draws.
    def load_batch(step):
        offset = int(np.random.default_rng([args.seed, step]).integers(0, n_sweep - B + 1))
        return pool.load(offset, offset + B)

    # Read the next shower batch while the current batch is processed.
    ex = ThreadPoolExecutor(max_workers=1)
    nxt = ex.submit(load_batch, start) if start < args.epochs else None
    for epoch in range(start, args.epochs):
        batch = nxt.result()
        if epoch + 1 < args.epochs:
            nxt = ex.submit(load_batch, epoch + 1)

        # Increase utility through the frozen model by moving detectors.
        optimizer.zero_grad(set_to_none=True)
        tot = kernel.pool_utility(xy_module, batch, surface, recon, args.chunk, backward=True)
        torch.nn.utils.clip_grad_norm_(xy_module.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        # Project moved detectors back onto the allowed surface.
        with torch.no_grad():
            e_new, n_new = kernel.project_to_mountain_ne(
                mountain, xy_module.x.detach().cpu(), xy_module.y.detach().cpu()
            )
            xy_module.x.data.copy_(e_new.to(device).to(xy_module.x.dtype))
            xy_module.y.data.copy_(n_new.to(device).to(xy_module.y.dtype))
        log_steps.append(dict(epoch=epoch + 1, recon_version=ver, **tot))

        # Retrain only after the detector-position update is complete.
        if args.refresh_every and (epoch + 1) % args.refresh_every == 0:
            k = (epoch + 1) // args.refresh_every
            t = time.time()
            a = int(np.random.default_rng([args.seed, REFRESH_TAG, k]).integers(0, n_sweep - M + 1))
            fresh = refresh.label(
                kernel,
                pool,
                a,
                a + M,
                xy_module,
                surface,
                args.score_chunk,
                bank["pool_blob"],
                device,
            )
            t_label = time.time() - t
            state, stats = refresh.fine_tune(
                recon, fresh, replay, anchor, cfg, seed=args.seed * 100_003 + k, device=device
            )

            # Save accepted weights before making that version live.
            if state is not None:
                ver += 1
                refresh.save_atomic(
                    refresh.checkpoint(
                        initial_ckpt,
                        state,
                        version=ver,
                        refresh=k,
                        step=k * args.refresh_every,
                        name=os.path.basename(out),
                    ),
                    os.path.join(out, "recons", f"v{ver:03d}.pt"),
                )
                refresh.install(recon, state)

            # Record validation improvement and forgetting for every attempt.
            refresh_log.append(
                dict(
                    k=k,
                    step=k * args.refresh_every,
                    offset=a,
                    n_fresh=len(fresh[0]),
                    label_seconds=t_label,
                    recon_version=ver,
                    **stats,
                )
            )
            log.info(
                "refresh %d at step %d: held %.5f -> %.5f (epoch %d) %s; anchor %.5f -> %.5f; label %.0f s, "
                "tune %.0f s",
                k,
                k * args.refresh_every,
                stats["held_before"],
                stats["held_best"],
                stats["best_epoch"],
                "ACCEPTED v%d" % ver if stats["accepted"] else "rejected",
                stats["anchor_before"],
                stats["anchor_after"],
                t_label,
                stats["seconds"],
            )

        # Score the current pair and keep the best layout/model version.
        if (epoch + 1) % args.score_every == 0 or epoch == args.epochs - 1:
            s = kernel.pool_utility(
                xy_module, score_set, surface, recon, args.score_chunk, backward=False
            )["U"]
            score_log.append(dict(epoch=epoch + 1, U_score=s, recon_version=ver))
            layout = dict(
                x=xy_module.x.detach().cpu().clone(),
                y=xy_module.y.detach().cpu().clone(),
                epoch=epoch + 1,
                U_score=s,
                recon_version=ver,
            )
            refresh.save_atomic(layout, os.path.join(out, "layouts", f"step_{epoch + 1:05d}.pt"))
            if s > best["U"]:
                best = dict(
                    U=s,
                    epoch=epoch + 1,
                    x=layout["x"],
                    y=layout["y"],
                    recon_version=ver,
                )
            log.info(
                "score %d: U %.3f (recon v%d)  best %.3f at %d",
                epoch + 1,
                s,
                ver,
                best["U"],
                best["epoch"],
            )

        if epoch == start or (epoch + 1) % args.log_every == 0:
            log.info("[%d/%d] batch U %.3f, recon v%d", epoch + 1, args.epochs, tot["U"], ver)

        # Save enough state to repeat only work after this checkpoint.
        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            refresh.save_atomic(
                dict(
                    xy=xy_module.state_dict(),
                    optimizer=optimizer.state_dict(),
                    epoch=epoch + 1,
                    log=log_steps,
                    score_log=score_log,
                    refresh_log=refresh_log,
                    best=best,
                    recon_version=ver,
                    config=vars(args),
                ),
                state_path,
            )
    ex.shutdown()

    # Retraining changes the selection score, so rescore every snapshot with the final model.
    if ver:
        best = None
        for f in sorted(glob.glob(os.path.join(out, "layouts", "step_*.pt"))):
            lay = torch.load(f, map_location="cpu", weights_only=False)
            xy_module.x.data.copy_(lay["x"].to(device).to(xy_module.x.dtype))
            xy_module.y.data.copy_(lay["y"].to(device).to(xy_module.y.dtype))
            s = kernel.pool_utility(
                xy_module, score_set, surface, recon, args.score_chunk, backward=False
            )["U"]
            if best is None or s > best["U"]:
                best = dict(U=s, epoch=lay["epoch"], x=lay["x"], y=lay["y"], recon_version=ver)
        log.info("rescored snapshots with recon v%d: best U %.3f at step %d", ver, best["U"], best["epoch"])

    # Export the selected layout, its model and the last trained model.
    refresh.save_atomic(
        dict(
            x=best["x"],
            y=best["y"],
            epoch=best["epoch"],
            U_score=best["U"],
            recon_version=best["recon_version"],
        ),
        os.path.join(out, "layout_best.pt"),
    )
    for tag, v in (("best", best["recon_version"]), ("final", ver)):
        source = args.recon if v == 0 else os.path.join(out, "recons", f"v{v:03d}.pt")
        refresh.save_atomic(
            torch.load(source, map_location="cpu", weights_only=False),
            os.path.join(out, f"recon_{tag}.pt"),
        )

    # Export the optimisation history and final result summary.
    with open(os.path.join(out, "log.json"), "w") as f:
        json.dump(dict(log=log_steps, score_log=score_log, refresh_log=refresh_log), f)
    n_acc = sum(r["accepted"] for r in refresh_log)
    log.info(
        "done: best U %.3f at step %d (recon v%d); %d of %d refreshes accepted",
        best["U"],
        best["epoch"],
        best["recon_version"],
        n_acc,
        len(refresh_log),
    )
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(
            dict(
                best_U=best["U"],
                best_epoch=best["epoch"],
                best_recon_version=best["recon_version"],
                final_recon_version=ver,
                refreshes=len(refresh_log),
                accepted=n_acc,
                config=vars(args),
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
