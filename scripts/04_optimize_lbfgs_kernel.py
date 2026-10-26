"""Stage 4: optimise the detector layout against the kernel (or the Stage 2 surrogate) and the frozen recon.

Per step: a fresh slice of --batch-events Stage-1 showers (the main pool), U = utility_of_xy (mode from
TAMBOPT_UTILITY), backward in sub-batches of --chunk, Adam on (East, North), projection onto the mesh. The best layout
is chosen on a fixed scoring set at the end of the pool (--n-score). Optional: chunked L-BFGS polish (--lbfgs-chunks),
cosmic background (--bg-window-ns), kNN 2-of-3 gate (--gate knn23), the Stage 2 surrogate as the response
(--response surrogate), frozen unseen detectors (--freeze-unseen). Resumes from opt_state.pt / lbfgs_state.pt.
Output: <OPT_FOLDER>_lbfgs_kernel_<suffix>/layout_best.pt, optimize_log.json, layout_density.png.

    python scripts/04_optimize_lbfgs_kernel.py --precompute-bw <species index>      # once per run (kernel widths)
    sbatch slurm/kernel_opt_run19.sh <grid|center> <utility> [lambda] [lr] [steps]   # see that script
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from _pathfix import V6_ROOT  # noqa: F401 — idempotent, registers v6 root

import numpy as np
import showerdata
import torch

import modules  # noqa: F401 — package import; keeps modules on the path
from modules.constants import (
    N_DETECTORS, GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, RECON_FOLDER, OPT_FOLDER,
    T_LOG_SCALE, KERNEL_INTERP_K, SIGMA_SPATIAL,
)
from modules.constants import (
    DUAL_SHOWER_CACHE_PATH, DATASET_FRACTION, NUM_SHOWERS, SPECIES_NAMES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER,
)
from modules.data.dataset_builder import compute_labels_batch, place_clouds, _load_corpus_metadata
from modules.showers.kernel import adaptive_bandwidths
from modules.surrogates import combine_species_outputs
from modules.layouts import LearnableXY
from modules.geometry import load_tr_mountain, SurfaceUpMap
from modules.geometry import project_to_mountain_ne, sample_initial_layout_ne
from modules.optimize import utility_of_xy
from modules.optimize.background import add_background, window_centres, knn_pairs, knn_gate, axis_distance
from modules.optimize.counting import expected_counts
from modules.optimize.objective import UTILITY_MODE, TWO_TIER_SCALE
from modules.optimize.cable import cable_length
from modules.surrogates import load_dual_surrogate
from modules.surrogates import build_recon_from_ckpt

# plots/lib/opt_plotting.py (loaded by path): plot_density_heatmap for layout_density.png.
_plt_spec = importlib.util.spec_from_file_location(
    "opt_plotting", os.path.join(V6_ROOT, "plots", "lib", "opt_plotting.py"))
_opt_plotting = importlib.util.module_from_spec(_plt_spec)
_plt_spec.loader.exec_module(_opt_plotting)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 4200
NAN_MAX_SKIPS = 30     # consecutive non-finite steps (each a fresh batch, same layout) before the run gives up
# Cable cost (TAMBOPT_CABLE_LAMBDA > 0): the objective becomes U - 30 (lambda L / L_ref + spacing / d_min^2), L the 3-D
# minimum-spanning-tree length over the detectors, L_ref that of the grid baseline grid_n99 (modules/optimize/cable.py).
# 0 = off: nothing below changes.
CABLE_LAMBDA = float(os.environ.get("TAMBOPT_CABLE_LAMBDA", "0"))
CABLE_DMIN = float(os.environ.get("TAMBOPT_CABLE_DMIN", "5"))


# ── Main pool: the Stage-1 events ─────────────────────────────────────────────
# Step 1 keeps the first k_sp events of each species block. Every step draws a fresh contiguous slice (corpus order
# is random: corr(index, log E) ~ 0); a fixed slice at the end is the scoring set that picks the best layout.
# Held-out scoring uses the separate held-out corpus, so nothing overlaps.
MAX_SHOWERS = (0 if DATASET_FRACTION >= 1.0 else
               int(DATASET_FRACTION * len(SPECIES_NAMES) * NUM_SHOWERS))


def bw_path(species):
    return os.path.join(TRAINING_DATASET_FOLDER,
                        f"kernel_bw_k{KERNEL_INTERP_K}_{species}.npy")


class MainPool:
    """Random access to the Stage-1 events' placed clouds and precomputed widths."""

    def __init__(self, mountain):
        self.meta = _load_corpus_metadata(mountain, DUAL_SHOWER_CACHE_PATH, MAX_SHOWERS)
        self.n = self.meta.k_sp
        self.bw = None
        if KERNEL_INTERP_K:
            paths = [bw_path(s) for s in SPECIES_NAMES]
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(f"bandwidth cache missing, run --precompute-bw: {missing}")
            self.bw = [np.load(p, mmap_mode="r") for p in paths]

    def clouds(self, s, a, b, with_frames=False):
        """Species s, events [a, b), placed per KERNEL_FRAME. Event ids (the
        circular-t2 seed) are the row within the species block, as in Stage 1."""
        m = self.meta
        pts = showerdata.load(DUAL_SHOWER_CACHE_PATH, start=s * m.per_sp + a,
                              stop=s * m.per_sp + b).points
        c = torch.as_tensor(pts, dtype=torch.float32)
        fr = place_clouds(c, m.positions[a:b], m.dirs[a:b], event_ids=torch.arange(a, b),
                          species=s, east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX)
        return (c, fr) if with_frames else c

    def load_prim(self, a, b):
        """Events [a, b): primaries only (the surrogate response needs no clouds)."""
        return [], [], self.meta.primaries[a:b], None

    def load(self, a, b):
        """Events [a, b): per-species clouds and widths (CPU), and primaries.
        Events with a non-finite cloud in any species are dropped (as load_events)."""
        placed = [self.clouds(s, a, b, with_frames=True) for s in range(len(SPECIES_NAMES))]
        clouds = [c for c, _ in placed]
        frames = placed[0][1]                     # per event, identical for every species
        # Widths are stored at the file's padded width; trailing slots are padding.
        bws = ([torch.from_numpy(np.asarray(w[a:b, :c.shape[1]], dtype=np.float32))
                for w, c in zip(self.bw, clouds)]
               if self.bw is not None else [None] * len(clouds))
        prim = self.meta.primaries[a:b]
        keep = torch.ones(b - a, dtype=torch.bool)
        for c in clouds:
            keep &= torch.isfinite(c).all(dim=(1, 2))
        if not keep.all():
            clouds = [c[keep] for c in clouds]
            bws = [w[keep] if w is not None else None for w in bws]
            prim = prim[keep]
            frames = frames[keep] if frames is not None else None
        return clouds, bws, prim, frames


class BatchKernel:
    """Kernel forward for one sub-batch of pool events, same call signature as
    KernelDualLabels: (primary_batch, xy_batch) -> (B, n_det, 2) log-encoded."""

    def __init__(self, clouds, bws, surface, differentiable, frames=None, species_transform=None):
        self.clouds, self.bws, self.frames = clouds, bws, frames
        self.surface, self.differentiable = surface, differentiable
        self.species_transform = species_transform        # e.g. counting.CountingTransform; None = unchanged

    def __call__(self, primary_batch, xy_batch):
        e_det, n_det = xy_batch[0, :, 0], xy_batch[0, :, 1]
        outs = []
        for c, w in zip(self.clouds, self.bws):
            E, T = compute_labels_batch(c, e_det, n_det, self.surface,
                                        differentiable=self.differentiable, bandwidth=w,
                                        frames=self.frames)
            outs.append(torch.stack([torch.log1p(E), torch.log1p(T * T_LOG_SCALE)], dim=-1))
        if UTILITY_MODE == "swgo_trigger":                # expected detected particles (2 m^2), with gradients
            r = axis_distance(e_det, n_det, self.surface, self.frames) if self.frames is not None else None
            self.mu = sum(expected_counts(outs, r))
        if self.species_transform is not None:
            outs = self.species_transform(outs, e_det, n_det, self.frames)
        return combine_species_outputs(*outs)


class SurrogateResponse:
    """Stage 2 hurdle surrogate as the detector response, thresholded at P(lit) >= p_min
    (MultiSpeciesSurrogate.forward_thresholded): same call signature as BatchKernel."""

    def __init__(self, surrogate, p_min):
        self.surrogate, self.p_min = surrogate, p_min

    def __call__(self, primary_batch, xy_batch):
        return self.surrogate.forward_thresholded(primary_batch, xy_batch, self.p_min)


class NoisyGated:
    """Wraps a response (BatchKernel / SurrogateResponse): adds cosmic-ray background in one readout window of
    `window_ns` around the shower's earliest hit (modules/optimize/background.py) and applies the kNN 2-of-3 gate
    (a detector is read out iff >= 2 of {itself, its 2 nearest neighbours} have E' > 0.001, else E = T = 0).
    The gate's forward value is always the exact rule; with `straight_through` its gradient is a soft count's.
    Keeps the read-out fraction and ghosts per event of the last call for logging."""

    def __init__(self, base, window_ns, gate, gen, t_fallback_us, straight_through):
        self.base, self.window_ns, self.gate, self.gen = base, window_ns, gate, gen
        self.t_fallback, self.st = t_fallback_us, straight_through
        self.read, self.ghosts, self.seen = 1.0, 0.0, None

    def __call__(self, primary_batch, xy_batch):
        out = self.base(primary_batch, xy_batch)
        self.mu = getattr(self.base, "mu", None)                          # swgo_trigger: expected counts, ungated
        E = torch.expm1(out[..., 0].clamp(min=0.0))
        t = torch.expm1(out[..., 1].clamp(min=0.0))                     # us
        timed = out[..., 1] > 0
        if self.window_ns > 0:
            t_c = window_centres(t.detach(), timed, self.t_fallback)
            E, t, _, has_bg = add_background(E, t, timed, t_c, self.window_ns, self.gen)
            ghost = has_bg & ~timed
        else:
            ghost = torch.zeros_like(timed)
        if self.gate == "knn23":
            hard, st = knn_gate(E.detach() if not self.st else E, knn_pairs(xy_batch[0, :, 0], xy_batch[0, :, 1]))
            E, t = E * (st if self.st else hard), t * hard
            self.read = float(hard.mean())
            self.ghosts = float((ghost & (hard > 0)).float().sum(1).mean())
        else:
            self.read, self.ghosts = 1.0, float(ghost.float().sum(1).mean())
        self.seen = (E.detach() > 1e-3).any(0)                            # (nd,) read out with signal in some shower
        return torch.stack([torch.log1p(E), torch.log1p(t)], dim=-1)


def pool_utility(xy_module, batch, surface, recon, chunk, backward, resp=None, gen=None):
    """U over a pool batch, sub-batch by sub-batch. U is a mean of per-event terms,
    so weighting each sub-batch by its share and calling backward per sub-batch
    gives the full-batch gradient with memory set by `chunk`, not the batch.
    `resp` (dict: surrogate, p_min, window_ns, gate, t_fallback_us) selects the response
    and the background / gate; `gen` is the background's random generator."""
    clouds, bws, prim, frames = batch
    N = prim.shape[0]
    tot = dict(U=0.0, r_mean=0.0, u_theta=0.0, u_phi=0.0, u_e=0.0, read=0.0, ghosts=0.0, seen=None)
    ctx = torch.enable_grad() if backward else torch.no_grad()
    with ctx:
        for lo in range(0, N, chunk):
            hi = min(lo + chunk, N)
            if resp is not None and resp["surrogate"] is not None:
                kern = SurrogateResponse(resp["surrogate"], resp["p_min"])
            else:
                kern = BatchKernel([c[lo:hi].to(DEVICE, non_blocking=True) for c in clouds],
                                   [w[lo:hi].to(DEVICE) if w is not None else None for w in bws],
                                   surface, differentiable=backward,
                                   frames=None if frames is None else frames[lo:hi].to(DEVICE))
            if resp is not None and (resp["window_ns"] > 0 or resp["gate"] != "none" or resp.get("track_seen")):
                kern = NoisyGated(kern, resp["window_ns"], resp["gate"], gen, resp["t_fallback_us"],
                                  straight_through=backward)
            x, y = xy_module()
            U, r, parts = utility_of_xy(x, y, prim[lo:hi].to(DEVICE), kern, recon)
            share = (hi - lo) / N
            if backward:
                (-share * U).backward()
            tot["U"] += share * float(U)
            tot["r_mean"] += share * float(r.mean())
            for k in ("u_theta", "u_phi", "u_e"):
                tot[k] += share * float(parts[k])
            tot["read"] += share * getattr(kern, "read", 1.0)
            tot["ghosts"] += share * getattr(kern, "ghosts", 0.0)
            sn = getattr(kern, "seen", None)
            if sn is not None:
                tot["seen"] = sn if tot["seen"] is None else (tot["seen"] | sn)
    return tot


def precompute_bandwidths(mountain, species_index, block=512):
    """Write one species' interpolated-kernel widths for the whole pool to a
    float16 .npy (resumable: a progress file records the next event)."""
    pool = MainPool.__new__(MainPool)
    pool.meta = _load_corpus_metadata(mountain, DUAL_SHOWER_CACHE_PATH, MAX_SHOWERS)
    pool.n, pool.bw = pool.meta.k_sp, None
    name = SPECIES_NAMES[species_index]
    final = bw_path(name)
    if os.path.exists(final):
        print(f"[bw] {final} exists")
        return
    part, prog = final + ".partial", final + ".progress"
    n_pts = showerdata.load(DUAL_SHOWER_CACHE_PATH, start=0, stop=1).points.shape[1]
    if os.path.exists(part):
        out = np.lib.format.open_memmap(part, mode="r+")
        start = int(open(prog).read()) if os.path.exists(prog) else 0
    else:
        out = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16,
                                        shape=(pool.n, n_pts))
        start = 0
    print(f"[bw] {name}: events {start}..{pool.n} of pool, k={KERNEL_INTERP_K} -> {final}")
    t0 = time.time()
    for a in range(start, pool.n, block):
        b = min(a + block, pool.n)
        c = pool.clouds(species_index, a, b).to(DEVICE)
        if c.shape[1] > n_pts:
            raise RuntimeError(f"events {a}..{b} padded to {c.shape[1]} > {n_pts} points")
        h = adaptive_bandwidths(c, KERNEL_INTERP_K, SIGMA_SPATIAL)
        out[a:b, :c.shape[1]] = h.clamp(max=6e4).cpu().numpy().astype(np.float16)
        out.flush()
        with open(prog, "w") as f:
            f.write(str(b))
        if (a // block) % 20 == 0:
            print(f"  [bw] {b}/{pool.n}  {(time.time() - t0) / (b - start):.4f}s/event", flush=True)
    del out
    os.replace(part, final)
    os.remove(prog)
    print(f"[bw] wrote {final}")


def lr_at(args, epoch):
    """--lr, or with --lr-final a log-linear decay from --lr (first step) to --lr-final (last step).
    A pure function of the step, so a requeued run resumes on the same schedule."""
    if args.lr_final is None or args.epochs <= 1:
        return args.lr
    f = min(max(epoch / (args.epochs - 1), 0.0), 1.0)
    return args.lr * (args.lr_final / args.lr) ** f


class CableCost:
    """The cable penalty in units of U, 30 (CABLE_LAMBDA L / L_ref + spacing / CABLE_DMIN^2); L_ref = the MST length of
    grid_n99 (the grid of make_triplets.py: seed 0, 99 detectors)."""

    def __init__(self, mountain, surface):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            e, n = init_layout(mountain, "grid", DEVICE, 99)
        self.surface = surface
        with torch.no_grad():
            self.l_ref = float(cable_length(e, n, surface, CABLE_DMIN)[0])
        print(f"[cable] lambda={CABLE_LAMBDA:g}  d_min={CABLE_DMIN:g} m  L_ref (grid_n99) = {self.l_ref / 1e3:.2f} km")

    def __call__(self, x, y):
        """(penalty tensor, L [m])."""
        L, h = cable_length(x, y, self.surface, CABLE_DMIN)
        return TWO_TIER_SCALE * (CABLE_LAMBDA * L / self.l_ref + h / CABLE_DMIN ** 2), float(L)


def add_cable(cable, xy_module, tot, backward):
    """Subtract the cable penalty from tot["U"] (backward: add its gradient); tot keeps U_util, cable_km, penalty."""
    with (torch.enable_grad() if backward else torch.no_grad()):
        pen, L = cable(*xy_module())
        if backward:
            pen.backward()
    tot.update(U_util=tot["U"], cable_km=L / 1e3, penalty=float(pen))
    tot["U"] -= float(pen)
    return tot


def run_main_pool(args, mountain, surface, recon, out_dir, resp=None):
    pool = MainPool(mountain)
    load = pool.load_prim if resp is not None and resp["surrogate"] is not None else pool.load
    n_score = args.n_score
    n_sweep = pool.n - n_score
    B = args.batch_events
    print(f"[pool] {pool.n} Stage-1 events; sweep {n_sweep}, scoring set {n_score} "
          f"(events {n_sweep}..{pool.n}); {B} fresh events per step")
    t0 = time.time()
    score_set = load(n_sweep, pool.n)
    print(f"[pool] scoring set loaded ({score_set[2].shape[0]} events) in {time.time() - t0:.0f}s")

    e_init, n_init = init_layout(mountain, args.init, DEVICE, args.n_det)
    xy_module = LearnableXY(e_init, n_init, device=str(DEVICE)).to(DEVICE)
    optimizer = torch.optim.Adam(xy_module.parameters(), lr=args.lr)

    def gen_for(seed):
        g = torch.Generator(device=DEVICE)
        g.manual_seed(int(seed))
        return g

    cable = CableCost(mountain, surface) if CABLE_LAMBDA > 0 else None

    def score():
        # the same background draws every time, so scores are comparable across steps
        tot = pool_utility(xy_module, score_set, surface, recon, args.score_chunk,
                           backward=False, resp=resp, gen=gen_for(args.seed * 1000003 + 999_999_937))
        return (add_cable(cable, xy_module, tot, False) if cable else tot)["U"]

    log, score_log = [], []
    start = 0
    state_path = os.path.join(out_dir, "opt_state.pt")
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=DEVICE, weights_only=False)
        xy_module.load_state_dict(st["xy"])
        optimizer.load_state_dict(st["optimizer"])
        start, log, score_log = st["epoch"], st["log"], st["score_log"]
        best_u, best_x, best_y = st["best_u"], st["best_x"], st["best_y"]
        print(f"[resume] {state_path} at epoch {start}, best scoring-set U={best_u:+.3f}")
    else:
        best_u = score()
        best_x, best_y = e_init.detach().cpu().clone(), n_init.detach().cpu().clone()
        score_log.append(dict(epoch=0, U_score=best_u))
        print(f"[score] init U={best_u:+.3f} on the scoring set")

    def offset(epoch):
        return int(np.random.default_rng([args.seed, epoch]).integers(0, n_sweep - B + 1))

    ex = ThreadPoolExecutor(max_workers=1)
    nxt = ex.submit(lambda a: load(a, a + B), offset(start)) if start < args.epochs else None
    t_loop = time.time()
    n_skip = 0
    for epoch in range(start, args.epochs):
        batch = nxt.result()
        if epoch + 1 < args.epochs:
            nxt = ex.submit(lambda a: load(a, a + B), offset(epoch + 1))

        for grp in optimizer.param_groups:
            grp["lr"] = lr_at(args, epoch)
        optimizer.zero_grad(set_to_none=True)
        tot = pool_utility(xy_module, batch, surface, recon, args.chunk, backward=True,
                           resp=resp, gen=gen_for(args.seed * 1000003 + epoch))   # per-step draws, resume-safe
        if cable:
            add_cable(cable, xy_module, tot, True)
        # NaN guard: a non-finite utility or gradient skips the step (no Adam update, so neither the positions nor
        # Adam's moments are poisoned); the first offending layout is saved; NAN_MAX_SKIPS in a row stop the run.
        n_bad = sum(int((~torch.isfinite(p.grad)).sum()) for p in xy_module.parameters() if p.grad is not None)
        if n_bad or not np.isfinite(tot["U"]):
            n_skip += 1
            tot["skipped"] = True
            print(f"  [nan-guard {epoch+1}] U={tot['U']} non-finite gradient entries={n_bad}: step skipped "
                  f"({n_skip} in a row)", flush=True)
            if not os.path.exists(os.path.join(out_dir, "nan_guard_layout.pt")):
                os.makedirs(out_dir, exist_ok=True)
                torch.save(dict(x=xy_module.x.detach().cpu(), y=xy_module.y.detach().cpu(), epoch=epoch, offset=offset(epoch)),
                           os.path.join(out_dir, "nan_guard_layout.pt"))
            optimizer.zero_grad(set_to_none=True)
            if n_skip >= NAN_MAX_SKIPS:
                raise SystemExit(f"[nan-guard] {n_skip} non-finite steps in a row at step {epoch+1}: stopping")
        else:
            n_skip = 0
            torch.nn.utils.clip_grad_norm_(xy_module.parameters(), max_norm=args.grad_clip)
            # --freeze-unseen: a detector with no signal in any shower of the batch keeps its position. Zeroing its
            # gradient is not enough (Adam's momentum would still move it), so its position is restored after the step.
            unseen = (~seen) if (args.freeze_unseen and (seen := tot["seen"]) is not None) else None
            if unseen is not None:
                x_old, y_old = xy_module.x.detach().clone(), xy_module.y.detach().clone()
            optimizer.step()
            if unseen is not None:
                with torch.no_grad():
                    xy_module.x.data[unseen] = x_old[unseen]
                    xy_module.y.data[unseen] = y_old[unseen]
                tot["frozen"] = int(unseen.sum())
        tot.pop("seen", None)
        with torch.no_grad():
            e_new, n_new = project_to_mountain_ne(mountain, xy_module.x.detach().cpu(),
                                                  xy_module.y.detach().cpu())
            xy_module.x.data.copy_(e_new.to(DEVICE).to(xy_module.x.dtype))
            xy_module.y.data.copy_(n_new.to(DEVICE).to(xy_module.y.dtype))
        log.append(dict(epoch=epoch + 1, **tot))

        if (epoch + 1) % args.score_every == 0 or epoch == args.epochs - 1:
            s = score()
            score_log.append(dict(epoch=epoch + 1, U_score=s))
            if s > best_u:
                best_u = s
                best_x = xy_module.x.detach().cpu().clone()
                best_y = xy_module.y.detach().cpu().clone()
            print(f"  [score {epoch+1}] U_score={s:+.3f}  best={best_u:+.3f}", flush=True)
        if epoch == start or (epoch + 1) % args.log_every == 0:
            print(f"  [{epoch+1:5d}/{args.epochs}] U_batch={tot['U']:+.3f}  "
                  f"(u_theta={tot['u_theta']:.3f} u_phi={tot['u_phi']:.3f} "
                  f"u_e={tot['u_e']:.3f})  r={tot['r_mean']:.3f}  read={tot['read']:.3f} ghosts={tot['ghosts']:.3f}  "
                  f"lr={lr_at(args, epoch):.3g}  frozen={tot.get('frozen', 0)}  "
                  + (f"U_util={tot['U_util']:+.3f} cable={tot['cable_km']:.2f}km  " if cable else "")
                  + f"{(time.time() - t_loop) / (epoch + 1 - start):.2f}s/epoch", flush=True)
        if (epoch + 1) % args.ckpt_every == 0:
            os.makedirs(out_dir, exist_ok=True)
            torch.save(dict(xy=xy_module.state_dict(), optimizer=optimizer.state_dict(),
                            epoch=epoch + 1, log=log, score_log=score_log, best_u=best_u,
                            best_x=best_x, best_y=best_y), state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)
    ex.shutdown()
    print(f"[adam done] best scoring-set U={best_u:+.3f}")

    lbfgs_log = []
    if args.lbfgs_chunks > 0:
        best_u, best_x, best_y, lbfgs_log = run_lbfgs(args, mountain, surface, recon, out_dir, resp, load, n_sweep,
                                                      xy_module, score, gen_for, best_u, best_x, best_y, cable)

    print(f"[done] best scoring-set U={best_u:+.3f}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(dict(x=best_x, y=best_y), os.path.join(out_dir, "layout_best.pt"))
    with open(os.path.join(out_dir, "optimize_log.json"), "w") as f:
        json.dump(dict(config=vars(args), log=log, score_log=score_log, lbfgs_log=lbfgs_log, best_u=best_u),
                  f, indent=2)
    print(f"[write] {out_dir}/layout_best.pt, optimize_log.json")
    try:
        plot_layout_density(best_x, best_y, mountain, out_dir)
        print(f"[write] {out_dir}/layout_density.png")
    except Exception as exc:
        print(f"[plot] layout_density skipped ({exc!r})")


def run_lbfgs(args, mountain, surface, recon, out_dir, resp, load, n_sweep, xy_module, score, gen_for,
              best_u, best_x, best_y, cable=None):
    """Chunked L-BFGS polish after Adam, as 04_optimize_lbfgs_ensemble.py (e44cdd0): starting from the Adam best, one
    torch.optim.LBFGS step (strong-Wolfe line search, up to --lbfgs-max-iter iterations) per FIXED chunk of
    --lbfgs-chunk-events pool showers with ONE frozen background draw (the generator is re-seeded at every closure, so
    the objective is deterministic within a chunk); the closure is pool_utility over sub-batches of --chunk showers
    (share-weighted backward = the full-chunk gradient). After each chunk: project onto the mountain and score on the
    fixed scoring set (its own fixed draw): better -> new best; within --lbfgs-tol (relative) of the best -> carry on;
    else revert to the best. --freeze-unseen zeroes the gradient of detectors with no signal in the chunk.
    Resumes per chunk (lbfgs_state.pt). Log: chunk, closures, U_chunk, U_score, best, kept, epoch (continuing after the
    Adam end, one per closure)."""
    state_path = os.path.join(out_dir, "lbfgs_state.pt")
    rng = np.random.default_rng([args.seed, 31337])
    n_slots = n_sweep // args.lbfgs_chunk_events                    # disjoint chunks: every shower used at most once
    if args.lbfgs_chunks > n_slots:
        raise ValueError(f"--lbfgs-chunks {args.lbfgs_chunks} x {args.lbfgs_chunk_events} exceeds the {n_sweep}-event sweep pool")
    offsets = [int(s) * args.lbfgs_chunk_events for s in rng.permutation(n_slots)[:args.lbfgs_chunks]]
    log, c0 = [], 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        log, c0, best_u, best_x, best_y = st["log"], st["chunk"], st["best_u"], st["best_x"], st["best_y"]
        xy = st["xy"]
        print(f"[lbfgs resume] chunk {c0}/{args.lbfgs_chunks}, best U={best_u:+.3f}", flush=True)
    else:
        xy = (best_x, best_y)
    with torch.no_grad():
        xy_module.x.data.copy_(xy[0].to(DEVICE)); xy_module.y.data.copy_(xy[1].to(DEVICE))
    for c in range(c0, args.lbfgs_chunks):
        batch = load(offsets[c], offsets[c] + args.lbfgs_chunk_events)
        seed_c = args.seed * 1000003 + 500_000_000 + c
        opt = torch.optim.LBFGS(xy_module.parameters(), lr=1.0, max_iter=args.lbfgs_max_iter, history_size=20,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-11, tolerance_change=1e-13)
        calls = dict(n=0, U=float("nan"), frozen=0)

        def closure():
            opt.zero_grad(set_to_none=False)
            tot = pool_utility(xy_module, batch, surface, recon, args.chunk, backward=True, resp=resp, gen=gen_for(seed_c))
            if cable:
                add_cable(cable, xy_module, tot, True)
            calls["n"] += 1; calls["U"] = tot["U"]
            for prm in xy_module.parameters():
                if prm.grad is not None:
                    bad = ~torch.isfinite(prm.grad)
                    if bad.any():
                        prm.grad[bad] = 0.0
                    if args.freeze_unseen and tot["seen"] is not None:
                        prm.grad[~tot["seen"]] = 0.0
                        calls["frozen"] = int((~tot["seen"]).sum())
            return torch.tensor(-tot["U"] if np.isfinite(tot["U"]) else 1e9, device=DEVICE)

        t0 = time.time()
        opt.step(closure)
        with torch.no_grad():
            e_new, n_new = project_to_mountain_ne(mountain, xy_module.x.detach().cpu(), xy_module.y.detach().cpu())
            xy_module.x.data.copy_(e_new.to(DEVICE).to(xy_module.x.dtype))
            xy_module.y.data.copy_(n_new.to(DEVICE).to(xy_module.y.dtype))
        s = score()
        if np.isfinite(s) and s > best_u:
            kept, best_u = "new best", s
            best_x, best_y = xy_module.x.detach().cpu().clone(), xy_module.y.detach().cpu().clone()
        elif np.isfinite(s) and s >= best_u - args.lbfgs_tol * abs(best_u):
            kept = "carry"
        else:
            kept = "revert"
            with torch.no_grad():
                xy_module.x.data.copy_(best_x.to(DEVICE)); xy_module.y.data.copy_(best_y.to(DEVICE))
        tot_closures = (log[-1]["closures_total"] if log else 0) + calls["n"]
        log.append(dict(chunk=c, offset=offsets[c], closures=calls["n"], closures_total=tot_closures,
                        epoch=args.epochs + tot_closures, U_chunk=calls["U"], U_score=s, best=best_u, kept=kept,
                        frozen=calls["frozen"]))
        print(f"  [lbfgs chunk {c + 1}/{args.lbfgs_chunks}] {calls['n']} closures  U_chunk={calls['U']:+.3f}  "
              f"U_score={s:+.3f}  best={best_u:+.3f}  {kept}  frozen={calls['frozen']}  {time.time() - t0:.0f}s", flush=True)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(dict(log=log, chunk=c + 1, best_u=best_u, best_x=best_x, best_y=best_y,
                        xy=(xy_module.x.detach().cpu().clone(), xy_module.y.detach().cpu().clone())), state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
    return best_u, best_x, best_y, log


def plot_layout_density(best_x, best_y, mountain, out_dir):
    """layout_density.png for one layout (plot_density_heatmap with a single run: Gaussian blobs at the detectors).
    Uses its own CPU surface (grid_sample needs surface and points on one device)."""
    surface = SurfaceUpMap.from_mountain(mountain).to("cpu")
    x_np, y_np = best_x.cpu().numpy(), best_y.cpu().numpy()
    aligned = np.stack([x_np, y_np], axis=-1)[None, :, :]
    _opt_plotting.plot_density_heatmap(
        aligned, x_np, y_np, mountain,
        os.path.join(out_dir, "layout_density.png"), surface=surface,
        count_word="K", count_suffix=" run (single chain, not an ensemble)")


def load_recon(recon_dir, device):
    """The frozen DeepSets recon."""
    ckpt = torch.load(os.path.join(recon_dir, "recon.pt"), map_location=device, weights_only=False)
    recon = build_recon_from_ckpt(ckpt, N_DETECTORS, device)
    cfg = ckpt.get("config", {})
    print(f"[load] recon.pt  model={cfg.get('model_type', 'mlp')}  "
          f"epoch={ckpt.get('epoch', '?')}  val={ckpt.get('val_total', '?')}  "
          f"recon_input={cfg.get('recon_input', '?')}  <- {recon_dir}")
    return recon


def init_layout(mountain, scheme, device, n_det=N_DETECTORS):
    e, n = sample_initial_layout_ne(mountain, n_units=n_det, scheme=scheme)
    e = torch.as_tensor(e, dtype=torch.float32)
    n = torch.as_tensor(n, dtype=torch.float32)
    e, n = project_to_mountain_ne(mountain, e, n)
    return e.to(device), n.to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", choices=("grid", "center"), default="center")
    ap.add_argument("--n-det", type=int, default=N_DETECTORS,
                    help="detectors in the layout; the DeepSets recon takes any count")
    ap.add_argument("--epochs", type=int, default=200, help="Adam steps")
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--lr-final", type=float, default=None,
                    help="decay the step size log-linearly from --lr to this over the run; unset = constant")
    ap.add_argument("--grad-clip", type=float, default=100.0)
    ap.add_argument("--batch-events", type=int, default=512, help="fresh pool showers per step")
    ap.add_argument("--chunk", type=int, default=8, help="showers per backward (memory only; the gradient is exact)")
    ap.add_argument("--n-score", type=int, default=1024, help="scoring-set showers (end of the pool)")
    ap.add_argument("--score-every", type=int, default=250)
    ap.add_argument("--score-chunk", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--ckpt-every", type=int, default=250,
                    help="save optimizer state every N steps; a rerun with the same --out-suffix resumes")
    ap.add_argument("--recon-folder", default=RECON_FOLDER + "_deepsets")
    ap.add_argument("--out-suffix", type=str, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--precompute-bw", type=int, default=None, metavar="SPECIES_INDEX",
                    help="write the pool's interpolated-kernel widths for one species (needed once) and exit")
    ap.add_argument("--bg-window-ns", type=float, default=0.0,
                    help="cosmic-ray background in one readout window of this length [ns] around the earliest hit "
                         "(modules/optimize/background.py); 0 = none")
    ap.add_argument("--gate", choices=("none", "knn23"), default="none",
                    help="knn23: a detector is read out only if >= 2 of {itself, its 2 nearest neighbours} have "
                         "E' > 0.001 (exact forward, straight-through soft gradient)")
    ap.add_argument("--t-fallback-us", type=float, default=27.7,
                    help="background window centre for showers with no timed detector (run 19 median timed T)")
    ap.add_argument("--response", choices=("kernel", "surrogate"), default="kernel",
                    help="detector response: the kernel, or the Stage 2 hurdle surrogate thresholded at "
                         "P(lit) >= --p-lit-min")
    ap.add_argument("--fnn-folder", default=FNN_FOLDER)
    ap.add_argument("--p-lit-min", type=float, default=0.5)
    ap.add_argument("--freeze-unseen", action="store_true",
                    help="a detector with no signal (E' > 0.001, read out) in any shower of the step's batch / "
                         "L-BFGS chunk keeps its position")
    ap.add_argument("--lbfgs-chunks", type=int, default=0,
                    help="after Adam: L-BFGS on this many fixed pool chunks, validated on the scoring set")
    ap.add_argument("--lbfgs-chunk-events", type=int, default=4096)
    ap.add_argument("--lbfgs-max-iter", type=int, default=40)
    ap.add_argument("--lbfgs-tol", type=float, default=0.05,
                    help="carry on from a chunk whose scoring-set U is within this relative tolerance of the best")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    out_dir = OPT_FOLDER + "_lbfgs_kernel" + (f"_{args.out_suffix}" if args.out_suffix else "")
    print(f"[stage 4] device={DEVICE}  utility={UTILITY_MODE}  response={args.response}  {args.n_det} detectors  "
          f"{args.epochs} steps x {args.batch_events} showers  -> {out_dir}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)

    if args.precompute_bw is not None:
        precompute_bandwidths(mountain, args.precompute_bw)
        return

    recon = load_recon(args.recon_folder, DEVICE)
    resp = None
    if args.response == "surrogate" or args.bg_window_ns > 0 or args.gate != "none" or args.freeze_unseen:
        resp = dict(track_seen=args.freeze_unseen,
                    surrogate=(load_dual_surrogate(args.fnn_folder, DEVICE, n_det=args.n_det)
                               if args.response == "surrogate" else None),
                    p_min=args.p_lit_min, window_ns=args.bg_window_ns, gate=args.gate,
                    t_fallback_us=args.t_fallback_us)
        print(f"[response] {args.response}  background window {args.bg_window_ns:g} ns  gate {args.gate}"
              + (f"  P(lit) >= {args.p_lit_min}" if args.response == "surrogate" else ""))
    run_main_pool(args, mountain, surface, recon, out_dir, resp)


if __name__ == "__main__":
    main()
