"""Analytic (physics-based) reconstruction, scored against the same val split
Stage 3 trains and validates on — the reconstruction-independent reference
docs/stage3-analytic-validation.md's Part A calls for. Nothing in this pipeline
has ever been compared against one: `true_utility.py`'s GENUINE/ARTIFACT verdict,
Stage 3's own val loss, all of it is "how well does this one frozen network do",
with no floor and no ceiling.

A Python/torch PORT of `/n/home05/zdimitrov/tambo/tambo_dir_reco/src/reco.jl`
(TamboDirReco.jl, a sibling repo, own git history) — not a Julia bridge. That
package consumes `Tambo.Frame` hit tables and needs Tambo.jl/TamboReader/Minuit2;
porting its three scoring functions to this corpus's `(x, y, E, T)` tensors is far
less work than an adapter plus a Julia dependency, and keeps every fit
differentiable in torch in case a later Stage-3 change wants one of them as an
input feature.

Five sub-analyses, run in this order because each gates the next:

  A0  Does the timing even support a front fit? Detector depth is
      z_cont = (EAST_ENTRY - East) / LAYER_EAST_DX with 500 m planes, so raw
      arrival time T is dominated by a near-deterministic ramp in East (see
      `modules/showers/kernel.py`: T is a leading-edge time, not a weighted
      mean, 0 = no hit). The question A1's plane fit depends on is whether the
      *transverse* (North, Up) structure explains anything ADDITIONAL beyond
      that ramp. If not, direction information in this corpus is carried by
      WHICH detectors fire, not WHEN, and A1/A2 are measuring noise.

  A1  Direction: a weighted plane-front fit, the linear-algebra form of
      `reco.jl`'s `plane_chisq_per_hit` (:22-40). Their version parametrizes the
      front by (theta, phi, t0) and minimizes a chi-square with Minuit because
      Tambo's convention needs the physical angles directly and a hard penalty
      for unphysical (upward) directions; but a plane wave's arrival-time model
      T_i = t0 + s . pos_i is LINEAR in the unknowns (t0, s) with s = n_hat / c,
      so it has a closed-form weighted least-squares solution -- no iterative
      solver, no angle parametrization, no local minima. n_hat = normalize(s)
      IS the fitted direction; |s| is not exactly 1/c for a noisy/finite array,
      which is itself a diagnostic (see the printed `|s|*c` summary).

      Weighting mirrors reco.jl's noise model (`sigma = 1km * sqrt(nhits)`,
      i.e. weight ~ nhits): this corpus has no nhits, so raw kernel deposit `E`
      is used as the multiplicity proxy (docs/THEORY.md SS10.4: a single fired
      detector's time is close to pure shower noise, so the fit must aggregate
      across detectors, weighted toward the better-sampled ones).

  A2  Origin: `reco.jl`'s `timing_llh_per_hit` (:42) adds a quadratic front-
      curvature term `pred_delay = a * r^2` on top of the plane fit -- exactly
      the "assume the front is a sphere" idea this file was requested to test.
      Ported via first principles rather than replaying their coordinate
      rotation: A1's own fit residual (t_i - t0 - s.pos_i) is BY CONSTRUCTION
      already the "what's left after removing the plane term" quantity their
      `delay` variable is built to be, so fitting `resid_i ~= a * r_i^2` (r_i =
      perpendicular distance from a deposit-weighted core to the A1 axis, same
      `get_core` weighting reco.jl uses) is the identical physics with one less
      moving part. The standard spherical-front correction is
      pred_delay ~= r^2 / (2 c R), so R = 1 / (2 c a) estimates the source
      distance -- compared here against the TRUE decay vertex, already a label
      (`primary[:, 5:8]`, ENU relative to array centre) that Stage 3 currently
      discards.

      Expect this to fail, informatively: at a ~1 km decay distance and a
      ~1 km array the curvature term is orders of magnitude below the plane
      term and per-detector timing is shower-noise dominated (docs/THEORY.md
      SS10.4). A clean negative here is itself the answer to "can the array
      localize the vertex from timing" -- worth knowing before anything asks
      Stage 4 to optimize for it.

  A3  Energy: `reco.jl`'s `Q_ldf` (:68) + `ldf_llh_per_hit` (:74), a lateral
      distribution function (LDF) fit -- deposit vs perpendicular distance from
      the shower axis, fit by maximizing a Poisson-like log-likelihood over
      (log_A, beta) with kappa fixed. Ported with the SAME un-hit-module term
      the Julia code has and this pipeline's recon has no analogue for: every
      one of the 100 detectors -- not just the ones that fired -- contributes
      `y_i log(y_pred_i) - y_pred_i` to the likelihood, so a DARK detector
      constrains the fit downward (real information: it bounds the footprint)
      exactly as it does in `run_minuit_on_ldf_llh`'s `unhit` argument.

      log_A is a fitted per-event "deposit density at r_ref" -- Q_ldf(r_ref)=1
      by construction, so exp(log_A) is that density directly. Calibrated to
      log_e via a 1-parameter linear fit on the TRAIN split (never touching
      val), then scored on val, so this is an honest out-of-sample comparison
      against recon's own log_e RMSE (0.341 dex on the archived run).

  A4  Multiplicity vs energy and vs distance: nobody has plotted N_lit against
      primary energy or axis-to-nearest-detector distance anywhere in this
      repo (checked every notebook, every plots/ script, every run-dir dump).
      It is load-bearing: `modules/optimize/objective.py`'s reconstructability
      gate spans p10=13/p50=48/p90=82 lit detectors and carries a TODO flagging
      its thresholds as stale (tuned before the photon component was summed
      in). This is the cheapest of the five -- no fitting, no iteration --
      and it says where reconstruction actually fails, physically.

Reads Step-1 outputs directly and reuses `shower_level_split` /
`load_kernel_responses` from `scripts/03_train_recon_deepsets.py` (imported by
path, the same trick `recon_regimes.py` uses) so this scores EXACTLY the same
val rows recon validated on -- including the blob-row filter, so a run of this
script after the blob-filtering fix lands is comparable to Stage 3's own
(now-corrected) validation metric. Read-only: no pipeline file is written.

Needs a GPU node for the corpus load (E.pt/T.pt are full-corpus, ~3.4 GB each
before the species-combine's temporaries) -- same footprint as
`recon_regimes.py` and `plots/training/aleatoric_floor.py`; run it there, e.g.:

    python plots/training/04_analytic_recon.py --n-val 20000 --n-train-calib 20000
"""
import argparse
import json
import os
import sys
import time
import importlib.util

# v6 root = parent of this file's plots/ dir, found by walking up to the
# _pathfix.py marker rather than counting parents (CWD-independent import).
_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = _HERE
while _V6 != os.path.dirname(_V6) and not os.path.exists(os.path.join(_V6, "_pathfix.py")):
    _V6 = os.path.dirname(_V6)
for _p in (_V6, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

import modules  # noqa: F401 — package import; keeps modules on the path
from modules.constants import (
    TRAINING_DATASET_FOLDER, RECON_FOLDER,
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, N_DETECTORS, T_LOG_SCALE,
    LOG_E_MIN, LOG_E_MAX,
)
from modules.geometry import load_tr_mountain, SurfaceUpMap
from modules.optimize import LAYOUT_THRESHOLD

# Import scripts/03_train_recon_deepsets.py by path (its filename starts with a
# digit, so it cannot be `import`ed by name) — the same trick recon_regimes.py
# uses, so this reuses the EXACT split/combination logic Stage 3 trains against
# rather than a re-derived copy that could silently drift from it.
_spec = importlib.util.spec_from_file_location(
    "_s3", os.path.join(_V6, "scripts", "03_train_recon_deepsets.py"))
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)

C_LIGHT = 299_792_458.0   # m/s — the plane/curvature fits' unit of speed
OUTPUT_FOLDER = RECON_FOLDER + "_deepsets"


# ── Core math (unit-tested against synthetic events before wiring to data) ──

def weighted_plane_fit(pos: torch.Tensor, t: torch.Tensor, w: torch.Tensor,
                       ridge: float = 1e-6):
    """Weighted least-squares fit of a plane-wave arrival-time model
    ``t_i = t0 + s . pos_i`` (`s = n_hat / c`), the linear form of
    `reco.jl`'s `plane_chisq_per_hit`. `w` is the multiplicity-proxy weight
    (0 excludes a detector, e.g. it never fired) — the weighted normal
    equations `(X^T W X) beta = X^T W t` are solved directly with a small
    ridge for numerical safety near-singular batches (near-planar arrays, or
    too few weighted detectors).

    Args:
        pos : (B, D, 3) detector [East, North, Up], metres.
        t   : (B, D) observed times, seconds.
        w   : (B, D) nonnegative weights.
    Returns:
        t0 (B,), s (B, 3), resid (B, D) = t - (t0 + s . pos),
        cond (B,) a SCALE-INVARIANT conditioning diagnostic: the min/max
        eigenvalue ratio of `A` normalized by its own diagonal (equivalent to
        the condition number of `A`'s "correlation matrix"). The raw ratio is
        NOT usable here — `A`'s diagonal entries span the constant, East/North/
        Up columns' units (metres, metres^2), so its eigenvalues differ by
        ~1e10 purely from units, not from anything about the actual fit.
        Verified against the real corpus (measured 2026-09-09): even properly
        normalized, `cond` is structurally small (median ~1e-6) because Up is
        a smooth deterministic function of (East, North) via the mountain's
        `SurfaceUpMap` — i.e. this array's 3 spatial columns are only
        ~2.something-dimensional, not fully independent. That is real
        geometry, not a numerical artifact, so `cond` is reported as a
        diagnostic (see A1's printed distribution) rather than used to reject
        events — the ridge term already keeps the solve well-posed regardless.
    """
    B, D, _ = pos.shape
    # Centre positions per event before the solve: East sits ~1400-1500 m
    # (near EAST_ENTRY), not near 0, so fitting the intercept t0 alongside a
    # slope on UNCENTRED East badly conditions that pair in float32 (t0 and
    # s_east become almost perfectly anti-correlated) — measured on the real
    # corpus (2026-09-09) giving R2 around -1e15, i.e. a "successful" solve
    # that is numerically garbage. Centering is a linear reparametrization
    # (t0 = t0' - s.mean_pos), so s/resid/cond are identical to the
    # uncentred solve in exact arithmetic; only the float32 rounding differs,
    # and it differs by orders of magnitude here.
    mean_pos = pos.mean(dim=1, keepdim=True)                # (B, 1, 3)
    pos_c = pos - mean_pos
    ones = torch.ones(B, D, 1, dtype=pos.dtype, device=pos.device)
    X = torch.cat([ones, pos_c], dim=-1)                     # (B, D, 4)
    Xw = X * w.unsqueeze(-1)
    A = torch.einsum("bdi,bdj->bij", Xw, X)                  # (B, 4, 4)
    b = torch.einsum("bdi,bd->bi", Xw, t)                    # (B, 4)
    eye4 = torch.eye(4, dtype=pos.dtype, device=pos.device).unsqueeze(0)
    trace = A.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-30)
    A_reg = A + ridge * trace.view(B, 1, 1) * eye4 / 4.0
    beta = torch.linalg.solve(A_reg, b.unsqueeze(-1)).squeeze(-1)   # (B, 4)
    t0_c, s = beta[:, 0], beta[:, 1:]
    pred = t0_c.unsqueeze(-1) + torch.einsum("bi,bdi->bd", s, pos_c)
    resid = t - pred
    t0 = t0_c - torch.einsum("bi,bi->b", s, mean_pos.squeeze(1))   # back to pos's own frame
    diag = A.diagonal(dim1=-2, dim2=-1).clamp(min=1e-30)      # (B, 4)
    d_inv = diag.sqrt().reciprocal()
    A_norm = A * d_inv.unsqueeze(-1) * d_inv.unsqueeze(-2)    # unit diagonal
    eigvals = torch.linalg.eigvalsh(A_norm)                   # (B, 4), ascending
    cond = eigvals[:, 0].clamp(min=0) / eigvals[:, -1].clamp(min=1e-30)
    return t0, s, resid, cond


def weighted_centroid(pos: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Deposit-weighted mean detector position — `reco.jl`'s `get_core`."""
    wsum = w.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    return (pos * w.unsqueeze(-1)).sum(dim=1) / wsum


def perp_distance(pos: torch.Tensor, core: torch.Tensor,
                  n_hat: torch.Tensor) -> torch.Tensor:
    """Perpendicular distance from each detector to the line through `core`
    along unit direction `n_hat`. pos: (B, D, 3), core/n_hat: (B, 3)."""
    d = pos - core.unsqueeze(1)
    proj = torch.einsum("bdi,bi->bd", d, n_hat)
    perp = d - proj.unsqueeze(-1) * n_hat.unsqueeze(1)
    return perp.norm(dim=-1)


def fit_curvature(r: torch.Tensor, resid: torch.Tensor,
                  w: torch.Tensor) -> torch.Tensor:
    """Single-parameter weighted LS: `resid_i ~= a * r_i^2`, closed form
    `a = sum(w r^2 resid) / sum(w r^4)` — the curvature coefficient behind
    `reco.jl`'s `timing_llh_per_hit`'s `pred_delay = a * r^2` (see module
    docstring for the derivation from A1's own residual)."""
    num = (w * r ** 2 * resid).sum(dim=-1)
    den = (w * r ** 4).sum(dim=-1).clamp(min=1e-30)
    return num / den


def Q_ldf(r: torch.Tensor, beta: torch.Tensor,
         kappa: float = 0.35, r_ref: float = 100.0) -> torch.Tensor:
    """`reco.jl`'s lateral distribution shape, `Q_ldf(r; beta, kappa, r_ref)`.
    `u^(-beta - kappa log10 u)`, `u = r / r_ref`. `Q_ldf(r_ref) = 1` exactly,
    so a fitted amplitude `exp(log_A)` is the deposit density AT r_ref."""
    u = (r / r_ref).clamp(min=1e-6)
    return u ** (-beta - kappa * torch.log10(u))


def fit_ldf_poisson(r: torch.Tensor, y: torch.Tensor,
                    steps: int = 300, lr: float = 0.05, kappa: float = 0.35,
                    r_ref: float = 100.0):
    """Batched Poisson-likelihood LDF fit over ALL `D` detectors (not just the
    fired ones) — the un-hit-module term `reco.jl`'s `run_minuit_on_ldf_llh`
    gets from its separate `unhit` argument falls out for free here: a dark
    detector has `y_i = 0`, so its likelihood term is just `-y_pred_i`,
    identical to their `ldf_llh_per_hit` on an unhit module.

    `(log_A, beta)` are free per-event scalars, fit jointly across the whole
    batch via Adam (no Minuit dependency; the model is smooth and low-
    dimensional, so a few hundred gradient steps converge reliably — verified
    against synthetic Poisson-noise data before this was wired to real
    events). `beta` is parametrized through a softplus + floor so it stays in
    `reco.jl`'s (0, 5]-ish physical range without a hard constraint.

    This runs its OWN internal optimization (unlike every other function in
    this module, which is a single forward computation) — so it wraps that
    loop in `torch.enable_grad()` regardless of the caller's ambient grad
    state. Without it, calling this from inside a `@torch.no_grad()` scope
    (a perfectly reasonable thing for a caller that only wants a read-only
    evaluation, e.g. an analytic-recon check bolted onto an optimizer loop)
    makes every tensor here inherit no_grad too, and `.backward()` below
    fails with "does not have a grad_fn" — caught wiring this into
    `scripts/04_optimize_lbfgs_kernel.py`'s `analytic_eval`.

    Returns: log_A (B,), beta (B,), both detached.
    """
    with torch.enable_grad():
        B = r.shape[0]
        log_A = torch.zeros(B, device=r.device, requires_grad=True)
        beta_raw = torch.zeros(B, device=r.device, requires_grad=True)
        opt = torch.optim.Adam([log_A, beta_raw], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            beta = torch.nn.functional.softplus(beta_raw) + 0.1
            y_pred = torch.exp(log_A).unsqueeze(-1) * Q_ldf(r, beta.unsqueeze(-1),
                                                            kappa=kappa, r_ref=r_ref)
            y_pred = y_pred.clamp(min=1e-12)
            nll = -(y * torch.log(y_pred) - y_pred).sum(dim=-1)
            nll.sum().backward()
            opt.step()
        beta = torch.nn.functional.softplus(beta_raw) + 0.1
    return log_A.detach(), beta.detach()


def r2_pearson(pred: torch.Tensor, true: torch.Tensor):
    """Per-column MSE / R2 / Pearson-r, in `recon_regimes.py`'s exact format
    (`R2 = 1 - mse / var(true)`) so the printed table sits directly beside
    `three_input_regimes_diagnostic.txt`."""
    out = []
    for i in range(true.shape[1]):
        p, t = pred[:, i], true[:, i]
        mse = ((p - t) ** 2).mean().item()
        var = t.var().item()
        r = torch.corrcoef(torch.stack([p, t]))[0, 1].item()
        out.append(dict(mse=mse, r2=1.0 - mse / max(var, 1e-30), pearson_r=r))
    return out


def linear_calibrate(x_train, y_train, x_val):
    """1-D OLS with intercept, fit on train, applied to val — `y ~= m x + b`.
    Used for A3's log_A -> log_e calibration, so val is never touched by the
    fit itself (an honest out-of-sample comparison)."""
    X = torch.stack([torch.ones_like(x_train), x_train], dim=-1)   # (N, 2)
    beta = torch.linalg.lstsq(X, y_train.unsqueeze(-1)).solution.squeeze(-1)
    b, m = beta[0], beta[1]
    return b + m * x_val, (float(m), float(b))


# ── Wiring to the real corpus ────────────────────────────────────────────────

def load_val_and_calib_idx(strat_ids, blob_ids, n_val, n_train_calib, seed):
    """The SAME shower-level split + blob filter Stage 3 trains on (after the
    blob-filtering fix — B0 in docs/stage3-analytic-validation.md), then a
    seeded subsample of each side down to a workable size."""
    train_idx, val_idx = s3.shower_level_split(strat_ids, 0.10, 1)
    keep = ~blob_ids.bool()
    train_idx = train_idx[keep[train_idx]]
    val_idx = val_idx[keep[val_idx]]
    g = torch.Generator().manual_seed(seed)
    val_sub = val_idx[torch.randperm(len(val_idx), generator=g)[:n_val]]
    train_sub = train_idx[torch.randperm(len(train_idx), generator=g)[:n_train_calib]]
    return train_sub, val_sub


def detector_positions(xy_rows: torch.Tensor, surface: SurfaceUpMap,
                       device: torch.device) -> torch.Tensor:
    """xy_rows: (N, D, 2) = (East, North) per row (each row carries its OWN
    layout). Returns (N, D, 3) = (East, North, Up), Up from the same
    differentiable SurfaceUpMap the kernel itself is evaluated through."""
    N, D, _ = xy_rows.shape
    east = xy_rows[..., 0].reshape(-1).to(device)
    north = xy_rows[..., 1].reshape(-1).to(device)
    up = surface(north, east).reshape(N, D)
    return torch.stack([xy_rows[..., 0].to(device), xy_rows[..., 1].to(device), up],
                       dim=-1)


def run_direction_and_curvature(pos, T_raw, E_raw, min_fired):
    """A0 (gate), A1 (direction) and A2 (curvature -> distance) for one batch
    of events. Returns a dict of per-event tensors plus the A0 summary."""
    N, D, _ = pos.shape
    fired = T_raw > 0
    n_fired = fired.sum(dim=-1)
    valid_count = n_fired >= min_fired

    # ---- A0: how much of T is the East ramp vs the transverse structure? ----
    # East-only design (North, Up columns zeroed) reuses the SAME weighted
    # solver as the full 3-D fit below, rather than a separate 1-D code path —
    # `weighted_plane_fit` degenerates cleanly to a 2-parameter (t0, s_east)
    # fit when the other two design columns carry no information.
    w0 = fired.float()
    pos_east_only = torch.zeros_like(pos)
    pos_east_only[..., 0] = pos[..., 0]
    _, _, resid_east, _ = weighted_plane_fit(pos_east_only, T_raw, w0)
    _, _, resid_full, _ = weighted_plane_fit(pos, T_raw, w0)
    ss_res_east = (resid_east * w0).pow(2).sum(dim=-1)
    ss_res_full = (resid_full * w0).pow(2).sum(dim=-1)
    w_mean = (T_raw * w0).sum(dim=-1, keepdim=True) / w0.sum(dim=-1, keepdim=True).clamp(min=1)
    ss_tot = ((T_raw - w_mean) * w0).pow(2).sum(dim=-1)
    r2_east = 1.0 - ss_res_east / ss_tot.clamp(min=1e-30)
    r2_full = 1.0 - ss_res_full / ss_tot.clamp(min=1e-30)
    # A handful of events have T ~constant across their fired detectors (all
    # arriving within float32 noise of each other), so ss_tot is genuinely
    # ~0 -- R2's 0/0-like blow-up there is not a numerical bug in the fit
    # (resid itself is tiny too, ~1e-5 s, a fine fit) but the R2 CONCEPT
    # not applying: there is no timing variance for the plane term to have
    # explained. Measured on the real corpus (2026-09-09): a few such events
    # in a 500-row sample gave R2 ~ -1e15/-1e18 and dominated the batch mean
    # by many orders of magnitude. Flagged via a floor RELATIVE to this
    # batch's own scale (not a magic absolute constant, since T's scale
    # varies with the corpus/units) rather than clamping ss_tot itself, which
    # would just move the same blow-up to a different, still-wrong number.
    # Scale reference from valid_count events only (a degenerate few-fired-
    # detector event's near-zero ss_tot would otherwise drag the median down
    # and let other degenerate events slip past the floor too).
    _ss_tot_scale = (ss_tot[valid_count].median() if valid_count.any()
                    else torch.tensor(1.0, device=ss_tot.device)).clamp(min=1e-30)
    ss_tot_ok = ss_tot > 1.0e-3 * _ss_tot_scale

    # ---- A1: weighted plane fit, deposit E as the multiplicity proxy ----
    w1 = torch.where(fired, E_raw.clamp(min=0), torch.zeros_like(E_raw))
    t0, s, resid, cond = weighted_plane_fit(pos, T_raw, w1)
    s_norm = s.norm(dim=-1).clamp(min=1e-30)
    n_hat = s / s_norm.unsqueeze(-1)
    speed_check = s_norm * C_LIGHT   # should be ~1 for a clean plane fit

    # `cond` (see weighted_plane_fit's docstring) is structurally small for
    # this array even on a good fit -- Up correlates with (East, North) via
    # the terrain -- so it is NOT used to reject events; only `cond > 0`
    # excludes the genuinely degenerate case (fewer independent positions
    # than free parameters, e.g. every fired detector at the same point).
    ok = valid_count & (cond > 0) & torch.isfinite(cond)

    # ---- A2: curvature -> source distance ----
    core = weighted_centroid(pos, w1)
    r_perp = perp_distance(pos, core, n_hat)
    a_curv = fit_curvature(r_perp, resid, w1)
    R_dist = 1.0 / (2.0 * C_LIGHT * a_curv.clamp(min=1e-30))
    R_dist = torch.where(a_curv > 0, R_dist, torch.full_like(R_dist, float("nan")))

    return dict(
        ok=ok, n_fired=n_fired, cond=cond,
        t0=t0, n_hat=n_hat, resid=resid, speed_check=speed_check,
        core=core, r_perp=r_perp, a_curv=a_curv, R_dist=R_dist,
        r2_east=r2_east, r2_full=r2_full, ss_tot_ok=ss_tot_ok,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=OUTPUT_FOLDER,
                    help="recon output dir; results are written alongside "
                         "recon.pt there, but nothing under it is read")
    ap.add_argument("--n-val", type=int, default=20_000)
    ap.add_argument("--n-train-calib", type=int, default=20_000)
    ap.add_argument("--min-fired", type=int, default=4,
                    help="minimum fired detectors for A1's plane fit "
                         "(4 unknowns: t0, sx, sy, sz)")
    ap.add_argument("--ldf-steps", type=int, default=300)
    ap.add_argument("--ldf-r-ref", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    print("=" * 72)
    print("v6/04_analytic_recon.py — analytic (plane/LDF) reconstruction, "
          "kernel ground truth")
    print("=" * 72)
    os.makedirs(args.folder, exist_ok=True)

    t0 = time.time()
    primary = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    strat_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    blob_path = os.path.join(TRAINING_DATASET_FOLDER, "blob_ids.pt")
    if not os.path.exists(blob_path):
        raise SystemExit(f"blob_ids.pt not found in {TRAINING_DATASET_FOLDER} — "
                         "re-run 01_build_dataset_northeast.py")
    blob_ids = torch.load(blob_path)
    print(f"[load] corpus metadata in {time.time() - t0:.1f}s  primary={tuple(primary.shape)}")

    train_sub, val_sub = load_val_and_calib_idx(
        strat_ids, blob_ids, args.n_val, args.n_train_calib, args.seed)
    print(f"[split] train-calib={len(train_sub)}  val={len(val_sub)}")

    t0 = time.time()
    E_kern, T_kern = s3.load_kernel_responses(TRAINING_DATASET_FOLDER)
    print(f"[kernel] combined species labels in {time.time() - t0:.1f}s")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                               east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                               n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(device)
    array_center = torch.as_tensor(mountain.centroids_ENU, dtype=torch.float32).mean(dim=0)

    def prep(idx):
        """Raw physical quantities for one index set, on `device`."""
        pos = detector_positions(xy[idx], surface, device)                # (N,D,3)
        E_raw = torch.expm1(E_kern[idx]).clamp(min=0).to(device)          # counts
        T_raw = (torch.expm1(T_kern[idx]).clamp(min=0) / T_LOG_SCALE).to(device)  # sec
        prim = primary[idx].to(device)
        return pos, E_raw, T_raw, prim

    # ---- A0/A1/A2 on the val split ----
    pos_v, E_v, T_v, prim_v = prep(val_sub)
    fits_v = run_direction_and_curvature(pos_v, T_v, E_v, args.min_fired)
    ok_v = fits_v["ok"]
    frac_ok = float(ok_v.float().mean())
    # A0's R2 is undefined (0/0-like), not just noisy, for an event whose
    # fired-detector T is ~constant (ss_tot ~0) -- ss_tot_ok excludes those
    # rather than let them dominate the mean by many orders of magnitude
    # (measured: a few such events gave R2 ~ -1e15/-1e18 on real data). Report
    # the exclusion count explicitly, and MEDIAN alongside mean since even
    # after that exclusion, per-event R2 is a fat-tailed quantity -- a plane
    # fit can genuinely fail badly on one noisy event without anything being
    # wrong with the arithmetic.
    ok_r2 = ok_v & fits_v["ss_tot_ok"]
    n_excluded = int(ok_v.sum()) - int(ok_r2.sum())
    r2e, r2f = fits_v["r2_east"][ok_r2], fits_v["r2_full"][ok_r2]
    print(f"[A0] R2(East ramp)      mean={r2e.mean():.4f}  median={r2e.median():.4f}")
    print(f"     R2(East+transverse) mean={r2f.mean():.4f}  median={r2f.median():.4f}  "
          f"(delta of medians={float(r2f.median() - r2e.median()):.4f})")
    print(f"     ({n_excluded}/{int(ok_v.sum())} otherwise-qualifying events excluded: "
          f"near-zero T variance across fired detectors, R2 undefined there)")
    print(f"[A1] qualifying events (>= {args.min_fired} fired, non-degenerate): "
          f"{frac_ok:.1%}  |s|*c mean={float(fits_v['speed_check'][ok_v].mean()):.3f} "
          f"(1.0 = a clean plane fit)")
    cond_q = torch.quantile(fits_v["cond"][ok_v], torch.tensor([0.1, 0.5, 0.9], device=ok_v.device))
    print(f"    conditioning (scale-invariant, see weighted_plane_fit docstring): "
          f"p10={cond_q[0]:.2e}  p50={cond_q[1]:.2e}  p90={cond_q[2]:.2e} "
          f"(structurally small -- Up correlates with East/North via the terrain)")

    true_dir_v = prim_v[:, :3]
    pred_dir_v = fits_v["n_hat"]
    # Global sign: the linear fit has no ambiguity (s is a literal regression
    # coefficient, not a chosen angle), but this port's East/North/Up axis
    # convention could still disagree with the corpus's dir_x/y/z convention by
    # an overall reflection. Fix ONE sign for the whole run from the sign of
    # the mean dot product, rather than per event (physics fixes this globally).
    sign = torch.sign((pred_dir_v[ok_v] * true_dir_v[ok_v]).sum(-1).mean())
    if sign == 0:
        sign = torch.tensor(1.0)
    pred_dir_v = pred_dir_v * sign
    dir_stats = r2_pearson(pred_dir_v[ok_v], true_dir_v[ok_v])
    for name, st in zip(("dir_x", "dir_y", "dir_z"), dir_stats):
        print(f"    {name:6s} mse={st['mse']:.5f}  R2={st['r2']:6.3f}  "
              f"pearson_r={st['pearson_r']:6.3f}")

    # ---- A2: source distance vs true decay-vertex distance ----
    # primary[:, 5:8] is the decay vertex relative to array centre (both ENU);
    # `core` (A1's deposit-weighted centroid) is already in absolute ENU, so
    # the comparison distance is simply true_vertex - core, both in ENU.
    true_vertex_v = prim_v[:, 5:8] + array_center.to(device)          # ENU
    true_dist_v = (true_vertex_v - fits_v["core"]).norm(dim=-1)
    ok_a2 = ok_v & torch.isfinite(fits_v["R_dist"])
    frac_a2 = float(ok_a2.float().mean())
    if ok_a2.any():
        R_err = (fits_v["R_dist"][ok_a2] - true_dist_v[ok_a2])
        print(f"[A2] {frac_a2:.1%} events give a positive curvature (a source "
              f"distance at all); median |error|={float(R_err.abs().median()):.0f} m "
              f"vs true distances ~{float(true_dist_v[ok_a2].median()):.0f} m")
    else:
        print("[A2] no event returned a positive curvature fit — the array "
              "cannot localize the vertex from timing (expected; see module docstring)")

    # ---- A3: LDF energy fit, calibrated on train, scored on val ----
    pos_t, E_t, T_t, prim_t = prep(train_sub)
    fits_t = run_direction_and_curvature(pos_t, T_t, E_t, args.min_fired)
    ok_t = fits_t["ok"]

    log_A_v, beta_v = fit_ldf_poisson(fits_v["r_perp"], E_v, steps=args.ldf_steps,
                                      r_ref=args.ldf_r_ref)
    log_A_t, beta_t = fit_ldf_poisson(fits_t["r_perp"], E_t, steps=args.ldf_steps,
                                      r_ref=args.ldf_r_ref)

    true_log_e_v = prim_v[:, 3] * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN   # decades
    true_log_e_t = prim_t[:, 3] * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN

    ok_a3_t = ok_t & torch.isfinite(log_A_t)
    ok_a3_v = ok_v & torch.isfinite(log_A_v)
    pred_log_e_v, (m_fit, b_fit) = linear_calibrate(
        log_A_t[ok_a3_t], true_log_e_t[ok_a3_t], log_A_v[ok_a3_v])
    dex_rmse = float(((pred_log_e_v - true_log_e_v[ok_a3_v]) ** 2).mean().sqrt())
    e_stats = r2_pearson(pred_log_e_v.unsqueeze(-1), true_log_e_v[ok_a3_v].unsqueeze(-1))[0]
    print(f"[A3] log_A -> log_e calibration (train): log_e = {m_fit:.4f}*log_A + {b_fit:.4f}")
    print(f"    log_e  mse={e_stats['mse']:.5f}  R2={e_stats['r2']:6.3f}  "
          f"pearson_r={e_stats['pearson_r']:6.3f}  RMSE={dex_rmse:.4f} dex "
          f"(recon's kernel-regime RMSE was 0.341 dex)")

    # Fixed-vs-marginal distance check: bin val by TRUE decay distance and
    # report the LDF's dex-RMSE per bin, vs the marginal number above -- tests
    # whether energy error shrinks once distance is controlled for (the
    # degeneracy hypothesis behind B4 in the plan).
    dist_v_a3 = true_dist_v[ok_a3_v]
    n_bins = 4
    try:
        edges = torch.quantile(dist_v_a3, torch.linspace(0, 1, n_bins + 1))
    except RuntimeError:
        edges = None
    dist_bin_rmse = []
    if edges is not None:
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            m = (dist_v_a3 >= lo) & (dist_v_a3 <= hi if i == n_bins - 1 else dist_v_a3 < hi)
            if m.sum() > 10:
                err = (pred_log_e_v[m] - true_log_e_v[ok_a3_v][m])
                dist_bin_rmse.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()),
                                          dex_rmse=float((err ** 2).mean().sqrt())))
    print(f"[A3] dex-RMSE by true decay distance: "
          + ", ".join(f"[{d['lo']:.0f},{d['hi']:.0f}]m n={d['n']} rmse={d['dex_rmse']:.3f}"
                       for d in dist_bin_rmse))

    # ---- A4: multiplicity vs energy and vs distance (no fitting) ----
    lit_thresh = (E_v > LAYOUT_THRESHOLD).sum(dim=-1).float()
    lit_nonzero = (E_v > 0).sum(dim=-1).float()
    true_e_v = torch.pow(10.0, true_log_e_v)   # GeV
    # Axis-to-nearest-detector distance from the TRUE axis (decay vertex +
    # true direction), the same quantity explore_trigger_counts.ipynb's
    # placement-closure cell computes as `axis_dist_m`.
    axis_dist_v = perp_distance(pos_v, true_vertex_v, true_dir_v).min(dim=-1).values
    corr_e = torch.corrcoef(torch.stack([lit_thresh, true_log_e_v]))[0, 1]
    corr_d = torch.corrcoef(torch.stack([lit_thresh, axis_dist_v]))[0, 1]
    print(f"[A4] N_lit vs log10(E): pearson_r={float(corr_e):.3f}   "
          f"N_lit vs axis distance: pearson_r={float(corr_d):.3f}")

    # ── Output: JSON + text table (recon_regimes.py format) + PNG ───────────
    result = dict(
        n_val=int(len(val_sub)), n_train_calib=int(len(train_sub)),
        min_fired=args.min_fired,
        A0=dict(r2_east_mean=float(r2e.mean()), r2_east_median=float(r2e.median()),
                r2_full_mean=float(r2f.mean()), r2_full_median=float(r2f.median()),
                n_excluded_zero_variance=n_excluded, n_qualifying=int(ok_v.sum())),
        A1=dict(qualifying_frac=frac_ok,
                mean_speed_check=float(fits_v["speed_check"][ok_v].mean()),
                sign_convention=float(sign),
                channels=dict(zip(("dir_x", "dir_y", "dir_z"), dir_stats))),
        A2=dict(qualifying_frac=frac_a2,
                median_abs_error_m=(float(R_err.abs().median()) if ok_a2.any() else None),
                median_true_distance_m=(float(true_dist_v[ok_a2].median()) if ok_a2.any() else None)),
        A3=dict(calibration=dict(m=m_fit, b=b_fit),
               log_e=e_stats, dex_rmse=dex_rmse,
               by_distance_bin=dist_bin_rmse),
        A4=dict(pearson_r_vs_logE=float(corr_e), pearson_r_vs_axis_dist=float(corr_d)),
    )
    json_path = os.path.join(args.folder, "analytic_recon_diagnostic.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[write] {json_path}")

    txt_path = os.path.join(args.folder, "analytic_recon_diagnostic.txt")
    with open(txt_path, "w") as f:
        f.write(f"[eval] {len(val_sub)} val rows (analytic reconstruction, kernel "
               f"ground truth, --min-fired={args.min_fired})\n\n")
        f.write("=== ANALYTIC (plane-front + LDF) reconstruction ===\n\n")
        f.write(f"  A0  R2(East ramp)       mean={result['A0']['r2_east_mean']:.4f}  "
               f"median={result['A0']['r2_east_median']:.4f}\n")
        f.write(f"      R2(East+transverse) mean={result['A0']['r2_full_mean']:.4f}  "
               f"median={result['A0']['r2_full_median']:.4f}  "
               f"({result['A0']['n_excluded_zero_variance']}/"
               f"{result['A0']['n_qualifying']} excluded: near-zero T variance)\n")
        f.write(f"  A1  qualifying={frac_ok:.1%}  |s|*c={result['A1']['mean_speed_check']:.3f}\n")
        for name, st in zip(("dir_x", "dir_y", "dir_z"), dir_stats):
            f.write(f"    {name:6s} mse={st['mse']:.5f}  R2={st['r2']:6.3f}  "
                   f"pearson_r={st['pearson_r']:6.3f}\n")
        f.write(f"  A2  qualifying={frac_a2:.1%}\n")
        f.write(f"  A3  log_e  mse={e_stats['mse']:.5f}  R2={e_stats['r2']:6.3f}  "
               f"pearson_r={e_stats['pearson_r']:6.3f}  RMSE={dex_rmse:.4f} dex\n")
        f.write(f"  A4  N_lit vs log10(E) r={result['A4']['pearson_r_vs_logE']:.3f}   "
               f"N_lit vs axis_dist r={result['A4']['pearson_r_vs_axis_dist']:.3f}\n")
    print(f"[write] {txt_path}")

    _plot(args.folder, fits_v, ok_v, true_dir_v, pred_dir_v, true_dist_v, ok_a2,
         true_log_e_v, pred_log_e_v, ok_a3_v, lit_thresh, true_e_v, axis_dist_v)


def _plot(folder, fits_v, ok_v, true_dir, pred_dir, true_dist, ok_a2,
         true_log_e, pred_log_e, ok_a3_v, lit_thresh, true_e, axis_dist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # A0: per-event R2 gain from the transverse terms. Same ss_tot_ok
    # exclusion as the printed A0 stats -- an event with ~0 T variance has
    # an undefined R2 (not just a noisy one) and would otherwise put a
    # handful of +-1e15-scale bars in an otherwise O(1)-ranged histogram.
    ok_r2_plot = ok_v & fits_v["ss_tot_ok"]
    gain = (fits_v["r2_full"] - fits_v["r2_east"])[ok_r2_plot].cpu().numpy()
    axes[0, 0].hist(gain, bins=40, color="C0")
    axes[0, 0].set_xlabel("R2(East+transverse) - R2(East only)")
    axes[0, 0].set_title("A0: transverse timing structure per event")

    # A1: angular error histogram
    cos = (pred_dir[ok_v] * true_dir[ok_v]).sum(-1).clamp(-1, 1)
    ang_err_deg = torch.rad2deg(torch.arccos(cos)).cpu().numpy()
    axes[0, 1].hist(ang_err_deg, bins=40, color="C1")
    axes[0, 1].set_xlabel("angular error [deg]")
    axes[0, 1].set_title("A1: plane-front direction error")

    # A2: fitted vs true source distance
    if ok_a2.any():
        axes[0, 2].scatter(true_dist[ok_a2].cpu().numpy(),
                           fits_v["R_dist"][ok_a2].cpu().numpy(), s=3, alpha=0.3)
        lo, hi = float(true_dist[ok_a2].min()), float(true_dist[ok_a2].max())
        axes[0, 2].plot([lo, hi], [lo, hi], "k--", lw=1)
        axes[0, 2].set_xlabel("true decay distance [m]")
        axes[0, 2].set_ylabel("fitted source distance [m]")
    else:
        axes[0, 2].text(0.5, 0.5, "no event gave a\npositive curvature fit",
                        ha="center", va="center", transform=axes[0, 2].transAxes)
    axes[0, 2].set_title("A2: curvature -> source distance")

    # A3: predicted vs true log_e
    axes[1, 0].scatter(true_log_e[ok_a3_v].cpu().numpy(), pred_log_e.cpu().numpy(),
                       s=3, alpha=0.3, c="C2")
    lo, hi = float(LOG_E_MIN), float(LOG_E_MAX)
    axes[1, 0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[1, 0].set_xlabel("true log10(E)"); axes[1, 0].set_ylabel("LDF-fit log10(E)")
    axes[1, 0].set_title("A3: LDF energy estimator")

    # A4a: N_lit vs energy (binned)
    logE = torch.log10(true_e.clamp(min=1.0))
    _binned_line(axes[1, 1], logE.cpu().numpy(), lit_thresh.cpu().numpy(), n_bins=15)
    axes[1, 1].set_xlabel("log10(E_true) [GeV]"); axes[1, 1].set_ylabel("N_lit (mean)")
    axes[1, 1].set_title("A4: multiplicity vs energy")

    # A4b: N_lit vs axis distance (binned)
    _binned_line(axes[1, 2], axis_dist.cpu().numpy(), lit_thresh.cpu().numpy(), n_bins=15)
    axes[1, 2].set_xlabel("axis-to-nearest-detector [m]"); axes[1, 2].set_ylabel("N_lit (mean)")
    axes[1, 2].set_title("A4: multiplicity vs axis distance")

    fig.tight_layout()
    path = os.path.join(folder, "analytic_recon_diagnostic.png")
    fig.savefig(path, dpi=110)
    print(f"[write] {path}")


def _binned_line(ax, x, y, n_bins=15):
    import numpy as np
    order = np.argsort(x)
    x, y = x[order], y[order]
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    centers, means, stds = [], [], []
    for i in range(n_bins):
        m = (x >= edges[i]) & (x <= edges[i + 1] if i == n_bins - 1 else x < edges[i + 1])
        if m.sum() < 5:
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        means.append(y[m].mean())
        stds.append(y[m].std())
    ax.errorbar(centers, means, yerr=stds, marker="o", ms=3, lw=1, capsize=2)


if __name__ == "__main__":
    main()
