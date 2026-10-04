"""Plane-aware detector response kernel.

Ground truth for the Step-1 labels: (E, T) per detector from a shower point
cloud.

The plane weight is triangular — 1 on an exact layer match, falling linearly to
0 one layer away:

    plane_w = relu(1 - |layer_p - z_cont|)
    kernel  = spatial_gaussian * plane_w      (B, max_points, n_det)

E is the deposit sum over that kernel and stays differentiable in the detector
positions and in z_cont. **T is the time of the EARLIEST DETECTED point** — a
leading edge, so it is a selection, piecewise constant in the detector position
and carrying no useful gradient. Nothing backprops through this kernel:
`compute_labels_batch` is `@torch.no_grad()` and every call site is label
generation, plotting, or eval. Gradients reach a layout through the *surrogate*
trained on these labels, which is smooth by construction.

T semantics (see docs/THEORY.md §3.6):

    T_d = min { t_p : e_p * kernel_pd > HIT_DEPOSIT_MIN }      seconds
    T_d = 0  when no point clears the threshold at d

`T_d == 0` does NOT imply `E_d == 0`: E sums arbitrarily many sub-threshold
deposits, so a detector can accumulate energy without any single detected point.

A leading edge is what a photodetector actually reports, and it makes the
multi-species combination a `min` over species — associative, and independent of
how many species the corpus is split into. The count-weighted mean it replaced
was neither: it read as an intensity x time (no normalization by the kernel
weight, divided by the PADDED point count), so it scaled with the corpus's
max_points and correlated 0.43 with the E channel.
"""

import torch

from ..constants import HIT_DEPOSIT_MIN, KERNEL_INTERP_MAX


@torch.no_grad()
def adaptive_bandwidths(samples: torch.Tensor, k: int, sigma_min: float,
                        max_pairs: int = 1 << 27, sigma_max: float = None) -> torch.Tensor:
    """Per-point Gaussian width for the interpolated kernel: the distance to the
    k-th nearest other real point IN THE SAME PLANE, floored at `sigma_min`.

    AllShowers points are macro-particles: a plane holds ~10^2-10^3 of them over
    a kilometre-scale footprint, so at a narrow sigma a detector between points
    reads ~0 purely because of how the generator sampled the shower. Widening
    each point to its local spacing turns the cloud into a smooth density
    estimate there, and leaves dense regions (spacing < sigma_min) untouched.

    Distances use the kernel's own transverse coordinates (columns 0, 1); planes
    are the rounded layer index (column 2). Padding rows (energy 0) are neither
    searched nor given a width beyond `sigma_min`. A plane with fewer than k+1
    real points uses its farthest available neighbour; a lone point keeps
    `sigma_min`. Depends only on the cloud, so gradients w.r.t. detector
    positions are unaffected.

    Searched plane by plane, vectorised over showers: each plane's points are
    gathered to the front of a (showers, n_plane) block, so the cost is the sum of
    per-plane squares rather than (points)^2 across all 24 planes. Distances are
    float64 on per-shower-centred coordinates, so the fast matmul form stays exact
    at metre scale. `max_pairs` bounds the (showers, n, n) block held at once.

    `sigma_max` caps the widths (default constants.KERNEL_INTERP_MAX, inf = no cap).

    Returns (B, P) widths in metres.
    """
    B, P, _ = samples.shape
    sigma_min = float(sigma_min)
    sigma_max = float(KERNEL_INTERP_MAX if sigma_max is None else sigma_max)
    h = torch.full((B, P), sigma_min, dtype=samples.dtype, device=samples.device)
    real = samples[..., 3] > 0
    if B == 0 or not bool(real.any()):
        return h
    layer = samples[..., 2].round().long()
    xy = samples[..., :2].double()
    w = real.unsqueeze(-1).double()
    xy = xy - (xy * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1.0)
    inf = torch.tensor(float("inf"), dtype=torch.float64, device=xy.device)

    for L in torch.unique(layer[real]).tolist():
        m = real & (layer == L)                                       # (B, P)
        cnt = m.sum(1)
        n = int(cnt.max())
        if n < 2:
            continue                                                  # lone points keep sigma_min
        sb = torch.nonzero(cnt > 1).squeeze(-1)                       # showers with >= 2 here
        # This plane's points first (stable, so original order), padded to n.
        order = torch.argsort((~m[sb]).to(torch.int8), dim=1, stable=True)[:, :n]
        valid = torch.arange(n, device=xy.device) < cnt[sb].unsqueeze(1)
        pts = torch.gather(xy[sb], 1, order.unsqueeze(-1).expand(-1, -1, 2))
        kk = min(int(k) + 1, n)
        bs = max(1, max_pairs // (n * n))
        for c0 in range(0, sb.numel(), bs):
            c1 = min(c0 + bs, sb.numel())
            v = valid[c0:c1]
            d = torch.cdist(pts[c0:c1], pts[c0:c1])                   # (bc, n, n) float64
            d = torch.where(v.unsqueeze(1), d, inf)                   # padding columns
            vals = d.topk(kk, dim=-1, largest=False).values           # ascending, self first
            # k-th neighbour if the plane has one, else its farthest; 0 if alone.
            r = torch.where(torch.isfinite(vals), vals, torch.zeros_like(vals)).amax(-1)
            r = r.clamp(min=sigma_min, max=max(sigma_max, sigma_min)).to(h.dtype)
            rows = sb[c0:c1]
            hsub = h[rows]
            idx = order[c0:c1]
            # Write only this plane's points; padding slots index other points.
            hsub.scatter_(1, idx, torch.where(v, r, hsub.gather(1, idx)))
            h[rows] = hsub
    return h


def GetCounts_planeaware(
    samples: torch.Tensor,
    x_det:   torch.Tensor,
    y_det:   torch.Tensor,
    z_cont:  torch.Tensor,
    SmearN_fn,
    fluxB_e:  torch.Tensor,
    TimeAverage_vectorized_fn,
    sigma:   float = 200.0,
    hit_deposit_min: float = HIT_DEPOSIT_MIN,
    bandwidth: torch.Tensor = None,
    t_mode: str = "leading",
) -> tuple:
    """Plane-aware count extraction + leading-edge arrival time.

    E is differentiable w.r.t. x_det, y_det and z_cont (and hence w.r.t. the
    learnable North/Up positions via the surface map). T is a min over the
    points that clear `hit_deposit_min` and is not.

    Args:
        samples  : (B, max_points, 5) point-cloud tensor with columns
                   [x, y, layer_index, energy, time].  Padding rows have energy=0.
        x_det    : (n_det,) North coordinates [m], requires_grad may be True.
        y_det    : (n_det,) Up   coordinates [m], requires_grad may be True.
        z_cont   : (n_det,) continuous plane index ∈ [0, n_planes-1],
                   derived as (East - east_min) / plane_dx.  requires_grad may be True.
        SmearN_fn, fluxB_e, TimeAverage_vectorized_fn :
                   accepted for interface compatibility; never called.
        sigma    : Gaussian spatial kernel width [m].
        hit_deposit_min : a point is DETECTED at a detector when its deposit
                   there (energy * kernel acceptance) exceeds this.
        bandwidth : optional (B, max_points) per-point width h >= sigma, from
                   `adaptive_bandwidths`. When given, the spatial factor is
                   (sigma/h)^2 * exp(-d^2 / 2h^2): area-normalised so every point
                   deposits the same total as with the fixed kernel, identical to
                   it wherever h == sigma. None (default) is the fixed kernel.
        t_mode   : "leading" (default) = earliest detected point, as a
                   photodetector triggers. "mean" = deposit-weighted mean time,
                   sum(e*w*t) / sum(e*w), where the detector's TOTAL deposit
                   clears `hit_deposit_min` (else 0). The mean averages over every
                   point the detector sees instead of taking an extreme value, so
                   it scatters less and varies smoothly with detector position.
                   Normalised by the detector's own deposit, unlike the retired
                   count-weighted mean (see module docstring), so it does not
                   scale with the corpus's point budget.

    Returns:
        (local_intensity, arrival_time) : each (B, n_det), raw (no
        post-processing). `arrival_time` is the earliest detected point's time
        in seconds, 0 at detectors with no detected point.
    """
    point_x = samples[..., 0]    # (B, P)
    point_y = samples[..., 1]    # (B, P)
    point_l = samples[..., 2]    # (B, P)  layer index (integer, but stored as float)
    point_e = samples[..., 3]    # (B, P)  energy
    point_t = samples[..., 4]    # (B, P)  time

    # dx, dy : (B, P, n_det)
    dx = point_x.unsqueeze(2) - x_det.unsqueeze(0).unsqueeze(0)
    dy = point_y.unsqueeze(2) - y_det.unsqueeze(0).unsqueeze(0)
    if bandwidth is None:
        spatial = torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))
    else:
        h2 = bandwidth.unsqueeze(2) ** 2                          # (B, P, 1)
        spatial = (sigma ** 2 / h2) * torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * h2))

    # delta_l : (B, P, n_det)
    delta_l = point_l.unsqueeze(2) - z_cont.unsqueeze(0).unsqueeze(0)
    plane_w = torch.relu(1.0 - delta_l.abs())

    kernel        = spatial * plane_w                              # (B, P, n_det)
    energy_kernel = point_e.unsqueeze(2) * kernel                 # (B, P, n_det)

    local_intensity = energy_kernel.sum(dim=1)                    # (B, n_det)

    if t_mode == "mean":
        # Deposit-weighted mean over every point the detector sees. Padding rows
        # carry energy 0, so they add nothing to either sum.
        lit = local_intensity > hit_deposit_min
        t_sum = (energy_kernel * point_t.unsqueeze(2)).sum(dim=1)
        mean_t = t_sum / local_intensity.clamp(min=torch.finfo(local_intensity.dtype).tiny)
        return local_intensity, torch.where(lit, mean_t, torch.zeros_like(mean_t))
    if t_mode != "leading":
        raise ValueError(f"t_mode must be 'leading' or 'mean', got {t_mode!r}")

    # Leading edge: the earliest point whose deposit clears the threshold.
    # Padding rows carry energy 0 (showerdata zero-fills), so they never clear
    # it and drop out on their own — no separate mask, and no dependence on the
    # corpus's padded width.
    detected     = energy_kernel > hit_deposit_min                 # (B, P, n_det)
    t_masked     = torch.where(detected, point_t.unsqueeze(2),
                               torch.full_like(energy_kernel, float("inf")))
    arrival_time = t_masked.amin(dim=1)                            # (B, n_det)
    # No detected point -> 0, the same "no signal" sentinel the E channel uses.
    arrival_time = torch.where(torch.isinf(arrival_time),
                               torch.zeros_like(arrival_time), arrival_time)

    return local_intensity, arrival_time
