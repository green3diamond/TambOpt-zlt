"""Plane-aware detector response kernel.

Ground truth for the Step-1 labels: (E, T) per detector from a shower point
cloud.

The plane weight is triangular — 1 on an exact layer match, falling linearly to
0 one layer away:

    plane_w = relu(1 - |layer_p - z_cont|)
    kernel  = spatial_gaussian * plane_w      (B, max_points, n_det)

E is the deposit sum over that kernel and stays differentiable in the detector
positions and in z_cont.

Units and normalisation of E (2026-10-21). Each AllShowers point is a CELL of the
CORSIKA 8 output (5 x 5 m for e/gamma, 10 x 10 m for mu; 10 / 20 m beyond 10 km),
and its "energy" is the SUMMED KINETIC ENERGY [GeV] of the particles in that cell
(TAMBO-opt postprocess_showers.py). No particle count is stored, and nothing on the
way divides by an area. The spatial weight has PEAK 1 (exp(-d^2 / 2 sigma^2)); its
total over the plane is 2 pi sigma^2 ~ 628 m^2 per point (sigma = 10 m). The
interpolated kernel's (sigma/h)^2 factor keeps that total at 2 pi sigma^2 for any
width h: a relative normalisation, not one to 1. So

    E ~ (kinetic-energy density [GeV / m^2]) x 2 pi sigma^2   [GeV]

i.e. a detector reads as if its area were ~628 m^2 (about 300x a 2 m^2 detector),
and E is energy, not particles. The recon and the surrogate learn on
z-scored log1p(E), so the scale mostly washes out for them; it matters wherever
E's absolute value is used: hit thresholds (E > 1, E > 0.001), HIT_DEPOSIT_MIN
(the leading-edge time), the background (modules/optimize/background.py) and any
Poisson / MIP probability. Expected charged particles in a detector of area A:
mu = E x (n_ch / eps) x A / (2 pi sigma^2), with n_ch / eps the charged particles
per GeV of kinetic energy (scripts/diagnostics/c8_charged_ke.py). **T is the time of the EARLIEST DETECTED point** — a
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

C_LIGHT = 299_792_458.0      # m/s, for carrying point times between planes (t_interp)


def _det_view(v):
    """(n_det,) -> (1, 1, n_det) shared by every shower; (B, n_det) -> (B, 1, n_det)."""
    if v.dim() == 1:
        return v.view(1, 1, -1)
    if v.dim() == 2:
        return v.unsqueeze(1)
    raise ValueError(f"detector coordinates must be (n_det,) or (B, n_det), got {tuple(v.shape)}")


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
    samples:   torch.Tensor,
    rho_det:   torch.Tensor,
    z_cont:    torch.Tensor,
    bandwidth: torch.Tensor,
    sigma:     float,
    hit_deposit_min: float = HIT_DEPOSIT_MIN,
    layer_dx:  float = 500.0,
) -> tuple:
    """Ring kernel in the shower frame: deposit E and leading-edge arrival time T per detector.

    E is differentiable w.r.t. rho_det and z_cont (and hence the detector positions); T is a min over
    the points that clear `hit_deposit_min` and is not.

    Each point is a RING of radius r = samples[..., 0] around the shower axis (circularly symmetric
    showers; place_clouds_shower), its energy spread evenly around it, and a detector is read by its
    distance rho from the axis. The spatial factor is the width-h Gaussian averaged around the ring,
    in closed form:
        (sigma/h)^2 exp(-(rho - r)^2 / 2h^2) i0e(rho r / h^2)        (i0e = exp(-x) I0(x))
    r = 0 gives the point kernel. (sigma/h)^2 keeps every point's deposit over the plane at
    2 pi sigma^2 per unit energy, whatever its width.

    Args:
        samples   : (B, max_points, 5) placed clouds, columns [r, 0, layer (depth / layer_dx), energy,
                    time]. Padding rows have energy 0.
        rho_det   : (n_det,) or (B, n_det) detector distance from the shower axis [m].
        z_cont    : same shape, detector depth along the axis / layer_dx.
        bandwidth : (B, max_points) per-point width h >= sigma (`adaptive_bandwidths`).
        sigma     : base Gaussian width [m].
        hit_deposit_min : a point is DETECTED at a detector when its deposit there (energy x kernel
                    weight) exceeds this.
        layer_dx  : plane spacing [m] (C8: 500 m along the axis).

    Plane weight relu(1 - |layer_p - z_cont|): linear between the two planes around the detector.
    T: each detected point's time is carried along the axis at c to the detector's depth,
    t = t_p + (z_det - z_p) * layer_dx / c (clamped >= 1e-12 s; 0 is "no hit"), and T is the
    earliest of them.

    Returns:
        (local_intensity, arrival_time) : each (B, n_det). `arrival_time` in seconds, 0 at detectors
        with no detected point.
    """
    point_r = samples[..., 0]    # (B, P)  ring radius
    point_l = samples[..., 2]    # (B, P)  layer index (integer, but stored as float)
    point_e = samples[..., 3]    # (B, P)  energy
    point_t = samples[..., 4]    # (B, P)  time

    # Detector coordinates: (n_det,) shared by every shower, or (B, n_det) per shower.
    rho_det, z_cont = (_det_view(v) for v in (rho_det, z_cont))

    h2 = bandwidth.unsqueeze(2) ** 2                                 # (B, P, 1)
    dr = point_r.unsqueeze(2) - rho_det                              # (B, P, n_det)
    rr = point_r.unsqueeze(2) * rho_det
    spatial = (sigma ** 2 / h2) * torch.exp(-dr ** 2 / (2.0 * h2)) * torch.special.i0e(rr / h2)

    delta_l = point_l.unsqueeze(2) - z_cont                          # (B, P, n_det)
    plane_w = torch.relu(1.0 - delta_l.abs())

    # Peak weight 1, total 2 pi sigma^2 (~628 m^2) per point: E is GeV over an
    # effective ~628 m^2, not a density and not a particle count (module docstring).
    energy_kernel = point_e.unsqueeze(2) * spatial * plane_w        # (B, P, n_det)
    local_intensity = energy_kernel.sum(dim=1)                      # (B, n_det)

    # Leading edge: the earliest point whose deposit clears the threshold. Padding rows carry
    # energy 0, so they never clear it. Times are carried along the axis at c to the detector's
    # depth; selection only (no gradient is wanted through T), so delta_l is detached.
    detected = energy_kernel > hit_deposit_min
    t_pt = (point_t.unsqueeze(2) - delta_l.detach() * (layer_dx / C_LIGHT)).clamp(min=1e-12)
    t_masked = torch.where(detected, t_pt, torch.full_like(energy_kernel, float("inf")))
    arrival_time = t_masked.amin(dim=1)                             # (B, n_det)
    # No detected point -> 0, the same "no signal" sentinel the E channel uses.
    arrival_time = torch.where(torch.isinf(arrival_time), torch.zeros_like(arrival_time), arrival_time)
    return local_intensity, arrival_time
