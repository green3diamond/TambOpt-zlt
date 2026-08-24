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

from ..constants import HIT_DEPOSIT_MIN



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
    spatial = torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))

    # delta_l : (B, P, n_det)
    delta_l = point_l.unsqueeze(2) - z_cont.unsqueeze(0).unsqueeze(0)
    plane_w = torch.relu(1.0 - delta_l.abs())

    kernel        = spatial * plane_w                              # (B, P, n_det)
    energy_kernel = point_e.unsqueeze(2) * kernel                 # (B, P, n_det)

    local_intensity = energy_kernel.sum(dim=1)                    # (B, n_det)

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
