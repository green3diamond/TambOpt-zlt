"""The individual terms the Step-4 objective sums into U."""

import torch


def reconstructability(events, layout_threshold=5e-2, tau_layout=5.,
                       reconstruct_threshold=130., tau_reconstruct=5.):
    """Differentiable reconstructability per event, in (0, 1).

    Two stacked sigmoids: soft-detect each detector above `layout_threshold`,
    then soft-threshold the detector count at `reconstruct_threshold`. The
    `tau_*` are sigmoid steepnesses. `events` is (n_events, n_detectors).
    """
    soft_detect = torch.sigmoid(tau_layout * (events - layout_threshold))
    n = torch.sum(soft_detect, dim=1)
    r = torch.sigmoid(tau_reconstruct * (n - reconstruct_threshold))
    return r


def U_PR(r):
    """Reconstructability term: sqrt of the summed per-event scores."""
    u = torch.sqrt(torch.sum(r) + 1e-6)
    return u


def _soft_cap(x, cap):
    """Smooth saturating cap: ~x for x << cap, -> cap as x -> inf.

    Pulls in the tail of the per-event reward — whose own 1/eps bound can sit far
    above the typical event, letting a lucky handful dominate the batch mean —
    without the gradient cliff a hard clamp would introduce.

    **Pick `cap` well ABOVE the median reward**, or the bulk of events saturate
    together and the term stops discriminating between layouts at all.
    """
    return cap * torch.tanh(x / cap)


def U_E(E_preds, E_trues, r, cap=None):
    """Energy term: inverse squared log-energy error, weighted by `r`.

    `cap` soft-caps the per-event reward 1/(err^2 + .01), whose hard ceiling is
    100; None leaves it uncapped. Pick it well above the median — see `_soft_cap`.
    """
    inv_err = 1.0 / ((torch.log10(E_preds) - torch.log10(E_trues)) ** 2 + .01)
    if cap is not None:
        inv_err = _soft_cap(inv_err, cap)
    u = torch.mean(r * inv_err)
    return u


def U_angle(angle_preds, angle_trues, r, cap=None, period=None):
    """Angular term: inverse squared angle error, weighted by `r`.

    `cap` soft-caps the per-event reward 1/(err^2 + .001), hard ceiling 1000;
    None leaves it uncapped. **theta and phi need very different caps** — their
    error distributions are not comparable.

    `period` wraps the difference into [-period/2, +period/2] before squaring,
    for angles that live on a circle: pass 2*pi for azimuth, leave None for
    zenith, which is confined to [0, pi]. Without it a prediction and a truth
    either side of the branch cut score as if almost a full turn apart, so with
    phi in [0, 2*pi) a true 0.023 rad error reads as 6.26 rad and the reward
    collapses from ~650 to 0.026, which also miscalibrates any cap tuned on it.
    """
    d = angle_preds - angle_trues
    if period is not None:
        d = torch.remainder(d + 0.5 * period, period) - 0.5 * period
    inv_err = 1.0 / (d ** 2 + .001)
    if cap is not None:
        inv_err = _soft_cap(inv_err, cap)
    u = torch.mean(r * inv_err)
    return u


def angle_between_deg(u, v):
    """Angle between direction vectors (..., 3) in degrees. Uses 2*atan2(|u-v|, |u+v|)
    on unit vectors, which keeps a finite gradient at perfect alignment (arccos of
    the dot product does not)."""
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return torch.rad2deg(2.0 * torch.atan2((u - v).norm(dim=-1), (u + v).norm(dim=-1)))


def U_two_tier(a_deg, e_err, theta_good=5.0, e_good=0.2, s_theta=1.0, s_e=0.05,
               theta_ref=20.0, e_ref=0.5, lam=0.3):
    """Two-tier utility in reconstruction-accuracy space.

    a_deg : per-shower angle error [deg]; e_err : per-shower |dlog10 E|.

        g = sigmoid((theta_good - a) / s_theta) * sigmoid((e_good - e) / s_e)
        q = 1 / (1 + (a / theta_ref)^2 + (e / e_ref)^2)
        U = mean[ g + lam * (1 - g) * q ]

    g is a soft "reconstructed well"; once a shower is good it stops paying, so
    the optimizer gains most by making more showers good. q gives the rest partial,
    bounded credit with diminishing returns; lam < 1 lets the count of good showers
    dominate. Returns (U, g) so callers can report the good fraction.
    """
    g = torch.sigmoid((theta_good - a_deg) / s_theta) * torch.sigmoid((e_good - e_err) / s_e)
    q = 1.0 / (1.0 + (a_deg / theta_ref) ** 2 + (e_err / e_ref) ** 2)
    return torch.mean(g + lam * (1.0 - g) * q), g


def U_three_tier(a_deg, det, w1=0.9, w5=0.3, wd=0.1, s1=0.2, s5=1.0):
    """Exclusive three-level utility (run 19): each shower scores its best level,
        w1 if within 1 deg, else w5 if within 5 deg, else wd if detected (det in (0, 1)), else 0.
    Soft: g1 = sigmoid((1 - a) / s1), g5 = sigmoid((5 - a) / s5),
        u = w1 g1 + w5 (1 - g1) g5 + wd (1 - g1) (1 - g5) det,   U = mean(u).
    Angle only (no energy condition). Returns (U, g1, g5)."""
    g1 = torch.sigmoid((1.0 - a_deg) / s1)
    g5 = torch.sigmoid((5.0 - a_deg) / s5)
    u = w1 * g1 + w5 * (1.0 - g1) * g5 + wd * (1.0 - g1) * (1.0 - g5) * det
    return torch.mean(u), g1, g5
