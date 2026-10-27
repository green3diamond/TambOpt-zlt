"""Per-shower summaries of native clouds with columns (x, y, layer, energy, time)."""
import numpy as np
import torch

from modules.constants import N_PLANES

ONSET = 0.05
NAMES = ("log10_energy", "n_points", "onset_layer", "mean_layer", "peak_width_m")
LABEL = {"log10_energy": "log10 total energy [GeV]", "n_points": "points",
         "onset_layer": "layer reaching 5% of energy", "mean_layer": "energy-weighted mean layer",
         "peak_width_m": "lateral width at peak layer [m]"}


def profile(clouds):
    """(B, N_PLANES) deposited energy per layer."""
    c = torch.as_tensor(clouds, dtype=torch.float64)
    e = c[..., 3].clamp(min=0)
    layer = c[..., 2].round().long().clamp(0, N_PLANES - 1)
    return torch.zeros(len(c), N_PLANES, dtype=torch.float64).scatter_add_(1, layer, e)


def summarize(clouds):
    """(B, P, 5) native clouds -> {name: (B,) array}; NaN where a shower has no finite positive energy."""
    c = torch.as_tensor(clouds, dtype=torch.float64)
    e = c[..., 3].clamp(min=0)
    layer = c[..., 2].round().long().clamp(0, N_PLANES - 1)
    prof = profile(c)
    total = prof.sum(1)
    frac = prof.cumsum(1) / total[:, None]
    peak = prof.argmax(1)
    w = e * (layer == peak[:, None])
    wsum = w.sum(1)
    cx = (w * c[..., 0]).sum(1) / wsum
    cy = (w * c[..., 1]).sum(1) / wsum
    spread = (w * ((c[..., 0] - cx[:, None]) ** 2 + (c[..., 1] - cy[:, None]) ** 2)).sum(1) / wsum
    out = {"log10_energy": torch.log10(total),
           "n_points": (e > 0).sum(1).double(),
           "onset_layer": (frac < ONSET).sum(1).double(),
           "mean_layer": (prof * torch.arange(N_PLANES, dtype=torch.float64)).sum(1) / total,
           "peak_width_m": spread.sqrt()}
    bad = ~torch.isfinite(total) | (total <= 0)
    return {k: np.where(bad.numpy(), np.nan, v.numpy()) for k, v in out.items()}


SCALARS = NAMES + ("peak_layer", "end_layer", "peak_time_mean", "peak_time_rms", "blob")
END = 0.95


def describe(clouds, chunk=256):
    """(B, P, 5) native clouds -> per-shower scalars (SCALARS) plus per-layer energy and lateral width (B, N_PLANES).

    Widths are the energy-weighted rms distance from each layer's centroid in (x, y); times are the energy-weighted
    mean and rms of column 4 in the peak layer; blob is the pipeline's degenerate-shower flag. NaN where a shower
    has no finite positive energy (blob stays as flagged)."""
    from modules.data.dataset_builder import flag_blob_showers
    out = {k: [] for k in SCALARS + ("profile", "width")}
    for lo in range(0, len(clouds), chunk):
        raw = torch.as_tensor(clouds[lo:lo + chunk])
        c = raw.double()
        e = c[..., 3].clamp(min=0)
        layer = c[..., 2].round().long().clamp(0, N_PLANES - 1)
        moment = lambda v: torch.zeros(len(c), N_PLANES, dtype=torch.float64).scatter_add_(1, layer, e * v)
        x, y, t = c[..., 0], c[..., 1], c[..., 4]
        s0, sx, sy, sr, st, stt = moment(1.0), moment(x), moment(y), moment(x * x + y * y), moment(t), moment(t * t)
        with np.errstate(invalid="ignore", divide="ignore"):
            width = (sr / s0 - (sx / s0) ** 2 - (sy / s0) ** 2).clamp(min=0).sqrt()
            total = s0.sum(1)
            frac = s0.cumsum(1) / total[:, None]
            peak = s0.argmax(1)
            at = lambda a: a.gather(1, peak[:, None])[:, 0]
            t_mean = at(st) / at(s0)
            vals = {"log10_energy": torch.log10(total), "n_points": (e > 0).sum(1).double(),
                    "onset_layer": (frac < ONSET).sum(1).double(), "end_layer": (frac < END).sum(1).double(),
                    "mean_layer": (s0 * torch.arange(N_PLANES, dtype=torch.float64)).sum(1) / total,
                    "peak_layer": peak.double(), "peak_width_m": at(width),
                    "peak_time_mean": t_mean, "peak_time_rms": (at(stt) / at(s0) - t_mean ** 2).clamp(min=0).sqrt()}
        bad = (~torch.isfinite(total) | (total <= 0)).numpy()
        for k, v in vals.items():
            out[k].append(np.where(bad, np.nan, v.numpy()))
        out["blob"].append(flag_blob_showers(raw.float()).numpy().astype(np.float64))
        out["profile"].append(np.where(bad[:, None], np.nan, s0.numpy()))
        out["width"].append(np.where(bad[:, None], np.nan, width.numpy()))
    return {k: np.concatenate(v) for k, v in out.items()}
