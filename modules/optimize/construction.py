"""Construction preferences as soft penalties on a layout, in units of the Stage 4 utility (x30 scale).

TAMBOPT_CONSTRUCT="slope=0.05,access=0.05,pads=0.05,contour=0.05" (any subset; unset = no penalty). Each term is
normalised to the grid baseline grid_n99: penalty = 30 lambda P(layout) / P(grid), so it costs 30 lambda U at the grid.

    slope    mean relu(s_i - s_max)^2, s_i = terrain slope [deg] under detector i (central differences of the
             SurfaceUpMap, step 5 m); s_max = TAMBOPT_SLOPE_MAX [deg], default the median slope under grid_n99
    access   mean 3-D distance [m] from detector i to the access point A = TAMBOPT_ACCESS_EN "east,north" [m],
             default the lowest face centroid of the detector region
    pads     mean squared distance to the nearest of K pad centres (TAMBOPT_PADS_K, default 20); the centres follow the
             layout by a few k-means (Lloyd) iterations on the detached positions each call, warm-started from the
             previous centres (seeded on grid_n99), so the gradient pulls detectors onto shared pads
    contour  mean squared height [m^2] of detector i off the nearest of N rows of constant height (TAMBOPT_CONTOUR_ROWS,
             default 8, evenly spaced between the region's lowest and highest face centroid): terraces / paths

    cost = ConstructionCost(mountain, surface, grid_xy)      # grid_xy = (east, north) of grid_n99
    pen, info = cost(x_east, y_north)                        # pen: tensor (differentiable), info: descriptive floats
"""
import os

import numpy as np
import torch

from .objective import TWO_TIER_SCALE

TERMS = ("slope", "access", "pads", "contour")
H_SLOPE = 5.0                                  # m, finite-difference step for the slope
LLOYD_ITERS = 3


def parse(spec=None):
    """{term: lambda} from TAMBOPT_CONSTRUCT ("slope=0.05,pads=0.1")."""
    spec = os.environ.get("TAMBOPT_CONSTRUCT", "") if spec is None else spec
    out = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        k, v = part.split("=")
        if k not in TERMS:
            raise ValueError(f"TAMBOPT_CONSTRUCT: unknown term {k!r} (known: {', '.join(TERMS)})")
        out[k] = float(v)
    return out


def slope_deg(x, y, surface, h=H_SLOPE):
    """Terrain slope [deg] at (east x, north y), differentiable; surface(north, east) is SurfaceUpMap."""
    dze = (surface(y, x + h) - surface(y, x - h)) / (2 * h)
    dzn = (surface(y + h, x) - surface(y - h, x)) / (2 * h)
    return torch.rad2deg(torch.atan(torch.sqrt(dze ** 2 + dzn ** 2 + 1e-12)))


class ConstructionCost:
    def __init__(self, mountain, surface, grid_xy, lambdas=None):
        self.lam = parse() if lambdas is None else lambdas
        self.surface = surface
        dev = surface.grid_up.device
        c = torch.as_tensor(np.asarray(mountain.centroids_ENU), dtype=torch.float32, device=dev)
        ge, gn = (t.to(dev).float() for t in grid_xy)
        with torch.no_grad():
            s = slope_deg(ge, gn, surface)
            self.s_max = float(os.environ.get("TAMBOPT_SLOPE_MAX", float(s.median())))
            acc = os.environ.get("TAMBOPT_ACCESS_EN")
            if acc:
                ae, an = (float(v) for v in acc.split(","))
                self.access = torch.tensor([ae, an, float(surface(torch.tensor([an], device=dev),
                                                                  torch.tensor([ae], device=dev)))], device=dev)
            else:
                self.access = c[c[:, 2].argmin()].clone()
            n_rows = int(os.environ.get("TAMBOPT_CONTOUR_ROWS", "8"))
            self.rows = torch.linspace(float(c[:, 2].min()), float(c[:, 2].max()), n_rows, device=dev)
            k = int(os.environ.get("TAMBOPT_PADS_K", "20"))
            self.centres = torch.stack([ge, gn], -1)[torch.linspace(0, len(ge) - 1, k).round().long()].clone()
            self.ref = {t: 1.0 for t in TERMS}
            ref = self.parts(ge, gn)[0]
            self.ref = {t: max(float(v), 1e-12) for t, v in ref.items()}
        print("[construct] " + "  ".join(f"{t}={self.lam[t]:g}" for t in self.lam)
              + f"  | s_max={self.s_max:.1f} deg  access=({float(self.access[0]):.0f}, {float(self.access[1]):.0f}, "
              f"{float(self.access[2]):.0f}) m  pads K={len(self.centres)}  rows={len(self.rows)} "
              f"({float(self.rows[0]):.0f}..{float(self.rows[-1]):.0f} m)  | grid: "
              + "  ".join(f"{t} {self.ref[t]:.3g}" for t in TERMS), flush=True)

    def _lloyd(self, p):
        """Update the pad centres on the detached (N, 2) positions; returns each detector's pad index."""
        c = self.centres.to(p.device)
        for _ in range(LLOYD_ITERS):
            a = torch.cdist(p, c).argmin(1)
            cnt = torch.bincount(a, minlength=len(c)).float()
            s = torch.zeros_like(c).index_add_(0, a, p)
            c = torch.where(cnt[:, None] > 0, s / cnt.clamp(min=1)[:, None], c)
        self.centres = c
        return torch.cdist(p, c).argmin(1)

    def parts(self, x, y):
        """({term: raw P}, {descriptive metric: float}); P differentiable in (x, y)."""
        z = self.surface(y, x)
        s = slope_deg(x, y, self.surface)
        p3 = torch.stack([x, y, z], -1)
        d_acc = (((p3 - self.access) ** 2).sum(-1) + 1e-6).sqrt()
        p2 = torch.stack([x, y], -1)
        a = self._lloyd(p2.detach())
        d2_pad = ((p2 - self.centres[a]) ** 2).sum(-1)
        d2_row = ((z[:, None] - self.rows[None]) ** 2).min(1).values
        P = dict(slope=(torch.relu(s - self.s_max) ** 2).mean(), access=d_acc.mean(), pads=d2_pad.mean(),
                 contour=d2_row.mean())
        info = dict(slope_mean=float(s.mean()), slope_over=float((s > self.s_max).float().mean()),
                    access_m=float(d_acc.mean()), pad_rms=float(d2_pad.mean().sqrt()),
                    row_rms=float(d2_row.mean().sqrt()))
        return P, info

    def __call__(self, x, y):
        P, info = self.parts(x, y)
        pen = sum(TWO_TIER_SCALE * lam * P[t] / self.ref[t] for t, lam in self.lam.items())
        info.update({f"pen_{t}": float(TWO_TIER_SCALE * lam * P[t] / self.ref[t]) for t, lam in self.lam.items()})
        return pen, info


def describe(x, y, surface, mountain, grid_xy):
    """Descriptive construction metrics of a fixed layout (no penalty), for validation tables."""
    cost = ConstructionCost.__new__(ConstructionCost)
    lam = dict.fromkeys(TERMS, 0.0)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        ConstructionCost.__init__(cost, mountain, surface, grid_xy, lam)
    with torch.no_grad():
        for _ in range(5):                       # settle the pad centres on this layout
            cost._lloyd(torch.stack([x, y], -1))
        return cost.parts(x, y)[1]


__all__ = ["ConstructionCost", "describe", "parse", "slope_deg", "TERMS"]
