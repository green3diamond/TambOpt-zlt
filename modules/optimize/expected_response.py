"""Expected detector response from Jay Agarwal's symbolic shower model (symbolic/, frozen model folder MODEL), as a
drop-in for the kernel in Stage 4 (pool_utility / utility_of_xy): same call (primary_batch, xy_batch) -> (B, n_det, 2)
[log1p E, log1p(T * T_LOG_SCALE)], and .mu, the expected detected particles per detector (all species).

Per shower and species (no particle draws, no in-plane noise):
  energy per plane   part 1 of the model: ONE draw of the sampler per shower (the shower-to-shower variation of the
                     longitudinal profile is kept; its plane-to-plane noise too, as drawn)
  ring energies      plane energy x rings law shares, stretched by the MEAN width law (no AR(1) width noise)
  particles          ring energy / mean particle energy (energy law), per unit area of the ring -> density [1/m^2]
  arrival delay      mean delay behind light on the axis, t = exp(A) r^p (k+1)^-q [ns] (times law 'delay')
These per-shower tables (log density per species, plane, ring) are precomputed in numpy (tables()), then read per
detector in torch, differentiable in the detector position:
  rho, depth   from detectors_in_frames (the kernel's shower frame; Up from the surface)
  density      kernel-smoothed (SMOOTH_H, see below) on a radial grid, linear in log density in ln r, linear in density between the two planes around
               the detector's depth; zero past the last plane, the first plane's before it
  mu_s         density x AREA x efficiency (photons EPS_GAMMA), as the counting model
  T            depth / c + mu-weighted mean of the species' delay at (rho, plane), seconds since the decay
The E channel carries the expected particle count (all species), not a kernel energy: the analytical recon reads mu
and T only; anything that thresholds E (E > 1) then thresholds counts.
"""
import os
import numpy as np
import torch
from scipy.special import softmax

from modules.constants import N_PLANES, SPECIES_NAMES, T_LOG_SCALE, ZENITH_MAX, ZENITH_MIN
from modules.data.dataset_builder import detectors_in_frames
from modules.optimize.counting import EPS_GAMMA

MODEL = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/symbolic_model"
AREA = 2.0                                       # m^2, background.AREA_M2
EFF = torch.tensor([1.0, 1.0, EPS_GAMMA])        # electron, muon, photon
C_LIGHT = 299792458.0
LOG_FLOOR = -60.0                                # log density of an empty ring
# Kernel-equivalent smoothing (default on): each ring's content spread as the kernel spreads a super-particle, a
# Gaussian of width SMOOTH_H averaged around its ring (the ring-averaged weight of modules/showers/kernel.py, at the
# kernel's minimum width), the ring represented by SUB radii uniform in area. The tables then hold the smoothed
# densities on the radial grid GRID_R instead of the 23 ring centres. TAMBOPT_XR_SMOOTH_H=0: unsmoothed rings.
SMOOTH_H = float(os.environ.get("TAMBOPT_XR_SMOOTH_H", "10.0"))      # m
SUB = 8
GRID_R = np.geomspace(1.0, 12600.0, 64)                              # m


def _sym():
    from symbolic import atmosphere, data, draw, energy, lateral, longitudinal, model, rings, rows, times, width
    return dict(atmosphere=atmosphere, data=data, draw=draw, energy=energy, lateral=lateral, longitudinal=longitudinal,
                model=model, rings=rings, rows=rows, times=times, width=width)


def load_model(folder=MODEL):
    return _sym()["model"].load(folder)


def ring_geometry():
    S = _sym()
    lo = S["data"].EDGES[:-1].astype(np.float64)
    hi = np.r_[S["data"].EDGES[1:-1], S["draw"].R_OPEN * S["data"].EDGES[-2]].astype(np.float64)
    return S["times"].LN_R.astype(np.float64), np.pi * (hi ** 2 - lo ** 2)


def radial_axis():
    """ln r of the tables' radial axis: the smoothing grid, or the ring centres when unsmoothed."""
    return np.log(GRID_R) if SMOOTH_H > 0 else ring_geometry()[0]


def smoothing_map(h=None):
    """(len(GRID_R), rings): density at GRID_R [1/m^2] per unit content of each ring, after the kernel's smoothing."""
    from scipy.special import i0e
    h = SMOOTH_H if h is None else h
    S = _sym()
    lo = S["data"].EDGES[:-1].astype(np.float64)
    hi = np.r_[S["data"].EDGES[1:-1], S["draw"].R_OPEN * S["data"].EDGES[-2]].astype(np.float64)
    out = np.zeros((len(GRID_R), len(lo)))
    g = GRID_R[:, None]
    for j in range(len(lo)):                     # sub-radii spaced at most h / 2, so the annulus is covered evenly
        m = max(SUB, int(np.ceil(2 * (hi[j] - lo[j]) / h)))
        r = np.sqrt(lo[j] ** 2 + (np.arange(m) + 0.5) / m * (hi[j] ** 2 - lo[j] ** 2))[None, :]
        out[:, j] = (np.exp(-(g - r) ** 2 / (2 * h * h)) * i0e(g * r / (h * h))).mean(1) / (2 * np.pi * h * h)
    return out


def tables(part1, laws, log10_e, dirs, cls, seed, energy_density=False):
    """Per shower: log particle density [1/m^2] (n, species, planes, rings) and the delay constants (n, species, 3).
    log10_e (n,), dirs (n, 3) travel directions, cls (n,) 0 EM / 1 hadronic, seed: part-1 draws. energy_density:
    also the log energy density [GeV/m^2], same shape as the particle density (diagnostics)."""
    S = _sym()
    n = len(log10_e)
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    zen = np.clip(np.degrees(np.arccos(np.clip(d[:, 2], -1, 1))), ZENITH_MIN, ZENITH_MAX)
    u = np.cos(np.radians(zen))
    X = S["atmosphere"].plane_depths(zen, N_PLANES)
    E, xmax = S["draw"].sample_energies(part1, log10_e, u, X, cls, np.random.default_rng(seed))
    _, area = ring_geometry()
    W = smoothing_map() if SMOOTH_H > 0 else None
    nr = len(radial_axis())
    logd = np.full((n, len(SPECIES_NAMES), N_PLANES, nr), LOG_FLOOR, np.float32)
    tconst = np.zeros((n, len(SPECIES_NAMES), 3), np.float32)
    loge = np.full_like(logd, LOG_FLOOR) if energy_density else None
    for i in range(n):
        cn = S["data"].CLASS[int(cls[i])]
        for a, s in enumerate(SPECIES_NAMES):
            law = laws[(s, cn)]
            tconst[i, a] = law["delay"]
            ks = np.nonzero(E[s][i] > 0)[0]
            if not len(ks):
                continue
            r_ref = np.array([S["lateral"].r_m_ref(k, u[i]) for k in ks])
            x = S["rows"].features(S["longitudinal"].age(X[i, ks], xmax[s][i]), np.full(len(ks), log10_e[i]),
                                   np.full(len(ks), u[i]), np.log(E[s][i, ks]), r_ref)
            ll = S["width"].mean(law["width"], ks, x[:, 3], x[:, 1])
            q = softmax(x @ law["rings"], axis=1)
            sh = np.stack([S["rings"].stretch(q[j], ll[j]) for j in range(len(ks))])
            mean_e = np.exp(S["energy"].tanh_sum(law["energy"], S["times"].expand(S["rows"].plane_inputs(x, ks))))
            n_part = E[s][i, ks][:, None] * sh / mean_e.reshape(len(ks), -1)
            e_ring = E[s][i, ks][:, None] * sh
            dn, de = (n_part @ W.T, e_ring @ W.T) if W is not None else (n_part / area, e_ring / area)
            with np.errstate(divide="ignore"):
                logd[i, a, ks] = np.maximum(np.log(dn), LOG_FLOOR)
                if energy_density:
                    loge[i, a, ks] = np.maximum(np.log(de), LOG_FLOOR)
    return (logd, tconst, loge) if energy_density else (logd, tconst)


class ExpectedResponse:
    """Kernel stand-in for one sub-batch: logd (B, 3, planes, radial axis), tconst (B, 3, 3), frames (B, 12)."""

    def __init__(self, logd, tconst, frames, surface):
        self.logd, self.tconst, self.frames, self.surface = logd, tconst, frames, surface
        ln_r = radial_axis()
        self.ln_r = torch.as_tensor(ln_r, dtype=torch.float32, device=logd.device)

    def __call__(self, primary_batch, xy_batch):
        e_det, n_det = xy_batch[0, :, 0], xy_batch[0, :, 1]
        rho, z = detectors_in_frames(e_det, n_det, self.surface, self.frames)       # (B, nd), z = depth / 500 m
        rho, z = rho.float(), z.float()
        B, nd = rho.shape
        kf = z - 1.0                                                                # plane k sits at (k + 1) x 500 m
        inside = kf <= N_PLANES - 1
        kc = kf.clamp(0.0, N_PLANES - 1 - 1e-4)
        k0 = kc.floor().long(); wk = kc - k0
        lr = torch.log(rho).clamp(float(self.ln_r[0]), float(self.ln_r[-1]) - 1e-6)
        j0 = (torch.searchsorted(self.ln_r, lr.contiguous(), right=True) - 1).clamp(0, len(self.ln_r) - 2)
        wj = (lr - self.ln_r[j0]) / (self.ln_r[j0 + 1] - self.ln_r[j0])
        b = torch.arange(B, device=rho.device)[:, None].expand(B, nd)
        mus, delays = [], []
        for a in range(len(SPECIES_NAMES)):
            L = self.logd[:, a]                                                     # (B, planes, rings)

            def at(k):
                return (1 - wj) * L[b, k, j0] + wj * L[b, k, j0 + 1]
            dens = (1 - wk) * torch.exp(at(k0)) + wk * torch.exp(at((k0 + 1).clamp(max=N_PLANES - 1)))
            mus.append(torch.where(inside, dens, torch.zeros_like(dens)) * AREA * float(EFF[a]))
            A, p, q = (self.tconst[:, a, c][:, None] for c in range(3))
            delays.append(torch.exp(torch.clamp(A + p * torch.log(rho) - q * torch.log(kc + 1.0), max=50.0)))  # ns
        mu_s = torch.stack(mus)                                                     # (3, B, nd)
        self._mu_s = mu_s
        self.mu = mu_s.sum(0)
        w = mu_s / self.mu.clamp(min=1e-30)
        T = (z * 500.0 / C_LIGHT + (w * torch.stack(delays)).sum(0) * 1e-9)        # s since the decay
        timed = self.mu > 1e-6
        T = torch.where(timed, T, torch.zeros_like(T))
        return torch.stack([torch.log1p(self.mu), torch.log1p(T * T_LOG_SCALE)], -1)
