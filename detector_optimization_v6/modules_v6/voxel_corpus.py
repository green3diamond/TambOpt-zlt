"""Voxel-compacted shower corpus for kernel-direct layout optimization.

The plane-aware Gaussian kernel (sigma = SIGMA_SPATIAL) cannot resolve
structure much finer than sigma, so placed clouds are pooled per
(North, Up, layer) voxel and the WHOLE corpus becomes GPU-resident — the
kernel then replaces the FNN surrogate inside the optimizer at full-corpus
exact-gradient cost (validated by plots/check_compaction_fidelity.py and
plots/check_kernel_grad_timing.py).

Per voxel: energy-weighted centroid (North, Up, layer), sum(energy), and the
RAW time sum. The kernel's arrival_time divides by the PADDED point count, so
the (Vmax / P) rescale that reproduces the full-cloud normalization is applied
at padding time (`pad_ragged`) — Vmax depends on the batch being padded.

Storage is ragged CSR per species: `vals (V_total, 5)` grouped by cloud +
`counts (n_pairs,)`, built once by `build_voxel_corpus` (streamed, chunked)
and loaded by `VoxelCorpus`. `KernelVoxelLabels` exposes a slice of events
with the dual FNN's exact `(primary_batch, xy_batch) -> (B, n_det, 2)` call
signature so `opt_core.utility_of_xy` is reused UNMODIFIED.
"""
import os

import torch

from .constants import (
    DUAL_SHOWER_CACHE_PATH, DUAL_POSITIONS_PATH, EAST_ENTRY, LAYER_EAST_DX,
    T_LOG_SCALE, TRAINING_DATASET_FOLDER,
)
from .fnn_surrogate_ne import place_clouds_enu, compute_labels_batch
from .dual_surrogate import combine_species_outputs

LAYER_BIN     = 0.25    # z/layer-axis bin [layer units]; the triangular plane
                        # weight is piecewise linear, so fine binning limits
                        # kink-straddling error
DEFAULT_DELTA = 31.25   # spatial voxel size [m]

# Point-corruption guard, root-caused via plots/inspect_voxel_corpus.py +
# plots/validate_limitations.py (check 2): a handful of muon corpus points
# carry finite-but-absurd position/energy (North offsets to ~2.9e5 m against a
# ~1-2 km mountain scale; energy sums to 6.6e34). These are FINITE, so
# torch.isfinite() alone (what the original per-point pipeline in
# fnn_surrogate_ne.build_training_pairs uses) never catches them; against any
# REAL detector layout their kernel weight underflows to exactly 0 (confirmed:
# the flagged events score 0.0 in the actual trained E.pt/T.pt), so the
# original pipeline is harmless to them by accident, not by design. This
# module's energy-weighted centroid, though, divides sums of e*position across
# many points in one voxel bin — if two such points of opposite sign land in
# the same bin, their products can hit +inf and -inf individually (float32 max
# ~3.4e38) and cancel to NaN, which then poisons dx = point_x - x_det for
# EVERY detector regardless of layout (this is what crashed the kernel-direct
# optimizer). Bounds below are generous (10x+ beyond the mountain/corpus scale,
# 10+ orders of magnitude below the observed corruption) so real signal is
# never touched.
SPATIAL_CAP = 1e4   # metres — mountain/corpus scale is O(1-2 km)
ENERGY_CAP  = 1e12  # electron's legitimate max reaches ~4.8e11; muon corruption reaches 1e34
TIME_CAP    = 1e-3  # seconds — legitimate scale is ~1e-7 to 1e-4; corruption reaches 1e13


def voxel_cache_path(delta: float = DEFAULT_DELTA) -> str:
    return os.path.join(TRAINING_DATASET_FOLDER, f"voxel_corpus_d{delta:g}.pt")


def voxelize(clouds: torch.Tensor, delta: float, layer_bin: float = LAYER_BIN,
            verbose: bool = False):
    """(B, P, 5) placed clouds -> ragged (vals (V, 5), counts (B,)).

    Placed columns: [North, Up, z_cont, energy, time]. Only energy-carrying
    points are pooled, AND only those within [SPATIAL_CAP, ENERGY_CAP,
    TIME_CAP] of a legitimate physical scale (see module docstring — this is
    a corruption guard, not a physics choice: dropped points contribute ~0 to
    any real layout anyway). `vals` rows are grouped by cloud in cloud order:
    [N_centroid, U_centroid, l_centroid, sum(e), RAW sum(t)] — apply
    `pad_ragged` before feeding the kernel."""
    B = clouds.shape[0]
    mask = (
        (clouds[..., 3] > 0)
        & torch.isfinite(clouds).all(dim=-1)
        & (clouds[..., 0].abs() < SPATIAL_CAP)
        & (clouds[..., 1].abs() < SPATIAL_CAP)
        & (clouds[..., 3].abs() < ENERGY_CAP)
        & (clouds[..., 4].abs() < TIME_CAP)
    )
    if verbose:
        n_dropped = int((clouds[..., 3] > 0).sum() - mask.sum())
        if n_dropped:
            print(f"[voxelize] dropped {n_dropped} corrupted point(s) "
                  f"(out of {int((clouds[..., 3] > 0).sum())} energy-carrying)")
    b_idx, p_idx = torch.nonzero(mask, as_tuple=True)
    pts = clouds[b_idx, p_idx]                                  # (M, 5)
    n, u, l, e, t = pts.unbind(dim=1)

    iN = torch.floor(n / delta).long()
    iU = torch.floor(u / delta).long()
    iL = torch.floor(l / layer_bin).long()
    for i in (iN, iU, iL):
        i -= i.min()
    stride = (iN.max() + 1) * (iU.max() + 1) * (iL.max() + 1)   # keys per cloud
    key = ((b_idx * (iN.max() + 1) + iN) * (iU.max() + 1) + iU) * (iL.max() + 1) + iL
    uniq, inv = torch.unique(key, return_inverse=True)

    sums = torch.zeros(uniq.numel(), 5, device=clouds.device)
    sums.index_add_(0, inv, torch.stack([e * n, e * u, e * l, e, t], dim=1))
    e_sum = sums[:, 3].clamp(min=1e-30)
    vals = torch.cat([sums[:, :3] / e_sum.unsqueeze(1),         # e-weighted centroid
                      sums[:, 3:4], sums[:, 4:5]], dim=1)       # sum(e), raw sum(t)

    vb = uniq // stride           # cloud of each voxel; uniq sorted -> vb sorted,
    counts = torch.bincount(vb, minlength=B)   # so `vals` is grouped by cloud
    return vals, counts


def pad_ragged(vals: torch.Tensor, counts: torch.Tensor, starts: torch.Tensor,
               P: int) -> torch.Tensor:
    """Gather ragged voxels into a padded (B, Vmax, 5) kernel-ready tensor.

    Padding rows are zeroed (zero energy and zero time sum contribute nothing
    to either kernel channel); col 4 is rescaled by (Vmax / P) so the kernel's
    padded-mean time normalization reproduces the full-cloud one."""
    B = counts.numel()
    Vmax = int(counts.max())
    ar = torch.arange(Vmax, device=vals.device)
    pos = (starts.unsqueeze(1) + ar.unsqueeze(0)).clamp(max=vals.shape[0] - 1)
    mask = ar.unsqueeze(0) < counts.unsqueeze(1)                # (B, Vmax)
    out = vals[pos] * mask.unsqueeze(-1)
    out[..., 4] *= Vmax / P
    return out


def build_voxel_corpus(delta: float = DEFAULT_DELTA, chunk: int = 1024,
                       device: str = "cuda", out_path: str = None) -> str:
    """Stream the dual corpus once, place, voxelize, save the ragged cache."""
    import showerdata   # deferred: heavy import, script-level dependency

    out_path = out_path or voxel_cache_path(delta)
    positions_all = torch.load(DUAL_POSITIONS_PATH)
    n_pairs = positions_all.shape[0] // 2

    acc = {s: ([], []) for s in ("elec", "muon")}               # vals, counts
    P = None
    for start in range(0, n_pairs, chunk):
        stop = min(start + chunk, n_pairs)
        subs = {"elec": showerdata.load(DUAL_SHOWER_CACHE_PATH, start=start, stop=stop),
                "muon": showerdata.load(DUAL_SHOWER_CACHE_PATH,
                                        start=n_pairs + start, stop=n_pairs + stop)}
        dirs = torch.as_tensor(subs["elec"].directions, dtype=torch.float32)
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)
        pos = positions_all[start:stop].float()
        for s, sub in subs.items():
            cl = torch.as_tensor(sub.points, dtype=torch.float32)
            P = cl.shape[1]
            place_clouds_enu(cl, pos, dirs, east_entry=EAST_ENTRY,
                             layer_east_dx=LAYER_EAST_DX)
            vals, counts = voxelize(cl.to(device), delta, verbose=True)
            acc[s][0].append(vals.cpu())
            acc[s][1].append(counts.cpu())
        if (start // chunk) % 10 == 0:
            print(f"[voxelize] {stop}/{n_pairs} pairs", flush=True)

    blob = dict(delta=delta, layer_bin=LAYER_BIN, P=P, n_pairs=n_pairs)
    for s in acc:
        blob[s + "_vals"] = torch.cat(acc[s][0])
        blob[s + "_counts"] = torch.cat(acc[s][1])
        print(f"[voxelize] {s}: {blob[s + '_vals'].shape[0]} voxels "
              f"({blob[s + '_vals'].shape[0] / n_pairs:.0f}/pair)")
    tmp = out_path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, out_path)                                   # preemption-atomic
    print(f"[voxelize] saved -> {out_path}")
    return out_path


class VoxelCorpus:
    """GPU-resident ragged voxel corpus (both species)."""

    def __init__(self, path: str, device):
        d = torch.load(path, map_location="cpu")
        self.delta, self.P, self.n_pairs = d["delta"], d["P"], d["n_pairs"]
        self.device = device
        for s in ("elec", "muon"):
            vals = d[s + "_vals"].to(device)
            counts = d[s + "_counts"].to(device)
            starts = torch.cumsum(counts, 0) - counts
            setattr(self, s, (vals, counts, starts))
        gib = (self.elec[0].numel() + self.muon[0].numel()) * 4 / 2 ** 30
        print(f"[voxel] {path}  delta={self.delta:g} m  pairs={self.n_pairs}  "
              f"resident {gib:.2f} GiB on {device}")

    def padded(self, idx: torch.Tensor, species: str) -> torch.Tensor:
        vals, counts, starts = getattr(self, species)
        return pad_ragged(vals, counts[idx], starts[idx], self.P)

    def pack_chunks(self, idx, slot_budget: int = 4_000_000,
                    max_events: int = 2048):
        """Split events `idx` into padding-efficient chunks for the kernel.

        Padded chunks cost n_events * (Vmax_elec + Vmax_muon) voxel slots and a
        single outlier cloud inflates the whole chunk (a ~34k-voxel muon cloud
        in a 512-event chunk OOMed a 40 GB A100). Sorting by size and capping
        each chunk's slot product keeps every chunk's autograd graph bounded:
        outliers go into small chunks, small clouds into large ones."""
        idx = torch.as_tensor(idx, device=self.device).reshape(-1)
        ec, mc = self.elec[1], self.muon[1]
        order = torch.argsort(ec[idx] + mc[idx], descending=True)
        chunks, cur = [], []
        max_e = max_m = 0
        for i in idx[order].tolist():
            me, mm = max(max_e, int(ec[i])), max(max_m, int(mc[i]))
            if cur and ((len(cur) + 1) * (me + mm) > slot_budget
                        or len(cur) >= max_events):
                chunks.append(torch.as_tensor(cur, device=self.device))
                cur, max_e, max_m = [], 0, 0
                me, mm = int(ec[i]), int(mc[i])
            cur.append(i)
            max_e, max_m = me, mm
        if cur:
            chunks.append(torch.as_tensor(cur, device=self.device))
        return chunks


class KernelVoxelLabels:
    """Kernel ground-truth label source over the voxel-corpus events `idx`,
    with the dual FNN's `(primary_batch, xy_batch) -> (B, n_det, 2)` signature
    (rows of a call share one layout). Gradient mode follows the caller:
    differentiable inside optimizer backprop, no-grad under torch.no_grad()."""

    def __init__(self, corpus: VoxelCorpus, idx: torch.Tensor, surface):
        idx = torch.as_tensor(idx, device=corpus.device)
        self.elec = corpus.padded(idx, "elec")
        self.muon = corpus.padded(idx, "muon")
        self.surface = surface

    def __call__(self, primary_batch, xy_batch):
        e_det, n_det = xy_batch[0, :, 0], xy_batch[0, :, 1]
        diff = torch.is_grad_enabled()
        E_e, T_e = compute_labels_batch(self.elec, e_det, n_det, self.surface,
                                        differentiable=diff)
        E_m, T_m = compute_labels_batch(self.muon, e_det, n_det, self.surface,
                                        differentiable=diff)
        # clamp(min=0) before log1p: a (numerically) negative count or summed
        # time would otherwise NaN the whole batch through log1p(x < -1).
        pred_e = torch.stack([torch.log1p(E_e.clamp(min=0)),
                              torch.log1p((T_e * T_LOG_SCALE).clamp(min=0))], dim=-1)
        pred_m = torch.stack([torch.log1p(E_m.clamp(min=0)),
                              torch.log1p((T_m * T_LOG_SCALE).clamp(min=0))], dim=-1)
        return combine_species_outputs(pred_e, pred_m)
