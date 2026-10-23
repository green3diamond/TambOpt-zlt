"""Prepare replay data and retrain a detector reconstruction network.

The network maps detector positions, expected counts and arrival-time signals to
shower direction and energy. label() computes those inputs for pool showers at
the current layout, optionally with every detector randomly displaced per small
batch of showers (jitter), so the new examples also show how the inputs change
as detectors move. fine_tune() trains a copy on the new examples mixed with
earlier ones (replay), taken from the given sources in order: typically the
latest refreshes first, then the original training data. A reserved slice of
the new examples selects the best epoch and decides whether to accept it. A
fixed validation sample (anchor) only measures forgetting. Normalisation from
the starting checkpoint stays unchanged.

Prepare the data file used by recon_loop/run.py with::

    python recon_loop/refresh.py --dataset /path/to/labelled_dataset \
        --recon /path/to/recon.pt --out /path/to/bank.pt

The dataset must contain the detector-response tensors written by the existing
data builder. Its shower ordering and training/validation split must match the
starting checkpoint and the optimisation corpus. The saved bank contains replay
examples, anchor examples and flags identifying unusable showers. See README.md
for the input files, training settings and complete launch commands.
"""

import argparse
import copy
import importlib.util
import os
import sys
import time
from dataclasses import asdict, dataclass

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

BUFFERS = ("in_mean", "in_std", "out_mean", "out_std")
LABEL_SLICE = 512  # Limit disk reads when labelling a refresh batch.


@dataclass
class Config:
    """Settings for one candidate-training attempt, separate from layout Adam."""

    lr: float = 1e-5
    epochs: int = 5
    batch: int = 256
    grad_clip: float = 10.0
    holdout: float = 0.1
    replay: float = 0.5

    def __post_init__(self):
        if not (0 < self.holdout < 1 and 0 <= self.replay < 1):
            raise ValueError("holdout must be in (0, 1), replay in [0, 1)")
        if self.lr < 0 or min(self.epochs, self.batch, self.grad_clip) <= 0:
            raise ValueError("lr must be nonnegative; epochs, batch and grad_clip positive")


def features(xy, ET):
    """Stack East, North, log-count and log-time channels into (showers, detectors, 4)."""
    return torch.stack([xy[..., 0], xy[..., 1], ET[..., 0], ET[..., 1]], dim=-1)


def loss(pred, target):
    """Sum the four mean squared errors: direction components and scaled log-energy."""
    return ((pred - target) ** 2).mean(0).sum()


@torch.no_grad()
def eval_loss(recon, feats, target, device, batch=4096):
    """Average the reconstruction error over a dataset without building gradients."""
    tot = 0.0
    for i in range(0, len(feats), batch):
        f, t = feats[i : i + batch].to(device), target[i : i + batch].to(device)
        tot += float(((recon(f) - t) ** 2).sum(0).sum())
    return tot / max(len(feats), 1)


def split(n, holdout, gen):
    """Return disjoint training and local-validation indices for this refresh."""
    if n < 2:
        raise ValueError("a refresh needs at least two usable showers")
    perm = torch.randperm(n, generator=gen)
    n_held = min(n - 1, max(1, int(round(holdout * n))))
    return perm[n_held:], perm[:n_held]


def _buffers(model):
    return {k: getattr(model, k).detach().clone() for k in BUFFERS}


def draw(sources, n, gen):
    """Take n random rows from (features, targets) sources, exhausting each before the next."""
    if n > sum(len(x) for x, _ in sources):
        raise ValueError(f"replay sources hold {sum(len(x) for x, _ in sources)} rows, refresh needs {n}")
    xs, ys = [], []
    for x, y in sources:
        idx = torch.randperm(len(x), generator=gen)[:n]
        xs.append(x[idx])
        ys.append(y[idx])
        n -= len(idx)
    return torch.cat(xs), torch.cat(ys)


def fine_tune(recon, fresh, replay, anchor, cfg, seed, device):
    """Train a candidate model and return (accepted weights or None, statistics).

    fresh and anchor are CPU (features, targets) tensor pairs; replay is a list
    of such pairs in priority order. A fraction cfg.replay of the training rows
    comes from earlier examples. All epochs run; the best local-validation epoch
    is returned only if it beats the incoming model. The live model and its
    normalisation buffers are unchanged.
    """
    # Separate local validation from the current layout's training rows.
    t0 = time.time()
    gen = torch.Generator().manual_seed(seed)
    tr, held = split(len(fresh[0]), cfg.holdout, gen)

    # Mix the requested share of earlier examples into the training set.
    n_rep = int(round(len(tr) * cfg.replay / (1.0 - cfg.replay))) if cfg.replay > 0 else 0
    rx, ry = draw(replay, n_rep, gen)
    x = torch.cat([fresh[0][tr], rx])
    y = torch.cat([fresh[1][tr], ry])
    hx, hy = fresh[0][held], fresh[1][held]

    # Train a copy; rejected candidates cannot change the live model.
    model = copy.deepcopy(recon)
    buf0 = _buffers(model)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.eval()

    # Measure the incoming model before trying any parameter updates.
    held0 = eval_loss(model, hx, hy, device)
    anchor0 = eval_loss(model, *anchor, device)
    best, best_state, best_epoch, curve = held0, None, 0, []

    # Train all epochs and retain the best local-validation state.
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x), cfg.batch):
            idx = perm[i : i + cfg.batch]
            l = loss(model(x[idx].to(device)), y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            opt.step()

        # Validation is evaluated without further training updates.
        model.eval()
        h = eval_loss(model, hx, hy, device)
        curve.append(h)
        if h < best:
            best, best_epoch = h, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Input/output scaling must remain identical to the starting model.
    if any(not torch.equal(getattr(model, k), v) for k, v in buf0.items()):
        raise RuntimeError("normalisation buffers changed during the refresh")

    # Anchor error monitors forgetting; it does not decide acceptance.
    anchor1 = anchor0
    if best_state is not None:
        model.load_state_dict(best_state)
        anchor1 = eval_loss(model, *anchor, device)

    # Return only an improvement, together with the full validation curve.
    stats = dict(
        config=asdict(cfg),
        n_fresh_train=len(tr),
        n_replay=n_rep,
        n_held=len(held),
        held_before=held0,
        held_curve=curve,
        held_best=best,
        best_epoch=best_epoch,
        accepted=best_state is not None,
        anchor_before=anchor0,
        anchor_after=anchor1,
        seconds=time.time() - t0,
    )
    return best_state, stats


def install(recon, state):
    """Install accepted weights, then freeze the model for detector-position updates."""
    buf0 = _buffers(recon)
    recon.load_state_dict(state)
    if any(not torch.equal(getattr(recon, k), v) for k, v in buf0.items()):
        raise RuntimeError("refreshed state carries different normalisation buffers")
    for p in recon.parameters():
        p.requires_grad_(False)
    recon.eval()


def checkpoint(initial_ckpt, state, **meta):
    """Package refreshed weights using the starting model's checkpoint format."""
    for k in BUFFERS:
        if not torch.equal(state[k].cpu(), initial_ckpt["state_dict"][k].cpu()):
            raise RuntimeError(f"state buffer {k} differs from the starting model's")
    return {
        **initial_ckpt,
        "state_dict": {k: v.detach().cpu().clone() for k, v in state.items()},
        "refresh": meta,
    }


def import_script(filename):
    """Import existing pipeline helpers without running their command-line entry point."""
    spec = importlib.util.spec_from_file_location(
        filename.removesuffix(".py"), os.path.join(REPO, "scripts", filename)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_atomic(obj, path):
    """Replace a checkpoint only after the temporary file has been written."""
    torch.save(obj, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def pool_blob(blob, species, strat):
    """Mark a shower unusable if any particle species has a degenerate response."""
    n_sp, n_st = int(species.max()) + 1, int(strat.max()) + 1
    b = blob.view(n_st, n_sp, -1).any(1)
    if not bool((b == b[0]).all()):
        raise ValueError("degenerate-shower flags differ between layout draws")
    return b[0].clone()


def select(mask, n, seed):
    """Choose a reproducible, sorted subset of eligible dataset rows."""
    idx = torch.nonzero(mask).squeeze(1)
    if n < 1 or len(idx) < n:
        raise ValueError(f"requested {n} rows from {len(idx)} eligible rows")
    return (
        idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed))[:n]]
        .sort()
        .values
    )


def kept_index(all_prim, kept_prim):
    """Map filtered shower targets back to their original pool indices."""
    idx, j = [], 0
    for i in range(len(all_prim)):
        if j < len(kept_prim) and torch.equal(all_prim[i], kept_prim[j]):
            idx.append(i)
            j += 1
    if j != len(kept_prim):
        raise RuntimeError("loaded primaries are not an ordered subset of the pool's")
    return torch.tensor(idx, dtype=torch.long)


def jitter_layout(x, y, sigma, gen, project):
    """Displace every detector by an independent Gaussian offset of width sigma, then project."""
    dx = torch.randn(len(x), generator=gen, dtype=x.dtype) * sigma
    dy = torch.randn(len(y), generator=gen, dtype=y.dtype) * sigma
    return project(x + dx, y + dy)


@torch.no_grad()
def label(kernel, pool, a, b, xy_module, surface, chunk, blob, device, jitter=0.0, gen=None, mountain=None):
    """Compute detector features and direction/energy targets for the current layout.

    With jitter > 0, each chunk of showers sees the layout with every detector moved
    by a Gaussian offset of that width (metres), projected back onto the allowed
    surface; gen makes the offsets reproducible and mountain defines the surface.
    """
    if jitter and (gen is None or mountain is None):
        raise ValueError("jitter needs a generator and the mountain surface")

    def project(e, n):
        e, n = kernel.project_to_mountain_ne(mountain, e.cpu(), n.cpu())
        return e.to(device).to(x.dtype), n.to(device).to(y.dtype)

    x, y = xy_module()
    feats, target = [], []
    for lo in range(a, b, LABEL_SLICE):
        hi = min(lo + LABEL_SLICE, b)

        # Keep exclusion flags aligned when the loader drops invalid showers.
        clouds, bws, prim, frames = pool.load(lo, hi)
        keep = ~blob[lo:hi][kept_index(pool.meta.primaries[lo:hi], prim)]
        f = []

        # Chunk kernel evaluation so the full refresh fits in GPU memory.
        for c0 in range(0, len(prim), chunk):
            c1 = min(c0 + chunk, len(prim))
            kern = kernel.BatchKernel(
                [c[c0:c1].to(device) for c in clouds],
                [w[c0:c1].to(device) if w is not None else None for w in bws],
                surface,
                differentiable=False,
                frames=None if frames is None else frames[c0:c1].to(device),
            )
            cx, cy = jitter_layout(x.cpu(), y.cpu(), jitter, gen, project) if jitter else (x, y)
            xy = torch.stack([cx, cy], -1).unsqueeze(0).expand(c1 - c0, -1, -1)
            f.append(features(xy, kern(None, xy)).cpu())

        # Store CPU training rows after removing flagged showers.
        feats.append(torch.cat(f)[keep])
        target.append(prim[:, :4].float()[keep])
    return torch.cat(feats), torch.cat(target)


def build_bank(dataset, recon, out, n_replay=200_000, n_anchor=4096):
    """Save earlier training examples and validation examples for network refreshes.

    Uses the existing reconstruction trainer's combined detector features and
    shower-level split. Keep one row per shower/layout and remove showers
    flagged degenerate in any species before choosing replay and anchor rows.
    """
    # Use the same feature construction and split as model training.
    trainer = import_script("03_train_recon_deepsets.py")

    def load(name):
        return torch.load(
            os.path.join(dataset, name + ".pt"), map_location="cpu", weights_only=False
        )

    # Check that checkpoint metadata agrees with its scaling buffers.
    ckpt = torch.load(recon, map_location="cpu", weights_only=False)
    for field, buf in zip(("input_mean", "input_std", "target_mean", "target_std"), BUFFERS):
        if not torch.equal(ckpt[field], ckpt["state_dict"][buf]):
            raise ValueError(f"checkpoint {field} differs from {buf}")

    # Build one feature vector per detector and the four prediction targets.
    E, T = trainer.load_kernel_responses(dataset)
    xy, primary = load("xy").float(), load("primary").float()
    species, strat, blob = (
        load("species_ids").long(),
        load("strategy_ids").long(),
        load("blob_ids").bool(),
    )
    feats = torch.stack([xy[..., 0], xy[..., 1], E, T], -1)

    # Keep every version/species of a shower on the same split side.
    tr, va = trainer.shower_level_split(strat, species, trainer.VAL_FRAC, trainer.SEED)
    pb = pool_blob(blob, species, strat)
    base = (species == 0) & ~pb[torch.arange(len(strat)) % len(pb)]
    in_tr, in_va = torch.zeros_like(base), torch.zeros_like(base)
    in_tr[tr], in_va[va] = True, True
    if bool((in_tr & in_va).any()):
        raise ValueError("training and validation rows overlap")

    # Choose earlier training rows and a separate forgetting-monitor sample.
    rep, anc = select(base & in_tr, n_replay, 0), select(base & in_va, n_anchor, 1)

    # Write a reusable CPU tensor bank without altering the input dataset.
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_atomic(
        dict(
            replay_feats=feats[rep],
            replay_target=primary[rep, :4],
            replay_rows=rep,
            anchor_feats=feats[anc],
            anchor_target=primary[anc, :4],
            anchor_rows=anc,
            pool_blob=pb,
            recon=os.path.abspath(recon),
            dataset=os.path.abspath(dataset),
        ),
        out,
    )
    print(
        f"[bank] replay {len(rep)}, anchor {len(anc)}, pool {len(pb)}, excluded {int(pb.sum())}: {out}"
    )


# The command-line entry point prepares a bank; training is called by run.py.
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Prepare replay and validation examples from labelled detector-response data."
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--out", required=True, help="output bank.pt")
    ap.add_argument("--n-replay", type=int, default=200_000)
    ap.add_argument("--n-anchor", type=int, default=4096)
    build_bank(**vars(ap.parse_args()))
