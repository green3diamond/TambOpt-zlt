"""A structured network for the mean particle energy per ring, built to be distilled piece by piece (SymTorch's method):
ln mean energy = c + f(ln r, ln k) + h(age, l, c, ln Ek, ln rm), f and h each a small tanh network, one fit per group and
fold on the fold's training showers, scored on its judged showers against the ring-by-ring law. The pieces are saved per
fold so each can be queried and searched on its own.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import energy_sr, rings_sr, times_sr, tiny_net

PIECES = {"f": (0, 2), "h": (1, 3, 4, 5, 6)}              # columns of rings_sr.VARIABLES each piece sees
WIDTH = 32
DEPTH = 2
LR = 3e-3
EPOCHS = 3000
PATIENCE = 200
SEEDS = 3
MEAN_BAR = 1.0                # the network must reach these ratios to the ring-by-ring law before it is distilled
GROUP_BAR = 1.25


def build(seed):
    import torch

    class Structured(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(seed)
            self.cols = {k: list(v) for k, v in PIECES.items()}
            self.nets = torch.nn.ModuleDict({k: mlp(len(v)) for k, v in PIECES.items()})
            self.c = torch.nn.Parameter(torch.zeros(1))

        def pieces(self, x):
            return {k: net(x[:, self.cols[k]]).squeeze(1) for k, net in self.nets.items()}

        def forward(self, x):
            return self.c + sum(self.pieces(x).values())

    def mlp(n_in):
        layers, n = [], n_in
        for _ in range(DEPTH):
            layers += [torch.nn.Linear(n, WIDTH), torch.nn.Tanh()]
            n = WIDTH
        return torch.nn.Sequential(*layers, torch.nn.Linear(n, 1))

    return Structured()


def train(X, y, val, seed, epochs=EPOCHS):
    """One structured network on the rows not in val, full batch -> (validation error, predict, saved state)."""
    import torch
    mu, sd = X[~val].mean(0), X[~val].std(0) + 1e-9
    y_mu, y_sd = y[~val].mean(), y[~val].std() + 1e-9
    tx = torch.as_tensor((X - mu) / sd, dtype=torch.float32)
    ty = torch.as_tensor((y - y_mu) / y_sd, dtype=torch.float32)
    net = build(seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    fit, hold = np.flatnonzero(~val), np.flatnonzero(val)
    best, state, wait = np.inf, None, 0
    for _ in range(epochs):
        opt.zero_grad()
        ((net(tx[fit]) - ty[fit]) ** 2).mean().backward()
        opt.step()
        with torch.no_grad():
            v = float(((net(tx[hold]) - ty[hold]) ** 2).mean())
        if v < best - 1e-6:
            best, state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(state)
    saved = {"state": state, "seed": seed, "mu": mu, "sd": sd, "y_mu": y_mu, "y_sd": y_sd}
    return best, lambda Q: predict(saved, Q), saved


def predict(saved, Q, piece=None):
    """ln mean energy at raw inputs Q, or one piece's share of it in ln units (c left out)."""
    import torch
    net = build(saved["seed"])
    net.load_state_dict(saved["state"])
    with torch.no_grad():
        x = torch.as_tensor((Q - saved["mu"]) / saved["sd"], dtype=torch.float32)
        out = net(x) if piece is None else net.pieces(x)[piece]
    out = out.numpy().astype(np.float64) * saved["y_sd"]
    return out + saved["y_mu"] if piece is None else out


def stage_fit(args, log):
    import torch
    torch.set_num_threads(args.workers)
    groups = times_sr.all_groups()
    run = os.path.join(args.out, args.name)
    scores, ratios, nets = {}, [], {f: {} for f in (0, 1)}
    reference = {}
    for g in groups:
        d = energy_sr.group_data(args, g)
        reference[g] = times_sr.score_rows(d["ref"], d)
        pred = np.full(d["y"].shape, np.nan)
        for f in (0, 1):
            rows, judged = d["fold"] != f, d["fold"] == f
            m = ((d["n"][rows] > 0) & np.isfinite(d["y"][rows])).ravel()
            X, y = times_sr.expand(d["v"][rows])[m], d["y"][rows].ravel()[m]
            showers = np.repeat(d["shower"][rows], times_sr.N_RING)[m]
            fits = [train(X, y, tiny_net.split_showers(showers, np.random.default_rng(100 * f + s)), s, args.epochs) for s in range(args.seeds)]
            best = min(fits, key=lambda q: q[0])
            pred[judged] = best[1](times_sr.expand(d["v"][judged])).reshape(-1, times_sr.N_RING)
            nets[f][g] = best[2]
        scores[g] = times_sr.score_rows(pred, d)
        if g in energy_sr.GATED:
            ratios.append(scores[g]["rmse"] / reference[g]["rmse"])
        log.info("%-18s law %.4f structured %.4f ratio %.2f", g, reference[g]["rmse"], scores[g]["rmse"], scores[g]["rmse"] / reference[g]["rmse"])
    for f in (0, 1):
        torch.save(nets[f], os.path.join(run, f"nets_fold{f}.pt"))
    passes = bool(np.mean(ratios) <= MEAN_BAR and max(ratios) <= GROUP_BAR)
    log.info("e/gamma ratio %s mean %.3f | passes %s", np.round(ratios, 2).tolist(), np.mean(ratios), passes)
    return {"scores": scores, "reference": reference, "ratios": ratios, "passes": passes,
            "numbers per group": sum(p.numel() for p in build(0).parameters())}


def main():
    from symbolic import logs
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=rings_sr.FOLDER)
    ap.add_argument("--name", default="struct_energy")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    run, log = logs.start(args.name, args)
    logs.save(run, stage_fit(args, log), args, log)


if __name__ == "__main__":
    main()
