"""A network small enough to be a formula: ln mean particle energy per ring = c + sum_u a_u tanh(w_u . x + b_u), one per
group and fold, for several numbers of terms U, scored on judged showers against the ring-by-ring law."""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import energy_sr, rings_sr, slot_net, times_sr

TERMS = (1, 2, 3, 4, 6, 8)
N_INPUTS = 7
LR = 3e-3
EPOCHS = 3000
PATIENCE = 200
SEEDS = 3


def numbers(u):
    return u * (N_INPUTS + 2) + 1


def train(X, y, val, u, seed, epochs=EPOCHS):
    """One tanh sum on the rows not in val, full batch -> (validation error, predict)."""
    import torch
    torch.manual_seed(seed)
    mu, sd = X[~val].mean(0), X[~val].std(0) + 1e-9
    y_mu, y_sd = y[~val].mean(), y[~val].std() + 1e-9
    tx = torch.as_tensor((X - mu) / sd, dtype=torch.float32)
    ty = torch.as_tensor((y - y_mu) / y_sd, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], u), torch.nn.Tanh(), torch.nn.Linear(u, 1))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    fit, hold = np.flatnonzero(~val), np.flatnonzero(val)
    best, state, wait = np.inf, None, 0
    for _ in range(epochs):
        opt.zero_grad()
        ((net(tx[fit]).squeeze(1) - ty[fit]) ** 2).mean().backward()
        opt.step()
        with torch.no_grad():
            v = float(((net(tx[hold]).squeeze(1) - ty[hold]) ** 2).mean())
        if v < best - 1e-6:
            best, state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(state)
    theta = constants(*(t.detach().numpy().astype(np.float64) for t in net.parameters()), mu, sd, y_mu, y_sd)
    return best, lambda Q: energy_sr.tanh_sum(theta, Q), theta


def constants(w, b, a, c, mu, sd, y_mu, y_sd):
    """The fitted sum on raw inputs and target, scalings folded in: [w (U x inputs), b (U), a (U), c], 9U + 1 numbers."""
    return np.concatenate([(w / sd).ravel(), b - (w / sd) @ mu, a.ravel() * y_sd, c * y_sd + y_mu])


def smallest_passing(ratios):
    """The smallest U whose ratios pass the formula bars. ratios: {U: [ratio per GATED group]}."""
    ok = [u for u in sorted(ratios) if times_sr.stands({"ln mean energy": ratios[u]})]
    return ok[0] if ok else None


def stage_fit(args, log):
    import torch
    torch.set_num_threads(args.workers)
    groups = times_sr.all_groups()
    data = {g: energy_sr.group_data(args, g) for g in groups}
    reference = {g: times_sr.score_rows(data[g]["ref"], data[g]) for g in groups}
    scores, ratios = {}, {}
    for u in args.terms:
        scores[u] = {}
        laws = {f: {"form": np.array("tanh")} for f in (0, 1)}
        for g, d in data.items():
            pred = np.full(d["y"].shape, np.nan)
            for f in (0, 1):
                train_rows, judged = d["fold"] != f, d["fold"] == f
                m = ((d["n"][train_rows] > 0) & np.isfinite(d["y"][train_rows])).ravel()
                X, y = times_sr.expand(d["v"][train_rows])[m], d["y"][train_rows].ravel()[m]
                showers = np.repeat(d["shower"][train_rows], times_sr.N_RING)[m]
                fits = [train(X, y, slot_net.split_showers(showers, np.random.default_rng(100 * f + s)), u, s, args.epochs) for s in range(args.seeds)]
                best = min(fits, key=lambda q: q[0])
                pred[judged] = best[1](times_sr.expand(d["v"][judged])).reshape(-1, times_sr.N_RING)
                laws[f][g.replace(" ", ".")] = best[2]
            scores[u][g] = times_sr.score_rows(pred, d)
        if args.save:
            for f in (0, 1):
                np.savez(os.path.join(args.out, args.name, f"fold{f}_U{u}_{energy_sr.LAW}"), **laws[f])
        ratios[u] = [scores[u][g]["rmse"] / reference[g]["rmse"] for g in energy_sr.GATED]
        log.info("U %d numbers %d total %d | %s | e/gamma ratio %s mean %.3f | stands %s", u, numbers(u), numbers(u) * len(groups),
                 " ".join(f"{g.split()[0][0]}{g.split()[1][0]} {scores[u][g]['rmse']:.3f}" for g in groups),
                 np.round(ratios[u], 2).tolist(), np.mean(ratios[u]), times_sr.stands({"ln mean energy": ratios[u]}))
    chosen = smallest_passing(ratios)
    log.info("law %s | smallest passing U %s", " ".join(f"{g.split()[0][0]}{g.split()[1][0]} {reference[g]['rmse']:.3f}" for g in groups), chosen)
    return {"scores": {str(u): v for u, v in scores.items()}, "reference": reference, "ratios": {str(u): v for u, v in ratios.items()}, "chosen": chosen}


def main():
    from symbolic import logs
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=rings_sr.FOLDER)
    ap.add_argument("--name", default="tiny_energy")
    ap.add_argument("--terms", type=int, nargs="+", default=list(TERMS))
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--save", action="store_true", help="per U and fold, the constants for draw.py --energy-law")
    args = ap.parse_args()
    run, log = logs.start(args.name, args)
    logs.save(run, stage_fit(args, log), args, log)


if __name__ == "__main__":
    main()
