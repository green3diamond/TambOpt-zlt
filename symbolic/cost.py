"""Sampling cost per shower on the scored test primaries: the per-plane energy sampler (CPU, one core) against
PointCountFM with its anti-clip re-roll (CPU, as in the pipeline) and the full neural chain (PointCountFM on CPU,
AllShowers on GPU), each timed on a second, warm call. Also the parameter counts of the networks.
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch

from symbolic import logs
from modules.constants import SPECIES_NAMES
from symbolic import chain, fit_profiles, planes, sampler, sources

FOLDER = "cost"


def timed(fn, repeats=2):
    """Seconds of the last of `repeats` calls (the first warms caches and compilation)."""
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return dt


def module_params(obj):
    """Parameters of every torch module held by obj (counted once)."""
    seen, n = set(), 0
    for v in vars(obj).values():
        if isinstance(v, torch.nn.Module):
            for p in v.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    n += p.numel()
    return int(n)


def stage_run(out, args, log):
    rows, E, D, lab = chain.primaries()
    n = len(rows)
    zt = np.load(os.path.join(args.out, fit_profiles.TEST))
    l, u, X = zt["meta.log10_E"][rows], zt["meta.cos_zenith"][rows], zt["X"][rows]
    res = {"n_primaries": n, "seconds_per_shower": {}, "params": {}}

    torch.set_num_threads(1)
    tr = sampler.load(os.path.join(args.out, fit_profiles.TRAIN))
    models = {c: sampler.fit(tr, c) for c in planes.CLASS}
    rng = np.random.default_rng(0)
    sym = lambda: [sampler.sample(models[c], l[lab == c], u[lab == c], X[lab == c], rng) for c in planes.CLASS]
    res["seconds_per_shower"]["symbolic sampler, 3 species, CPU 1 core"] = timed(sym) / n
    log.info("symbolic sampler: %.3g s per shower (all species)", res["seconds_per_shower"]["symbolic sampler, 3 species, CPU 1 core"])

    torch.set_default_dtype(torch.float32)          # the sampler's modules set float64; the networks run in float32
    s0 = sources.stage0()
    from allshowers.generator import Generator
    energies, directions, labels = torch.as_tensor(E), torch.as_tensor(D), torch.as_tensor(lab).to(torch.int64)
    ids = torch.arange(n)
    pcfm_total, chain_total = {}, {}
    for threads in (1, args.threads):
        torch.set_num_threads(threads)
        t = 0.0
        for s in SPECIES_NAMES:
            pcfm = os.path.join(sources.STAGED, s, "pcfm_compiled.pt")

            def counts():
                k = s0.run_point_count_fm(model_path=pcfm, energies=energies, directions=directions, labels=labels)
                s0.resample_overclip(pcfm, energies, directions, labels, k, cap=sources.CAP[s],
                                     max_clip_frac=s0.MAX_CLIP_FRAC, max_retries=s0.MAX_PCFM_RETRIES)
            t += timed(counts)
        pcfm_total[threads] = t / n
        log.info("PointCountFM + re-roll, 3 species, CPU %d threads: %.3g s per shower", threads, t / n)
    for s in SPECIES_NAMES:
        res["params"][f"PointCountFM {s}"] = int(sum(p.numel() for p in torch.jit.load(
            os.path.join(sources.STAGED, s, "pcfm_compiled.pt"), map_location="cpu").parameters()))
    torch.set_num_threads(args.threads)
    t = 0.0
    for s in SPECIES_NAMES:
        sources.weights(s)
        gen = Generator(run_dir=os.path.join(sources.STAGED, s), num_timesteps=s0.NUM_TIMESTEPS, compile=True,
                        solver=s0.SOLVER)
        gen.max_points = sources.CAP[s]
        res["params"][f"AllShowers {s}"] = module_params(gen)
        pcfm = os.path.join(sources.STAGED, s, "pcfm_compiled.pt")
        dt = timed(lambda: s0._gen_chunk(gen, pcfm, {"max_points": sources.CAP[s]}, energies, directions, labels,
                                         ids, sources.CAP[s]))
        log.info("full chain %s: %.3g s per shower", s, dt / n)
        t += dt
        del gen
        torch.cuda.empty_cache()
    res["seconds_per_shower"]["PointCountFM + re-roll, 3 species, CPU 1 thread"] = pcfm_total[1]
    res["seconds_per_shower"][f"PointCountFM + re-roll, 3 species, CPU {args.threads} threads"] = pcfm_total[args.threads]
    res["seconds_per_shower"][f"full chain, 3 species, CPU {args.threads} threads + GPU"] = t / n
    res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    log.info("params %s", res["params"])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER, "run"), args)
    logs.save(out, stage_run(out, args, log), args, log)


if __name__ == "__main__":
    main()
