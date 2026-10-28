"""Interpretability of the per-plane energy sampler: parameter count, the laws of every profile number in one table,
and the physics scalings (elongation rate d Xmax / d log10 E, peak size d log10 E_max / d log10 E) from sampler draws
against the same slopes on CORSIKA's fitted test showers.

Rule (study_plan.md 1f): PASS per species and class if the sampler's slope is within 2 se of CORSIKA's.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import logs
from modules.constants import SPECIES_NAMES
from symbolic import fit_profiles, longitudinal, planes, sampler

FOLDER = "laws"
N_DRAWS = 8
NUMBERS = ("log10 Emax/E", "Xmax", "ln L", "R", "start depth")
XMAX, PEAK = 1, 0                        # columns within a species block
GRID_L = (5.25, 6.0)                     # log10 E where the median laws are tabulated


def slope(y, l, u):
    """Coefficient of log10 E (with its se) in y ~ 1 + log10 E + cos zenith."""
    A = np.column_stack([np.ones_like(l), l, u])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    cov = np.linalg.inv(A.T @ A) * (r @ r) / (len(y) - A.shape[1])
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def n_params(model):
    """Fitted numbers: margin coefficients, copula correlations, per-plane bias and sd, AR(1) rho."""
    margins = sum(m.W.numel() for m in model["margins"])
    k = len(model["C"])
    pl = sum(len(p["bias"]) + len(p["sd"]) + 1 for p in model["planes"].values())
    return {"margins": int(margins), "copula": k * (k - 1) // 2, "planes": int(pl),
            "total": int(margins + k * (k - 1) // 2 + pl)}


def law_table(model, u_mid):
    """Median and 16-84% width of every number at GRID_L and the central cos zenith."""
    D = model["design"](np.array(GRID_L), np.full(len(GRID_L), u_mid))
    rows = {}
    for i, s in enumerate(SPECIES_NAMES):
        for j, nm in enumerate(NUMBERS):
            m = model["margins"][sampler.BLOCK * i + j]
            med = m.ppf(D, np.full(len(GRID_L), 0.5))
            w = m.ppf(D, np.full(len(GRID_L), 0.841)) - m.ppf(D, np.full(len(GRID_L), 0.159))
            rows[f"{s} {nm}"] = {f"median at {l}": float(a) for l, a in zip(GRID_L, med)}
            rows[f"{s} {nm}"].update({f"width at {l}": float(a) for l, a in zip(GRID_L, w)})
            rows[f"{s} {nm}"]["median slope per decade"] = float((med[1] - med[0]) / (GRID_L[1] - GRID_L[0]))
    return rows


def stage_run(out, args, log):
    tr = sampler.load(os.path.join(args.out, fit_profiles.TRAIN))
    te = sampler.load(os.path.join(args.out, fit_profiles.TEST))
    rng = np.random.default_rng(args.seed)
    res = {"rule": __doc__.split("Rule")[1].strip(), "classes": {}}
    for c, cn in planes.CLASS.items():
        model = sampler.fit(tr, c)
        t = sampler.class_rows(te, c)
        X = te["X"][t["rows"]]
        lat = np.concatenate([sampler.sample(model, t["l"], t["u"], X, rng)[1] for _ in range(N_DRAWS)])
        l8, u8 = np.tile(t["l"], N_DRAWS), np.tile(t["u"], N_DRAWS)
        R = res["classes"][cn] = {"n_test": len(t["rows"]), "params": n_params(model),
                                  "laws": law_table(model, float(np.median(t["u"]))), "scaling": {}}
        for i, s in enumerate(SPECIES_NAMES):
            for key, col, add in (("elongation rate [g/cm2 per decade]", XMAX, 0.0),
                                  ("d log10 Emax / d log10 E", PEAK, 1.0)):
                b = sampler.BLOCK * i + col
                cs, cse = slope(t["X"][:, b], t["l"], t["u"])
                ss, sse = slope(lat[:, b], l8, u8)
                z = (ss - cs) / np.hypot(sse, cse)
                R["scaling"][f"{s} {key}"] = {"CORSIKA": [cs + add, cse], "sampler": [ss + add, sse], "z": float(z),
                                              "verdict": "PASS" if abs(z) <= 2 else "FAIL"}
                log.info("%s %s %s: CORSIKA %.3f +- %.3f, sampler %.3f +- %.3f, z %.2f", cn, s, key, cs + add, cse,
                         ss + add, sse, z)
        log.info("%s: %d fitted numbers %s", cn, R["params"]["total"], R["params"])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER, "run"), args)
    logs.save(out, stage_run(out, args, log), args, log)


if __name__ == "__main__":
    main()
