"""Census: where the run-12 corpus asks the generator to work (energy, direction, class), against its CORSIKA training set."""
import argparse
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import h5py
import numpy as np
import torch

from analysis_core import logs, plots
from data_lane import sources
from modules.constants import HELDOUT_SHOWER_CACHE_PATH, HOLDOUT_FRAC, HOLDOUT_SEED, LOG_E_MIN, SPECIES_NAMES
from modules.showers import load_tau_primaries
from modules.surrogates.fnn import encode_primary

LAB = sources.LAB
TAU_FILE = f"{LAB}/jagarwal/tau_wholesky_n750k.h5"
PRIMARY = "test_v6_run_01_northeast/primary.pt"
BAND_CANDIDATES = (7.0, 8.0)          # LOG_E_MAX values the corpus could have been filtered with
SEED_CANDIDATES = range(4)            # EM/hadronic label seeds tried against the stored held-out classes
EM_PDG = (11, -11, 111)
EDGES = {"log10_E": np.arange(5.0, 8.01, 0.25), "zenith": np.arange(0.0, 180.1, 5.0), "azimuth": np.arange(0.0, 360.1, 30.0)}
THIN = (1, 2, 5, 10, 20, 50, 100)


def angles(directions):
    """Zenith and azimuth [deg] of (N, 3) direction vectors, as the training files define them (Up = column 2, azimuth from column 0)."""
    d = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(d[:, 2], -1, 1))), np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360


def cells(log_e, zen, az):
    """(N, 3) bin index per event; -1 where outside the binned range."""
    idx = []
    for v, e in zip((log_e, zen, az), EDGES.values()):
        i = np.digitize(v, e) - 1
        idx.append(np.where((i >= 0) & (i < len(e) - 1), i, -1))
    return np.stack(idx, 1)


def cell_counts(train_cells):
    shape = tuple(len(e) - 1 for e in EDGES.values())
    ok = (train_cells >= 0).all(1)
    return np.bincount(np.ravel_multi_index(train_cells[ok].T, shape), minlength=np.prod(shape)).reshape(shape)


def lookup(counts, c):
    """Training count in each event's cell; 0 outside the binned range."""
    ok = (c >= 0).all(1)
    out = np.zeros(len(c), np.int64)
    out[ok] = counts[tuple(c[ok].T)]
    return out


def support(train, events):
    """Out-of-range flags and the training count in each event's cell. `train`, `events`: dicts of log10_E, zenith, azimuth."""
    lo = {k: float(train[k].min()) for k in ("log10_E", "zenith")}
    hi = {k: float(train[k].max()) for k in ("log10_E", "zenith")}
    out = {k: (events[k] < lo[k]) | (events[k] > hi[k]) for k in ("log10_E", "zenith")}
    counts = cell_counts(cells(train["log10_E"], train["zenith"], train["azimuth"]))
    n = lookup(counts, cells(events["log10_E"], events["zenith"], events["azimuth"]))
    return {"range": {"lo": lo, "hi": hi}, "out_energy": out["log10_E"], "out_zenith": out["zenith"],
            "out_any": out["log10_E"] | out["zenith"], "cell_count": n}


def thin_curve(n):
    return {str(t): float((n < t).mean()) for t in THIN}


def training(split):
    """Primaries of one CORSIKA split (rows matched across species), with the EM/hadronic class from actual_pdg."""
    meta = None
    for s in SPECIES_NAMES:
        with h5py.File(sources.corsika_path(s, split), "r") as f:
            m = {k: f[k][...] for k in ("shower_ids", "energies", "directions", "pdg", "actual_pdg")}
        if meta is None:
            meta = m
        elif not all(np.array_equal(meta[k], m[k]) for k in m):
            raise ValueError(f"{split}: {s} rows differ from {SPECIES_NAMES[0]}")
    zen, az = angles(meta["directions"].astype(np.float64))
    em = np.isin(meta["actual_pdg"], EM_PDG)
    return {"log10_E": np.log10(meta["energies"][:, 0].astype(np.float64)), "zenith": zen, "azimuth": az,
            "label": meta["pdg"].astype(np.int64), "em": em, "actual_pdg": meta["actual_pdg"]}


def heldout_cache():
    """Metadata of the stored run-12 held-out corpus, first species block (the blocks share primaries)."""
    with h5py.File(HELDOUT_SHOWER_CACHE_PATH, "r") as f:
        m = {k: f[k][...] for k in ("shower_ids", "energies", "directions", "pdg")}
    n = len(m["pdg"]) // len(SPECIES_NAMES)
    for k in range(1, len(SPECIES_NAMES)):
        for key in m:
            if not np.array_equal(m[key][:n], m[key][k * n:(k + 1) * n]):
                raise ValueError(f"held-out cache: species block {k} {key} differs from block 0")
    return {k: v[:n] for k, v in m.items()}


def rebuild_corpus(stored, log):
    """The corpus Step 0 drew, rebuilt with the pipeline's own loader and split; must equal the stored held-out rows exactly."""
    n_ho = len(stored["pdg"])
    for e_max in BAND_CANDIDATES:
        prim = load_tau_primaries(TAU_FILE, e_min=10 ** LOG_E_MIN, e_max=10 ** e_max, n=0, seed=0)
        n_total = len(prim["energies"])
        if max(1, int(round(HOLDOUT_FRAC * n_total))) == n_ho:
            break
        log.info("band log10 E <= %.1f gives %d events, held-out would be %d, stored %d", e_max, n_total,
                 round(HOLDOUT_FRAC * n_total), n_ho)
    else:
        raise ValueError(f"no energy band reproduces the {n_ho} stored held-out events")
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(HOLDOUT_SEED))
    ho = perm[:n_ho].sort().values.numpy()
    tr = perm[n_ho:].sort().values.numpy()
    for k, ours in (("shower_ids", ho), ("energies", prim["energies"][ho].numpy()),
                    ("directions", prim["directions"][ho].numpy())):
        if not np.array_equal(stored[k], ours.astype(stored[k].dtype)):
            raise ValueError(f"rebuilt held-out {k} differ from the stored cache")
    for seed in SEED_CANDIDATES:
        labels = load_tau_primaries(TAU_FILE, e_min=10 ** LOG_E_MIN, e_max=10 ** e_max, n=0, seed=seed)["labels"].numpy()
        if np.array_equal(labels[ho], stored["pdg"]):
            break
    else:
        raise ValueError("no label seed reproduces the stored held-out classes")
    with h5py.File(TAU_FILE, "r") as f:
        energy, oneweight = f["energy"][...], f["oneweight"][...]
    tau_row = np.nonzero((energy >= 10 ** LOG_E_MIN) & (energy <= 10 ** e_max))[0]
    log.info("corpus: band log10 E %.1f to %.1f, %d of %d taus in band (all kept), %d held out, label seed %d; "
             "held-out ids, energies, directions and classes equal the stored cache", LOG_E_MIN, e_max, n_total,
             len(energy), n_ho, seed)
    return prim, labels, tau_row, oneweight[tau_row], tr, ho, {"log10_E_max": e_max, "label_seed": seed,
                                                                "n_in_band": n_total, "n_tau_file": len(energy)}


def match_rows(ours, theirs):
    """Index into `ours` of each row of `theirs` (byte-equal float32 rows), -1 where none; `ours` rows must be unique."""
    key = lambda a: np.ascontiguousarray(a.astype(np.float32)).view(f"V{a.shape[1] * 4}").ravel()
    _, inv = np.unique(np.concatenate([key(ours), key(theirs)]), return_inverse=True)
    inv = inv.ravel()
    lut = np.full(inv.max() + 1, -1, np.int64)
    lut[inv[:len(ours)]] = np.arange(len(ours))
    if len(np.unique(inv[:len(ours)])) != len(ours):
        raise ValueError("rebuilt events are not unique in the matching columns")
    return lut[inv[len(ours):]]


def check_primary(path, prim, labels, tr, e_max, log):
    """Every primary.pt row must be one rebuilt training event (direction, class, normalised energy); returns the array centre."""
    p = torch.load(path).numpy()
    ours = encode_primary(prim["directions"][tr], prim["energies"][tr], torch.as_tensor(labels[tr]),
                          prim["positions"][tr], torch.zeros(3)).numpy()     # cols 5-7: absolute vertex
    idx = match_rows(ours[:, [0, 1, 2, 4]], p[:, [0, 1, 2, 4]])
    if (idx < 0).any():
        raise ValueError(f"{(idx < 0).sum()} of {len(p)} primary.pt rows match no rebuilt training event")
    n_seen = len(np.unique(idx))
    log_e = np.log10(prim["energies"][tr, 0].numpy().astype(np.float64))[idx]
    norm_err = {m: float(np.abs(p[:, 3] - (log_e - LOG_E_MIN) / (m - LOG_E_MIN)).max()) for m in BAND_CANDIDATES}
    centre = ours[idx, 5:] - p[:, 5:]
    spread = float(np.abs(centre - np.median(centre, 0)).max())
    if n_seen != len(tr) or spread > 1.0 or norm_err[e_max] > 1e-5:
        raise ValueError(f"primary.pt covers {n_seen} of {len(tr)} training events; centre spread {spread:.3g} m; "
                         f"energy normalisation error by LOG_E_MAX {norm_err}")
    log.info("primary.pt: %d rows = %d training events x %.3g rows each; all match in direction and class; energy column "
             "normalised with LOG_E_MAX %.1f (max error %s); array centre %s (max spread %.2g m)", len(p), n_seen,
             len(p) / n_seen, e_max, norm_err, np.round(np.median(centre, 0), 1).tolist(), spread)
    return np.median(centre, 0)


def oneweight_summary(w, ev):
    by = {}
    for k in ("log10_E", "zenith"):
        e = EDGES[k]
        i = np.digitize(ev[k], e) - 1
        by[k] = {f"{e[b]:g}": [int((i == b).sum()), float(np.median(w[i == b]))] for b in range(len(e) - 1) if (i == b).any()}
    return {"quantiles_1_16_50_84_99": np.percentile(w, [1, 16, 50, 84, 99]).tolist(),
            "effective_fraction": float(w.sum() ** 2 / (len(w) * (w ** 2).sum())),
            "count_and_median_by_bin": by}


def figure(train, ev, sup_class):
    fig, axes = plots.plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (x, ex) in zip(axes[:2], (("log10_E", EDGES["log10_E"]), ("azimuth", EDGES["azimuth"]))):
        h, _, _ = np.histogram2d(ev[x], ev["zenith"], bins=(ex, EDGES["zenith"]))
        ax.pcolormesh(ex, EDGES["zenith"], np.log10(np.where(h > 0, h, np.nan)).T, cmap="Greys")
        lo, hi = np.percentile(train[x], [1, 99])
        zl, zh = train["zenith"].min(), train["zenith"].max()
        ax.plot([lo, hi, hi, lo, lo], [zl, zl, zh, zh, zl], color="C3", lw=1.5, label="training 1st to 99th percentile")
        ax.set(xlabel={"log10_E": "log10 E [GeV]", "azimuth": "azimuth [deg]"}[x], ylabel="zenith [deg]",
               ylim=(max(0, ev["zenith"].min() - 5), min(180, ev["zenith"].max() + 5)),
               title=f"Corpus events (log10 count) in {x.replace('log10_E', 'energy')} and zenith")
        ax.legend(fontsize=8, loc="upper right")
    t = np.array(THIN)
    for name, n, style in (("all events, both classes' training", sup_class["all"], "k-"),
                           ("EM events, EM training", sup_class["em"], "C0--"),
                           ("hadronic events, hadronic training", sup_class["hadronic"], "C1--")):
        axes[2].plot(t, [(n < x).mean() for x in t], style, marker="o", label=name)
    axes[2].set(xscale="log", xlabel="N training showers in the cell", ylabel="fraction of corpus events",
                title="Corpus events in cells with fewer than N training showers", ylim=(0, None))
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out, log = logs.start(os.path.join("data_lane", "census"), args)

    train, test = training("train"), training("test")
    labels_seen = {int(l): sorted(set(np.abs(train["actual_pdg"][train["label"] == l]).tolist())) for l in np.unique(train["label"])}
    log.info("training: %d showers, class label -> |actual pdg| %s, EM fraction %.3f; test %d, EM fraction %.3f",
             len(train["label"]), labels_seen, train["em"].mean(), len(test["label"]), test["em"].mean())
    em_label = [l for l, p in labels_seen.items() if set(p) <= {11, 111}]
    if len(em_label) != 1:
        raise ValueError(f"training class labels do not split into EM and hadronic: {labels_seen}")
    em_label = em_label[0]

    stored = heldout_cache()
    prim, labels, tau_row, oneweight, tr, ho, corpus_info = rebuild_corpus(stored, log)
    centre = check_primary(os.path.join(args.run_dir, PRIMARY), prim, labels, tr, corpus_info["log10_E_max"], log)

    zen, az = angles(prim["directions"].numpy().astype(np.float64))
    ev = {"log10_E": np.log10(prim["energies"][:, 0].numpy().astype(np.float64)), "zenith": zen, "azimuth": az}
    split = np.zeros(len(zen), np.int64)
    split[ho] = 1
    em = labels == em_label
    sup = support(train, ev)
    sup_test = support(train, test)
    n_class = np.where(em, support({k: v[train["em"]] for k, v in train.items()}, ev)["cell_count"],
                       support({k: v[~train["em"]] for k, v in train.items()}, ev)["cell_count"])
    rel = prim["positions"].numpy() - centre
    dist = np.linalg.norm(rel, axis=1)

    res = {"corpus": {**corpus_info, "n_train": len(tr), "n_heldout": len(ho), "array_centre_ENU": centre.tolist()},
           "training_range": sup["range"],
           "class": {"training_em_label": em_label, "training_em_fraction": float(train["em"].mean()),
                     "test_em_fraction": float(test["em"].mean()), "corpus_em_fraction": float(em.mean())},
           "out_of_range": {}, "thin_cells": {}, "thin_cells_own_class": {},
           "binning": {k: [float(e[0]), float(e[1] - e[0]), float(e[-1])] for k, e in EDGES.items()},
           "test_set_out_of_range": float(sup_test["out_any"].mean()), "test_set_thin_cells": thin_curve(sup_test["cell_count"]),
           "oneweight": oneweight_summary(oneweight, ev),
           "vertex_distance_m_quantiles_1_16_50_84_99": np.percentile(dist, [1, 16, 50, 84, 99]).tolist(),
           "corpus_quantiles_1_16_50_84_99": {k: np.percentile(v, [1, 16, 50, 84, 99]).tolist() for k, v in ev.items()},
           "training_log10_E_quantiles_50_90_99_99.9": np.percentile(train["log10_E"], [50, 90, 99, 99.9]).tolist(),
           "per_energy_bin": {f"{e:g}": [int(a), int(b)] for e, a, b in zip(
               EDGES["log10_E"], np.histogram(train["log10_E"], EDGES["log10_E"])[0], np.histogram(ev["log10_E"], EDGES["log10_E"])[0])}}
    for name, m in (("all", np.ones_like(em)), ("train_rows", split == 0), ("heldout_rows", split == 1),
                    ("em", em), ("hadronic", ~em)):
        res["out_of_range"][name] = {k: float(sup[k][m].mean()) for k in ("out_energy", "out_zenith", "out_any")}
        res["thin_cells"][name] = thin_curve(sup["cell_count"][m])
        res["thin_cells_own_class"][name] = thin_curve(n_class[m])
    for k in ("out_of_range", "thin_cells", "thin_cells_own_class"):
        log.info("%s: %s", k, res[k])
    log.info("class: %s", res["class"])
    log.info("training log10 E quantiles 50/90/99/99.9: %s; per energy bin [training, corpus]: %s",
             np.round(res["training_log10_E_quantiles_50_90_99_99.9"], 3).tolist(), res["per_energy_bin"])
    log.info("test set against training: out of range %.4f, thin %s", res["test_set_out_of_range"], res["test_set_thin_cells"])
    log.info("oneweight: quantiles %s, effective fraction %.3f", res["oneweight"]["quantiles_1_16_50_84_99"],
             res["oneweight"]["effective_fraction"])
    log.info("vertex distance from array centre [m] quantiles: %s", res["vertex_distance_m_quantiles_1_16_50_84_99"])

    examples = {"held-out event 0": int(ho[0]), "largest zenith": int(np.argmax(zen)), "highest energy": int(np.argmax(ev["log10_E"]))}
    for name, i in examples.items():
        c = cells(ev["log10_E"][i:i + 1], zen[i:i + 1], az[i:i + 1])[0]
        log.info("example %s: corpus row %d (%s), tau file row %d, log10 E %.4f, direction %s, zenith %.2f, azimuth %.2f, "
                 "class %d (%s), oneweight %.4g, cell %s, training showers in cell %d (own class %d), out of range %s, "
                 "vertex rel. centre %s m", name, i, ["train", "held-out"][split[i]], tau_row[i], ev["log10_E"][i],
                 np.round(prim["directions"][i].numpy(), 4).tolist(), zen[i], az[i], labels[i], ["hadronic", "EM"][int(em[i])],
                 oneweight[i], c.tolist(), sup["cell_count"][i], n_class[i], bool(sup["out_any"][i]), np.round(rel[i], 1).tolist())
    res["examples"] = examples

    cols = {"tau_row": tau_row, "heldout": split, "log10_E": ev["log10_E"], "zenith_deg": zen, "azimuth_deg": az,
            "class": labels, "em": em.astype(np.int64), "oneweight": oneweight, "out_energy": sup["out_energy"].astype(np.int64),
            "out_zenith": sup["out_zenith"].astype(np.int64), "cell_count": sup["cell_count"], "cell_count_own_class": n_class,
            "rel_E": rel[:, 0], "rel_N": rel[:, 1], "rel_U": rel[:, 2]}
    np.savez(os.path.join(out, "corpus_events.npz"), **cols)
    np.savez(os.path.join(out, "cell_counts.npz"), train=cell_counts(cells(train["log10_E"], train["zenith"], train["azimuth"])),
             corpus=cell_counts(cells(ev["log10_E"], zen, az)), **{f"edges_{k}": e for k, e in EDGES.items()})
    path = os.path.join(out, "corpus_events.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in zip(*cols.values()):
            w.writerow([f"{v:.6g}" if isinstance(v, (float, np.floating)) else int(v) for v in row])
    log.info("wrote %s (%d bytes)", path, os.path.getsize(path))
    plots.save(figure(train, ev, {"all": sup["cell_count"], "em": n_class[em], "hadronic": n_class[~em]}), out, "census", log)
    logs.save(out, res, args, log)


if __name__ == "__main__":
    main()
