"""Is the degenerate-shower ("blob") bug ours, or upstream's?

The AllShowers generator lives in a sibling repo, /n/home05/zdimitrov/tambo/
TAMBO-opt, which is a fork of hamzahanif2210/TAMBO-opt with local commits on
top. A pristine upstream clone sits at TAMBO-opt-Hamza. This compares them.

Reading the diff first: only `allshowers/generator.py` and
`allshowers/transformer.py` differ on the generation path -- `preprocessing.py`
(the exp inverse energy transform), `flow_matching.py`, `ode_solvers.py` and
`util/allshowers_related/generate_showers.py` are byte-identical. So this script
only has to settle whether those two files change the numbers.

They cannot be imported into one interpreter: both repos ship a package named
`allshowers`. Hence the mode-per-process design -- run `gen` twice, then
`compare` the two saved files.

    python tests/compare_upstream_generator.py mask-check
    python tests/compare_upstream_generator.py prep --species muon --n 32 --out prep.pt
    python tests/compare_upstream_generator.py gen  --impl local      --prep prep.pt --out a.pt
    python tests/compare_upstream_generator.py gen  --impl hamza      --prep prep.pt --out b.pt
    python tests/compare_upstream_generator.py compare a.pt b.pt

Nothing in modules/, scripts/, or either sibling repo is modified by any mode.
"""
import argparse
import glob
import hashlib
import importlib.util
import os
import shutil
import sys
import time
import warnings

import numpy as np
import torch
import yaml

# Same two flags modules/showers/generate.py sets, duplicated because the
# `hamza` mode must not import that file (it injects the LOCAL repo onto
# sys.path). Without them, inductor cannot lower flex_attention's mask subgraph
# under dynamic shapes on torch 2.9.1+cu128.
import torch._dynamo
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.assume_static_by_default = True

# Step 0 sets this before generating, and it changes float32 matmul numerics --
# so it must be set here too, identically for both impls, or it would be a
# confound rather than a controlled variable.
torch.set_float32_matmul_precision("high")

LOCAL_ROOT = "/n/home05/zdimitrov/tambo/TAMBO-opt"
HAMZA_ROOT = "/n/home05/zdimitrov/tambo/TAMBO-opt-Hamza"

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(_HERE)
STEP0 = os.path.join(REPO, "scripts", "00_generate_data_dual_species.py")

BEST = ("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/"
        "detector_optimization_v6/checkpoints")
RESULTS = ("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/"
           "detector_optimization_v6/tests/upstream_comparison")

# Mirrors SPECIES in scripts/00_generate_data_dual_species.py. Duplicated rather
# than imported because importing that module pulls in `allshowers` from the
# LOCAL repo, which the `hamza` mode must avoid.
SPECIES = {
    "electron": dict(run="20260519_185649_Electron-Allshower",
                     pcfm="20260521_040716_Electron-PointCountFM", max_points=4096),
    "muon":     dict(run="20260520_160031_Muons-Allshower",
                     pcfm="20260521_043912_Muon-PointCountFM",     max_points=25088),
    "photon":   dict(run="20260724_074020_Photon-Allshower",
                     pcfm="20260727_041023_Photon-PointCountFM",   max_points=8064),
}

N_PLANES = 24
# Median point energy: ~3.3 for a healthy shower, 1e8-4e9 for a degenerate one
# (tests/README.md). Eight orders of clean air, so the threshold is not delicate.
BLOB_MEDIAN_E = 1e3


# ── staging ──────────────────────────────────────────────────────────────────

def stage(species, dst, pre_ln):
    """A Generator-loadable run-dir, mirroring stage_run_dir() in Step 0.

    Reimplemented here for one reason: `pre_ln`. Step 0 hardcodes it True, but
    upstream's Transformer has no such parameter and `Transformer(**params)`
    would raise TypeError on the key -- so the upstream run-dir needs a conf
    WITHOUT it (pass pre_ln=None). Upstream is hardcoded pre-LN anyway, so the
    two staged confs request the same behaviour by different means.
    """
    cfg = SPECIES[species]
    src = os.path.join(BEST, cfg["run"])
    os.makedirs(os.path.join(dst, "weights"), exist_ok=True)
    os.makedirs(os.path.join(dst, "preprocessing"), exist_ok=True)

    with open(os.path.join(src, "conf.yaml")) as f:
        conf = yaml.safe_load(f)
    if pre_ln is None:
        conf["model"].pop("pre_ln", None)
    else:
        conf["model"]["pre_ln"] = bool(pre_ln)
    with open(os.path.join(dst, "conf.yaml"), "w") as f:
        yaml.safe_dump(conf, f, sort_keys=False)

    weights = os.path.join(dst, "weights", "best.pt")
    trafos  = os.path.join(dst, "preprocessing", "trafos.pt")
    if not os.path.exists(trafos):
        shutil.copy2(os.path.join(src, "preprocessing", "trafos.pt"), trafos)
    if not os.path.exists(weights):
        ck = sorted(glob.glob(os.path.join(src, "checkpoints", "best_epoch_*.pt")))
        if not ck:
            raise FileNotFoundError(f"no checkpoints/best_epoch_*.pt in {src}")
        raw = torch.load(ck[-1], map_location="cpu", weights_only=False)
        sd = _extract_sd(raw)
        torch.save(sd, weights)
    print(f"[stage] {species} pre_ln={pre_ln} -> {dst}")
    return dst


_WRAP_KEYS = ("model", "model_state_dict", "state_dict", "ema", "ema_model",
              "flow", "net", "network", "weights")


def _extract_sd(obj):
    """Flat {name: tensor} from a checkpoint, raw or wrapped. Same probe ORDER as
    Step 0's _extract_state_dict -- raw first, then the wrapper keys -- so the
    weights staged here are the tensors Step 0 would have staged."""
    def _flat(d):
        return isinstance(d, dict) and d and all(torch.is_tensor(x) for x in d.values())
    if _flat(obj):
        return obj
    if isinstance(obj, dict):
        for k in _WRAP_KEYS:
            if _flat(obj.get(k)):
                return obj[k]
        for k, v in obj.items():
            if _flat(v):
                return v
        raise RuntimeError(f"no state_dict in checkpoint; keys = {list(obj.keys())}")
    raise RuntimeError(f"unexpected checkpoint object type: {type(obj)}")


def _step0():
    """Import Step 0 by path (its filename starts with a digit). LOCAL-repo only
    -- it imports `allshowers`, so never call this from the `hamza` mode."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    sys.path.insert(0, REPO)
    spec = importlib.util.spec_from_file_location("step0", STEP0)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── the time-aware forward (the shim) ────────────────────────────────────────

def time_forward(gen, energies, num_points, angles, label=None):
    """Upstream's Generator.forward, extended to sample 4 features and return 5
    columns -- the mechanical port of the local generator.py change and nothing
    else. Byte-for-byte the with_time branch of TAMBO-opt/allshowers/generator.py.

    Applied to BOTH impls in `gen` mode so the forward is not itself a variable:
    what differs between runs is only the Generator/Transformer underneath.
    """
    if gen.expects_angles:
        condition = torch.concatenate(
            [gen.cond_trafo(energies * gen.resize_factor), angles], dim=-1)
    else:
        condition = gen.cond_trafo(energies)
    layer = torch.zeros((condition.shape[0], gen.max_points, 1), dtype=torch.int32)
    mask  = torch.zeros((condition.shape[0], gen.max_points, 1), dtype=torch.bool)
    for i in range(condition.shape[0]):
        total_points = torch.sum(num_points[i])
        layer_i = torch.repeat_interleave(num_points[i])
        if total_points > gen.max_points:
            warnings.warn(f"num points {total_points} exceeds max points "
                          f"{gen.max_points}, truncating")
            total_points = gen.max_points
            layer_i = layer_i[: gen.max_points]
        layer[i, :total_points, 0] = layer_i
        mask[i, :total_points, 0] = True
    layer = layer.to(condition.device)
    mask = mask.to(condition.device)
    raw = gen.flow.sample(
        shape=(condition.shape[0], gen.max_points, 4),
        num_timesteps=gen.num_timesteps, cond=condition,
        num_points=num_points, layer=layer, mask=mask, label=label,
    )
    out = torch.zeros((condition.shape[0], gen.max_points, 5), device=raw.device)
    out[:, :, :2] = gen.samples_coordinate_trafo.inverse(raw[:, :, :2])
    out[:, :, 2]  = layer.squeeze(2)
    out[:, :, 3]  = gen.samples_energy_trafo.inverse(raw[:, :, 2])
    out[:, :, 4]  = gen.samples_time_trafo.inverse(raw[:, :, 3])
    out[~mask.repeat(1, 1, 5)] = 0
    return out


def attach_time_trafo(gen, run_dir, device):
    """Give an upstream Generator the samples_time_trafo it never composes.
    No-op when the Generator already built one (the local repo does)."""
    if getattr(gen, "samples_time_trafo", None) is not None:
        return
    from allshowers.preprocessing import compose
    with open(os.path.join(run_dir, "conf.yaml")) as f:
        data_params = yaml.safe_load(f)["data"]
    trafo = compose(data_params.get("samples_time_trafo"))
    state = torch.load(os.path.join(run_dir, "preprocessing", "trafos.pt"),
                       map_location="cpu", weights_only=True)
    trafo.load_state_dict(state["samples_time_trafo"])
    gen.samples_time_trafo = trafo.to(device)


# ── statistics ───────────────────────────────────────────────────────────────

def shower_stats(pts, e_prim):
    """Per-shower diagnostics, same conventions as tests/repro_blob_showers.py
    and plots/showers/inspect_rows.py::row_stats."""
    real = pts[pts[:, 3] > 0]
    if not len(real):
        return dict(n=0, median_e=0.0, total=0.0, tot_over_prim=0.0,
                    max_plane_over_prim=0.0, top1=0.0, top100=0.0,
                    x_span=0.0, y_span=0.0)
    e = real[:, 3].astype(np.float64)
    lay = np.clip(np.rint(real[:, 2]).astype(int), 0, N_PLANES - 1)
    plane_e = np.bincount(lay, weights=e, minlength=N_PLANES)
    srt = np.sort(e)[::-1]
    return dict(n=int(len(real)), median_e=float(np.median(e)), total=float(e.sum()),
                tot_over_prim=float(e.sum() / max(e_prim, 1e-30)),
                max_plane_over_prim=float(plane_e.max() / max(e_prim, 1e-30)),
                top1=float(srt[0] / max(e.sum(), 1e-30)),
                top100=float(srt[:100].sum() / max(e.sum(), 1e-30)),
                x_span=float(real[:, 0].max() - real[:, 0].min()),
                y_span=float(real[:, 1].max() - real[:, 1].min()))


def table(samples, energies):
    s = np.asarray(samples, dtype=np.float32)
    ep = np.asarray(energies, dtype=np.float64).reshape(-1)
    return [shower_stats(s[i], ep[i]) for i in range(len(s))]


def rate_line(rows, label):
    med = np.array([r["median_e"] for r in rows])
    bad = med > BLOB_MEDIAN_E
    return (f"{label:<24} n={len(rows):>5}  blobs={int(bad.sum()):>4} "
            f"({100*bad.mean():6.2f}%)  median(median_e)={np.median(med):.4g}  "
            f"max(median_e)={med.max():.4g}")


# ── modes ────────────────────────────────────────────────────────────────────

def cmd_mask_check(args):
    """Do the two compute_mask implementations agree at the num_layer_cond these
    checkpoints actually use? Both transformer.py files are import-standalone, so
    unlike the generators they CAN be loaded side by side in one process."""
    def load(root, name):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(root, "allshowers", "transformer.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        # Return the raw closure instead of a block-sparse BlockMask: block masks
        # are coarser than the predicate, so comparing them could hide a
        # per-element difference.
        m.create_block_mask = lambda mask_mod, **kw: mask_mod
        return m

    loc, ham = load(LOCAL_ROOT, "t_local"), load(HAMZA_ROOT, "t_hamza")
    g = torch.Generator().manual_seed(0)
    B, L = 4, 96
    npad = torch.randint(L // 2, L, (B,), generator=g)
    padding = torch.zeros(B, L, 1, dtype=torch.bool)
    for b in range(B):
        padding[b, :npad[b], 0] = True
    layer = torch.randint(0, N_PLANES, (B, L, 1), generator=g, dtype=torch.int32)

    b_i = torch.arange(B).view(B, 1, 1)
    q_i = torch.arange(L).view(1, L, 1)
    k_i = torch.arange(L).view(1, 1, L)

    print(f"{'num_layer_cond':>16}  {'equal':>7}  {'differing elements':>20}   note")
    ok = True
    for nlc in (4, 8, -1):
        a = loc.compute_mask(padding.clone(), layer.clone(), nlc)(b_i, None, q_i, k_i)
        b = ham.compute_mask(padding.clone(), layer.clone(), nlc)(b_i, None, q_i, k_i)
        a, b = torch.broadcast_tensors(a, b)
        same = bool(torch.equal(a, b))
        note = {4: "electron/muon", 8: "photon", -1: "unused by these checkpoints"}[nlc]
        print(f"{nlc:>16}  {str(same):>7}  {int((a != b).sum()):>20}   {note}")
        if nlc >= 0:
            ok &= same
    print("\nRESULT:", "the mask difference is inert for all three checkpoints"
          if ok else "*** the masks DIFFER at a num_layer_cond in use ***")
    return 0 if ok else 1


def cmd_prep(args):
    """Primaries + PointCountFM + anti-clip, saved once so both impls generate
    from byte-identical inputs. Removes PCFM's own RNG as a confound."""
    step0 = _step0()
    from modules.constants import (LOG_E_MIN, LOG_E_MAX, TAU_WHOLESKY_PATH,
                                   ZENITH_MIN, ZENITH_MAX, AZIMUTH_MIN, AZIMUTH_MAX)
    from modules.showers import load_tau_primaries
    from allshowers.generate_showers import sample_primary_particles

    cfg = SPECIES[args.species]
    if args.primaries == "tau":
        prim = load_tau_primaries(TAU_WHOLESKY_PATH, e_min=10**LOG_E_MIN,
                                  e_max=10**LOG_E_MAX, n=args.n, seed=args.seed)
    else:
        prim = sample_primary_particles(
            e_min=10**LOG_E_MIN, e_max=10**LOG_E_MAX, zenith_min=ZENITH_MIN,
            zenith_max=ZENITH_MAX, azimuth_min=AZIMUTH_MIN, azimuth_max=AZIMUTH_MAX,
            n=args.n, seed=args.seed)
    energies   = prim["energies"][:args.n]
    directions = prim["directions"][:args.n]
    labels     = prim["labels"][:args.n].to(torch.int64)

    pcfm = os.path.join(BEST, cfg["pcfm"], "compiled.pt")
    from allshowers.generate_showers import run_point_count_fm
    torch.manual_seed(args.seed)
    num_points = run_point_count_fm(model_path=pcfm, energies=energies,
                                    directions=directions, labels=labels)
    if args.anti_clip:
        num_points = step0.resample_overclip(
            pcfm, energies, directions, labels, num_points,
            cap=int(cfg["max_points"]), max_clip_frac=step0.MAX_CLIP_FRAC,
            max_retries=step0.MAX_PCFM_RETRIES)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(dict(species=args.species, seed=args.seed, primaries=args.primaries,
                    anti_clip=args.anti_clip, energies=energies,
                    directions=directions, labels=labels, num_points=num_points),
               args.out)
    tot = num_points.sum(1)
    print(f"[prep] {args.species} n={len(energies)} primaries={args.primaries} "
          f"anti_clip={args.anti_clip}\n"
          f"       points/shower: min={int(tot.min())} median={int(tot.median())} "
          f"max={int(tot.max())}  cap={cfg['max_points']}  "
          f"at-or-over-cap={int((tot >= cfg['max_points']).sum())}\n"
          f"[prep] -> {args.out}")
    return 0


def cmd_gen(args):
    root = {"local": LOCAL_ROOT, "local-shim": LOCAL_ROOT, "hamza": HAMZA_ROOT}[args.impl]
    sys.path.insert(0, root)
    from allshowers.generator import Generator

    p = torch.load(args.prep, map_location="cpu", weights_only=False)
    species = p["species"]
    cfg = SPECIES[species]
    device = torch.device(args.device)

    # Upstream's Transformer has no pre_ln parameter, so its conf must not carry
    # the key; it is hardcoded pre-LN, which is what pre_ln=True asks the local
    # one for. Same behaviour, requested two ways.
    pre_ln = None if args.impl == "hamza" else args.pre_ln
    staged = stage(species, os.path.join(args.stage_root, f"{args.impl}_{species}"
                                         + ("" if pre_ln in (None, True) else "_postln")),
                   pre_ln)

    gen = Generator(run_dir=staged, num_timesteps=args.timesteps,
                    compile=not args.no_compile, solver=args.solver).to(device)
    gen.max_points = int(cfg["max_points"])
    attach_time_trafo(gen, staged, device)
    native = args.impl == "local" and getattr(gen, "with_time", False)
    print(f"[gen] impl={args.impl} species={species} pre_ln={pre_ln} "
          f"forward={'native with_time' if native else 'shim time_forward'} "
          f"max_points={gen.max_points}")

    n = len(p["energies"])
    batch = args.batch
    # Per-shower digest of the raw float32 bytes. Bit-identity is then checkable
    # without keeping the samples: at 12k muons the tensors are ~5 GB per run.
    outs, digests, rows, t0, i = [], [], [], time.time(), 0
    while i < n:
        j = min(i + batch, n)
        # Reseed per window from the window START index, so the noise a shower
        # sees depends only on its position -- not on how the run was batched.
        torch.manual_seed(args.seed + i)
        torch.cuda.manual_seed_all(args.seed + i)
        try:
            with torch.no_grad():
                out = time_forward(
                    gen, p["energies"][i:j].to(device), p["num_points"][i:j].to(device),
                    p["directions"][i:j].to(device), p["labels"][i:j].to(device),
                ).detach().float().cpu()
        except torch.OutOfMemoryError:
            if batch == 1:
                raise
            torch.cuda.empty_cache()
            batch = max(1, batch // 2)
            print(f"  [oom] retrying at batch {batch}", flush=True)
            continue
        outs.append(out if not args.stats_only else None)
        arr = out.numpy()
        digests.extend(hashlib.blake2b(arr[k].tobytes(), digest_size=16).hexdigest()
                       for k in range(len(arr)))
        rows.extend(table(arr, p["energies"][i:j].numpy()))
        i = j
        print(f"  [gen] {i}/{n}  {time.time()-t0:.0f}s", flush=True)
    samples = None if args.stats_only else torch.cat(outs)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(dict(impl=args.impl, species=species, pre_ln=pre_ln, seed=args.seed,
                    batch=args.batch, prep=os.path.abspath(args.prep),
                    samples=samples, digests=digests,
                    energies=p["energies"], stats=rows), args.out)
    print(rate_line(rows, f"{args.impl}/{species}"))
    shape = "stats+digests only" if samples is None else str(tuple(samples.shape))
    print(f"[gen] -> {args.out}  {shape}  {time.time()-t0:.0f}s")
    return 0


def cmd_compare(args):
    a = torch.load(args.a, map_location="cpu", weights_only=False)
    b = torch.load(args.b, map_location="cpu", weights_only=False)
    la = f"{a['impl']}/pre_ln={a['pre_ln']}"
    lb = f"{b['impl']}/pre_ln={b['pre_ln']}"
    print(f"A = {la:<28} {args.a}")
    print(f"B = {lb:<28} {args.b}")
    if a["species"] != b["species"]:
        print(f"!! different species ({a['species']} vs {b['species']})")
    if a["prep"] != b["prep"]:
        print(f"!! different prep files -- inputs are not the same primaries")

    sa, sb = a.get("samples"), b.get("samples")
    if sa is not None and sb is not None:
        print(f"\nshape {tuple(sa.shape)} vs {tuple(sb.shape)}")
        if sa.shape != sb.shape:
            print("SHAPES DIFFER -- not comparable elementwise")
            identical = False
        else:
            identical = bool(torch.equal(sa, sb))
            d = (sa - sb).abs()
            print(f"bit-identical : {identical}")
            print(f"max |A-B|     : {d.max().item():.6g}")
            print(f"max rel diff  : {(d / sa.abs().clamp(min=1e-30)).max().item():.6g}")
            print(f"elements diff : {int((sa != sb).sum())} / {sa.numel()}")
    else:
        # A --stats-only run keeps a blake2b digest of each shower's raw float32
        # bytes instead of the tensor. Equal digests == equal bytes, so this is
        # the same claim, just not able to quantify a mismatch if one appears.
        da, db = a.get("digests"), b.get("digests")
        if not da or not db:
            print("\nno samples and no digests -- cannot check bit-identity")
            identical = False
        else:
            bad = [i for i, (x, y) in enumerate(zip(da, db)) if x != y]
            identical = len(da) == len(db) and not bad
            print(f"\nper-shower digests: {len(da)} vs {len(db)}")
            print(f"bit-identical : {identical}")
            print(f"showers diff  : {len(bad)} / {min(len(da), len(db))}")
            for i in bad[:10]:
                print(f"  row {i:>6}  A {da[i]}  B {db[i]}")
    if identical:
        print("\n=> The two implementations produce THE SAME SHOWERS. Any blob "
              "present in one is present in the other, at the same rate, by "
              "construction.")

    print()
    print(rate_line(a["stats"], la))
    print(rate_line(b["stats"], lb))

    ma = np.array([r["median_e"] for r in a["stats"]])
    mb = np.array([r["median_e"] for r in b["stats"]])
    dis = np.nonzero((ma > BLOB_MEDIAN_E) != (mb > BLOB_MEDIAN_E))[0]
    if len(dis):
        print(f"\nshowers classified differently ({len(dis)}):")
        for i in dis[:20]:
            print(f"  row {i:>5}  A median_e={ma[i]:.4g}  B median_e={mb[i]:.4g}")
    elif len(ma) == len(mb):
        print("\nsame showers flagged as blobs in both.")

    worst = np.argsort(-np.maximum(ma, mb))[:args.top]
    print(f"\nmost extreme {len(worst)} showers:")
    print(f"{'row':>6}{'A median_e':>13}{'B median_e':>13}{'A tot/Ep':>11}"
          f"{'A top100%':>11}{'A span km':>11}{'A n':>8}")
    for i in worst:
        ra, rb = a["stats"][i], b["stats"][i]
        span = max(ra.get("x_span", 0.0), ra.get("y_span", 0.0)) / 1e3
        print(f"{i:>6}{ra['median_e']:>13.4g}{rb['median_e']:>13.4g}"
              f"{ra['tot_over_prim']:>11.4g}{100*ra.get('top100', 0):>11.1f}"
              f"{span:>11.2f}{ra['n']:>8}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("mask-check", help="compare compute_mask across the two repos (CPU)")

    p = sub.add_parser("prep", help="primaries + PointCountFM, shared by both impls")
    p.add_argument("--species", required=True, choices=list(SPECIES))
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--primaries", choices=("tau", "synthetic"), default="tau")
    p.add_argument("--anti-clip", action="store_true", default=True)
    p.add_argument("--no-anti-clip", dest="anti_clip", action="store_false",
                   help="skip the over-cap PointCountFM re-roll (Step 0 applies it)")
    p.add_argument("--out", required=True)

    g = sub.add_parser("gen", help="generate showers with one implementation")
    g.add_argument("--impl", required=True, choices=("local", "local-shim", "hamza"))
    g.add_argument("--prep", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--timesteps", type=int, default=16)
    g.add_argument("--solver", default="midpoint")
    g.add_argument("--device", default="cuda")
    g.add_argument("--no-compile", action="store_true")
    g.add_argument("--stats-only", action="store_true",
                   help="keep per-shower stats + digests instead of the samples "
                        "(the tensors are ~5 GB per 12k muons); bit-identity is "
                        "still checkable, digest by digest")
    g.add_argument("--pre-ln", dest="pre_ln", type=lambda s: s.lower() == "true",
                   default=True, help="local impls only; false = the known-bad "
                                      "post-LN pairing, as a positive control")
    g.add_argument("--stage-root", default=os.path.join(RESULTS, "staged"))

    c = sub.add_parser("compare", help="bit-identity + blob-rate comparison")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--top", type=int, default=10)

    args = ap.parse_args()
    return {"mask-check": cmd_mask_check, "prep": cmd_prep,
            "gen": cmd_gen, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
