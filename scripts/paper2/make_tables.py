"""Paper 2 (TAMBO layout study): every results table and the two layout figures, generated from the score JSONs that
score_all.py writes. Each table is written complete (environment, column spec, header, caption, label) to
<out>/tables_2/<name>.tex, so the chapters only \\input it; stale .tex files in <out>/tables_2 are removed. The layout
figures go to <out>/figures/results_2/.

    python scripts/paper2/make_tables.py --out <overleaf clone> [--res <score dir>]

Scores (in --res, default $PAPER2_RES or the lab folder below):
  scoresd{,bg1,bg10}_20_*.json   Malata, flow-matching showers, run recon, triggered metrics (*_cnt)
  scores{19_,bg1_19_,bg10_19_}*  12 km plane, flow-matching showers (older scoring: all-shower metrics only)
  scoresft_19_all.json           12 km flow-matching showers, fine-tuned recon (only its N_trig is used)
  scoresc8ft{,bg1,bg10}_<run>_*  CORSIKA 8 test showers, recon fine-tuned on CORSIKA 8 (ft_c8.py)
  scores{,c8}ana{,v2,v3}_20_*    analytical reconstruction (score_all.py with RECO=v1|v2|v3, ANA_HIT=mu)
Malata accuracies are over triggered showers, 12 km accuracies over all showers (trigsel.py).
"""
import argparse, glob, json, os, re, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trigsel import trig_view

LAB = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings"
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="paper folder (Overleaf clone)")
ap.add_argument("--res", default=os.environ.get("PAPER2_RES", f"{LAB}/res2"), help="score JSON folder")
args = ap.parse_args()
D = args.res
TAB = os.path.join(args.out, "tables_2"); FIG = os.path.join(args.out, "figures", "results_2")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SITES = f"{LAB}/sites"                                          # plane.npz / malata.npz: detector-region meshes

UT = ["orig", "twotier_l0p3", "twotier_l0", "threetier", "trig5_poisson", "trig5_step", "trig1_poisson"]
PART = {"access": "access", "contour": "contour rows", "pads": "pads", "slope": "slope", "eall": "all $E$",
        "e500_567": r"$\log E$ 5.00--5.67", "e567_633": r"$\log E$ 5.67--6.33", "e633_700": r"$\log E$ 6.33--7.00"}
FIXED = {"grid": "Grid", "ugrid": "Uniform grid", "centre": "Centre cluster"}

# layouts of the layout figures (4 x 4: fixed on top, optimized below); underlined in the tables
SEL = {
    19: ["grid", "ugrid", "disk_r6000", "annulus_rin2000", "core_r300", "clu20x5_r500", "tri33_r10",
         "tri33disk_R1000_r25", "opt_current_lr10_grid", "opt_current_lr10_center_p1", "opt_twotier_l0p3_lr10_grid",
         "opt_twotier_l0p3_lr10_center", "opt_threetier_090301_lr10_grid", "opt_threetier_090301_lr10_center",
         "opt_twotier_l0_lr10_grid", "opt_current_lr10_ugrid_p1"],
    20: ["grid", "ugrid", "disk_r400", "disk_r150", "tri33_r25", "tri33disk_R150_r25", "opt_current_lr10_center_p1",
         "opt_current_lr10_ugrid_p1", "opt_swgotrig_l0p3_k3_a2_lr10_grid_b2048",
         "opt_swgotrig_l0p3_k3_a2_lr10_center_b2048", "opt_swgotrig_l0p3_k3_a2_step_lr10_center_b2048",
         "opt_swgotrig_l0p3_k3_a2_lr10_grid_b2048_cab0p1", "opt_ker5_eall", "opt_trg_eall", "opt_ker5_contour",
         "opt_ker5_access"],
}


# ── layout names ─────────────────────────────────────────────────────────────────────────────────────────────────
def fixed_label(n):
    if n in FIXED:
        return FIXED[n]
    m = re.match(r"disk_r(\d+)", n)
    if m:
        return f"Disk {int(m[1]) / 1000:g}\\,km" if int(m[1]) >= 1000 else f"Disk {m[1]}\\,m"
    m = re.match(r"annulus_rin(\d+)", n)
    if m:
        return f"Annulus {int(m[1]) / 1000:g}--6\\,km"
    m = re.match(r"tri33disk_R(\d+)_r(\d+)", n)
    if m:
        return f"33 triplets ({m[2]}\\,m) in {m[1]}\\,m disk"
    m = re.match(r"tri33_r(\d+)", n)
    if m:
        return f"33 triplets ({m[1]}\\,m)"
    if n.startswith("core_r"):
        return f"Core {n[6:]}\\,m"
    if n.startswith("clu20x5"):
        return "20 clusters of 5"
    return n


def optimized_for(n):
    """utility key a layout was optimized for (underlined in the utility tables), or None"""
    s = n[4:]
    if s.startswith("current"):
        return "orig"
    if s.startswith("twotier"):
        return "twotier_l0p3" if "_l0p3" in s else "twotier_l0"
    if s.startswith("threetier"):
        return "threetier"
    if s.startswith("swgotrig"):
        return None if "_a20" in s else ("trig5_step" if "_step" in s else "trig5_poisson")
    if re.match(r"(ker5|sur5)", s):
        return "trig5_poisson"
    return None


_CFG = {}


def run_cfg(path):
    """(config, number of L-BFGS rounds) of the optimisation run that wrote the layout file"""
    if path not in _CFG:
        f = os.path.join(os.path.dirname(path), "optimize_log.json")
        lg = json.load(open(f)) if os.path.exists(f) else {}
        _CFG[path] = (lg.get("config", {}), len(lg.get("lbfgs_log") or []))
    return _CFG[path]


def cells(n, rec):
    """layout description as four cells: optimizer, utility, test case, start. Optimizer and start come from the
    run's config (runs without an 'optim' entry predate the option and used Adam)."""
    if not n.startswith("opt_"):
        return ["fixed", "", fixed_label(n), ""]
    c, nlb = run_cfg(rec["path"])
    s = n[4:]
    opt = {"sgd": "SGD", "adam": "Adam"}.get(c.get("optim", "adam"), str(c.get("optim")))
    opt += " + L-BFGS" if nlb else ""       # spaces: \\uline only breaks at spaces
    init = str(c.get("init", ""))
    start = {"grid": "grid", "center": "centre"}.get(init, "unif.\\ grid" if init.endswith("ugrid.pt") else init)
    if s.startswith("current"):
        ut = "Orig."
    elif s.startswith("twotier"):
        ut = "2-tier $\\lambda{=}0.3$" if "_l0p3" in s else "2-tier $\\lambda{=}0$"
    elif s.startswith("threetier"):
        ut = "3-tier"
    elif s.startswith("swgotrig"):
        ut = "Trig.\\ $5^\\circ$ step" if "_step" in s else "Trig.\\ $5^\\circ$ P"
    elif re.match(r"(ker5|sur5)", s):
        ut = "Trig.\\ $5^\\circ$ P"
    elif s.startswith("trg"):
        ut = "Trigger only"
    elif s.startswith("pen"):
        ut = "Penalty only"
    else:
        ut = ""
    case = []
    if c.get("response") == "surrogate":
        case.append("surr.")
    bg = re.search(r"bg(\d+)us", s)
    us = int(bg[1]) if bg else (c.get("bg_window_ns") or 0) / 1000
    if us:
        case.append(f"{us:g}\\,$\\mu$s bg" + (" + kNN" if ("knn" in s or c.get("gate") not in (None, "none")) else ""))
    if "lrd1" in s or c.get("lr_final"):
        case.append("lr decay")
    if "_a20" in s:
        case.append("20\\,m$^2$")
    m = re.search(r"cab(\d+p\d+)", s)
    if m:
        case.append(f"cable {m[1].replace('p', '.')}")
    m = re.match(r"(?:ker5|sur5_1us|sur5|trg|pen)_(\w+)", s)
    if m:
        case.append(PART.get(m[1], m[1].replace("_", r"\_")))
    if s.startswith("regress"):
        case.append("code test (2 steps)")
    return [opt, ut, ", ".join(case), start]


def fixed_key(n):
    m = re.search(r"(\d+)", n)
    v = int(m[1]) if m else 0
    for i, p in enumerate(("grid", "ugrid", "centre", "disk", "annulus", "core", "clu", "tri33_", "tri33disk")):
        if n == p or (n.startswith(p) and p not in ("grid", "ugrid", "centre")):
            return (i, v)
    return (99, v)


def opt_key(n):
    s = n[4:]
    for i, p in enumerate(("current", "twotier_l0p3", "twotier_l0", "threetier", "swgotrig", "ker5", "sur5_1us",
                           "sur5", "trg", "pen")):
        if s.startswith(p):
            return (i, s)
    return (99, s)


# ── cells ────────────────────────────────────────────────────────────────────────────────────────────────────────
G0 = 0.01           # log colour scale: relative shortfall g from the column best, t = 1 - log(1+g/G0)/log(1+gmax/G0)


def heat(s, v, best, gmax, high_good):
    """3-colour scale (red - yellow - green, lightened) as a \\hm{colour}{text} box; logarithmic in the relative
    shortfall from the column best: values near the best are spread out, low ones compressed."""
    if v != v:
        return s
    g = (best - v) / abs(best) if high_good else (v - best) / abs(best)
    t = 1.0 if gmax <= 0 else 1 - np.log1p(max(g, 0) / G0) / np.log1p(gmax / G0)
    c = f"red!{200 * (0.5 - t):.0f}!yellow" if t < 0.5 else f"green!{200 * (t - 0.5):.0f}!yellow"
    return f"\\hm{{{c}!45!white}}{{{s}}}"


def rng_of(d, k, sc):
    """(best, largest relative shortfall, higher is better) over all layouts of d"""
    vals = [d[n][k] * sc for n in d if d[n][k] == d[n][k]]
    if not vals:
        return 0.0, 0.0, True
    if k == "med":
        b = min(vals)
        return b, (max(vals) - b) / abs(b), False
    b = max(vals)
    return b, ((b - min(vals)) / abs(b) if b else 0.0), True


def hcell(d, n, k, sc, fmt, colour=True):
    """one cell of source d: bold if the best of d, coloured on d's scale; '--' if missing or NaN"""
    v = d[n][k] * sc if n in d else np.nan
    if v != v:
        return "--"
    s = fmt.format(v)
    if not colour:
        return s
    b, gmax, hi = rng_of(d, k, sc)
    return heat(f"\\textbf{{{s}}}" if abs(v - b) < 1e-12 else s, v, b, gmax, hi)


def pair(fm, c8, n, k, sc, fmt, colour=True):
    """flow matching [CORSIKA 8]"""
    s = hcell(fm, n, k, sc, fmt, colour)
    return s + (" [" + hcell(c8, n, k, sc, fmt, colour) + "]" if n in c8 else "")


def label(n, rec, site):
    """the four description cells; layouts of the layout figure underlined (\\uline breaks across lines)"""
    cs = cells(n, rec)
    if n in SEL[site]:
        cs = [f"\\uline{{{c}}}" if c else c for c in cs]
    return " & ".join(cs)


ANG = [("w1", 100, "{:.1f}"), ("w2.5", 100, "{:.1f}"), ("w5", 100, "{:.1f}"), ("med", 1, "{:.2f}"),
       ("e02", 100, "{:.1f}"), ("trig", 100, "{:.1f}"), ("ntrig", 1, "{:.0f}")]


def ang_row(n, fm, c8, site):
    return label(n, fm[n], site) + " & " + " & ".join(
        pair(fm, c8, n, k, sc, f, colour=k != "ntrig") for k, sc, f in ANG) + r" \\"


def util_row(n, fm, c8, site, cols):
    uk = optimized_for(n) if n.startswith("opt_") else None
    cs = []
    for k in cols:
        c = pair(fm, c8, n, k, 1, "{:.2f}" if k == "orig" else "{:.3f}")
        cs.append(f"\\underline{{{c}}}" if k == uk else c)
    return label(n, fm[n], site) + " & " + " & ".join(cs) + r" \\"


def body(names, row):
    """rows of the layouts in order, a rule between fixed and optimized"""
    out = []
    for i, n in enumerate(names):
        if i and n.startswith("opt_") and not names[i - 1].startswith("opt_"):
            out.append(r"\hline")
        out.append(row(n))
    return "\n".join(out)


def stacked(blocks, ncol):
    """[(title, rows)] -> rows with an italic title row per block and a double rule between blocks"""
    out = []
    for i, (title, rows) in enumerate(blocks):
        if i:
            out.append(r"\hline\hline")
        out.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (ncol, title))
        out.append(rows)
    return "\n".join(out)


# ── scores ───────────────────────────────────────────────────────────────────────────────────────────────────────
def load(site, pre):
    """(flow matching, CORSIKA 8) records of one run and background mode ("", "bg1_", "bg10_")"""
    tag = pre.rstrip("_")
    res = {}
    new = sorted(glob.glob(f"{D}/scoresd{tag}_{site}_*.json"))
    for p in new or [q for q in sorted(glob.glob(f"{D}/scores{pre}{site}_*.json")) if "c8test" not in q]:
        res.update(json.load(open(p)))
    res = {n: v for n, v in trig_view(res, acc=site != 19).items() if "smoke" not in n}
    if site == 19 and not pre and os.path.exists(f"{D}/scoresft_19_all.json"):
        # N_trig of the 12 km flow-matching showers from the fine-tuned-recon scoring of the same showers and layouts
        # (trigger counts do not depend on the recon; the trigger rates of the two scorings agree to 1e-8)
        ft = json.load(open(f"{D}/scoresft_19_all.json"))
        for n, r in res.items():
            if n in ft and r["ntrig"] != r["ntrig"] and abs(ft[n]["trig"] - r["trig"]) < 1e-6:
                r["ntrig"] = float(ft[n]["n_cnt"])
    c8 = {}
    for p in sorted(glob.glob(f"{D}/scoresc8ft{tag}_{site}_*.json")):
        c8.update(json.load(open(p)))
    c8 = {n: v for n, v in trig_view(c8, acc=site != 19).items() if "smoke" not in n}
    return res, c8


def order(res):
    fixed = sorted([n for n in res if not n.startswith("opt_")], key=fixed_key)
    return fixed + sorted([n for n in res if n.startswith("opt_")], key=opt_key)


def is_trigopt19(n):        # 12 km trigger-optimized layouts: appendix
    return "swgotrig" in n


def is_setup19(n):          # 12 km disk / annulus / triplet layouts not in the figure: Sec. 4.1
    return not n.startswith("opt_") and n not in SEL[19] and re.match(r"(disk|annulus|tri33)", n)


# ── LaTeX ────────────────────────────────────────────────────────────────────────────────────────────────────────
DESC = (r">{\raggedright\arraybackslash}p{1.1cm}>{\raggedright\arraybackslash}p{1.2cm}"     # optimizer, utility
        r">{\raggedright\arraybackslash}p{1.35cm}>{\raggedright\arraybackslash}p{0.85cm}")    # test case, start
DESC_HEAD = "Optimizer & Utility & Test case & Start"
H_ANG = DESC_HEAD + r" & $<1^\circ$ & $<2.5^\circ$ & $<5^\circ$ & Median [$^\circ$] & $|\Delta\log_{10}E|<0.2$ & Trigger & $N_\mathrm{trig}$ \\"
H_U4 = DESC_HEAD + r" & Orig. & 2-tier $\lambda{=}0.3$ & 2-tier $\lambda{=}0$ & 3-tier \\"
H_U7 = DESC_HEAD + r" & Orig. & 2-tier $\lambda{=}0.3$ & 2-tier $\lambda{=}0$ & 3-tier & Trig.\ $5^\circ$ P & Trig.\ $5^\circ$ step & Trig.\ $1^\circ$ P \\"
H_TU = DESC_HEAD + r" & Trig.\ $5^\circ$ P & Trig.\ $5^\circ$ step & Trig.\ $1^\circ$ P \\"
H_BG = (r" & & & & \multicolumn{3}{c}{$<1^\circ$ [\%]} & \multicolumn{3}{c}{Median [$^\circ$]} & \multicolumn{3}{c}{Trigger [\%]} \\" + "\n"
        + DESC_HEAD + r" & none & 1\,$\mu$s & 10\,$\mu$s & none & 1\,$\mu$s & 10\,$\mu$s & none & 1\,$\mu$s & 10\,$\mu$s \\")
H_ANAV = (r" & & & & \multicolumn{2}{c}{DeepSets} & \multicolumn{2}{c}{Analytical v1} & \multicolumn{2}{c}{Analytical v2} & \multicolumn{2}{c}{Analytical v3} \\" + "\n"
          + DESC_HEAD + r" & $<1^\circ$ & Median [$^\circ$] & $<1^\circ$ & Median [$^\circ$] & $<1^\circ$ & Median [$^\circ$] & $<1^\circ$ & Median [$^\circ$] \\")
H_ANA = (r" & & & & \multicolumn{4}{c}{DeepSets} & \multicolumn{4}{c}{Analytical v1} \\" + "\n"
         + DESC_HEAD + r" & $<1^\circ$ & $<2.5^\circ$ & $<5^\circ$ & Median [$^\circ$] & $<1^\circ$ & $<2.5^\circ$ & $<5^\circ$ & Median [$^\circ$] \\")
WRITTEN = []


def write(name, kind, ndata, head, rows, cap, lab, sep="2pt"):
    """complete table -> tables_2/<name>.tex; kind 'long' (longtable over pages) or 'float' (table float)"""
    spec = DESC + "r" * ndata
    if kind == "long":
        tex = (f"\\begingroup\\tiny\\setlength{{\\tabcolsep}}{{{sep}}}\n\\begin{{longtable}}{{{spec}}}\n"
               f"\\caption{{{cap}}}\n\\label{{{lab}}}\\\\\n\\hline\n{head}\n\\hline\n\\endfirsthead\n"
               f"\\hline\n{head}\n\\hline\n\\endhead\n\\hline\n\\endfoot\n{rows}\n\\end{{longtable}}\n\\endgroup\n")
    else:
        tex = (f"\\begin{{table}}\n\\centering\\tiny\\setlength{{\\tabcolsep}}{{{sep}}}\n\\begin{{tabular}}{{{spec}}}\n"
               f"\\hline\n{head}\n\\hline\n{rows}\n\\\\[-1.5ex]\\hline\n\\end{{tabular}}\n"
               f"\\caption{{{cap}}}\n\\label{{{lab}}}\n\\end{{table}}\n")
    open(os.path.join(TAB, name + ".tex"), "w").write(tex)
    WRITTEN.append(name + ".tex")


# ── figures ──────────────────────────────────────────────────────────────────────────────────────────────────────
def plain(s):
    s = s.replace("\\,", " ").replace("\\ ", " ").replace("$\\mu$", "µ").replace("{=}", "=").replace("--", "–")
    return s.replace("$", "").replace("^\\circ", "°").replace("\\lambda", "λ").replace("^2", "²")


def figure(site, res, names):
    """4 x 4 layout figure; panels in table order; titles: share within 1 and 5 deg (flow matching)"""
    site_npz = np.load(f"{SITES}/{'plane' if site == 19 else 'malata'}.npz")
    tris = site_npz["tris"]; sc = 1000.0 if site == 19 else 1.0
    cE, cN = tris[:, :, 0].mean(1) / sc, tris[:, :, 1].mean(1) / sc
    fig, ax = plt.subplots(4, 4, figsize=(10, 10.6), constrained_layout=True)
    panels = [n for n in names if n in SEL[site]] if site == 19 else SEL[site]
    for a, n in zip(ax.flat, panels):
        if n not in res:
            a.axis("off"); continue
        L = torch.load(res[n]["path"], map_location="cpu", weights_only=False)
        x, y = np.asarray(L["x"]) / sc, np.asarray(L["y"]) / sc
        a.scatter(cE, cN, s=(0.3 if site == 19 else 4), color="0.85", lw=0, rasterized=True)
        a.scatter(x, y, s=7, color="tab:blue" if n.startswith("opt_") else "0.15", lw=0)
        o, u, cs, st = cells(n, res[n])
        lab = plain(fixed_label(n) if not n.startswith("opt_") else ", ".join(x for x in (u, cs, o, st) if x))
        a.set_title(f"{lab}\n1°: {100 * res[n]['w1']:.1f}%  5°: {100 * res[n]['w5']:.1f}%", fontsize=7.5)
        a.set_aspect("equal"); a.tick_params(labelsize=6.5)
    u = "km" if site == 19 else "m"
    for a in ax[-1]:
        a.set_xlabel(f"East [{u}]", fontsize=7.5)
    for a in ax[:, 0]:
        a.set_ylabel(f"North [{u}]", fontsize=7.5)
    fig.savefig(f"{FIG}/layouts{site}.pdf", dpi=300)
    plt.close(fig)


# ── tables ───────────────────────────────────────────────────────────────────────────────────────────────────────
S19, S20 = r"12\,km plane (run 19)", "Malata (run 20)"
B1, B10, NONE = r"1\,$\mu$s background + kNN gate", r"10\,$\mu$s background + kNN gate", "no background"
W = {s: {p: load(s, p) for p in ("", "bg1_", "bg10_")} for s in (19, 20)}
NAMES = {s: order(W[s][""][0]) for s in (19, 20)}

# 12 km plane
fm, c8 = W[19][""]
main19 = [n for n in NAMES[19] if not is_trigopt19(n) and not is_setup19(n)]
setup19 = [n for n in NAMES[19] if is_setup19(n)]
trig19 = [n for n in NAMES[19] if is_trigopt19(n)]
U4 = UT[:4]
write("r19_ang", "long", 7, H_ANG, body(main19, lambda n: ang_row(n, fm, c8, 19)),
      f"{S19}, no background: angular scores (all showers), energy score and trigger of every layout.", "tab:r19_ang")
write("r19_util", "long", 4, H_U4, body(main19, lambda n: util_row(n, fm, c8, 19, U4)),
      f"{S19}, no background: accuracy utilities (Sec.~\\ref{{sec:utilities}}) of every layout on the showers of "
      f"Table~\\ref{{tab:r19_ang}}; all but the original utility are per-shower means in $[0,1]$.", "tab:r19_util")
write("r19_cmp_ang", "float", 7, H_ANG, body(setup19, lambda n: ang_row(n, fm, c8, 19)),
      f"{S19}, no background: disk, annulus and triplet layouts not in Table~\\ref{{tab:r19_ang}}; angular scores (all "
      f"showers), energy score and trigger. Bold and colour over all layouts of the run.", "tab:r19_cmp_ang")
write("r19_cmp_util", "float", 4, H_U4, body(setup19, lambda n: util_row(n, fm, c8, 19, U4)),
      f"{S19}, no background: accuracy utilities of the layouts of Table~\\ref{{tab:r19_cmp_ang}}, as in "
      f"Table~\\ref{{tab:r19_util}}.", "tab:r19_cmp_util")
bg_names19 = [n for n in NAMES[19] if not is_trigopt19(n)]
write("r19_bg_ang", "long", 7, H_ANG,
      stacked([(t, body(bg_names19, lambda n, p=p: ang_row(n, *W[19][p], 19))) for t, p in ((B1, "bg1_"), (B10, "bg10_"))], 11),
      f"{S19}, cosmic background + kNN gate: angular scores (all showers), energy score and trigger, as in "
      f"Table~\\ref{{tab:r19_ang}}.", "tab:r19_bg_ang")
write("r19_bg_util", "long", 4, H_U4,
      stacked([(t, body(bg_names19, lambda n, p=p: util_row(n, *W[19][p], 19, U4))) for t, p in ((B1, "bg1_"), (B10, "bg10_"))], 8),
      f"{S19}, cosmic background + kNN gate: accuracy utilities, as in Table~\\ref{{tab:r19_util}}.", "tab:r19_bg_util")
modes = ((NONE, ""), (B1, "bg1_"), (B10, "bg10_"))
write("r19_trig_ang", "float", 7, H_ANG,
      stacked([(t, body(trig19, lambda n, p=p: ang_row(n, *W[19][p], 19))) for t, p in modes], 11),
      f"{S19}: angular scores (all showers), energy score and trigger of the trigger-optimized layouts.", "tab:r19_trig_ang")
write("r19_trig_util", "float", 4, H_U4,
      stacked([(t, body(trig19, lambda n, p=p: util_row(n, *W[19][p], 19, U4))) for t, p in modes], 8),
      f"{S19}: accuracy utilities of the trigger-optimized layouts.", "tab:r19_trig_util")
write("r19_trigut", "long", 3, H_TU,
      stacked([(t, body(bg_names19, lambda n, p=p: util_row(n, *W[19][p], 19, UT[4:])) + "\n\\hline\n"
                + body(trig19, lambda n, p=p: util_row(n, *W[19][p], 19, UT[4:]))) for t, p in modes], 7),
      f"{S19}: trigger utility at $5^\\circ$ (Poisson and step trigger) and $1^\\circ$ (Poisson), per-shower means in "
      f"$[0,1]$, of every layout; in each block fixed layouts, then accuracy-optimized, then trigger-optimized.",
      "tab:r19_trigut")

# Malata
fm, c8 = W[20][""]
write("r20_ang", "long", 7, H_ANG, body(NAMES[20], lambda n: ang_row(n, fm, c8, 20)),
      f"{S20}, no background: angular scores (triggered showers), energy score and trigger of every layout.", "tab:r20_ang")
write("r20_util", "long", 7, H_U7, body(NAMES[20], lambda n: util_row(n, fm, c8, 20, UT)),
      f"{S20}, no background: utilities (Sec.~\\ref{{sec:utilities}}) of every layout on the showers of "
      f"Table~\\ref{{tab:r20_ang}}; all but the original utility are per-shower means in $[0,1]$.", "tab:r20_util",
      sep="1.5pt")
write("r20_bg_ang", "long", 7, H_ANG,
      stacked([(t, body(NAMES[20], lambda n, p=p: ang_row(n, *W[20][p], 20))) for t, p in ((B1, "bg1_"), (B10, "bg10_"))], 11),
      f"{S20}, cosmic background + kNN gate: angular scores (triggered showers), energy score and trigger, as in "
      f"Table~\\ref{{tab:r20_ang}}.", "tab:r20_bg_ang")
write("r20_bg_util", "long", 7, H_U7,
      stacked([(t, body(NAMES[20], lambda n, p=p: util_row(n, *W[20][p], 20, UT))) for t, p in ((B1, "bg1_"), (B10, "bg10_"))], 11),
      f"{S20}, cosmic background + kNN gate: utilities, as in Table~\\ref{{tab:r20_util}}.", "tab:r20_bg_util",
      sep="1.5pt")


# background comparison: layouts of the figure, no background / 1 us / 10 us side by side
def bg_row(n, site):
    cs = [pair(*W[site][p], n, k, sc, f) for k, sc, f in (("w1", 100, "{:.1f}"), ("med", 1, "{:.2f}"),
                                                          ("trig", 100, "{:.1f}")) for p in ("", "bg1_", "bg10_")]
    return label(n, W[site][""][0][n], site).replace("\\uline", "") + " & " + " & ".join(cs) + r" \\"


for site, sname, sel_txt in ((19, S19, "share of all showers within $1^\\circ$, median angular error"),
                             (20, S20, "share of the triggered showers within $1^\\circ$, their median angular error")):
    figs = [n for n in NAMES[site] if n in SEL[site]]
    write(f"r{site}_bg", "float", 9, H_BG, body(figs, lambda n, s=site: bg_row(n, s)),
          f"{sname}: layouts of Fig.~\\ref{{fig:r{site}_layouts}} without background (none) and with a 1 or "
          f"10\\,$\\mu$s readout window and the kNN gate; {sel_txt} and expected trigger rate.", f"tab:r{site}_bg", sep="1.1pt")

# analytical reconstruction (Malata, no background): DeepSets vs TamboDirReco port, triggered showers
def tv(pat):
    return trig_view({n: v for p in sorted(glob.glob(f"{D}/{pat}")) for n, v in json.load(open(p)).items()
                      if "smoke" not in n})


SRC = {"ds": W[20][""], "v1": (tv("scoresana_20_*.json"), tv("scoresc8ana_20_*.json")),
       "v2": (tv("scoresanav2_20_*.json"), tv("scoresc8anav2_20_*.json")),
       "v3": (tv("scoresanav3_20_*.json"), tv("scoresc8anav3_20_*.json"))}


def ana_row(n, cols, ul):
    lab = label(n, W[20][""][0][n], 20)
    if not ul:
        lab = lab.replace("\\uline", "")
    return lab + " & " + " & ".join(pair(*SRC[m], n, k, sc, f) for m, k, sc, f in cols) + r" \\"


short = lambda m: [(m, "w1", 100, "{:.1f}"), (m, "med", 1, "{:.2f}")]
full = lambda m: [(m, "w1", 100, "{:.1f}"), (m, "w2.5", 100, "{:.1f}"), (m, "w5", 100, "{:.1f}"), (m, "med", 1, "{:.2f}")]
figs20 = [n for n in NAMES[20] if n in SEL[20]]
write("r20_anav", "float", 8, H_ANAV,
      body(figs20, lambda n: ana_row(n, short("ds") + short("v1") + short("v2") + short("v3"), False)),
      f"{S20}, no background: DeepSets vs the analytical reconstruction (v1, v2, v3) on the triggered showers, layouts "
      f"of Fig.~\\ref{{fig:r20_layouts}}; share within $1^\\circ$ [\\%] and median angular error. v2 and v3 were run on "
      f"these layouts only.", "tab:r20_anav")
write("r20_ana", "long", 8, H_ANA, body(NAMES[20], lambda n: ana_row(n, full("ds") + full("v1"), True)),
      f"{S20}, no background: DeepSets vs analytical v1 (Sec.~\\ref{{sec:analytic}}) on the triggered showers, every "
      f"layout.", "tab:r20_ana")

# figures
figure(19, W[19][""][0], main19)
figure(20, W[20][""][0], NAMES[20])

# stale generated tables
for f in sorted(os.listdir(TAB)):
    if f.endswith(".tex") and f not in WRITTEN:
        os.remove(os.path.join(TAB, f)); print("removed stale", f)
print(f"{len(WRITTEN)} tables -> {TAB}; layouts19.pdf, layouts20.pdf -> {FIG}")
