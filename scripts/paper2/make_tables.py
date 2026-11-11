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
UT_HEAD = [r"Orig.", r"2-tier $\lambda{=}0.3$", r"2-tier $\lambda{=}0$", r"3-tier", r"Trig.\ $5^\circ$ P",
           r"Trig.\ $5^\circ$ step", r"Trig.\ $1^\circ$ P"]
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
    """one cell: optimizer | utility | test case | start, '--' where a field does not apply; layouts of the layout
    figure underlined (\\uline breaks across lines)"""
    o, u, c, st = cells(n, rec)
    if not n.startswith("opt_"):
        o = ""
    lab = " $|$ ".join(x if x else "--" for x in (o, u, c, st))
    return f"\\uline{{{lab}}}" if n in SEL[site] else lab


ANG = [("w1", 100, "{:.1f}"), ("w2.5", 100, "{:.1f}"), ("w5", 100, "{:.1f}"), ("med", 1, "{:.2f}"),
       ("e02", 100, "{:.1f}"), ("trig", 100, "{:.1f}"), ("ntrig", 1, "{:.0f}")]


def name_label(n, site):
    """fixed-layout tables: the layout name, underlined if in the layout figure"""
    return f"\\uline{{{fixed_label(n)}}}" if n in SEL[site] else fixed_label(n)


ANG_MAIN = [c for c in ANG if c[0] != "w2.5"]          # main text: the 2.5 deg share is in the appendix tables (_all)


def ang_row(n, fm, c8, site, name=False, cols=ANG):
    return (name_label(n, site) if name else label(n, fm[n], site)) + " & " + " & ".join(
        pair(fm, c8, n, k, sc, f, colour=k != "ntrig") for k, sc, f in cols) + r" \\"


def util_row(n, fm, c8, site, cols, name=False):
    uk = optimized_for(n) if n.startswith("opt_") else None
    cs = []
    for k in cols:
        c = pair(fm, c8, n, k, 1, "{:.2f}" if k == "orig" else "{:.3f}")
        cs.append(f"\\underline{{{c}}}" if k == uk else c)
    return (name_label(n, site) if name else label(n, fm[n], site)) + " & " + " & ".join(cs) + r" \\"


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


def is_setup19(n):          # 12 km disk / annulus / triplet layouts not in the figure: only in the sweep table
    return not n.startswith("opt_") and n not in SEL[19] and re.match(r"(disk|annulus|tri33)", n)


# ── LaTeX ────────────────────────────────────────────────────────────────────────────────────────────────────────
DESC = r">{\raggedright\arraybackslash}p{4.2cm}"           # layout: optimizer | utility | test case | start
DESC_HEAD = r"Layout: optimizer $|$ utility $|$ test case $|$ start"
H_ANG = DESC_HEAD + r" & $<1^\circ$ & $<2.5^\circ$ & $<5^\circ$ & Median [$^\circ$] & $|\Delta\log_{10}E|<0.2$ & Trigger & $N_\mathrm{trig}$ \\"
H_ANG_MAIN = H_ANG.replace(r" & $<2.5^\circ$", "")
H_U4 = DESC_HEAD + r" & Orig. & 2-tier $\lambda{=}0.3$ & 2-tier $\lambda{=}0$ & 3-tier \\"
H_U7 = DESC_HEAD + r" & Orig. & 2-tier $\lambda{=}0.3$ & 2-tier $\lambda{=}0$ & 3-tier & Trig.\ $5^\circ$ P & Trig.\ $5^\circ$ step & Trig.\ $1^\circ$ P \\"
H_TU = DESC_HEAD + r" & Trig.\ $5^\circ$ P & Trig.\ $5^\circ$ step & Trig.\ $1^\circ$ P \\"
H_BG = (r" & \multicolumn{3}{c}{$<1^\circ$ [\%]} & \multicolumn{3}{c}{Median [$^\circ$]} & \multicolumn{3}{c}{Trigger [\%]} \\" + "\n"
        + DESC_HEAD + r" & none & 1\,$\mu$s & 10\,$\mu$s & none & 1\,$\mu$s & 10\,$\mu$s & none & 1\,$\mu$s & 10\,$\mu$s \\")
H_ANAV = (r" & \multicolumn{2}{c}{DeepSets} & \multicolumn{2}{c}{Analytical v1} & \multicolumn{2}{c}{Analytical v2} & \multicolumn{2}{c}{Analytical v3} \\" + "\n"
          + DESC_HEAD + r" & $<1^\circ$ & Median [$^\circ$] & $<1^\circ$ & Median [$^\circ$] & $<1^\circ$ & Median [$^\circ$] & $<1^\circ$ & Median [$^\circ$] \\")
H_ANA = (r" & \multicolumn{4}{c}{DeepSets} & \multicolumn{4}{c}{Analytical v1} \\" + "\n"
         + DESC_HEAD + r" & $<1^\circ$ & $<2.5^\circ$ & $<5^\circ$ & Median [$^\circ$] & $<1^\circ$ & $<2.5^\circ$ & $<5^\circ$ & Median [$^\circ$] \\")
WRITTEN = []


def write(name, kind, ndata, head, rows, cap, lab, sep="2pt", desc=None, landscape=False):
    """complete table -> tables_2/<name>.tex; kind 'long' (longtable over pages) or 'float' (table float)"""
    spec = (desc or DESC) + "r" * ndata
    if kind == "long":
        tex = (f"\\begingroup\\tiny\\setlength{{\\tabcolsep}}{{{sep}}}\n\\begin{{longtable}}{{{spec}}}\n"
               f"\\caption{{{cap}}}\n\\label{{{lab}}}\\\\\n\\hline\n{head}\n\\hline\n\\endfirsthead\n"
               f"\\hline\n{head}\n\\hline\n\\endhead\n\\hline\n\\endfoot\n{rows}\n\\end{{longtable}}\n\\endgroup\n")
    else:
        tex = (f"\\begin{{table}}\n\\centering\\tiny\\setlength{{\\tabcolsep}}{{{sep}}}\n\\begin{{tabular}}{{{spec}}}\n"
               f"\\hline\n{head}\n\\hline\n{rows}\n\\\\[-1.5ex]\\hline\n\\end{{tabular}}\n"
               f"\\caption{{{cap}}}\n\\label{{{lab}}}\n\\end{{table}}\n")
    if landscape:       # rotated page(s) (pdflscape): wide tables
        tex = "\\begin{landscape}\n" + tex + "\\end{landscape}\n"
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
trig19 = [n for n in NAMES[19] if is_trigopt19(n)]
U4 = UT[:4]
COLHEAD = {"w1": r"$<1^\circ$", "w2.5": r"$<2.5^\circ$", "w5": r"$<5^\circ$", "med": r"Median [$^\circ$]",
           "e02": r"$|\Delta\log_{10}E|<0.2$", "trig": "Trigger", "ntrig": r"$N_\mathrm{trig}$"}
APPX = {19: ["w2.5", "trig"], 20: ["w2.5"]}      # main-text scores only in the appendix tables (_all)


def scores_table(site, names, uts, sep, wdesc, landscape=True):
    """one table per site: every main-text layout (fixed above the line, optimized below), the scores not in APPX
    and the utilities uts"""
    f_, c_ = W[site][""]
    cols = [c for c in ANG if c[0] not in APPX[site]]
    head = (r" & \multicolumn{%d}{c}{Scores} & \multicolumn{%d}{c}{Utilities} \\" % (len(cols), len(uts)) + "\n"
            + DESC_HEAD + " & " + " & ".join(COLHEAD[k] for k, _, _ in cols) + " & "
            + " & ".join(UT_HEAD[UT.index(k)] for k in uts) + r" \\")

    def row(n):
        u = util_row(n, f_, c_, site, uts)
        return ang_row(n, f_, c_, site, cols=cols)[:-3] + u[len(label(n, f_[n], site)):]
    sname, sh = (S19, "all showers") if site == 19 else (S20, "triggered showers")
    moved = "share within $2.5^\\circ$ and trigger rate" if "trig" in APPX[site] else "share within $2.5^\\circ$"
    write(f"r{site}_scores", "long", len(cols) + len(uts), head, body(names, row),
          f"{sname}, no background: angular scores ({sh}), energy score{'' if 'trig' in APPX[site] else ', trigger'} and "
          f"utilities (Sec.~\\ref{{sec:utilities}}; per-shower means in $[0,1]$ except the original utility) of every "
          f"layout; fixed layouts above the line, optimized below. The {moved}: App.~\\ref{{app:all}}.",
          f"tab:r{site}_scores", sep=sep, desc=r">{\raggedright\arraybackslash}p{%s}" % wdesc, landscape=landscape)


scores_table(19, main19, U4, "1pt", "2.9cm", landscape=False)
# Malata
fm, c8 = W[20][""]
scores_table(20, NAMES[20], UT, "1pt", "2.7cm")
# background comparison: layouts of the figure, no background / 1 us / 10 us side by side
BG_COLS = {19: (("w1", 100, "{:.1f}"), ("med", 1, "{:.2f}")),            # 12 km: trigger rate in the appendix
           20: (("w1", 100, "{:.1f}"), ("med", 1, "{:.2f}"), ("trig", 100, "{:.1f}"))}


def bg_row(n, site):
    cs = [pair(*W[site][p], n, k, sc, f) for k, sc, f in BG_COLS[site] for p in ("", "bg1_", "bg10_")]
    return label(n, W[site][""][0][n], site).replace("\\uline", "") + " & " + " & ".join(cs) + r" \\"


for site, sname, sel_txt in ((19, S19, "share of all showers within $1^\\circ$ and median angular error (trigger rate: "
                                       "App.~\\ref{app:all})"),
                             (20, S20, "share of the triggered showers within $1^\\circ$, their median angular error and "
                                       "expected trigger rate")):
    figs = [n for n in NAMES[site] if n in SEL[site]]
    nb = len(BG_COLS[site])
    write(f"r{site}_bg", "float", 3 * nb, H_BG if nb == 3 else H_BG.split(r" & \multicolumn{3}{c}{Trigger [\%]}")[0] + r" \\"
          + "\n" + DESC_HEAD + r" & none & 1\,$\mu$s & 10\,$\mu$s & none & 1\,$\mu$s & 10\,$\mu$s \\",
          body(figs, lambda n, s=site: bg_row(n, s)),
          f"{sname}: layouts of Fig.~\\ref{{fig:r{site}_layouts}} without background (none) and with a 1 or "
          f"10\\,$\\mu$s readout window and the kNN gate; {sel_txt}.", f"tab:r{site}_bg", sep="1.1pt")

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

# layout sweeps of the fixed-layout experiments (Sec. 4.1 plots): every layout, one block per sweep
def sweep_label(n):
    m = re.fullmatch(r"disk_r(\d+)", n)
    if m:
        return f"Disk {int(m[1]) / 1000:g}\\,km" if int(m[1]) >= 1000 else f"Disk {m[1]}\\,m"
    m = re.fullmatch(r"annulus_rin(\d+)_R(\d+)_n\d+", n)
    if m:
        return (f"Annulus {int(m[1]) / 1000:g}--{int(m[2]) / 1000:g}\\,km" if int(m[1])
                else f"Disk {int(m[2]) / 1000:g}\\,km (no hole)")
    m = re.fullmatch(r"tri33disk_R(\d+)_r(\d+)", n)
    if m:
        return f"33 triplets ({m[2]}\\,m) in {int(m[1]) / 1000:g}\\,km disk"
    m = re.fullmatch(r"tri33_r(\d+)", n)
    if m:
        return f"33 triplets ({m[1]}\\,m)"
    m = re.fullmatch(r"clu(\d+)x(\d+)_r(\d+)", n)
    if m:
        return f"{m[1]} clusters of {m[2]} ({m[3]}\\,m)"
    return FIXED.get(n, n)


def sweeps(run, cols):
    """rows of every sweep layout (grid for reference, then one block per sweep) for the columns cols"""
    tag = "ext" if run == 19 else "extd_"
    fm = trig_view({k: v for p in sorted(glob.glob(f"{D}/{tag}{run}_*.json")) for k, v in json.load(open(p)).items()},
                   acc=run != 19)
    c8 = trig_view({k: v for p in sorted(glob.glob(f"{D}/extc8ft_{run}_*.json")) for k, v in json.load(open(p)).items()},
                   acc=run != 19)
    num = lambda n: [int(x) for x in re.findall(r"\d+", n)]
    groups = [("disk size", sorted([n for n in fm if n.startswith("disk_r")], key=num)),
              ("disk with a hole", sorted([n for n in fm if n.startswith("annulus")], key=lambda n: (-num(n)[1], num(n)[0]))),
              ("clustering", sorted([n for n in fm if n.startswith("tri33_")], key=num)
               + sorted([n for n in fm if n.startswith("tri33disk")], key=num)
               + sorted([n for n in fm if n.startswith("clu")], key=num))]
    row = lambda n: sweep_label(n) + " & " + " & ".join(pair(fm, c8, n, k, sc, f, colour=k != "ntrig")
                                                        for k, sc, f in cols) + r" \\"
    ref = r"\multicolumn{%d}{l}{\textit{reference}} \\" % (1 + len(cols)) + "\n" + row("grid") + "\n\\hline\\hline\n"
    return ref + stacked([(t, "\n".join(row(n) for n in g)) for t, g in groups if g], 1 + len(cols))


write("r19_sweeps", "long", 11, r"Layout & $<1^\circ$ & $<2.5^\circ$ & $<5^\circ$ & Median [$^\circ$] & "
      r"$|\Delta\log_{10}E|<0.2$ & Trigger & $N_\mathrm{trig}$ & Orig. & 2-tier $\lambda{=}0.3$ & 2-tier $\lambda{=}0$ & "
      r"3-tier \\", sweeps(19, ANG + [(k, 1, "{:.2f}" if k == "orig" else "{:.3f}") for k in UT[:4]]),
      f"{S19}, no background: every layout of the sweeps of Figs.~\\ref{{fig:exp19}} and~\\ref{{fig:exp19_layouts}} "
      f"(99 or 100 detectors), grid for reference; angular scores (all showers), energy score, trigger and accuracy "
      f"utilities (Sec.~\\ref{{sec:utilities}}). Bold and colour over the layouts of this table.", "tab:r19_sweeps",
      desc="l", landscape=True)

# appendix: complete scores of every layout, one block per background window (bold and colour per block)
MODES = ((NONE, ""), (B1, "bg1_"), (B10, "bg10_"))


def full_body(site, p, row):
    names = [n for n in NAMES[site] if not (site == 19 and is_trigopt19(n))]
    out = body(names, lambda n: row(n, p))
    if site == 19:          # 12 km trigger-optimized layouts below a second line
        out += "\n\\hline\n" + body([n for n in NAMES[19] if is_trigopt19(n)], lambda n: row(n, p))
    return out


for site, sname, sh in ((19, S19, "all showers"), (20, S20, "triggered showers")):
    extra = "; trigger-optimized layouts below the second line" if site == 19 else ""
    win = (r"without background and with a 1 or 10\,$\mu$s readout window and the kNN gate "
           f"(Sec.~\\ref{{sec:r{site}_bg}}); bold and colour per block{extra}")
    write(f"r{site}_all_ang", "long", 7, H_ANG,
          stacked([(t, full_body(site, p, lambda n, p_: ang_row(n, *W[site][p_], site))) for t, p in MODES], 8),
          f"{sname}: angular scores ({sh}), energy score and trigger of every layout, {win}.", f"tab:r{site}_all_ang")
    write(f"r{site}_all_util", "long", 7, H_U7,
          stacked([(t, full_body(site, p, lambda n, p_: util_row(n, *W[site][p_], site, UT))) for t, p in MODES], 8),
          f"{sname}: utilities (Sec.~\\ref{{sec:utilities}}; per-shower means in $[0,1]$ except the original utility) "
          f"of every layout, {win}.", f"tab:r{site}_all_util", sep="1.5pt")

# figures
figure(19, W[19][""][0], main19)
figure(20, W[20][""][0], NAMES[20])

# stale generated tables
for f in sorted(os.listdir(TAB)):
    if f.endswith(".tex") and f not in WRITTEN:
        os.remove(os.path.join(TAB, f)); print("removed stale", f)
print(f"{len(WRITTEN)} tables -> {TAB}; layouts19.pdf, layouts20.pdf -> {FIG}")
