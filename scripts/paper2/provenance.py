"""Paper 2 provenance: where every layout, optimisation run, score file, table and figure comes from.

    python scripts/paper2/provenance.py [--res <score dir>] [--out docs/paper2_provenance.md]

Reads the layout lists (list19ns / list20ns / ext19 / ext20 in the score folder), every optimisation run's
optimize_log.json (config, L-BFGS rounds, scoring log), the score JSONs and a snapshot of the job history
(sacct_nov.txt in the score folder; sacct keeps only a limited date range, so the snapshot is kept) and writes one
Markdown file. Paths and settings are read from the files; nothing is typed in by hand except the code map and the
paper-element map below, which mirror make_tables.py and plot_exp.py.
"""
import argparse, glob, json, os, re

LAB = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov"
V6 = f"{LAB}/detector_optimization_v6"
ap = argparse.ArgumentParser()
ap.add_argument("--res", default=os.environ.get("PAPER2_RES", f"{LAB}/tmp_rings/res2"))
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs",
                                              "paper2_provenance.md"))
args = ap.parse_args()
D = args.res
RUNS = {19: ("12 km plane", f"{V6}/19_plane45S_12km_showerframe_400k"), 20: ("Malata", f"{V6}/20_malata_showerframe")}

# ── code (repository TambOpt, branch cleanup) ────────────────────────────────────────────────────────────────────
CODE = [
    ("scripts/04_optimize_lbfgs_kernel.py", "Stage 4 layout optimisation (Adam / SGD, optional L-BFGS polish); kernel or "
     "surrogate response; utility chosen by TAMBOPT_UTILITY and related environment variables; writes "
     "<RUN>/test_v6_run_04_optimize_lbfgs_kernel_<suffix>/{layout_best.pt, optimize_log.json, opt_state.pt}. "
     "--pool c8 / --recon-type analytic added for the CORSIKA 8 + analytical runs (c8ana_*)"),
    ("scripts/03_train_recon_deepsets.py", "DeepSets reconstruction (<RUN>/test_v6_run_03_recentered_deepsets)"),
    ("scripts/paper2/ft_c8.py", "fine-tune of the run's DeepSets on CORSIKA 8 training showers -> tmp_rings/c8ft/ft19, ft20"),
    ("scripts/paper2/score_all.py", "scores layouts on held-out showers (noise-free kernel): flow matching (default) or "
     "CORSIKA 8 test showers (SRC=c8), DeepSets (RECON_DIR) or analytical v1/v2/v3 (RECO, ANA_HIT=mu), background "
     "(BG_NS + kNN gate) -> score JSONs in tmp_rings/res2"),
    ("scripts/paper2/trigsel.py", "triggered view: Malata accuracies over triggered showers, 12 km over all showers"),
    ("modules/optimize/analytic_reco.py", "TamboDirReco port (v1, v2, v3) used by score_all.py RECO=..."),
    ("modules/optimize/analytic_diff.py", "differentiable v1 (Gauss-Newton) for --recon-type analytic"),
    ("scripts/paper2/make_tables.py", "every table (complete LaTeX) and the layout / constraint figures -> Overleaf "
     "tables_2/, figures/results_2/"),
    ("scripts/paper2/plot_exp.py", "sweep figures exp19 / exp20 and their layout figures"),
    ("scripts/paper2/make_paper2.sh", "driver: make_tables.py + plot_exp.py into the Overleaf clone"),
    ("scripts/paper2/check_paper2.sh", "compile check of the chapters at the journal page size"),
    ("scripts/paper2/provenance.py", "this file"),
]
LAYOUT_WRITERS = {
    "diagnostics/triplets/layouts": "plots/analysis/run19/triplets/make_triplets.py (grid_n99, disk_r*, tri33_r*); "
                                    "on Malata called from slurm/run20_submit.sh",
    "diagnostics/counting/layouts": "plots/analysis/run19/counting/make_annuli.py (annulus_rin*_R*_n*), "
                                    "plots/analysis/run19/counting/make_region_triplets.py (tri33disk_R*_r*)",
    "diagnostics/tests/layouts": "plots/analysis/run19/tests/make_layouts.py (core_r*, clu*, disk_r*_n*)",
    "diagnostics/paper1": "uniform grid ugrid.pt (start of the paper-1 runs, slurm/paper1_opt_submit.sh); writer not "
                          "in the repository",
    "diagnostics/opt/layouts": "start layouts of the optimiser (start_center.pt); writer not in the repository",
    "tmp_rings/optres": "Malata start layouts (malata_start_center.pt) of tmp_rings/optres/score.sh; writer not in "
                        "the repository",
}
# paper element -> score files (mirrors make_tables.py / plot_exp.py)
ELEMENTS = [
    ("Table 1 (12 km main) and Fig. 3", "scores19_{done,part0-3}.json, scoresbg{1,10}_19_all.json (flow matching, "
     "run recon, all showers); scoresc8ft{,bg1,bg10}_19_all.json (CORSIKA 8, fine-tuned recon); N_trig of flow "
     "matching from scoresft_19_all.json (same showers and layouts)"),
    ("Table 2 (Malata main) and Fig. 6", "scoresd{,bg1,bg10}_20_all.json (flow matching, triggered metrics); "
     "scoresc8ft{,bg1,bg10}_20_all.json (CORSIKA 8, fine-tuned recon)"),
    ("Constraint table and figure (Malata)", "scoresd_20_all.json, scoresc8ft_20_all.json; cable length computed "
     "in make_tables.py from the layout file and the detector-region mesh (tmp_rings/sites/malata.npz)"),
    ("Analytical table v1/v2/v3 (Malata)", "DeepSets: scoresd_20_all.json, scoresc8ft_20_all.json; v1: "
     "scores{,c8}ana_20_all.json overridden by scores{,c8}ana_20_zfix*.json; v2 / v3: scores{,c8}ana{v2,v3}_20_sel.json"),
    ("Fig. 4 / Fig. 7 (sweeps) and their layout figures", "ext19_all.json (12 km flow matching, all showers), "
     "extd_20_all.json (Malata flow matching), extc8ft_{19,20}_all.json (CORSIKA 8); uniform-grid reference line from "
     "the main score files"),
    ("Appendix: all layouts, complete scores", "the main score files above, all layouts and all background windows"),
    ("Appendix: sweeps", "ext19_all.json, extc8ft_19_all.json"),
    ("Appendix: analytical v1", "scores{,c8}ana_20_*.json"),
]


def jobs_for(fname, lines):
    """producing jobs of a score file from the job-history snapshot"""
    stem = os.path.basename(fname)[:-5]
    out = []
    for l in lines:
        f = l.split("|", 4)                  # the submit line itself contains "|" (--constraint=a100|h100)
        if len(f) < 5:
            continue
        sub = f[4]
        hit = stem in sub
        m = re.match(r"scores(c8_)?(19|20)_part(\d)$", stem)
        if m and "runq.sh" in sub:
            hit = (f"SITE={m[2]}," in sub and f"PART={m[3]}," in sub and ("SRC=c8" in sub) == bool(m[1]))
        m = re.match(r"scores(c8)?ana(v2|v3)_20_sel$", stem)
        if m:
            hit = f"scores{m[1] or ''}ana${{v}}_20_sel" in sub
        if hit:
            env = " ".join(sorted(set(re.findall(r"\b(RECO=\w+|SRC=\w+|BG_NS=\d+|ANA_HIT=\w+|RECON_DIR=\S+|PART=\d)", sub))))
            env = env.replace(f"{LAB}/tmp_rings/", "")
            out.append(f"{f[0]} `{f[1]}` {f[2][:10]} {f[3].split()[0]}" + (f" ({env})" if env else ""))
    return out


def score_files():
    lines = open(f"{D}/sacct_nov.txt").read().splitlines() if os.path.exists(f"{D}/sacct_nov.txt") else []
    pats = ["scores19_done.json", "scores19_part*.json", "scoresbg1_19_all.json", "scoresbg10_19_all.json",
            "scoresft_19_all.json", "ext19_all.json", "scoresc8ft_19_all.json", "scoresc8ftbg1_19_all.json",
            "scoresc8ftbg10_19_all.json", "extc8ft_19_all.json", "scoresd_20_all.json", "scoresdbg1_20_all.json",
            "scoresdbg10_20_all.json", "extd_20_all.json", "scoresc8ft_20_all.json", "scoresc8ftbg1_20_all.json",
            "scoresc8ftbg10_20_all.json", "extc8ft_20_all.json", "scoresana_20_all.json", "scoresana_20_zfix*.json",
            "scoresc8ana_20_all.json", "scoresc8ana_20_zfix.json", "scoresanav2_20_sel.json", "scoresanav3_20_sel.json",
            "scoresc8anav2_20_sel.json", "scoresc8anav3_20_sel.json"]
    rows = []
    for pat in pats:
        for p in sorted(glob.glob(f"{D}/{pat}")):
            d = json.load(open(p))
            n0 = next(iter(d.values()), {})
            rows.append((os.path.basename(p), len(d), n0.get("n", "?"), "w1_cnt" in n0, jobs_for(p, lines)))
    return rows


def layouts(run):
    out = []
    for lst in (f"list{run}ns.txt", f"ext{run}.txt"):
        for item in open(f"{D}/{lst}").read().split():
            name, path = item.split("=", 1)
            out.append((lst, name, path))
    return out


def opt_info(path):
    d = os.path.dirname(path)
    f = os.path.join(d, "optimize_log.json")
    if not os.path.exists(f):
        return None
    L = json.load(open(f))
    c = L.get("config", {})
    init = str(c.get("init", ""))
    init = "uniform grid" if init.endswith("ugrid.pt") else ("centre" if init == "center" else os.path.basename(init))
    sl = [(e["epoch"], e["U_score"]) for e in L.get("score_log", []) if "U_score" in e]
    best = max(sl, key=lambda t: t[1]) if sl else (None, None)
    bg = c.get("bg_window_ns") or 0
    return dict(folder=os.path.basename(d), optim=c.get("optim", "adam (option not recorded)"),
                lbfgs=len(L.get("lbfgs_log") or []), response=c.get("response", "kernel (not recorded)"),
                init=init, steps=c.get("epochs"), batch=c.get("batch_events", c.get("n_events")), lr=c.get("lr"),
                lr_final=c.get("lr_final"), bg=f"{bg / 1000:g} us" if bg else "none", gate=c.get("gate", "none"),
                counting=c.get("counting", "none"), seed=c.get("seed"), best=best[0], last=sl[-1][0] if sl else None)


def main():
    o = ["# Paper 2 provenance", "",
         "Generated by `scripts/paper2/provenance.py` from the layout lists, the runs' `optimize_log.json`, the score "
         f"JSONs and the job-history snapshot in `{D}`. Repository: TambOpt, branch `cleanup`.", ""]
    o += ["## Code", "", "| Script | Role |", "|---|---|"] + [f"| `{p}` | {r} |" for p, r in CODE] + [""]
    o += ["## Run inputs", ""]
    for r, (nm, loc) in RUNS.items():
        o += [f"**{nm} (run {r})**: run folder `{loc}`", "",
              f"- flow-matching corpus: `{loc}/v6_run_00/cashed_showers_tau_dual.pt` (5,000 held-out showers scored)",
              f"- DeepSets recon: `{loc}/test_v6_run_03_recentered_deepsets`",
              f"- CORSIKA 8 recon (fine-tuned): `{LAB}/tmp_rings/c8ft/ft{r}/recon.pt` (scripts/paper2/ft_c8.py)",
              f"- CORSIKA 8 test showers: `{LAB.replace('zdimitrov', 'hhanif')}/tambo_simulations_for_training/h5_files_v3/"
              "combined_{electrons,muons,photons}_test.h5`, placed at the run's tau with the closest direction (< 3 deg)",
              f"- run guard: `slurm/run{r}_guard.sh` (sets the run's constants)", ""]
    o += ["## Paper elements", "", "| Element | Score files (in the score folder) |", "|---|---|"]
    o += [f"| {e} | {s} |" for e, s in ELEMENTS] + [""]
    o += ["## Score files", "", "| File | Layouts | Showers | Triggered metrics | Producing jobs (id, name, submitted, state, "
          "settings) |", "|---|---|---|---|---|"]
    for f, nl, ns, trig, jobs in score_files():
        o.append(f"| `{f}` | {nl} | {ns} | {'yes' if trig else 'no (all showers)'} | "
                 + ("<br>".join(jobs) if jobs else "not in the job snapshot") + " |")
    o += ["", "`scores19_done.json`: the 5 layouts scored by job 58272514 (`s19`, scripts run.sh, SITE=19, writes "
          "scores19.json; cancelled 2026-11-06 and renamed), then split over the four `runq.sh` part jobs.",
          "", "Score files written by more than one job: the scoring skips layouts already in the file, so a "
          "timed-out job's layouts were kept and the resubmission scored the rest. `*_zfix*` files rescored the "
          "analytical v1 layouts first run before the analytic_reco fix and override the `_all` entries.", ""]
    o += ["## Fixed layouts", "", "| Folder | Written by |", "|---|---|"]
    o += [f"| `{k}` | {v} |" for k, v in LAYOUT_WRITERS.items()] + [""]
    for r, (nm, loc) in RUNS.items():
        rows = layouts(r)
        fixed = [(l, n, p) for l, n, p in rows if "test_v6_run_04" not in p]
        opt = [(l, n, p) for l, n, p in rows if "test_v6_run_04" in p]
        o += [f"### {nm} (run {r}): fixed layouts", "", "| Name | List | File |", "|---|---|---|"]
        o += [f"| {n} | {l} | `{p.replace(loc + '/', '<RUN>/').replace(LAB + '/', '')}` |" for l, n, p in fixed] + [""]
        o += [f"### {nm} (run {r}): optimisation runs", "",
              f"Run folders: `{loc}/test_v6_run_04_optimize_lbfgs_kernel_<suffix>/`; layout used: `layout_best.pt` (best "
              "scoring-set utility). Utility from the run name (the environment variables are not stored in the config).",
              "", "| Name | Optimizer | L-BFGS rounds | Response | Start | Steps x batch | lr | Background | Gate | "
              "Counting | Best step / last |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for l, n, p in opt:
            i = opt_info(p)
            if i is None:
                o.append(f"| {n} | (no optimize_log.json) | | | | | | | | | |")
                continue
            lr = f"{i['lr']}" + (f" -> {i['lr_final']}" if i["lr_final"] else "")
            o.append(f"| {n[4:]} | {i['optim']} | {i['lbfgs']} | {i['response']} | {i['init']} | {i['steps']} x "
                     f"{i['batch']} | {lr} | {i['bg']} | {i['gate']} | {i['counting']} | {i['best']} / {i['last']} |")
        o.append("")
    c8 = sorted(glob.glob(f"{V6}/20_malata_showerframe/test_v6_run_04_optimize_lbfgs_kernel_c8ana_*"))
    o += ["## Running (not in the paper yet)", "",
          "CORSIKA 8 + differentiable analytical v1 runs (Malata, 1 deg trigger utility, --pool c8 --recon-type analytic): "
          "jobs c8ana_ugrid / c8ana_center, logs `tmp_rings/c8opt/opt_{ugrid,center}.out`, run folders "
          + (", ".join(f"`{os.path.basename(x)}`" for x in c8) if c8 else "not created yet") + ".", ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(o))
    print(f"wrote {os.path.abspath(args.out)} ({len(o)} lines)")


main()
