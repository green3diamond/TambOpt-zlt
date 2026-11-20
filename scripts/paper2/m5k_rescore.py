"""Paper 2 scores of the rerun optimizations (slurm/rerun_mean5k_submit.sh on branch rerun-mean5k: mean arrival time
without noise, 5,120 showers per step, plateau schedule; folders <old run folder>_m5k, the 12 km centre starts
continued in <old run folder>_m5kc). The scorer is unchanged (score_all.py, noise-free kernel, default time model).

    python m5k_rescore.py lists <res2> <res3>   per score family: the rerun layouts that family's res2 files hold,
                                                under their old keys with the new layout files -> <res3>/m5k/<fam>.txt;
                                                prints "family|site|environment" for the submitter
    python m5k_rescore.py build <res2> <res3>   <res3> = copy of <res2> with those layouts' records replaced by the
                                                new scores (<res3>/m5k/<fam>.json); *_e02.json files likewise

Layouts listed for a family that its res2 files do not hold are added to its first file.
A family is the prefix of a score file name (scoresd_20_all.json, scoresd_20_e02.json -> scoresd_20); only the
families make_tables.py reads are rescored.
"""
import glob, json, os, re, shutil, sys

LAB = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings"
FAM = {19: ["scores", "scoresbg1", "scoresbg10", "scoresbg1ng", "scoresbg10ng",
            "scoresc8ft", "scoresc8ftbg1", "scoresc8ftbg10", "scoresc8ftbg1ng", "scoresc8ftbg10ng"],
       20: ["scoresd", "scoresdbg1", "scoresdbg10", "scoresdbg1ng", "scoresdbg10ng",
            "scoresc8ft", "scoresc8ftbg1", "scoresc8ftbg10", "scoresc8ftbg1ng", "scoresc8ftbg10ng",
            "scoresana", "scoresc8ana", "scoresanav2", "scoresanav3", "scoresc8anav2", "scoresc8anav3"]}


def env(fam, site):
    s, e = fam[len("scores"):], []
    if "c8" in s:
        e.append("SRC=c8")
    if "c8ft" in s:
        e.append(f"RECON_DIR={LAB}/c8ft/ft{site}")
    m = re.search(r"ana(v\d)?", s)
    if m:
        e += [f"RECO={m.group(1) or 'v1'}", "ANA_HIT=mu"]
    m = re.search(r"bg(\d+)", s)
    if m:
        e.append(f"BG_NS={int(m.group(1)) * 1000}")
    if s.endswith("ng"):
        e.append("BG_GATE=none")
    return " ".join(e)


def files(D, fam, site):
    """score files of a family: <fam>_<site>_*.json (12 km flow matching: scores19_*.json)"""
    pat = f"{D}/scores{site}_*.json" if fam == "scores" else f"{D}/{fam}_{site}_*.json"
    return sorted(glob.glob(pat))


def new_path(p):
    d = os.path.dirname(p)
    for t in ("_m5kc", "_m5k"):
        if os.path.exists(f"{d}{t}/layout_best.pt"):
            return f"{d}{t}/layout_best.pt"
    return None


def fam_id(fam, site):
    return f"{fam}_{site}"


def lists(D, O):
    os.makedirs(f"{O}/m5k", exist_ok=True)
    for site, fams in FAM.items():
        for fam in fams:
            old = {}
            for f in files(D, fam, site):
                old.update(json.load(open(f)))
            pairs = {n: new_path(r["path"]) for n, r in old.items() if n.startswith("opt_") and "smoke" not in n}
            pairs = {n: p for n, p in pairs.items() if p}
            if not pairs:
                continue
            open(f"{O}/m5k/{fam_id(fam, site)}.txt", "w").write("".join(f"{n}={p}\n" for n, p in sorted(pairs.items())))
            print(f"{fam_id(fam, site)}|{site}|{env(fam, site)}|{len(pairs)}")


def build(D, O):
    for f in glob.glob(f"{D}/*"):
        if os.path.isfile(f) and not os.path.exists(f"{O}/{os.path.basename(f)}"):
            shutil.copy2(f, O)
    n_rep = 0
    for site, fams in FAM.items():
        for fam in fams:
            nf = f"{O}/m5k/{fam_id(fam, site)}.json"
            lf = f"{O}/m5k/{fam_id(fam, site)}.txt"
            if not os.path.exists(lf):
                continue
            want = [l.split("=", 1)[0] for l in open(lf).read().split()]
            new = json.load(open(nf)) if os.path.exists(nf) else {}
            miss = [n for n in want if n not in new]
            if miss:
                sys.exit(f"{nf}: {len(miss)} of {len(want)} layouts not scored yet ({miss[:3]} ...)")
            fs = files(D, fam, site)
            held = set().union(*(json.load(open(f)) for f in fs))
            for i, f in enumerate(fs):
                d = json.load(open(f))
                if i == 0:      # layouts this family had not scored (e.g. a new headline layout): into its first file
                    d.update({n: None for n in want if n not in held})
                hit = [n for n in d if n in new]
                for n in hit:
                    d[n] = new[n]
                json.dump(d, open(f"{O}/{os.path.basename(f)}", "w"), indent=1)
                n_rep += len(hit)
                print(f"{os.path.basename(f)}: {len(hit)} replaced")
    print(f"{n_rep} records replaced")


if __name__ == "__main__":
    {"lists": lists, "build": build}[sys.argv[1]](sys.argv[2], sys.argv[3])
