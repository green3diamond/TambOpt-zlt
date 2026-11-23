"""Rescore every layout of the paper 2 background families at another cosmic background rate (TAMBOPT_BG_RATE,
modules/optimize/background.py). The default 1.826e-6 /m^2/ns is SWGO's all-particle rate at ~5 km; the muon rate at
3-4 km is ~4e-7 (audit). Same scorer and settings as res3 (score_all.py via m5k_rescore.env), only the rate differs.

    python bgrate_rescore.py lists <res3> <out>   per background family: all layouts its res3 files hold
                                                  -> <out>/bgrate/<fam>.txt; prints "family|site|environment|n"
    python bgrate_rescore.py build <res3> <out>   <out> = copy of <res3> with the background families' records
                                                  replaced by <out>/bgrate/<fam>.json (also the *_e02.json files)
Submit with bgrate_rescore.sh.
"""
import glob, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5k_rescore import FAM, env, files  # noqa: E402

BG = {site: [f for f in fams if "bg" in f] for site, fams in FAM.items()}


def fid(fam, site):
    return f"{fam}_{site}"


def lists(D, O):
    os.makedirs(f"{O}/bgrate", exist_ok=True)
    for site, fams in BG.items():
        for fam in fams:
            rec = {}
            for f in files(D, fam, site):
                rec.update({n: r for n, r in json.load(open(f)).items() if r and "smoke" not in n})
            if not rec:
                continue
            open(f"{O}/bgrate/{fid(fam, site)}.txt", "w").write("".join(f"{n}={r['path']}\n" for n, r in sorted(rec.items())))
            print(f"{fid(fam, site)}|{site}|{env(fam, site)}|{len(rec)}")


def build(D, O):
    for f in glob.glob(f"{D}/*"):
        if os.path.isfile(f) and not os.path.exists(f"{O}/{os.path.basename(f)}"):
            shutil.copy2(f, O)
    n_rep = 0
    for site, fams in BG.items():
        for fam in fams:
            lf, nf = f"{O}/bgrate/{fid(fam, site)}.txt", f"{O}/bgrate/{fid(fam, site)}.json"
            if not os.path.exists(lf):
                continue
            want = [l.split("=", 1)[0] for l in open(lf).read().split()]
            new = json.load(open(nf)) if os.path.exists(nf) else {}
            miss = [n for n in want if n not in new]
            if miss:
                sys.exit(f"{nf}: {len(miss)} of {len(want)} layouts not scored yet ({miss[:3]} ...)")
            for f in files(D, fam, site):
                d = json.load(open(f))
                hit = [n for n in d if n in new]
                for n in hit:
                    d[n] = new[n]
                json.dump(d, open(f"{O}/{os.path.basename(f)}", "w"), indent=1)
                n_rep += len(hit)
                print(f"{os.path.basename(f)}: {len(hit)} replaced")
    print(f"{n_rep} records replaced")


if __name__ == "__main__":
    {"lists": lists, "build": build}[sys.argv[1]](sys.argv[2], sys.argv[3])
