"""Score folder with the analytical reconstruction in LlhamaFit's model: <out> = <res> with every scores{,c8}ana*_20
file replaced by the CPU fits on the saved hits (ana_fit_hits.py, tmp_rings/llhama/fit/<src>_llhama_<v>_<layout>.json).
v1 -> scores{,c8}ana_20_all.json, v2 / v3 -> scores{,c8}anav{2,3}_20_sel.json; the old zfix files are dropped (they
would override). Fields an analytical record of <res> has and the fit does not (utilities, e02) are NaN, as there.

    python3 scripts/paper2/llhama_build_res.py <res3> <res5> [tmp_rings/llhama]
"""
import glob, json, os, shutil, sys

res, out = sys.argv[1], sys.argv[2]
L = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(res)), "llhama")
if os.path.exists(out):
    sys.exit(f"{out} exists")
shutil.copytree(res, out)
old = sorted(glob.glob(f"{out}/scores*ana*_20_*.json"))
template = {}
for f in old:
    for r in json.load(open(f)).values():
        template.update({k: float("nan") for k, v in r.items() if isinstance(v, float)})
    os.remove(f)
for src, pre in (("fm", "scoresana"), ("c8", "scoresc8ana")):
    for v, name in (("v1", f"{pre}_20_all"), ("v2", f"{pre}v2_20_sel"), ("v3", f"{pre}v3_20_sel")):
        recs = {}
        for f in sorted(glob.glob(f"{L}/fit/{src}_llhama_{v}_*.json")):
            for n, r in json.load(open(f)).items():
                recs[n] = dict(template, **r)
        if recs:
            json.dump(recs, open(f"{out}/{name}.json", "w"), indent=1)
            print(f"{name}.json: {len(recs)} layouts")
print("removed:", ", ".join(os.path.basename(f) for f in old))
