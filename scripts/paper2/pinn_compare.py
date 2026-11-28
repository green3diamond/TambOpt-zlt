"""PINN test vs the paper: triggered showers, share within 1 / 5 deg [%] and median [deg] per layout.
CORSIKA 8 and flow matching. Columns: DeepSets (paper, res3 scoresc8ft_20_all / scoresd_20_all), analytical v1 (paper, TamboDirReco port), v1 and v3 in LlhamaFit's
model (tmp_rings/llhama/fit), and the PINN fine-tunes (tmp_rings/pinn/score_c8_<run>.json).

    python3 scripts/paper2/pinn_compare.py [tmp_rings] > tmp_rings/pinn/compare.md
"""
import glob, json, sys

T = sys.argv[1] if len(sys.argv) > 1 else "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings"


def merged(pat):
    o = {}
    for p in sorted(glob.glob(pat)):
        for n, v in json.load(open(p)).items():
            o.setdefault(n, {}).update(v)
    return o


COLS = {"c8": [("DeepSets (paper)", merged(f"{T}/res3/scoresc8ft_20_all.json")),
               ("v1 (paper)", merged(f"{T}/res3/scoresc8ana_20_*.json")),
               ("v1 LlhamaFit", merged(f"{T}/llhama/fit/c8_llhama_v1_*.json")),
               ("v3 LlhamaFit", merged(f"{T}/llhama/fit/c8_llhama_v3_*.json"))],
        "fm": [("DeepSets (paper)", merged(f"{T}/res3/scoresd_20_all.json")),
               ("v1 (paper)", merged(f"{T}/res3/scoresana_20_*.json")),
               ("v1 LlhamaFit", merged(f"{T}/llhama/fit/fm_llhama_v1_*.json")),
               ("v3 LlhamaFit", merged(f"{T}/llhama/fit/fm_llhama_v3_*.json"))]}
for src in COLS:
    COLS[src] += [(f"PINN {r}", merged(f"{T}/pinn/score_{src}_{r}.json")) for r in ("lam0c", "lam0.01c", "lam0.3")]
names = [a.split("=", 1)[0] for a in open(f"{T}/llhama/list.txt").read().split()]
for src, label in (("c8", "CORSIKA 8"), ("fm", "flow matching")):
    for k, s, fmt, title in (("w1", 100, "{:.1f}", "share within 1 deg [%]"),
                             ("w5", 100, "{:.1f}", "share within 5 deg [%]"), ("med", 1, "{:.2f}", "median [deg]")):
        print(f"\n### {title}, {label}, triggered showers\n")
        print("| Layout | " + " | ".join(c[0] for c in COLS[src]) + " |")
        print("|---" * (len(COLS[src]) + 1) + "|")
        for n in names:
            print(f"| {n} | " + " | ".join(fmt.format(s * d[n][f"{k}_cnt"]) if n in d and f"{k}_cnt" in d[n] else "-"
                                           for _, d in COLS[src]) + " |")
