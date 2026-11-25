"""Analytical reconstruction, paper (TamboDirReco port) vs LlhamaFit model, on the layouts of Table r20_anav.

Triggered showers (>= 3 detectors with mu >= 1), share within 1 and 5 deg [%] and median [deg], flow matching [CORSIKA 8].
Paper v1: res3 scores{,c8}ana_20_*.json; the other columns: ana_fit_hits.py on the saved hits,
tmp_rings/llhama/fit/{fm,c8}_{tdr,llhama}_{v1,v3}_<layout>.json.

    python3 scripts/paper2/llhama_compare.py [tmp_rings] > tmp_rings/llhama/compare.md
"""
import glob
import json
import sys

T = sys.argv[1] if len(sys.argv) > 1 else "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings"
L = f"{T}/llhama"


def merged(pat):
    o = {}
    for p in sorted(glob.glob(pat)):
        for n, v in json.load(open(p)).items():
            o.setdefault(n, {}).update(v)
    return o


def load(f):
    try:
        return json.load(open(f))
    except FileNotFoundError:
        return {}


F = lambda src, m, v: merged(f"{L}/fit/{src}_{m}_{v}_*.json")
COLS = [("DeepSets", merged(f"{T}/res3/scoresd_20_all.json"), merged(f"{T}/res3/scoresc8ft_20_all.json")),
        ("v1 paper", merged(f"{T}/res3/scoresana_20_*.json"), merged(f"{T}/res3/scoresc8ana_20_*.json")),
        ("v1 TamboDirReco (CPU rerun)", F("fm", "tdr", "v1"), F("c8", "tdr", "v1")),
        ("v1 LlhamaFit", F("fm", "llhama", "v1"), F("c8", "llhama", "v1")),
        ("v3 TamboDirReco", F("fm", "tdr", "v3"), F("c8", "tdr", "v3")),
        ("v3 LlhamaFit", F("fm", "llhama", "v3"), F("c8", "llhama", "v3"))]
names = [a.split("=", 1)[0] for a in open(f"{L}/list.txt").read().split()]


def cell(fm, c8, n, k, s, fmt):
    g = lambda d: fmt.format(s * d[n][f"{k}_cnt"]) if n in d and f"{k}_cnt" in d[n] else "-"
    return f"{g(fm)} [{g(c8)}]"


for k, s, fmt, title in (("w1", 100, "{:.1f}", "share within 1 deg [%]"), ("w5", 100, "{:.1f}", "share within 5 deg [%]"),
                         ("med", 1, "{:.2f}", "median [deg]")):
    print(f"\n### {title}, triggered showers, flow matching [CORSIKA 8]\n")
    print("| Layout | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---" * (len(COLS) + 1) + "|")
    for n in names:
        print(f"| {n} | " + " | ".join(cell(fm, c8, n, k, s, fmt) for _, fm, c8 in COLS) + " |")
