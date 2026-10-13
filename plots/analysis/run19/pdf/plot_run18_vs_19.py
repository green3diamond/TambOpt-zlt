"""Run 18 (6 km plane, relative-time reltime1 recon, 1,536 showers) vs run 19 (12 km plane, reltime1, 5,000 showers):
share within 1 / 2.5 / 5 deg vs centred-disk radius per detector count, and the full-plane grid vs detector count."""
import json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
P = os.path.dirname(os.path.abspath(__file__))
V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/"
Z18 = V6 + "18_plane45S_6km_showerframe_200k/diagnostics/zoo_compare/reltime1_wd003/"
T19 = V6 + "19_plane45S_12km_showerframe_400k/diagnostics/tests/"
TH = (1, 2.5, 5); KEY = {1: "within1", 2.5: "within2p5", 5: "within5"}
NC = {10: "#c23b6b", 20: "#eb6834", 50: "#1baf7a", 100: "#2a78d6"}


def d18(th):
    out = {}
    for st in ("disks_n10", "core_test_ndet", "angleonly"):
        for k, v in json.load(open(f"{Z18}{st}/scores_pad_{th:g}deg.json")).items():
            m = re.fullmatch(r"disk_r(\d+)(?:_n(\d+))?", k)
            if m:
                out[(int(m.group(2) or 100), int(m.group(1)))] = 100 * v["frac_good_t"]
    return out


S19 = json.load(open(T19 + "scores.json"))["scores"]
def d19(th):
    out = {}
    for k, v in S19.items():
        m = re.fullmatch(r"disk_r(\d+)_n(\d+)", k)
        if m:
            out[(int(m.group(2)), int(m.group(1)))] = 100 * v[KEY[th]]
    return out


fig, axs = plt.subplots(2, 3, figsize=(21, 11.5))
for j, th in enumerate(TH):
    a, b = d18(th), d19(th)
    ax = axs[0, j]
    for n, col in NC.items():
        r = sorted(rr for nn, rr in a if nn == n); ax.plot([x / 1e3 for x in r], [a[(n, x)] for x in r], "o--", color=col, lw=1.4, ms=4, alpha=0.8)
        r = sorted(rr for nn, rr in b if nn == n); ax.plot([x / 1e3 for x in r], [b[(n, x)] for x in r], "o-", color=col, lw=2.2, ms=5, label=f"{n} det.")
    ax.axvline(2.1, color="#52514e", lw=0.8, ls="--"); ax.axvline(4.2, color="#52514e", lw=0.8, ls=":")
    ax.set(xscale="log", xlabel="disk radius [km] (centred)", ylabel=f"within {th:g}° [%]",
           title=f"Within {th:g}° · solid run 19 (12 km), dashed run 18 (6 km)")
    ax.grid(alpha=0.25, lw=0.5)
axs[0, 0].legend(frameon=False, fontsize=10, title="N–S half-width: 2.1 km (18, dashed line), 4.2 km (19, dotted)", title_fontsize=8.5)
G18 = json.load(open(Z18 + "detcount/detector_count_grid.json")); G19 = json.load(open(T19 + "detcount/detector_count_grid.json"))
GK = {1: "good1_angle", 2.5: "good25_angle", 5: "good5_angle"}
ns = sorted(int(k) for k in G19)
for j, th in enumerate(TH):
    ax = axs[1, j]
    ax.plot(ns, [100 * G18[str(n)][GK[th]] for n in ns], "s--", color="#9b9a95", lw=1.8, label="run 18 grid (6 km plane)")
    ax.plot(ns, [100 * G19[str(n)][GK[th]] for n in ns], "o-", color="#2a78d6", lw=2.2, label="run 19 grid (12 km plane)")
    ax.plot([n * 4 for n in ns if n * 4 <= 100], [100 * G18[str(n)][GK[th]] for n in ns if n * 4 <= 100], "^:", color="#eb6834", lw=1.5,
            label="run 18 grid with N/4 detectors, drawn at N (same detectors per area)")
    ax.set(xscale="log", xlabel="detectors in the full-plane grid", ylabel=f"within {th:g}° [%]", title=f"Full-plane grid · within {th:g}°")
    ax.set_xticks(ns); ax.set_xticklabels(ns); ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=9)
fig.suptitle("Run 18 vs run 19 (both: DeepSets, reltime1 input, wd 0.03; each scored on its own held-out showers)", fontsize=14)
fig.tight_layout(); fig.savefig(f"{P}/run18_vs_run19.png", dpi=105, facecolor="white", bbox_inches="tight")
print("[write]", f"{P}/run18_vs_run19.png")
