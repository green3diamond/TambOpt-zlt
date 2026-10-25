"""Merge score_knn.py outputs of held-out chunks (--event-start) into one: every metric is a shower-weighted mean over
the chunks (n_showers summed). Medians (ang_med, de_med, de_med_trig) become weighted means of the chunk medians
(approximate). Prints the tables of the swgo_trigger validation.

    python plots/analysis/run19/triplets/merge_scores.py <out.json> <chunk.json> [<chunk.json> ...]
"""
import json
import sys

out, files = sys.argv[1], sys.argv[2:]
chunks = [json.load(open(f)) for f in files]
merged = {}
for name in chunks[0]["scores"]:
    merged[name] = {}
    for rd, by_w in chunks[0]["scores"][name].items():
        if not isinstance(by_w, dict):
            continue
        merged[name][rd] = {}
        for w in by_w:
            rows = [c["scores"][name][rd][w] for c in chunks]
            n = [r["n_showers"] for r in rows]
            tot = sum(n)
            m = dict(n_showers=tot, n_det=rows[0]["n_det"])
            for k, v in rows[0].items():
                if k in m or not isinstance(v, (int, float)):
                    continue
                vals = [(r[k], ni) for r, ni in zip(rows, n) if r.get(k) == r.get(k)]   # skip NaN
                m[k] = sum(x * ni for x, ni in vals) / max(sum(ni for _, ni in vals), 1)
            merged[name][rd][w] = m
json.dump(dict(chunks=files, event_starts=[c.get("event_start") for c in chunks], windows_ns=chunks[0]["windows_ns"],
               scores=merged), open(out, "w"), indent=1)
print(f"[write] {out}  ({len(files)} chunks)")

for rd in ("none", "knn"):
    for w in merged[next(iter(merged))][rd]:
        n = merged[next(iter(merged))][rd][w]["n_showers"]
        print(f"\n== readout {rd}, window {w} ns  ({n} showers)")
        print(f"{'layout':26s} {'1deg':>6s} {'2.5':>6s} {'5':>6s} {'oldU':>6s} {'newU':>7s} {'newUstep':>8s} {'trigP':>6s}")
        for name, v in merged.items():
            r = v[rd][w]
            print(f"{name:26s} {100*r['notrig_1']:6.1f} {100*r['notrig_2.5']:6.1f} {100*r['notrig_5']:6.1f} "
                  f"{r['u_two_tier']:6.3f} {r['u_swgo']:7.4f} {r['u_swgo_step']:8.4f} {100*r['trig_poisson']:6.1f}")
