"""Triggered-only view of score_all.py records: accuracies (w1, w2.5, w5, med) taken over the triggered showers
(>= 3 detectors with expected count mu >= 1 and a time; keys *_cnt). Fewer than MIN_N triggered showers -> NaN.
Records without *_cnt keys (older scoring) are left as they are (all showers); ntrig = NaN for them."""
MIN_N = 20
ACC = ("w1", "w2.5", "w5", "med")


def trig_view(d, acc=True):
    """acc=False: keep the all-shower accuracies, only add ntrig."""
    out = {}
    for n, r in d.items():
        r = dict(r)
        if "w1_cnt" in r:
            ok = r["n_cnt"] >= MIN_N
            for k in (ACC if acc else ()):
                r[k] = r[k + "_cnt"] if ok else float("nan")
            r["ntrig"] = float(r["n_cnt"])
        else:
            r["ntrig"] = float("nan")
        out[n] = r
    return out
