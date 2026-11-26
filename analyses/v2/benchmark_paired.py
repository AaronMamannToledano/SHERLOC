"""benchmark_paired.py — paired SHERLOC-vs-method comparisons on the 100 benchmark splits (C-index, iAUC, IBS,
calibration error), Nadeau-Bengio corrected t-test. Output: results/posthoc_benchmark/paired_vs_sherloc.csv"""
import numpy as np, pandas as pd
from scipy import stats
from article_figures import load_bench, DS, METHODS

def nb_test(d, frac=0.3):
    d = np.asarray(d, float); n = len(d)
    se = np.sqrt(d.var(ddof=1) * (1 / n + frac / (1 - frac)))
    t = d.mean() / se
    return d.mean(), 2 * stats.t.sf(abs(t), n - 1)

rows = []
for ds, _ in DS:
    b = load_bench(ds)
    for var in ("raw", "recalibrated"):
        x = b[b.variant == var].set_index(["method", "seed"])
        sh = x.loc["SHERLOC"]
        for m, lab, fam in METHODS:
            if m == "SHERLOC" or m not in x.index.get_level_values(0):
                continue
            o = x.loc[m]; common = sh.index.intersection(o.index)
            r = {"dataset": ds, "variant": var, "method": lab, "n_splits": len(common)}
            for k in ("C_index_Uno", "iAUC", "IBS", "calib_err"):
                if k not in o or o[k].isna().all():
                    continue
                d = sh.loc[common, k] - o.loc[common, k]
                r[f"d_{k}"], r[f"p_{k}"] = nb_test(d)
                r[f"win_{k}"] = float((d < 0).mean() if k in ("IBS", "calib_err") else (d > 0).mean())
                r[f"SHERLOC_{k}"], r[f"method_{k}"] = sh.loc[common, k].mean(), o.loc[common, k].mean()
            rows.append(r)
P = pd.DataFrame(rows)
P.to_csv("results/posthoc_benchmark/paired_vs_sherloc.csv", index=False)
pd.set_option("display.width", 250)
print(P[P.variant == "raw"][["dataset", "method", "d_C_index_Uno", "p_C_index_Uno", "win_C_index_Uno", "d_iAUC",
                              "p_iAUC", "win_iAUC"]].round(3).to_string(index=False))
print(P[P.variant == "recalibrated"][["dataset", "method", "d_IBS", "p_IBS", "win_IBS"]].round(4).to_string(index=False))

# IBS: recalibrated SHERLOC vs each competitor in its better variant (raw or recalibrated)
rows = []
for ds, _ in DS:
    b = load_bench(ds)
    sh = b[(b.method == "SHERLOC") & (b.variant == "recalibrated")].set_index("seed")
    for m, lab, fam in METHODS:
        if m == "SHERLOC":
            continue
        o = b[b.method == m]
        v = o.groupby("variant").IBS.mean().idxmin()
        o = o[o.variant == v].set_index("seed"); common = sh.index.intersection(o.index)
        for k in ("IBS", "calib_err"):
            if k not in o or o[k].isna().all():
                continue
            d = sh.loc[common, k] - o.loc[common, k]
            dd, p = nb_test(d)
            rows.append({"dataset": ds, "method": lab, "metric": k, "variant_used": v, "SHERLOC": sh.loc[common, k].mean(),
                         "method_value": o.loc[common, k].mean(), "diff": dd, "p": p, "SHERLOC_better": float((d < 0).mean())})
Q = pd.DataFrame(rows); Q.to_csv("results/posthoc_benchmark/paired_ibs_best_variant.csv", index=False)
print(Q.round(4).to_string(index=False))
