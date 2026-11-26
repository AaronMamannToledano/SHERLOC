"""jm_summary.py — joint model (JMbayes2 v2) vs SHERLOC on the 100 benchmark splits, all datasets.

Reads the per-split joint-model metrics (jm_metrics_*.py) and SHERLOC's per-split metrics from the post-hoc
benchmark (C-index / iAUC as trained, IBS after out-of-fold recalibration), and writes
results/jm_summary_all_datasets.csv (used by gen_tables.py). Run from analyses/v2:  python jm/jm_summary.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats


def nb_paired(a, b, test_frac):
    """Mean paired difference and Nadeau-Bengio corrected t-test p-value (same as benchmark.nb_paired)."""
    d = np.asarray(a) - np.asarray(b); n = len(d)
    se = np.sqrt(d.var(ddof=1) * (1 / n + test_frac / (1 - test_frac)))
    return d.mean(), 2 * (1 - stats.t.cdf(abs(d.mean() / se), n - 1))

TF = 0.3
JM = {"impower150_4tp": "results/jm_impower150/4tp/JointModel_v2_metrics_per_split.csv",
      "impower150_3tp": "results/jm_impower150/3tp/JointModel_v2_metrics_per_split.csv",
      "impower150_2tp": "results/jm_impower150/2tp/JointModel_v2_metrics_per_split.csv",
      "pada1": "results/benchmark_response_longitudinal/JointModel_v2_metrics_per_split.csv"}

rows = []
for ds, f in JM.items():
    j = pd.read_csv(f).set_index("seed")
    sh = pd.concat(pd.read_csv(p) for p in sorted(Path(f"results/posthoc_benchmark/{ds}").glob("SHERLOC_seeds*.csv")))
    raw = sh[sh.variant == "raw"].set_index("seed"); rec = sh[sh.variant == "recalibrated"].set_index("seed")
    s = j.index.intersection(raw.index)
    r = {"dataset": ds, "n": len(s), "C": j.loc[s, "C_index_Uno"].mean(), "iAUC": j.loc[s, "iAUC"].mean(),
         "IBS": j.loc[s, "IBS"].mean(), "calib": j.loc[s, [c for c in j.columns if c.startswith("calib")][0]].mean()}
    for k, a, b in [("C", raw.loc[s, "C_index_Uno"], j.loc[s, "C_index_Uno"]),
                    ("iAUC", raw.loc[s, "iAUC"], j.loc[s, "iAUC"]),
                    ("IBS", rec.loc[s, "IBS"], j.loc[s, "IBS"])]:
        d, p = nb_paired(a.to_numpy(), b.to_numpy(), TF)
        better = (a.to_numpy() > b.to_numpy()) if k != "IBS" else (a.to_numpy() < b.to_numpy())
        r.update({f"d{k}_SHERLOC_minus_JM": d, f"p_{k}": p, f"frac_SH_better_{k}": better.mean()})
    if "max_Rhat_all" in j:
        r.update({"Rhat_median": j.loc[s, "max_Rhat_all"].median(), "Rhat>1.1": int((j.loc[s, "max_Rhat_all"] > 1.1).sum())})
    rows.append(r)
pd.DataFrame(rows).to_csv("results/jm_summary_all_datasets.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())
