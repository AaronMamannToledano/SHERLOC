"""Summary + paired Nadeau-Bengio tests (vs SHERLOC calibrated, response longitudinal) on the splits each method has."""
from pathlib import Path
import pandas as pd
from benchmark import nb_paired
O = Path("results/benchmark_response_longitudinal")
ref = pd.read_csv("results/final/SHERLOC_calibrated_response_longitudinal_LandmarkOS_C2_metrics_per_split.csv")
ref = ref.assign(seed=range(len(ref))) if "seed" not in ref else ref
runs = {"SHERLOC calibrated (scale 2)": ref,
        "SHERLOC (scale 1)": pd.read_csv("results/final/SHERLOC_response_longitudinal_LandmarkOS_C2_metrics_per_split.csv")}
names = {"Featurewise_Transformer": "Feature-wise Transformer", "Temporal_Transformer": "Temporal Transformer",
         "RNN": "RNN", "GRU": "GRU", "MLP": "MLP (DeepSurv)", "RSF": "RSF", "ridge": "Ridge Cox (repo, unscaled)",
         "lasso": "Lasso Cox (repo, unscaled)", "elastic_net": "Elastic-net Cox (repo, unscaled)",
         "ridge_scaled": "Ridge Cox (standardised)", "lasso_scaled": "Lasso Cox (standardised)",
         "elastic_net_scaled": "Elastic-net Cox (standardised)", "FPCRF": "FPCRF", "JointModel": "Joint model (JMbayes2)", "JointModel_v2": "Joint model v2 (JMbayes2, 7 markers)",
         "JointModel_v2_fast": "Joint model v2 fast (noise SD 0.2, 6000 it.)"}
for k, n in names.items():
    f = O / f"{k}_metrics_per_split.csv"
    if f.exists():
        runs[n] = pd.read_csv(f)
rows = []
for n, r in runs.items():
    r = r.assign(seed=range(len(r))) if "seed" not in r else r
    row = {"method": n, "n_splits": len(r)}
    for m in ["C_index_Uno", "iAUC", "IBS", "calib_err_24m"]:
        row[m] = r[m].mean()
    if not n.startswith("SHERLOC calibrated"):
        a = ref.set_index("seed").loc[r.seed]
        for m in ["C_index_Uno", "iAUC", "IBS"]:
            d, p = nb_paired(a[m].values, r[m].values, 0.3)
            better = (a[m].values > r[m].values).mean() if m != "IBS" else (a[m].values < r[m].values).mean()
            row[f"d_{m}_SHERLOCcal_minus"] = d; row[f"p_{m}"] = p; row[f"frac_SHERLOCcal_better_{m}"] = better
    rows.append(row)
s = pd.DataFrame(rows); s.to_csv(O / "summary_with_paired_tests.csv", index=False)
pd.set_option("display.width", 250)
print(s[["method", "n_splits", "C_index_Uno", "iAUC", "IBS", "calib_err_24m", "d_C_index_Uno_SHERLOCcal_minus", "p_C_index_Uno",
         "frac_SHERLOCcal_better_C_index_Uno", "p_iAUC", "p_IBS"]].round(4).to_string(index=False))
