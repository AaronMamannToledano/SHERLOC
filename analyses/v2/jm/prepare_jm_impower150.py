"""
prepare_jm_impower150.py — inputs of the joint model v2 (train_jm_impower150.R) for IMpower150 OS, 4 / 3 / 2 timepoints.
Same patients, splits (df.sample(frac=0.3, random_state=4*seed)), clinical covariates and frozen MSK-CHORD encoder as
every other method (posthoc_benchmark.tabular). Markers, chosen with the PADA-1 v2 rule:
  3 most frequent genes at BL (gene-level max VAF, %): TP53, KRAS, SPTA1
  mean VAF (mean_of_AF x 100, %) and number of drivers (n_kl)
  the 2 dimensions of the frozen encoder at each timepoint
time = nominal months from BL (21-day cycles: C2D1 0.69, C3D1 1.38, C4D1 2.07; no sample dates in IMpower150);
t_LM = time of the landmark sample (last timepoint); OS_time_BL = landmark OS + t_LM.
Usage: python prepare_jm_impower150.py --tp 4tp     -> results/jm_impower150/<tp>/IMP150_JM_long.csv, IMP150_JM_test_indices.csv
"""
import argparse
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analyses/v2
import posthoc_benchmark as B

CYCLE = 21 / 30.44
ap = argparse.ArgumentParser(); ap.add_argument("--tp", required=True, choices=["4tp", "3tp", "2tp"])
a = ap.parse_args()
cfg, win, df, lat, genes, gene_cols, panel_cols, cln_cols = B.tabular(f"impower150_{a.tp}")
tps = cfg["timepoints"]; T = len(tps)
top3 = (df[[f"{g}_BL" for g in genes]] > 0).mean().sort_values(ascending=False).index.str[:-3][:3].tolist()
L = lat.to_numpy().reshape(len(df), 2, T)                    # encoder output is (n, dim, timepoint)
t_lm = (T - 1) * CYCLE
rows = []
for k, tp in enumerate(tps):
    d = pd.DataFrame({"index": np.arange(len(df)), "timepoint": tp, "time": k * CYCLE})
    for g in top3:
        d[f"gene_{g}"] = df[f"{g}_{tp}"].to_numpy()
    d["mean_VAF"] = 100 * df[f"{tp}_mean_of_AF"].to_numpy()
    d["nb_driver"] = df[f"{tp}_n_kl"].to_numpy()
    d["enc0"], d["enc1"] = L[:, 0, k], L[:, 1, k]
    rows.append(d)
static = pd.DataFrame({"index": np.arange(len(df)), "OS_event": df.OS_event.to_numpy(),
                       "OS_time_BL": (df.OS_rebaseline + t_lm).to_numpy(), "t_LM": t_lm,
                       "OS_landmark": df.OS_rebaseline.to_numpy()})
for c in cln_cols:
    static[c] = df[c].to_numpy()
long = pd.concat(rows).merge(static, on="index").sort_values(["index", "time"])
out = Path(f"results/jm_impower150/{a.tp}"); out.mkdir(parents=True, exist_ok=True)
long.to_csv(out / "IMP150_JM_long.csv", index=False)
pos = {p: i for i, p in enumerate(df.index)}
idx = [[pos[p] for p in df.sample(frac=0.3, random_state=4 * s).index] for s in range(100)]
pd.DataFrame(idx).to_csv(out / "IMP150_JM_test_indices.csv", index=False)
print(a.tp, long.shape, "patients", len(df), "deaths", int(df.OS_event.sum()), "genes", top3, "t_LM", round(t_lm, 2))
