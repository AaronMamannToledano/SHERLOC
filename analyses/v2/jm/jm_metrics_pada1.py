"""
jm_metrics_pada1.py — SHERLOC metrics for the joint model from train_jm_pada1.R predictions.
pred_CIF(t) is the cumulative event probability from BL, conditional on survival to Tstart = 1.7 months.
For each test patient the survival curve on the landmark scale (months after C2) is S(u) = 1 - CIF(t_C2 + u),
and the risk score is CIF at 24 months after C2 (alternatives give the same ranking up to ties).
Metrics: Uno C (tau 36), iAUC and IBS 3-36 months, calibration error at 24 months — same code as all methods.
Usage: python jm_metrics_pada1.py [pred_dir] [out_csv]   (defaults: v1 folder and JointModel_metrics_per_split.csv)
"""
from pathlib import Path
import numpy as np, pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analyses/v2
from mtl_benchmark import metrics
D = Path("results/benchmark_response_longitudinal/jm")
P = Path(sys.argv[1]) if len(sys.argv) > 1 else D                     # folder with pred_split*.csv
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/benchmark_response_longitudinal/JointModel_metrics_per_split.csv"
long = pd.read_csv(D / "PADA1_JM_long.csv")
pat = long.drop_duplicates("index").set_index("index")
idx = pd.read_csv(D / "PADA1_JM_test_indices.csv")
rows = []
grid_lm = np.arange(0, 36.25, 0.25)
for f in sorted(P.glob("pred_split*.csv")):
    s = int(f.stem[-3:])
    pr = pd.read_csv(f)
    test = idx.iloc[s].astype(int).tolist()
    train = [i for i in pat.index if i not in set(test)]
    d_te = pat.loc[test].assign(OS_t=lambda d: d.OS_landmark, OS_e=lambda d: d.OS_event)
    d_tr = pat.loc[train].assign(OS_t=lambda d: d.OS_landmark, OS_e=lambda d: d.OS_event)
    piv = pr.pivot_table(index="time", columns="index", values="pred_CIF").sort_index()
    S = pd.DataFrame(index=grid_lm, columns=test, dtype=float)
    for i in test:
        tt = grid_lm + pat.loc[i, "t_C2"]
        S[i] = 1 - np.interp(tt, piv.index.to_numpy(), piv[i].to_numpy())
    risk = 1 - S.loc[24.0].to_numpy()
    m = metrics(d_tr, d_te, risk, S, "OS", (3.0, 36.0, 24.0))
    rows.append({"seed": s, "C_index_Uno": m["C_index_Uno"], "iAUC": m["iAUC"], "IBS": m["IBS"],
                 "calib_err_24m": m["calib_err"], "pred_S24_range": m["pred_S_range"]})
r = pd.DataFrame(rows)
r.to_csv(OUT, index=False)
print(len(r), "splits"); print(r.drop(columns="seed").mean().round(4).to_dict())
