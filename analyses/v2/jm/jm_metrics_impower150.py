"""
jm_metrics_impower150.py — SHERLOC metrics for the joint model v2 on IMpower150 (train_jm_impower150.R predictions).
pred_CIF(t) is the cumulative event probability from BL, conditional on survival to the landmark (t_LM). Landmark-scale
survival S(u) = 1 - CIF(t_LM + u); risk = CIF at 24 months after the landmark. Same metric code and windows as every
IMpower150 method (Uno C tau 30, iAUC / IBS 3-30 months, calibration at 24 months).
Usage: python jm_metrics_impower150.py <4tp|3tp|2tp>  -> results/jm_impower150/<tp>/JointModel_v2_metrics_per_split.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analyses/v2
from mtl_benchmark import metrics

tp = sys.argv[1]
D = Path("results/jm_impower150") / tp
long = pd.read_csv(D / "IMP150_JM_long.csv")
pat = long.drop_duplicates("index").set_index("index")
idx = pd.read_csv(D / "IMP150_JM_test_indices.csv")
t_lm = float(pat.t_LM.iloc[0])
grid = np.arange(0, 30.25, 0.25)
rows = []
for f in sorted((D / "jm_v2").glob("pred_split*.csv")):
    s = int(f.stem[-3:])
    pr = pd.read_csv(f)
    test = idx.iloc[s].astype(int).tolist()
    train = [i for i in pat.index if i not in set(test)]
    d_te = pat.loc[test].assign(OS_t=lambda d: d.OS_landmark, OS_e=lambda d: d.OS_event)
    d_tr = pat.loc[train].assign(OS_t=lambda d: d.OS_landmark, OS_e=lambda d: d.OS_event)
    piv = pr.pivot_table(index="time", columns="index", values="pred_CIF").sort_index()
    S = pd.DataFrame({i: 1 - np.interp(grid + t_lm, piv.index.to_numpy(), piv[i].to_numpy()) for i in test},
                     index=grid)
    m = metrics(d_tr, d_te, 1 - S.loc[24.0].to_numpy(), S, "OS", (3.0, 30.0, 24.0))
    res = pd.read_csv(D / "jm_v2" / f"res_split{s:03d}.csv").iloc[0]
    rows.append({"seed": s, "C_index_Uno": m["C_index_Uno"], "iAUC": m["iAUC"], "IBS": m["IBS"],
                 "calib_err_24m": m["calib_err"], "pred_S24_range": m["pred_S_range"],
                 "max_Rhat_all": res.max_Rhat_all, "minutes": res.minutes})
r = pd.DataFrame(rows)
r.to_csv(D / "JointModel_v2_metrics_per_split.csv", index=False)
print(tp, len(r), "splits"); print(r.drop(columns="seed").mean().round(4).to_dict())
