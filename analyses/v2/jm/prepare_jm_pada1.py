"""
prepare_jm_pada1.py — inputs for the joint model (train_jm_pada1.R), following scripts/prepare_jm_data.py of the
SHERLOC repo, with the same inputs as every other method in competitors_pada1.py:
  long format, one row per patient x timepoint (BL, C2): gene-level max VAF (10 genes), panel features (mean VAF,
  nb drivers > 0.5 %, 7 longitudinal response features); static: cln_* (9 clinical), Encoding_* (frozen MSK-CHORD
  encoder at BL and C2), OS_event, OS time from BL (= landmark OS + months BL -> C2).
  time = months from BL (BL = 0, C2 = actual collection delay; the repo uses fixed nominal times).
  Test indices: the same 100 splits as SHERLOC (df.sample(frac=0.3, random_state=4*seed)).
Outputs: results/benchmark_response_longitudinal/jm/PADA1_JM_long.csv, PADA1_JM_test_indices.csv
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import competitors_pada1 as C
cfg, df, lat, genes, gene_cols, panel_cols, cln_cols = C.tabular()
out = Path("results/benchmark_response_longitudinal/jm"); out.mkdir(parents=True, exist_ok=True)
cln = pd.read_csv(os.environ.get("PADA1_CLINICAL", "pada1_raw/cln/curated/cln_curated.tsv"), sep="\t").set_index("Subject_Id")
t_c2 = (cln.Days_Between_Samples.reindex(df.index) / 30.44).fillna(28 / 30.44)
panel = cfg["model"]["panel_features"]
rows = []
for k, tp in enumerate(cfg["timepoints"]):
    d = pd.DataFrame({"index": np.arange(len(df)), "timepoint": tp,
                      "time": 0.0 if tp == "BL" else t_c2.to_numpy()})
    for g in genes:
        d[g] = df[f"{g}_{tp}"].to_numpy()
    for p in panel:
        d[p] = df[f"{tp}_{p}"].to_numpy()
    rows.append(d)
long = pd.concat(rows)
static = pd.DataFrame({"index": np.arange(len(df)), "OS_event": df.OS_event.to_numpy(),
                       "OS_time_BL": (df.OS_rebaseline + t_c2).to_numpy(), "t_C2": t_c2.to_numpy(),
                       "OS_landmark": df.OS_rebaseline.to_numpy()})
for c in cln_cols:
    static[c] = df[c].to_numpy()
L = lat.to_numpy()
for i in range(L.shape[1]):
    static[f"Encoding_{i}"] = L[:, i]
long = long.merge(static, on="index").sort_values(["index", "time"])
long.to_csv(out / "PADA1_JM_long.csv", index=False)
pos = {p: i for i, p in enumerate(df.index)}
idx = [[pos[p] for p in df.sample(frac=0.3, random_state=4 * s).index] for s in range(100)]
pd.DataFrame(idx).to_csv(out / "PADA1_JM_test_indices.csv", index=False)
print(long.shape, "markers:", genes + panel, "max C2 month", round(t_c2.max(), 2))
