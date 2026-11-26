"""
posthoc_benchmark_summary.py — one table per dataset from results/posthoc_benchmark/<dataset>/*.csv:
C-index and iAUC (identical raw / recalibrated), IBS and calibration error raw and after the same 5-fold OOF
post-hoc recalibration, median calibration slope, and Nadeau-Bengio paired tests vs recalibrated SHERLOC.
Usage: python posthoc_benchmark_summary.py [dataset ...]
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
from benchmark import nb_paired

ROOT = Path("results/posthoc_benchmark")
NAMES = {"SHERLOC": "SHERLOC", "Featurewise_Transformer": "Feature-wise Transformer",
         "Temporal_Transformer": "Temporal Transformer", "RNN": "RNN", "GRU": "GRU", "MLP": "MLP (DeepSurv)",
         "ridge_scaled": "Ridge Cox", "lasso_scaled": "Lasso Cox", "elastic_net_scaled": "Elastic-net Cox",
         "RSF": "RSF", "FPCRF": "FPCRF"}


def summarise(ds, test_frac=0.3):
    files = glob.glob(str(ROOT / ds / "*_seeds*.csv"))
    if not files:
        return None
    d = pd.concat(pd.read_csv(f) for f in files).drop_duplicates(["method", "seed", "variant"])
    raw = d[d.variant == "raw"].set_index(["method", "seed"])
    rec = d[d.variant == "recalibrated"].set_index(["method", "seed"])
    ref = rec.loc["SHERLOC"] if "SHERLOC" in rec.index.get_level_values(0) else None
    rows = []
    for m in [k for k in NAMES if k in raw.index.get_level_values(0)]:
        r, c = raw.loc[m], rec.loc[m]
        row = {"method": NAMES[m], "n_splits": len(r), "C_index": r.C_index_Uno.mean(), "iAUC": r.iAUC.mean(),
               "IBS_raw": r.IBS.mean(), "IBS_recal": c.IBS.mean(),
               "calib_raw": r.calib_err.mean(), "calib_recal": c.calib_err.mean(), "beta_median": c.beta.median()}
        if ref is not None and m != "SHERLOC":
            s = ref.index.intersection(r.index)
            for k, src_m, src_ref in (("C", r.C_index_Uno, ref.C_index_Uno), ("IBS_recal", c.IBS, ref.IBS)):
                diff, p = nb_paired(src_ref.loc[s].values, src_m.loc[s].values, test_frac)
                row[f"d{k}_SHERLOC_minus_method"] = diff; row[f"p_{k}"] = p
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / ds / "summary.csv", index=False)
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for ds in (sys.argv[1:] or ["impower150_4tp", "impower150_3tp", "pada1", "impower150_2tp"]):
        t = summarise(ds)
        if t is not None:
            print("==", ds); print(t.round(4).to_string(index=False))


def combined(datasets=("impower150_4tp", "impower150_3tp", "impower150_2tp", "pada1")):
    """Long table: dataset x method, with the IBS / calibration of the better variant flagged."""
    out = []
    for ds in datasets:
        t = summarise(ds)
        if t is not None:
            out.append(t.assign(dataset=ds))
    t = pd.concat(out)
    t["IBS_best"] = t[["IBS_raw", "IBS_recal"]].min(axis=1)
    t["IBS_best_variant"] = np.where(t.IBS_recal < t.IBS_raw, "recalibrated", "raw")
    t.to_csv(ROOT / "summary_all_datasets.csv", index=False)
    return t
