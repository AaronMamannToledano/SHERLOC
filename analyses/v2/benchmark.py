"""
benchmark.py — SHERLOC vs the other ctDNA-dynamics encoders on PADA-1 (landmark OS at C2).

All methods share:
  - the same frozen genomic-interaction encoder (MSK-CHORD pretraining, config.yaml),
  - the same inputs (gene-level, panel-level, clinical; config.yaml `model`),
  - the NaN-safe Cox loss,
  - the same 100 reporting splits (seeds 0-99, random_state = 4*seed).

Training settings: SHERLOC keeps those selected in experiments.py. Every competitor gets
the same training grid, evaluated on the 40 tuning splits (seeds 100-139); its best
setting (highest mean Uno C) is then run on the reporting splits.

Usage
-----
  python benchmark.py --config config.yaml
Outputs in results/benchmark/
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from train_sherloc_pada1 import METHODS, load_cfg, run_cv, summarise

TUNING_SEEDS = range(100, 140)
TRAIN_GRID = {
    "fixed_ep60":  {"val_frac": 0,   "epochs": 60},
    "fixed_ep100": {"val_frac": 0,   "epochs": 100},
    "earlystop_val0.2": {"val_frac": 0.2, "epochs": 256, "patience": 20},   # repo default
    "earlystop_val0.3": {"val_frac": 0.3, "epochs": 256, "patience": 20},
}


def with_training(cfg, method, setting):
    c = copy.deepcopy(cfg)
    c["model"]["method"] = method
    c["training"].update(TRAIN_GRID[setting] if setting else {})
    return c


def nb_paired(a, b, test_frac):
    d = np.asarray(a) - np.asarray(b); n = len(d)
    se = np.sqrt(d.var(ddof=1) * (1 / n + test_frac / (1 - test_frac)))
    return d.mean(), 2 * (1 - stats.t.cdf(abs(d.mean() / se), n - 1))


def main(cfg, n_seeds):
    out = Path(cfg["_P"]["output_dir"]) / "benchmark"; out.mkdir(parents=True, exist_ok=True)
    tf = cfg["training"]["test_frac"]

    # 1. training-setting selection for competitors (tuning splits)
    sel_rows, chosen = [], {"Proposed_Method": None}
    for m in METHODS:
        if m == "Proposed_Method":
            continue
        best = None
        for s in TRAIN_GRID:
            r, _, _ = run_cv(with_training(cfg, m, s), TUNING_SEEDS, verbose=False)
            row = {"method": m, "training": s, "C_index_Uno": r.C_index_Uno.mean(),
                   "iAUC": r.iAUC.mean(), "IBS": r.IBS.mean(), "epochs_median": r.epochs.median()}
            sel_rows.append(row); print("[tuning]", {k: (round(v, 4) if isinstance(v, float) else v)
                                                      for k, v in row.items()}, flush=True)
            if best is None or row["C_index_Uno"] > best[1]:
                best = (s, row["C_index_Uno"])
        chosen[m] = best[0]
        pd.DataFrame(sel_rows).to_csv(out / "training_selection_tuning_splits.csv", index=False)

    # 2. reporting splits
    per_split, summ = {}, []
    for m, s in chosen.items():
        c = with_training(cfg, m, s)
        r, preds, _ = run_cv(c, range(n_seeds), verbose=False)
        r.insert(0, "method", m)
        r.to_csv(out / f"{m}_metrics_per_split.csv", index=False)
        preds.to_csv(out / f"{m}_test_risk_scores.csv", index=False)
        per_split[m] = r
        sm = summarise(r, tf); sm.insert(0, "training", s or "config.yaml (SHERLOC tuned)"); sm.insert(0, "method", m)
        summ.append(sm)
        print(f"[report] {m:24s} ({s})  C={r.C_index_Uno.mean():.4f}  iAUC={r.iAUC.mean():.4f}  "
              f"IBS={r.IBS.mean():.4f}", flush=True)
    summ = pd.concat(summ, ignore_index=True)
    summ.to_csv(out / "benchmark_summary_long.csv", index=False)

    wide = summ.pivot_table(index="method", columns="metric", values=["mean", "sd"])
    wide.columns = [f"{m}_{s}" for s, m in wide.columns]
    wide = wide[[f"{m}_{s}" for m in ["C_index_Uno", "iAUC", "IBS"] for s in ["mean", "sd"]]]
    wide.insert(0, "training", pd.Series(chosen).fillna("config.yaml (SHERLOC tuned)"))
    wide = wide.sort_values("C_index_Uno_mean", ascending=False)
    wide.to_csv(out / "benchmark_summary.csv")

    # 3. SHERLOC vs each competitor (paired, Nadeau-Bengio corrected t-test)
    cmp = []
    for m in chosen:
        if m == "Proposed_Method":
            continue
        for met in ["C_index_Uno", "iAUC", "IBS"]:
            d, p = nb_paired(per_split["Proposed_Method"][met], per_split[m][met], tf)
            win = (per_split["Proposed_Method"][met] > per_split[m][met]).mean() if met != "IBS" \
                else (per_split["Proposed_Method"][met] < per_split[m][met]).mean()
            cmp.append({"competitor": m, "metric": met, "SHERLOC_minus_competitor": d,
                        "p_NB_corrected": p, "frac_splits_SHERLOC_better": win})
    pd.DataFrame(cmp).to_csv(out / "SHERLOC_vs_competitors.csv", index=False)
    print("\n", wide.round(4).to_string())
    print("\n", pd.DataFrame(cmp).round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n_seeds", type=int, default=100)
    a = ap.parse_args()
    main(load_cfg(a.config), a.n_seeds)
