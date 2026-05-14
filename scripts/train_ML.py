"""
train_ML.py — Machine learning survival models with a pre-trained genomic encoder.

Pipeline
--------
1. Pretrain a genomic-interaction encoder (MLP_SolidBiopsy) on the pretraining
   cohort (e.g. MSK-CHORD) using the Cox PH loss.
2. Extract per-patient latent gene features from the frozen encoder at every
   configured longitudinal timepoint.
3. Concatenate latent features with longitudinal ctDNA and clinical features.
4. Run Monte-Carlo Cross-Validation (default: 100 seeds) on the target cohort
   (e.g. IMpower150) with the selected model:
   - ``penalized_cox``: ElasticNet / Lasso / Ridge-penalised Cox regression
     (CoxnetSurvivalAnalysis), with grid search over regularisation strength.
   - ``RSF``:           Random Survival Forest, with grid search over tree depth
     and minimum split size.
5. Report C-index (IPCW), cumulative AUC, and integrated Brier score per seed.

Usage
-----
  python train_ML.py                               # reads config.yaml
  python train_ML.py --config my_cfg.yaml
  python train_ML.py --method RSF
  python train_ML.py --method penalized_cox --sub_method lasso
  python train_ML.py --timepoints BL C2D1
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sksurv.metrics
import sksurv.util
from sksurv.metrics import integrated_brier_score
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import GridSearchCV, KFold

from scripts.utils import (
    load_config,
    resolve_path,
    validate_config,
    load_dataframe,
    prepare_pretraining_data,
    pretrain_encoder,
    extract_latent_features,
    prepare_target_data,
)

# L1 mixing ratios for each penalised Cox sub-method
_L1_RATIO = {
    "elastic_net": 0.5,
    "lasso":       1.0,
    "ridge":       1e-5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature assembly
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_matrix(
    df: pd.DataFrame,
    agg_feature_cols: list[str],
    gene_list_restricted: list[str],
    timepoints: list[str],
    surv_time: str,
    surv_event: str,
    lat_features: torch.Tensor,
) -> pd.DataFrame:
    """
    Assemble the flat feature matrix used by scikit-survival models.

    Columns included:
    - Longitudinal gene-level VAF features (gene × timepoint)
    - Longitudinal panel-level aggregated ctDNA features
    - Latent genomic-encoder features (flattened across timepoints)
    - Survival target columns (kept for ``get_target`` calls)

    Args:
        df:                   Subset of the target-cohort DataFrame.
        agg_feature_cols:     Panel-level aggregated feature column names.
        gene_list_restricted: Genes mutated in ≥5 % of target-cohort patients.
        timepoints:           Ordered list of timepoint labels.
        surv_time:            Survival time column name.
        surv_event:           Survival event column name.
        lat_features:         Latent encoder tensor,
                              shape ``(n_patients, encoder_dim, n_timepoints)``.

    Returns:
        Flat DataFrame with all features and survival columns; row order
        matches ``df`` (after ``reset_index``).
    """
    df = df.reset_index(drop=True)

    # Gene-level VAF columns: <gene>_<timepoint>
    gene_tp_cols = [
        f"{g}_{tp}"
        for g in gene_list_restricted
        for tp in timepoints
        if f"{g}_{tp}" in df.columns
    ]
    survival_cols = [surv_time, surv_event]
    keep = [c for c in gene_tp_cols + agg_feature_cols + survival_cols if c in df.columns]
    df_out = df[keep].copy()

    # Append flattened latent encoder features
    lat_flat = lat_features.reshape(len(df), -1).numpy()
    lat_df   = pd.DataFrame(lat_flat).add_prefix("latent_")
    df_out = pd.concat([df_out.reset_index(drop=True), lat_df], axis=1)
    df_out.columns = df_out.columns.map(str)

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# ML model training and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def ml_model(cfg: dict) -> None:
    """
    Full ML survival pipeline: pretraining → feature extraction →
    Monte-Carlo CV → result export.

    Args:
        cfg: Loaded and validated configuration dict.
    """
    # ── unpack config ─────────────────────────────────────────────────────────
    ml_cfg     = cfg["ml"]
    method     = ml_cfg["method"]
    sub_method = ml_cfg.get("sub_method", "")
    timepoints = cfg["timepoints"]
    n_tp       = len(timepoints)
    eval_cfg   = cfg["evaluation"]
    tsurv      = cfg["target_survival"]
    surv_time  = tsurv["time"]
    surv_event = tsurv["event"]
    tr_hp      = cfg["training"]          # test_frac reused from DL section

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{method}_{sub_method}" if sub_method else method
    result_path = out_dir / f"{label}_{n_tp}tp_metrics.csv"

    print(f"[INFO] Method:     {method}" + (f" ({sub_method})" if sub_method else ""))
    print(f"[INFO] Timepoints: {timepoints}")
    print(f"[INFO] Results →   {result_path}")

    # ── pretraining ───────────────────────────────────────────────────────────
    print("[INFO] Loading pretraining data …")
    df_pretrain, gene_names, cln_names = prepare_pretraining_data(cfg)
    print(
        f"[INFO] Pretraining: {len(df_pretrain)} patients | "
        f"{len(gene_names)} genes | {len(cln_names)} clinical features"
    )

    print("[INFO] Pretraining genomic-interaction encoder …")
    net_SB = pretrain_encoder(df_pretrain, gene_names, cln_names, cfg)
    print("[INFO] Pretraining complete.")

    # ── target dataset ────────────────────────────────────────────────────────
    df_list_genes = load_dataframe(
        resolve_path(cfg["paths"]["genes_kept_restricted"], n_tp), index_col=0
    )
    if "gene_kept_restricted" not in df_list_genes.columns:
        raise ValueError(
            "genes_kept_restricted file must contain a 'gene_kept_restricted' "
            "column. Check paths.genes_kept_restricted in config.yaml."
        )
    gene_list_restricted = df_list_genes["gene_kept_restricted"].unique().tolist()

    print("[INFO] Loading target cohort data …")
    df_target, agg_feature_cols = prepare_target_data(cfg)
    print(f"[INFO] Target cohort: {len(df_target)} patients")

    # ── helpers ───────────────────────────────────────────────────────────────
    get_target = lambda df: (df[surv_time].values, df[surv_event].values)
    t_min_ibs  = eval_cfg["time_min_ibs"]
    t_min_auc  = eval_cfg["time_min_auc"]
    t_max      = eval_cfg["time_max"]

    # ── Monte-Carlo Cross-Validation ──────────────────────────────────────────
    C_INDEX, AUC, IBS = [], [], []

    print(f"[INFO] Running {eval_cfg['n_seeds']} Monte-Carlo CV iterations …")
    for seed in tqdm(range(eval_cfg["n_seeds"])):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Train / test split (no validation set needed for ML models)
        df_test  = df_target.sample(frac=tr_hp["test_frac"], random_state=seed)
        df_train = df_target.drop(df_test.index)

        # Extract latent encoder features for train and test sets
        lat_train = extract_latent_features(df_train, net_SB, gene_names, timepoints)
        lat_test  = extract_latent_features(df_test,  net_SB, gene_names, timepoints)

        # Assemble flat feature matrices
        df_train_feat = _build_feature_matrix(
            df_train, agg_feature_cols, gene_list_restricted,
            timepoints, surv_time, surv_event, lat_train
        )
        df_test_feat = _build_feature_matrix(
            df_test, agg_feature_cols, gene_list_restricted,
            timepoints, surv_time, surv_event, lat_test
        )

        feature_cols = [
            c for c in df_train_feat.columns
            if c != surv_time and c != surv_event
        ]

        # Survival targets
        dur_train, ev_train = get_target(df_train_feat)
        dur_test,  ev_test  = get_target(df_test_feat)
        y_train = sksurv.util.Surv.from_arrays(event=ev_train, time=dur_train)
        y_test  = sksurv.util.Surv.from_arrays(event=ev_test,  time=dur_test)

        cv = KFold(n_splits=4, shuffle=True, random_state=seed)

        # ── model selection and grid search ───────────────────────────
        if method == "penalized_cox":
            l1_ratio = (
                ml_cfg["penalized_cox"].get("l1_ratio_override") or
                _L1_RATIO[sub_method]
            )
            cox_hp = ml_cfg["penalized_cox"]
            # Step 1: fit a path to get candidate alphas
            estimator = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                n_alphas=cox_hp["n_alphas"],
                max_iter=cox_hp["max_iter"],
                alpha_min_ratio=cox_hp["alpha_min_ratio"],
            )
            estimator.fit(df_train_feat[feature_cols], y_train)
            # Step 2: grid search over the candidate alphas
            gcv = GridSearchCV(
                CoxnetSurvivalAnalysis(
                    l1_ratio=l1_ratio,
                    max_iter=cox_hp["max_iter"],
                    fit_baseline_model=True,
                ),
                param_grid={"alphas": [[v] for v in estimator.alphas_]},
                cv=cv,
                error_score=0.5,
                n_jobs=1,
            ).fit(df_train_feat[feature_cols], y_train)

        elif method == "RSF":
            rsf_hp = ml_cfg["rsf"]
            gcv = GridSearchCV(
                RandomSurvivalForest(),
                param_grid={
                    "max_depth":         rsf_hp["max_depth"],
                    "min_samples_split": rsf_hp["min_samples_split"],
                },
                cv=cv,
                error_score=0.5,
                n_jobs=rsf_hp["n_jobs"],
            ).fit(df_train_feat[feature_cols], y_train)

        best_model = gcv.best_estimator_

        # ── evaluation ────────────────────────────────────────────────
        risk_scores = best_model.predict(df_test_feat[feature_cols])

        c_uno = sksurv.metrics.concordance_index_ipcw(
            y_train, y_test, estimate=risk_scores
        )[0]
        C_INDEX.append(c_uno)

        times_auc = np.arange(t_min_auc, t_max)
        auc = sksurv.metrics.cumulative_dynamic_auc(
            y_train, y_test, estimate=risk_scores, times=times_auc
        )[1]
        AUC.append(auc)

        # IBS: use model's survival function evaluated on a shared time grid
        survs      = best_model.predict_survival_function(df_test_feat[feature_cols])
        base_times = survs[0].x
        times_ibs  = base_times[
            (base_times >= t_min_ibs)
            & (base_times <= t_max)
        ]
        preds = np.array([[fn(t) for t in times_ibs] for fn in survs])
        ibs = integrated_brier_score(
            y_train, y_test, estimate=preds, times=times_ibs
        )
        IBS.append(ibs)

    # ── summary and export ────────────────────────────────────────────────────
    df_results = pd.DataFrame({"C-index": C_INDEX, "AUC": AUC, "IBS": IBS})

    print("\n[INFO] Summary across seeds:")
    print(df_results.describe().round(4))

    df_results.to_csv(result_path, index=True)
    print(f"\n[INFO] Results saved to {result_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ML survival models with pre-trained genomic encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    p.add_argument(
        "--method", choices=["RSF", "penalized_cox"],
        help="Override ml.method from config.yaml"
    )
    p.add_argument(
        "--sub_method", choices=["elastic_net", "lasso", "ridge"],
        help="Override ml.sub_method from config.yaml (penalized_cox only)"
    )
    p.add_argument(
        "--timepoints", nargs="+",
        help="Override timepoints from config.yaml, e.g. --timepoints BL C2D1"
    )
    p.add_argument(
        "--output_dir",
        help="Override paths.output_dir from config.yaml"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not Path(args.config).exists():
        sys.exit(
            f"[ERROR] Config file not found: {args.config}\n"
            "Copy config.yaml to your working directory and edit the paths section."
        )

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.method:
        cfg["ml"]["method"] = args.method
    if args.sub_method:
        cfg["ml"]["sub_method"] = args.sub_method
    if args.timepoints:
        cfg["timepoints"] = args.timepoints
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir

    validate_config(cfg, mode="ml")
    ml_model(cfg)
