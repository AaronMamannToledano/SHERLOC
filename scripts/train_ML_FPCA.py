"""
train_ML_FPCA.py — ML survival models with FPCA+PCA longitudinal feature
                   extraction and a pre-trained genomic encoder.

Pipeline
--------
1. Pretrain a genomic-interaction encoder (MLP_SolidBiopsy) on the pretraining
   cohort (e.g. MSK-CHORD) using the Cox PH loss.                 [utils]
2. Extract per-patient latent gene features from the frozen encoder at every
   configured longitudinal timepoint.                             [utils]
3. For each longitudinal feature channel (gene-level VAF + panel-level ctDNA):
   a. Represent it as a functional object (FDataGrid) on an evenly-spaced grid
      whose spacing is inferred from the timepoint collection schedule.
   b. Fit FPCA with ``n_fpca_components`` components on the training set;
      transform both train and test.
   c. Select the top ``n_top_features`` FPCA scores by permutation importance
      (Random Survival Forest fitted on training data).
   d. Apply StandardScaler then PCA(``n_pca_components``) on the selection.
4. Concatenate FPCA-PCA scores with clinical and latent encoder features.
5. Run Monte-Carlo Cross-Validation (default: 100 seeds) on the target cohort
   (e.g. IMpower150) with the selected ML model:
   - ``penalized_cox``: ElasticNet / Lasso / Ridge-penalised Cox regression
     (CoxnetSurvivalAnalysis), with grid search over regularisation strength.
   - ``RSF``:           Random Survival Forest, with grid search over tree depth
     and minimum split size.
6. Report C-index (IPCW), cumulative AUC, and integrated Brier score per seed.

Key differences vs train_ML.py
-------------------------------
train_ML.py feeds raw gene-level VAF columns + latent encoder features directly
to the survival model.  train_ML_FPCA.py instead summarises the longitudinal
trajectory of every feature with FPCA, then further compresses those FPCA
scores with PCA, giving the survival model a compact functional summary rather
than raw time-series values.

Usage
-----
  python train_ML_FPCA.py                              # reads config.yaml
  python train_ML_FPCA.py --config my_cfg.yaml
  python train_ML_FPCA.py --method RSF
  python train_ML_FPCA.py --method penalized_cox --sub_method elastic_net
  python train_ML_FPCA.py --timepoints BL C2D1
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
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA

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

# L1 mixing ratios for each penalised Cox sub-method (mirrors train_ML.py)
_L1_RATIO = {
    "elastic_net": 0.5,
    "lasso":       1.0,
    "ridge":       1e-5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Longitudinal tensor extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_longitudinal_tensors(
    df: pd.DataFrame,
    gene_list_restricted: list[str],
    panel_feature_suffixes: list[str],
    timepoints: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build dense gene-VAF and panel-aggregated tensors for an entire split.

    This is a vectorised, Dataset-free replacement for the notebook's
    ``molecular_survival`` DataLoader approach; it produces the same arrays
    that the original loop collected batch-wise.

    Args:
        df:                    Target-cohort DataFrame (any index).
        gene_list_restricted:  Genes mutated in ≥5 % of target-cohort patients.
        panel_feature_suffixes: Panel-level feature suffixes (e.g. ``_max_of_AF``);
                               columns are looked up as ``<timepoint><suffix>``.
        timepoints:            Ordered list of timepoint labels.

    Returns:
        Tuple ``(gene_tensor, panel_tensor)`` with shapes
        ``(n_patients, n_genes, n_timepoints)`` and
        ``(n_patients, n_panel_features, n_timepoints)``.
    """
    df = df.reset_index(drop=True)
    n    = len(df)
    n_tp = len(timepoints)

    # Gene-level VAF: (n_patients, n_genes, n_timepoints)
    gene_arr = np.zeros((n, len(gene_list_restricted), n_tp), dtype=np.float32)
    for g_idx, gene in enumerate(gene_list_restricted):
        for t_idx, tp in enumerate(timepoints):
            col = f"{gene}_{tp}"
            if col in df.columns:
                gene_arr[:, g_idx, t_idx] = df[col].to_numpy(dtype=np.float32)

    # Panel-level aggregated: (n_patients, n_panel_features, n_timepoints)
    panel_arr = np.zeros((n, len(panel_feature_suffixes), n_tp), dtype=np.float32)
    for f_idx, sf in enumerate(panel_feature_suffixes):
        for t_idx, tp in enumerate(timepoints):
            col = f"{tp}{sf}"
            if col in df.columns:
                panel_arr[:, f_idx, t_idx] = df[col].to_numpy(dtype=np.float32)

    return torch.from_numpy(gene_arr), torch.from_numpy(panel_arr)


# ─────────────────────────────────────────────────────────────────────────────
# FPCA grid
# ─────────────────────────────────────────────────────────────────────────────

def _fpca_grid_points(timepoints: list[str]) -> list[float]:
    """
    Build an evenly-spaced numeric grid for FDataGrid matching the collection
    schedule encoded in the timepoint list.

    The original notebook used a fixed 0.75-month spacing between timepoints
    (BL→0, C2D1→0.75, C3D1→1.50, C4D1→2.25).  This function generalises that
    convention to any number of timepoints.

    Args:
        timepoints: Ordered list of timepoint labels.

    Returns:
        List of floats of length ``len(timepoints)``.
    """
    return [i * 0.75 for i in range(len(timepoints))]


# ─────────────────────────────────────────────────────────────────────────────
# FPCA + PCA feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fpca_pca_features(
    gene_train: torch.Tensor,
    panel_train: torch.Tensor,
    gene_test: torch.Tensor,
    panel_test: torch.Tensor,
    y_train: sksurv.util.Surv,
    timepoints: list[str],
    fpca_cfg: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply per-feature FPCA, permutation-importance-based selection,
    StandardScaler, and PCA to the longitudinal feature tensors.

    This is a clean, config-driven replacement for the notebook's
    ``FPCA_PCA_features`` function.

    Steps
    -----
    1. Concatenate gene-VAF and panel-aggregated tensors along the feature axis
       → shape ``(n_patients, n_features, n_timepoints)``.
    2. For each feature, fit ``FDataGrid`` on training data and apply FPCA
       (``n_fpca_components`` components per feature).  Transform both sets.
    3. Fit a ``RandomSurvivalForest`` on all FPCA scores; rank by permutation
       importance; keep the top ``n_top_features``.
    4. Apply ``StandardScaler`` then ``PCA(n_pca_components)`` on the selection.

    Args:
        gene_train:   Gene-VAF tensor for train split,
                      shape ``(n_train, n_genes, n_timepoints)``.
        panel_train:  Panel-aggregated tensor for train split,
                      shape ``(n_train, n_panel, n_timepoints)``.
        gene_test:    Gene-VAF tensor for test split.
        panel_test:   Panel-aggregated tensor for test split.
        y_train:      scikit-survival structured array for the training split.
        timepoints:   Ordered list of timepoint labels.
        fpca_cfg:     Sub-dict ``cfg["ml"]["fpca_pca"]`` with keys
                      ``n_fpca_components``, ``n_top_features``,
                      ``n_pca_components``, ``n_rsf_estimators_importance``.
        seed:         Random seed for the importance RSF and PCA.

    Returns:
        Tuple ``(pca_train, pca_test)``, each an ndarray of shape
        ``(n_patients, n_pca_components)``.
    """
    n_fpca   = fpca_cfg["n_fpca_components"]
    n_top    = fpca_cfg["n_top_features"]
    n_pca    = fpca_cfg["n_pca_components"]
    n_est    = fpca_cfg["n_rsf_estimators_importance"]
    grid_pts = _fpca_grid_points(timepoints)

    # Stack gene and panel channels along the feature axis
    # Shape: (n_patients, n_genes + n_panel, n_timepoints)
    long_train = torch.cat((gene_train.float(), panel_train.float()), dim=1)
    long_test  = torch.cat((gene_test.float(),  panel_test.float()),  dim=1)
    n_features = long_train.shape[1]

    scores_train = np.empty((long_train.shape[0], n_features, n_fpca), dtype=np.float32)
    scores_test  = np.empty((long_test.shape[0],  n_features, n_fpca), dtype=np.float32)

    for feat_idx in range(n_features):
        fd_train = FDataGrid(
            data_matrix=long_train[:, feat_idx, :].numpy(),
            grid_points=grid_pts,
        )
        fd_test = FDataGrid(
            data_matrix=long_test[:, feat_idx, :].numpy(),
            grid_points=grid_pts,
        )
        fpca = FPCA(n_components=n_fpca)
        scores_train[:, feat_idx, :] = fpca.fit_transform(fd_train)
        scores_test[:, feat_idx,  :] = fpca.transform(fd_test)

    # Flatten to (n_patients, n_features * n_fpca_components)
    scores_train_flat = scores_train.reshape(scores_train.shape[0], -1)
    scores_test_flat  = scores_test.reshape(scores_test.shape[0],   -1)
    fpca_col_names = [
        f"FPCA_feat_{i}" for i in range(scores_train_flat.shape[1])
    ]

    # ── Permutation importance ─────────────────────────────────────────────
    rsf_imp = RandomSurvivalForest(n_estimators=n_est, random_state=seed)
    rsf_imp.fit(scores_train_flat, y_train)
    imp = permutation_importance(
        rsf_imp, scores_train_flat, y_train,
        n_repeats=10, random_state=seed,
    )
    top_indices = np.argsort(imp.importances_mean)[::-1][:n_top]

    train_sel = scores_train_flat[:, top_indices]
    test_sel  = scores_test_flat[:,  top_indices]

    # ── StandardScaler + PCA ───────────────────────────────────────────────
    # Guard: n_pca cannot exceed the number of selected features
    effective_pca = min(n_pca, train_sel.shape[1])
    if effective_pca < n_pca:
        print(
            f"[WARN] n_pca_components ({n_pca}) > n_top_features ({n_top}); "
            f"capping at {effective_pca}."
        )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_sel)
    test_scaled  = scaler.transform(test_sel)

    pca = PCA(n_components=effective_pca, random_state=seed)
    pca_train = pca.fit_transform(train_scaled)
    pca_test  = pca.transform(test_scaled)

    return pca_train, pca_test


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix assembly
# ─────────────────────────────────────────────────────────────────────────────

def _build_fpca_feature_matrix(
    pca_scores: np.ndarray,
    df_split: pd.DataFrame,
    clinical_features: list[str],
    lat_features: torch.Tensor,
    surv_time: str,
    surv_event: str,
) -> pd.DataFrame:
    """
    Assemble the flat feature DataFrame consumed by scikit-survival models.

    Columns included (in order):
    - FPCA-PCA longitudinal scores.
    - Clinical features (static, one value per patient).
    - Latent genomic-encoder features (flattened across timepoints).
    - Survival target columns (kept for ``get_target`` calls).

    This is a config-driven replacement for the notebook's
    ``create_dataframe_FPCA_with_static_features`` function.

    Args:
        pca_scores:        FPCA-PCA array, shape ``(n_patients, n_pca_components)``.
        df_split:          Target-cohort DataFrame for this split.
        clinical_features: Column names for clinical features.
        lat_features:      Latent encoder tensor,
                           shape ``(n_patients, encoder_dim, n_timepoints)``.
        surv_time:         Survival time column name.
        surv_event:        Survival event column name.

    Returns:
        Flat DataFrame with all features and survival columns; all column names
        are cast to strings for compatibility with scikit-survival.
    """
    df_split = df_split.reset_index(drop=True)

    df_scores = pd.DataFrame(
        pca_scores,
        columns=[f"FPCA_PCA_feat_{i}" for i in range(pca_scores.shape[1])],
    )
    df_cln = df_split[clinical_features].reset_index(drop=True)

    lat_flat = lat_features.reshape(len(df_split), -1).numpy()
    df_lat   = pd.DataFrame(lat_flat).add_prefix("latent_")

    df_surv  = df_split[[surv_time, surv_event]].reset_index(drop=True)

    df_out = pd.concat(
        [df_scores, df_cln, df_lat, df_surv], axis=1
    )
    df_out.columns = df_out.columns.map(str)
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# ML model training and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def ml_fpca_model(cfg: dict) -> None:
    """
    Full ML-FPCA survival pipeline: pretraining → feature extraction →
    FPCA+PCA → Monte-Carlo CV → result export.

    All data loading, encoder pretraining, and latent feature extraction are
    delegated to ``utils``; this function owns the FPCA pipeline and the
    scikit-survival training loop.

    Args:
        cfg: Loaded and validated configuration dict.
    """
    # ── unpack config ─────────────────────────────────────────────────────────
    ml_cfg     = cfg["ml"]
    method     = ml_cfg["method"]
    sub_method = ml_cfg.get("sub_method", "")
    fpca_cfg   = ml_cfg["fpca_pca"]
    timepoints = cfg["timepoints"]
    n_tp       = len(timepoints)
    eval_cfg   = cfg["evaluation"]
    tsurv      = cfg["target_survival"]
    surv_time  = tsurv["time"]
    surv_event = tsurv["event"]
    t_cln      = cfg["target_clinical_features"]
    pf_suf     = cfg["panel_feature_suffixes"]
    tr_hp      = cfg["training"]

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    label       = f"{method}_{sub_method}" if sub_method else method
    result_path = out_dir / f"FPCA_{label}_{n_tp}tp_metrics.csv"

    t_min_ibs = eval_cfg["time_min_ibs"]
    t_min_auc = eval_cfg["time_min_auc"]
    t_max     = eval_cfg["time_max"]

    print(f"[INFO] Method:     FPCA + {method}" + (f" ({sub_method})" if sub_method else ""))
    print(f"[INFO] Timepoints: {timepoints}")
    print(f"[INFO] Results →   {result_path}")

    # ── 1. Pretraining ────────────────────────────────────────────────────────
    print("[INFO] Loading pretraining data …")
    df_pretrain, gene_names, cln_names = prepare_pretraining_data(cfg)
    print(
        f"[INFO] Pretraining: {len(df_pretrain)} patients | "
        f"{len(gene_names)} genes | {len(cln_names)} clinical features"
    )

    print("[INFO] Pretraining genomic-interaction encoder …")
    net_SB = pretrain_encoder(df_pretrain, gene_names, cln_names, cfg)
    print("[INFO] Pretraining complete.")

    # ── 2. Target dataset ─────────────────────────────────────────────────────
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
    df_target, _ = prepare_target_data(cfg)
    print(f"[INFO] Target cohort: {len(df_target)} patients")

    missing_cln = [c for c in t_cln if c not in df_target.columns]
    if missing_cln:
        raise ValueError(
            f"Target dataset is missing clinical columns: {missing_cln}\n"
            "Update target_clinical_features in config.yaml."
        )

    # ── helpers ───────────────────────────────────────────────────────────────
    get_target = lambda df: (df[surv_time].values, df[surv_event].values)

    # ── 3. Monte-Carlo Cross-Validation ──────────────────────────────────────
    C_INDEX, AUC, IBS = [], [], []

    print(f"[INFO] Running {eval_cfg['n_seeds']} Monte-Carlo CV iterations …")
    for seed in tqdm(range(eval_cfg["n_seeds"])):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ── Train / test split ────────────────────────────────────────
        # No validation set is needed for ML models.
        df_test  = df_target.sample(frac=tr_hp["test_frac"], random_state=eval_cfg["seed_multiplier"]*seed)
        df_train = df_target.drop(df_test.index)

        # ── Latent encoder features (utils) ───────────────────────────
        lat_train = extract_latent_features(df_train, net_SB, gene_names, timepoints)
        lat_test  = extract_latent_features(df_test,  net_SB, gene_names, timepoints)

        # ── Longitudinal tensors for FPCA ─────────────────────────────
        # Shape: (n_patients, n_features, n_timepoints)
        gene_train, panel_train = _extract_longitudinal_tensors(
            df_train, gene_list_restricted, pf_suf, timepoints
        )
        gene_test, panel_test = _extract_longitudinal_tensors(
            df_test, gene_list_restricted, pf_suf, timepoints
        )

        # ── Survival targets (needed by FPCA importance step) ─────────
        dur_train, ev_train = get_target(df_train)
        dur_test,  ev_test  = get_target(df_test)
        y_train = sksurv.util.Surv.from_arrays(event=ev_train, time=dur_train)
        y_test  = sksurv.util.Surv.from_arrays(event=ev_test,  time=dur_test)

        # ── FPCA + PCA ────────────────────────────────────────────────
        pca_train, pca_test = _compute_fpca_pca_features(
            gene_train, panel_train,
            gene_test,  panel_test,
            y_train, timepoints, fpca_cfg, seed,
        )

        # ── Flat feature matrices ─────────────────────────────────────
        df_train_feat = _build_fpca_feature_matrix(
            pca_train, df_train, t_cln, lat_train, surv_time, surv_event
        )
        df_test_feat = _build_fpca_feature_matrix(
            pca_test, df_test, t_cln, lat_test, surv_time, surv_event
        )

        feature_cols = [
            c for c in df_train_feat.columns
            if c != surv_time and c != surv_event
        ]

        cv = KFold(n_splits=4, shuffle=True, random_state=seed)

        # ── Model selection and grid search ───────────────────────────
        if method == "penalized_cox":
            l1_ratio = (
                ml_cfg["penalized_cox"].get("l1_ratio_override") or
                _L1_RATIO[sub_method]
            )
            cox_hp = ml_cfg["penalized_cox"]
            # Step 1: fit a regularisation path to get candidate alphas
            estimator = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                n_alphas=cox_hp["n_alphas"],
                max_iter=cox_hp["max_iter"],
                alpha_min_ratio=cox_hp["alpha_min_ratio"],
            )
            estimator.fit(df_train_feat[feature_cols], y_train)
            # Step 2: grid-search over the candidate alphas
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

        # ── Evaluation ────────────────────────────────────────────────
        risk_scores = best_model.predict(df_test_feat[feature_cols])

        c_uno = sksurv.metrics.concordance_index_ipcw(
            y_train, y_test, estimate=risk_scores, tau=t_max
        )[0]
        C_INDEX.append(c_uno)

        times_auc = np.arange(t_min_auc, t_max)
        auc = sksurv.metrics.cumulative_dynamic_auc(
            y_train, y_test, estimate=risk_scores, times=times_auc
        )[1]
        AUC.append(auc)

        # IBS requires the model's survival function on a shared time grid
        survs      = best_model.predict_survival_function(df_test_feat[feature_cols])
        base_times = survs[0].x
        times_ibs  = base_times[
            (base_times >= t_min_ibs) & (base_times <= t_max)
        ]
        preds = np.array([[fn(t) for t in times_ibs] for fn in survs])
        ibs = integrated_brier_score(
            y_train, y_test, estimate=preds, times=times_ibs
        )
        IBS.append(ibs)

    # ── Summary and export ────────────────────────────────────────────────────
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
        description="ML survival models with FPCA+PCA and a pre-trained genomic encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    p.add_argument(
        "--method", choices=["RSF", "penalized_cox"],
        help="Override ml.method from config.yaml",
    )
    p.add_argument(
        "--sub_method", choices=["elastic_net", "lasso", "ridge"],
        help="Override ml.sub_method from config.yaml (penalized_cox only)",
    )
    p.add_argument(
        "--timepoints", nargs="+",
        help="Override timepoints from config.yaml, e.g. --timepoints BL C2D1",
    )
    p.add_argument(
        "--output_dir",
        help="Override paths.output_dir from config.yaml",
    )
    p.add_argument(
        "--n_fpca_components", type=int,
        help="Override ml.fpca_pca.n_fpca_components from config.yaml",
    )
    p.add_argument(
        "--n_top_features", type=int,
        help="Override ml.fpca_pca.n_top_features from config.yaml",
    )
    p.add_argument(
        "--n_pca_components", type=int,
        help="Override ml.fpca_pca.n_pca_components from config.yaml",
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
    if args.n_fpca_components:
        cfg["ml"]["fpca_pca"]["n_fpca_components"] = args.n_fpca_components
    if args.n_top_features:
        cfg["ml"]["fpca_pca"]["n_top_features"] = args.n_top_features
    if args.n_pca_components:
        cfg["ml"]["fpca_pca"]["n_pca_components"] = args.n_pca_components

    # Reuse the same ML config validation as train_ML.py (method + sub_method)
    validate_config(cfg, mode="ml")
    ml_fpca_model(cfg)
