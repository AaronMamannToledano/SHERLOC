"""
prepare_jm_data.py — Build the two input files required by train_jm.R.

Pipeline
--------
1. Pretrain a genomic-interaction encoder (MLP_SolidBiopsy) on the
   pretraining cohort (e.g. MSK-CHORD) using the Cox PH loss.
2. Extract latent encoder representations for every patient in the
   target cohort at every configured longitudinal timepoint.
3. Pivot the wide target-cohort table to long format (one row per
   patient × timepoint), stacking gene-level VAF features and panel-
   level aggregated ctDNA features as separate longitudinal variables.
4. Attach per-patient static columns: clinical covariates, survival
   labels (OS_event, TTEOS_rebaseline_BL), and the flattened latent
   encoder representations.
5. Generate the same 100 Monte-Carlo train/test splits that train_ML.py
   uses (same DataFrame, same seeds, same test_frac) and record the
   integer patient indices for each test set.
6. Write both outputs to paths.output_dir.

Outputs  (paths defined in config.yaml joint_model section)
-------
  joint_model.data_file    — long-format table consumed by JMbayes2 in
                             train_jm.R (one row per patient × timepoint).
  joint_model.indices_file — 100 × n_test integer matrix; each row
                             contains the test-patient indices for one seed.

Column conventions in the long-format output
--------------------------------------------
  index                   — integer patient identifier (0 … n_patients - 1).
  timepoint               — categorical timepoint label (e.g. "BL", "C2D1").
  <gene>                  — max VAF at this timepoint for each gene in the
                            restricted gene list (gene_kept_restricted).
  <panel_feat>            — aggregated ctDNA panel feature (e.g. max_of_AF).
  cln_*                   — clinical covariates (constant across timepoints).
  OS_event                — binary survival event indicator.
  TTEOS_rebaseline_BL     — baseline-referenced OS time (joint_model.survival_time_col).
  Encoding_0,Encoding_1,… — flattened latent encoder features, 
                            which must match joint_model.encoder_prefix.

Usage
-----
  python prepare_jm_data.py                        # reads config.yaml
  python prepare_jm_data.py --config my_cfg.yaml
  python prepare_jm_data.py --timepoints BL C2D1
  python prepare_jm_data.py --output_dir results/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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


# ─────────────────────────────────────────────────────────────────────────────
# Patient-ID helpers (prepare_target_data drops the ID column; we reconstruct
# the ordering by replicating its inner merge on a minimal subset of columns)
# ─────────────────────────────────────────────────────────────────────────────

def _get_ordered_patient_ids(cfg: dict) -> pd.Series:
    """
    Return the patient IDs in the same row order as the DataFrame returned
    by ``prepare_target_data``.

    ``prepare_target_data`` drops the patient-ID column after an inner merge
    of the gene-VAF file with the panel-ctDNA file.  This function replicates
    that merge on the ID column only so callers can align any external per-
    patient data (e.g. TTEOS_rebaseline_BL) with ``df_target`` rows.

    Args:
        cfg: Loaded configuration dict.

    Returns:
        pandas Series of length n_patients with the patient IDs, indexed 0 … n-1.
    """
    n_tp    = len(cfg["timepoints"])
    pid_col = cfg["target_patient_id"]

    # gene_ctdna: patient IDs stored as row names (index_col=0 → index)
    df_gene = load_dataframe(
        resolve_path(cfg["paths"]["gene_ctdna"], n_tp), index_col=0
    )
    df_gene = df_gene.reset_index(drop=False).rename(columns={"index": pid_col})

    # panel_ctdna: row names are R row numbers (index_col=0 → index);
    # patient IDs are a regular column
    df_panel = load_dataframe(
        resolve_path(cfg["paths"]["panel_ctdna"], n_tp), index_col=0
    )

    # Replicate prepare_target_data's inner merge (row order is determined here)
    df_ids = pd.merge(df_gene[[pid_col]], df_panel[[pid_col]], on=pid_col, how="inner")
    return df_ids[pid_col].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Wide → long pivot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pivot_genes_to_long(
    df_target: pd.DataFrame,
    gene_list: list[str],
    timepoints: list[str],
) -> pd.DataFrame:
    """
    Pivot gene-level VAF columns from wide to long format.

    Expected source column name pattern:  ``{gene}_{timepoint}``  (e.g. TP53_BL)
    Output columns: ``index``, ``timepoint``, then one column per gene.
    Missing source columns are filled with 0.

    Args:
        df_target:  Wide-format target-cohort DataFrame (rows indexed 0 … n-1).
        gene_list:  Genes to include (e.g. ``gene_list_restricted``).
        timepoints: Ordered list of timepoint labels.

    Returns:
        Long-format DataFrame with len(df_target) × len(timepoints) rows.
    """
    slices = []
    for tp in timepoints:
        df_tp = pd.DataFrame({
            "index":     np.arange(len(df_target)),
            "timepoint": tp,
        })
        for gene in gene_list:
            col = f"{gene}_{tp}"
            df_tp[gene] = (
                df_target[col].to_numpy(dtype=np.float32)
                if col in df_target.columns
                else 0.0
            )
        slices.append(df_tp)
    return pd.concat(slices, axis=0, ignore_index=True)


def _pivot_agg_to_long(
    df_target: pd.DataFrame,
    features_agg_base: list[str],
    timepoints: list[str],
) -> pd.DataFrame:
    """
    Pivot panel-level aggregated ctDNA features from wide to long format.

    Expected source column name pattern:  ``{timepoint}_{feature}``  (e.g. BL_max_of_AF)
    Output columns: ``index``, ``timepoint``, then one column per panel feature.
    Missing source columns are filled with 0.

    Args:
        df_target:         Wide-format target-cohort DataFrame.
        features_agg_base: Base feature names (without timepoint prefix or leading ``_``),
                           e.g. ``["max_of_AF", "mean_of_AF", "cfDNA_conc", "n_kl"]``.
        timepoints:        Ordered list of timepoint labels.

    Returns:
        Long-format DataFrame with len(df_target) × len(timepoints) rows.
    """
    slices = []
    for tp in timepoints:
        df_tp = pd.DataFrame({
            "index":     np.arange(len(df_target)),
            "timepoint": tp,
        })
        for feat in features_agg_base:
            col = f"{tp}_{feat}"
            df_tp[feat] = (
                df_target[col].to_numpy(dtype=np.float32)
                if col in df_target.columns
                else 0.0
            )
        slices.append(df_tp)
    return pd.concat(slices, axis=0, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Static (per-patient, timepoint-invariant) feature table
# ─────────────────────────────────────────────────────────────────────────────

def _build_static_df(
    df_target: pd.DataFrame,
    lat_features: torch.Tensor,
    tteos_bl: pd.Series,
    surv_event: str,
) -> pd.DataFrame:
    """
    Build a per-patient static feature table to be merged into the long-format
    output.  These columns are constant across timepoints for each patient.

    Columns returned:
    - ``index``           — integer patient identifier matching the long tables.
    - ``cln_*``           — all clinical covariate columns from ``df_target``.
    - ``<surv_event>``    — binary event indicator.
    - ``<tteos_bl.name>`` — baseline-referenced OS time.
    - ``Encoding_*``      — flattened latent encoder features.  

    Args:
        df_target:    Wide-format target-cohort DataFrame, rows indexed 0 … n-1.
        lat_features: Encoder output tensor, shape (n, encoder_dim, n_timepoints).
        tteos_bl:     Baseline OS time Series (same length / order as df_target),
                      with a name attribute equal to the desired output column name.
        surv_event:   Survival event column name in df_target.

    Returns:
        DataFrame with n_patients rows, indexed 0 … n-1.
    """
    n = len(df_target)
    cln_cols = [c for c in df_target.columns if c.startswith("cln_")]

    df_static = (
        df_target[cln_cols + [surv_event]]
        .copy()
        .reset_index(drop=True)
    )
    df_static["index"]     = np.arange(n)
    df_static[tteos_bl.name] = tteos_bl.to_numpy()

    # Flatten (n, encoder_dim, n_tp) → (n, encoder_dim * n_tp)
    # Column names become "Encoding_0", "Encoding_1", …
    lat_flat = lat_features.reshape(n, -1).numpy()
    lat_df = pd.DataFrame(
        lat_flat,
        columns=[f"Encoding_{i}" for i in range(lat_flat.shape[1])],
    )
    lat_df["index"] = np.arange(n)

    return pd.merge(df_static, lat_df, on="index", how="inner")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_jm_inputs(cfg: dict) -> None:
    """
    Full pipeline: pretrain encoder → extract features → pivot → save.

    Args:
        cfg: Loaded and validated configuration dict.
    """
    timepoints = cfg["timepoints"]
    n_tp       = len(timepoints)
    pid_col    = cfg["target_patient_id"]
    tsurv      = cfg["target_survival"]
    surv_event = tsurv["event"]
    jm_cfg     = cfg["joint_model"]
    eval_cfg   = cfg["evaluation"]
    tr_hp      = cfg["training"]

    # ── Output paths ──────────────────────────────────────────────────
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve {n_timepoints} and keep only the filename (output_dir is the
    # authoritative location; the template may point elsewhere)
    data_path    = out_dir / Path(resolve_path(jm_cfg["data_file"],    n_tp)).name
    indices_path = out_dir / Path(resolve_path(jm_cfg["indices_file"], n_tp)).name

    print(f"[INFO] Timepoints:       {timepoints}")
    print(f"[INFO] Output data    →  {data_path}")
    print(f"[INFO] Output indices →  {indices_path}")

    # ── 1. Load restricted gene list ──────────────────────────────────
    df_gene_list = load_dataframe(
        resolve_path(cfg["paths"]["genes_kept_restricted"], n_tp), index_col=0
    )
    if "gene_kept_restricted" not in df_gene_list.columns:
        raise ValueError(
            "genes_kept_restricted file must contain a 'gene_kept_restricted' "
            "column.  Check paths.genes_kept_restricted in config.yaml."
        )
    gene_list_restricted = df_gene_list["gene_kept_restricted"].unique().tolist()
    print(f"[INFO] Restricted gene list: {len(gene_list_restricted)} genes")

    # ── 2. Pretrain genomic-interaction encoder ───────────────────────
    print("[INFO] Loading pretraining cohort …")
    df_pretrain, gene_names, cln_names = prepare_pretraining_data(cfg)
    print(
        f"[INFO] Pretraining cohort: {len(df_pretrain)} patients | "
        f"{len(gene_names)} genes | {len(cln_names)} clinical features"
    )

    print("[INFO] Pretraining encoder …")
    net_SB = pretrain_encoder(df_pretrain, gene_names, cln_names, cfg)
    net_SB.eval()
    print("[INFO] Pretraining complete.")

    # ── 3. Load target cohort ─────────────────────────────────────────
    print("[INFO] Loading target cohort …")
    df_target, _ = prepare_target_data(cfg)
    n_patients   = len(df_target)
    print(f"[INFO] Target cohort: {n_patients} patients")

    # Recover patient IDs (prepare_target_data drops the ID column)
    patient_ids = _get_ordered_patient_ids(cfg)
    if len(patient_ids) != n_patients:
        raise RuntimeError(
            f"Patient-ID count ({len(patient_ids)}) does not match "
            f"df_target row count ({n_patients}).  Ensure panel_ctdna and "
            f"gene_ctdna files have a consistent set of patients."
        )

    # ── 4. Extract latent encoder features for the full target cohort ─
    print("[INFO] Extracting latent encoder representations …")
    lat_features = extract_latent_features(
        df_target, net_SB, gene_names, timepoints
    )
    # Shape: (n_patients, encoder_dim, n_tp)

    # ── 5. Load baseline OS time ──────────────────────────────────────
    # TTEOS_rebaseline_BL is the BL-referenced survival time required by the
    # joint model (different from the per-cycle OS_rebaseline in df_target).
    tteos_bl_col = jm_cfg["survival_time_col"]   # "TTEOS_rebaseline_BL"

    raw_cln_path = cfg["paths"].get("target_clinical_raw", "")
    if not raw_cln_path:
        raise ValueError(
            "paths.target_clinical_raw is not set in config.yaml.\n"
            "Add:  target_clinical_raw: 'path/to/"
            "anonymized_imp150_ctDNA_patient_clinical_toShare.csv'"
        )
    df_cln_raw = load_dataframe(raw_cln_path)
    if pid_col not in df_cln_raw.columns or tteos_bl_col not in df_cln_raw.columns:
        raise ValueError(
            f"target_clinical_raw file must contain columns "
            f"'{pid_col}' and '{tteos_bl_col}'.  "
            "Check paths.target_clinical_raw and joint_model.survival_time_col "
            "in config.yaml."
        )
    tteos_lookup = df_cln_raw.set_index(pid_col)[tteos_bl_col]
    tteos_bl     = patient_ids.map(tteos_lookup).rename(tteos_bl_col)

    n_missing = tteos_bl.isna().sum()
    if n_missing > 0:
        print(f"[WARN] {n_missing} patients have no '{tteos_bl_col}'; filling with 0.")
        tteos_bl = tteos_bl.fillna(0.0)

    # ── 6. Derive base aggregated-feature names ───────────────────────
    # panel_feature_suffixes: ["_max_of_AF", ...] → ["max_of_AF", ...]
    # (strip the leading "_"; the timepoint prefix is also removed during pivot)
    features_agg_base = [sf.lstrip("_") for sf in cfg["panel_feature_suffixes"]]

    # ── 7. Build long-format DataFrame ────────────────────────────────
    print("[INFO] Pivoting gene-level VAF features to long format …")
    df_genes_long = _pivot_genes_to_long(
        df_target, gene_list_restricted, timepoints
    )

    print("[INFO] Pivoting panel aggregated features to long format …")
    df_agg_long = _pivot_agg_to_long(
        df_target, features_agg_base, timepoints
    )

    # Merge gene long + agg long on (patient index, timepoint)
    df_long = pd.merge(
        df_genes_long, df_agg_long,
        on=["index", "timepoint"], how="inner"
    )

    # Build and merge static (per-patient, timepoint-invariant) columns
    print("[INFO] Attaching clinical, survival, and encoder columns …")
    df_static = _build_static_df(df_target, lat_features, tteos_bl, surv_event)
    df_long   = pd.merge(df_long, df_static, on="index", how="inner")

    print(
        f"[INFO] Long-format table: "
        f"{df_long.shape[0]} rows × {df_long.shape[1]} columns"
    )

    # ── 8. Generate Monte-Carlo CV test-set indices ───────────────────
    # Uses the same DataFrame and random-state scheme as train_ML.py so that
    # train_jm.R is evaluated on exactly the same test splits.
    print(f"[INFO] Generating {eval_cfg['n_seeds']} train/test index splits …")
    ids_tests: list[list[int]] = []
    for seed in tqdm(range(eval_cfg["n_seeds"])):
        test_idx = (
            df_target
            .sample(frac=tr_hp["test_frac"], random_state=eval_cfg["seed_multiplier"]*seed)
            .index.tolist()
        )
        ids_tests.append(test_idx)

    df_indices = pd.DataFrame(ids_tests)

    # ── 9. Save outputs ───────────────────────────────────────────────
    df_long.to_csv(data_path, index=False)
    print(f"[INFO] Saved: {data_path}")

    # row.names=1 in R's read.csv expects the seed index in the first column
    df_indices.to_csv(indices_path, index=True)
    print(f"[INFO] Saved: {indices_path}")

    print("[INFO] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare joint-model inputs for train_jm.R",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    p.add_argument(
        "--timepoints", nargs="+",
        help="Override timepoints from config.yaml, e.g. --timepoints BL C2D1",
    )
    p.add_argument(
        "--output_dir",
        help="Override paths.output_dir from config.yaml",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not Path(args.config).exists():
        sys.exit(
            f"[ERROR] Config file not found: {args.config}\n"
            "Copy config.yaml to your working directory and edit the paths section."
        )

    cfg = load_config(args.config)

    # CLI overrides
    if args.timepoints:
        cfg["timepoints"] = args.timepoints
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir

    # Reuse the ML validation (covers all shared required fields)
    validate_config(cfg, mode="ml")

    prepare_jm_inputs(cfg)
