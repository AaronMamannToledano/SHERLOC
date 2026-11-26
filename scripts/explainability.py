"""
explainability.py — Feature-importance analysis for the trained SHERLOC model.

Pipeline
--------
1. Pretrain the genomic-interaction encoder (MLP_SolidBiopsy) on the
   pretraining cohort.                                       [utils]
2. Load the longitudinal target cohort and extract latent gene features
   from the frozen encoder.                                  [utils]
3. Train SHERLOC once on a fixed, reproducible train/val split (seed from
   config) using the full training procedure from train_DL.py.
4. Bootstrap output-layer retraining (default: 100 iterations):
   - Sample 80 % of the training set without replacement.
   - Clone SHERLOC and load the trained weights.
   - Re-initialise only the output layer (fc_1); freeze everything else.
   - Retrain fc_1 on the bootstrap sample with early stopping.
   - Record the shared gene-filter weights (filter1) and fc_1 weights.
5. Compute corrected confidence intervals and resampled t-test p-values
   (Nadeau & Bengio correction) across bootstrap samples.
6. Assemble and export a results DataFrame indexed by human-readable
   feature names (configured in config.yaml).

Usage
-----
  python explainability.py                        # reads config.yaml
  python explainability.py --config my_cfg.yaml
  python explainability.py --timepoints BL C2D1
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchtuples as tt
from pycox.models import CoxPH
from sherloc.losses import SafeCoxPHLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── DL model ─────────────────────────────────────────────────────────────────
from sherloc.ctDNA_Dynamics_Encoder.SHERLOC import SHERLOC

# ── shared pipeline utilities ─────────────────────────────────────────────────
from scripts.utils import (
    load_config,
    resolve_path,
    validate_config,
    load_dataframe,
    prepare_pretraining_data,
    pretrain_encoder,
    extract_latent_features,
    prepare_target_data,
    calculate_corrected_confidence_intervals
)

# ── DL-specific dataset ──────────────────────────────────────────────────────
from train_DL import LongitudinalSurvivalDataset

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(df_split, gene_list, pf_suf, t_cln,
                 timepoints, surv_time, surv_evt, lat):
    """Convenience wrapper: Dataset → single-batch DataLoader."""
    ds = LongitudinalSurvivalDataset(
        df_split, gene_list, pf_suf, t_cln,
        timepoints, surv_time, surv_evt, lat,
    )
    return DataLoader(ds, batch_size=len(df_split), shuffle=False), ds


def _extract_batch(loader):
    """Pull the single full batch from a full-batch DataLoader."""
    for batch in loader:
        return batch


def _encoder_repr_names(timepoints: list, n_dims: int) -> list:
    """
    Generate display names for the pre-trained encoder output dimensions.

    For each representation dimension, names cycle through all timepoints
    in order, e.g. for n_dims=2 and timepoints ['BL','C2D1']:
        ['BL Representation 1', 'C2D1 Representation 1',
         'BL Representation 2', 'C2D1 Representation 2']
    """
    return [
        f"{tp} Representation {dim}"
        for dim in range(1, n_dims + 1)
        for tp in timepoints
    ]


def _validate_explainability_config(cfg: dict) -> None:
    """Raise ValueError for missing or inconsistent explainability config."""
    exp_cfg  = cfg.get("explainability", {})
    pf_suf   = cfg.get("panel_feature_suffixes", [])
    t_cln    = cfg.get("target_clinical_features", [])

    panel_names = exp_cfg.get("panel_feature_display_names", [])
    cln_names   = exp_cfg.get("clinical_feature_display_names", [])
    n_dims      = exp_cfg.get("encoder_repr_dims", 2)

    expected_panel = len(pf_suf) * n_dims
    if panel_names and len(panel_names) != expected_panel:
        raise ValueError(
            f"explainability.panel_feature_display_names has {len(panel_names)} "
            f"entries but expected {expected_panel} "
            f"({len(pf_suf)} panel suffixes × {n_dims} encoder dims). "
            "Update panel_feature_display_names in config.yaml."
        )

    if cln_names and len(cln_names) != len(t_cln):
        raise ValueError(
            f"explainability.clinical_feature_display_names has {len(cln_names)} "
            f"entries but target_clinical_features has {len(t_cln)}. "
            "They must have the same length and order."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def explainability(cfg: dict) -> None:
    """
    Full explainability pipeline: pretraining -> SHERLOC training ->
    bootstrap output-layer retraining -> CI / p-value export.

    Args:
        cfg: Loaded and validated configuration dict.
    """
    # ── unpack config ─────────────────────────────────────────────────────────
    timepoints = cfg["timepoints"]
    d_channel  = len(timepoints)
    n_tp       = d_channel
    tsurv      = cfg["target_survival"]
    t_cln      = cfg["target_clinical_features"]
    pf_suf     = cfg["panel_feature_suffixes"]
    surv_time  = tsurv["time"]
    surv_evt   = tsurv["event"]
    tr_hp      = cfg["training"]
    exp_cfg    = cfg["explainability"]

    seed           = exp_cfg["seed"]
    n_bootstrap    = exp_cfg["n_bootstrap"]
    bootstrap_frac = exp_cfg["bootstrap_frac"]
    n_dims         = exp_cfg.get("encoder_repr_dims", 2)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "SHERLOC_explainability.csv"

    print(f"[INFO] Timepoints:  {timepoints}")
    print(f"[INFO] Fixed seed:  {seed}")
    print(f"[INFO] Bootstrap:   {n_bootstrap} iterations")
    print(f"[INFO] Results ->   {result_path}")

    # ── 1. Pretraining (utils) ────────────────────────────────────────────────
    print("[INFO] Loading pretraining data ...")
    df_pretrain, gene_names, cln_names = prepare_pretraining_data(cfg)
    print(
        f"[INFO] Pretraining: {len(df_pretrain)} patients | "
        f"{len(gene_names)} genes | {len(cln_names)} clinical features"
    )

    print("[INFO] Pretraining genomic-interaction encoder ...")
    net_SB = pretrain_encoder(df_pretrain, gene_names, cln_names, cfg)
    print("[INFO] Pretraining complete.")

    # ── 2. Target cohort (utils) ──────────────────────────────────────────────
    df_list_genes = load_dataframe(
        resolve_path(cfg["paths"]["genes_kept_restricted"], n_tp), index_col=0
    )
    if "gene_kept_restricted" not in df_list_genes.columns:
        raise ValueError(
            "genes_kept_restricted file must contain a 'gene_kept_restricted' "
            "column. Check paths.genes_kept_restricted in config.yaml."
        )
    gene_list_restricted = df_list_genes["gene_kept_restricted"].unique().tolist()

    print("[INFO] Loading target cohort data ...")
    df_target, _ = prepare_target_data(cfg)
    print(f"[INFO] Target cohort: {len(df_target)} patients")

    missing_cln = [c for c in t_cln if c not in df_target.columns]
    if missing_cln:
        raise ValueError(
            f"Target dataset is missing clinical columns: {missing_cln}\n"
            "Update target_clinical_features in config.yaml."
        )

    # ── 3. Fixed-seed train / val / test split ────────────────────────────────
    np.random.seed(seed)
    torch.manual_seed(seed)

    df_test      = df_target.sample(frac=exp_cfg["test_frac"], random_state=seed)
    df_train_val = df_target.drop(df_test.index)
    df_val       = df_train_val.sample(frac=exp_cfg["val_frac"], random_state=seed)
    df_train     = df_train_val.drop(df_val.index)

    lat_train = extract_latent_features(df_train, net_SB, gene_names, timepoints)
    lat_val   = extract_latent_features(df_val,   net_SB, gene_names, timepoints)

    trainloader, train_ds = _make_loader(
        df_train, gene_list_restricted, pf_suf, t_cln,
        timepoints, surv_time, surv_evt, lat_train,
    )
    valloader, _ = _make_loader(
        df_val, gene_list_restricted, pf_suf, t_cln,
        timepoints, surv_time, surv_evt, lat_val,
    )

    # ── 4. Train SHERLOC once on the full training set ────────────────────────
    # Input dimensions are derived from the dataset to stay consistent with data
    d_input      = train_ds[0][0].shape[0]   # number of genes
    d_agg_input  = train_ds[0][3].shape[0]   # number of panel-level features
    d_other_input = train_ds[0][2].shape[1]  # number of clinical features
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = SHERLOC(
        d_input=d_input,
        d_agg_input=d_agg_input,
        d_other_input=d_other_input,
        d_channel=d_channel,
        d_output=1,
        device=DEVICE,
    )
    model = CoxPH(net, tt.optim.Adam, loss=SafeCoxPHLoss())
    model.optimizer.set_lr(tr_hp["lr"])

    train_batch = _extract_batch(trainloader)
    inputs, targets, other_inputs, agg_inputs, gene_enc = train_batch

    val_batch = _extract_batch(valloader)
    vi, vt, vo, va, vge = val_batch
    val_data = tt.tuplefy([vi.float(), va.float(), vo.float(), vge.float()], vt)

    print("[INFO] Training SHERLOC on the target cohort ...")
    model.fit(
        [inputs.float(), agg_inputs.float(),
         other_inputs.float(), gene_enc.float()],
        targets,
        tr_hp["batch_size"],
        tr_hp["epochs"],
        [tt.callbacks.EarlyStopping(patience=tr_hp["patience"])],
        verbose=False,
        val_data=val_data.repeat(10).cat(),
    )
    print("[INFO] SHERLOC training complete.")

    # ── 5. Bootstrap output-layer retraining ──────────────────────────────────
    # For each bootstrap sample:
    #   (a) clone the trained SHERLOC and load its weights
    #   (b) re-initialise fc_1 (output layer)
    #   (c) freeze all parameters except fc_1
    #   (d) retrain fc_1 on the bootstrap sample
    #   (e) record filter1 weights (shared gene representation) and fc_1 weights
    weight_time_models = []   # filter1 weights per bootstrap
    weights_models     = []   # fc_1 weights per bootstrap

    print(f"[INFO] Running {n_bootstrap} bootstrap output-layer retraining iterations ...")
    for random_state in tqdm(range(n_bootstrap)):

        df_sub_train = df_train.sample(
            frac=bootstrap_frac, replace=False, random_state=10 * random_state
        )
        df_sub_val = df_train.drop(df_sub_train.index)

        lat_sub_train = extract_latent_features(df_sub_train, net_SB, gene_names, timepoints)
        lat_sub_val   = extract_latent_features(df_sub_val,   net_SB, gene_names, timepoints)

        sub_trainloader, _ = _make_loader(
            df_sub_train, gene_list_restricted, pf_suf, t_cln,
            timepoints, surv_time, surv_evt, lat_sub_train,
        )
        sub_valloader, _ = _make_loader(
            df_sub_val, gene_list_restricted, pf_suf, t_cln,
            timepoints, surv_time, surv_evt, lat_sub_val,
        )

        # Clone model and load trained weights
        model_bt_net = SHERLOC(
            d_input=d_input,
            d_agg_input=d_agg_input,
            d_other_input=d_other_input,
            d_channel=d_channel,
            d_output=1,
            device=DEVICE,
        ).to(DEVICE)
        model_bt_net.load_state_dict(model.net.state_dict())

        # Re-initialise output layer
        nn.init.xavier_uniform_(model_bt_net.fc_1.weight)
        nn.init.zeros_(model_bt_net.fc_1.bias)

        # Freeze all parameters, then unfreeze only fc_1
        for p in model_bt_net.parameters():
            p.requires_grad = False
        for p in model_bt_net.fc_1.parameters():
            p.requires_grad = True

        optimizer_bt = optim.Adam(
            filter(lambda p: p.requires_grad, model_bt_net.parameters()),
            lr=exp_cfg["lr"],
        )
        model_bt = CoxPH(model_bt_net, optimizer_bt, loss=SafeCoxPHLoss())

        sub_train_batch = _extract_batch(sub_trainloader)
        bt_inputs, bt_targets, bt_other, bt_agg, bt_gene_enc = sub_train_batch

        sub_val_batch = _extract_batch(sub_valloader)
        bv_i, bv_t, bv_o, bv_a, bv_ge = sub_val_batch
        bt_val = tt.tuplefy(
            [bv_i.float(), bv_a.float(), bv_o.float(), bv_ge.float()], bv_t
        )

        model_bt.fit(
            [bt_inputs.float(), bt_agg.float(),
             bt_other.float(), bt_gene_enc.float()],
            bt_targets,
            exp_cfg["batch_size"],
            exp_cfg["epochs"],
            [tt.callbacks.EarlyStopping(patience=exp_cfg["patience"])],
            verbose=False,
            val_data=bt_val.repeat(10).cat(),
        )

        weight_time_models.append(
            model_bt.net.filter1.weight.detach().cpu().numpy().flatten()
        )
        weights_models.append(
            model_bt.net.fc_1.weight.detach().cpu().numpy().flatten()
        )

    # ── 6. Confidence intervals and p-values (Nadeau & Bengio correction) ─────
    df_bt_weight_time = pd.DataFrame(weight_time_models)
    df_bt_weights     = pd.DataFrame(weights_models)

    resultats_ic = calculate_corrected_confidence_intervals(
        df_bt_weights,
        n_train=len(df_sub_train),
        n_test=len(df_sub_val),
    )

    # ── 7. Build human-readable feature index ─────────────────────────────────
    # Order must match SHERLOC's fc_1 input concatenation:
    #   genes (shared repr.) | panel features | clinical | encoder representations
    panel_display   = exp_cfg.get("panel_feature_display_names") 
    cln_display = exp_cfg.get("clinical_feature_display_names") 
    encoder_display = _encoder_repr_names(timepoints, n_dims)

    feature_names = gene_list_restricted + panel_display + cln_display + encoder_display

    if len(feature_names) != resultats_ic.shape[0]:
        raise ValueError(
            f"Feature name list has {len(feature_names)} entries but the "
            f"results DataFrame has {resultats_ic.shape[0]} rows.\n"
            "Check that panel_feature_display_names and "
            "clinical_feature_display_names in config.yaml are consistent "
            "with panel_feature_suffixes and target_clinical_features."
        )
    resultats_ic.index = feature_names

    # ── 8. Add per-timepoint weights from the trained model ───────────────────
    # Gene shared filter weights (filter1): one weight per timepoint per gene
    df_gene_time = pd.DataFrame(
        model.net.filter1.weight.detach().cpu().numpy(),
        columns=timepoints,
    )
    for tp in timepoints:
        resultats_ic.loc[gene_list_restricted, tp] = df_gene_time[tp].values[0]

    # Panel feature-specific MLP weights: one 2-D projection per panel feature
    panel_weights_rows = []
    for j in range(len(model.net.mlps)):
        w = pd.DataFrame(
            model.net.mlps[j][0].weight.detach().cpu().numpy(),
            columns=timepoints,
        )
        panel_weights_rows.append(w)
    df_panel_weights = pd.concat(panel_weights_rows)
    df_panel_weights.index = panel_display

    for tp in timepoints:
        for feat in panel_display:
            resultats_ic.loc[feat, tp] = df_panel_weights.loc[feat, tp]

    resultats_ic = resultats_ic.fillna(0)

    # ── 9. Export ─────────────────────────────────────────────────────────────
    df_out = resultats_ic.reset_index(drop=False)
    df_out.to_csv(result_path, index=True)
    print(f"\n[INFO] Results saved to {result_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SHERLOC explainability — bootstrap weight CI analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    p.add_argument(
        "--timepoints", nargs="+",
        help="Override timepoints, e.g. --timepoints BL C2D1"
    )
    p.add_argument(
        "--output_dir",
        help="Override paths.output_dir from config.yaml"
    )
    p.add_argument(
        "--seed", type=int,
        help="Override explainability.seed from config.yaml"
    )
    p.add_argument(
        "--n_bootstrap", type=int,
        help="Override explainability.n_bootstrap from config.yaml"
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
    if args.timepoints:
        cfg["timepoints"] = args.timepoints
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["explainability"]["seed"] = args.seed
    if args.n_bootstrap is not None:
        cfg["explainability"]["n_bootstrap"] = args.n_bootstrap

    validate_config(cfg, mode="dl")        # checks paths, timepoints
    _validate_explainability_config(cfg)   # checks display-name lengths
    explainability(cfg)
