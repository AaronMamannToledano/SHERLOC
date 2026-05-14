"""
train_DL.py — Deep learning survival models with a pre-trained genomic encoder.

Pipeline
--------
1. Pretrain a genomic-interaction encoder (MLP_SolidBiopsy) on the pretraining
   cohort (e.g. MSK-CHORD) using the Cox PH loss.           [shared_utils]
2. Extract per-patient latent gene features from the frozen encoder at every
   configured longitudinal timepoint.                        [shared_utils]
3. Load and merge the longitudinal target-cohort files.      [shared_utils]
4. Run Monte-Carlo Cross-Validation (default: 100 seeds) on the target cohort
   (e.g. IMpower150) with the selected deep learning model:
   - ``Proposed_Method``        : SHERLOC
   - ``Featurewise_Transformer``: feature-wise self-attention transformer
   - ``Temporal_Transformer``   : temporal self-attention transformer
   - ``RNN``                    : recurrent neural network
   - ``GRU``                    : gated recurrent unit
   - ``MLP``                    : MLP DeepSurv baseline
5. Report C-index (IPCW), cumulative AUC, and integrated Brier score per seed.

Usage
-----
  python train_DL.py                             # reads config.yaml
  python train_DL.py --config my_cfg.yaml        # custom config file
  python train_DL.py --method GRU                # override a single field
  python train_DL.py --timepoints BL C2D1 C3D1   # three-timepoint run

See config.yaml for the full list of tuneable settings.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxPH
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sksurv.metrics
import sksurv.util
from sksurv.metrics import integrated_brier_score

# ── DL model architectures ────────────────────────────────────────────────────
from sherloc.ctDNA_Dynamics_Encoder.SHERLOC import SHERLOC
from sherloc.ctDNA_Dynamics_Encoder.Featurewise_Transformer import Featurewise_Transformer
from sherloc.ctDNA_Dynamics_Encoder.Temporal_Transformer import Temporal_Transformer
from sherloc.ctDNA_Dynamics_Encoder.RNN import RNN_timesteps
from sherloc.ctDNA_Dynamics_Encoder.GRU import GRU_timesteps
from sherloc.ctDNA_Dynamics_Encoder.MLP_DeepSurv import MLP_DeepSurv

# ── shared pipeline utilities ─────────────────────────────────────────────────
# Data loading, pretraining, and latent feature extraction are all centralised
# in shared_utils so that train_DL.py and train_ML.py stay in sync without
# code duplication.
from shared_utils import (
    load_config,
    resolve_path,
    validate_config,
    load_dataframe,
    prepare_pretraining_data,
    pretrain_encoder,
    extract_latent_features,
    prepare_target_data,
)

VALID_METHODS = {
    "Proposed_Method":         SHERLOC,
    "Featurewise_Transformer": Featurewise_Transformer,
    "Temporal_Transformer":    Temporal_Transformer,
    "RNN":                     RNN_timesteps,
    "GRU":                     GRU_timesteps,
    "MLP":                     MLP_DeepSurv,
}


# ─────────────────────────────────────────────────────────────────────────────
# DL-specific dataset
# ─────────────────────────────────────────────────────────────────────────────

class LongitudinalSurvivalDataset(Dataset):
    """
    PyTorch Dataset for the main DL survival model on the target cohort.

    Combines four input modalities per patient:
    - Gene-level VAF features across longitudinal timepoints.
    - Panel-level aggregated ctDNA features across longitudinal timepoints.
    - Clinical features.
    - Pre-computed latent encoder features (from the frozen MLP_SolidBiopsy).

    Args:
        df:                  Target-cohort DataFrame (``reset_index`` applied
                             internally).
        gene_list:           Genes included in the ctDNA-dynamics encoder
                             (mutated in >= 5% of target-cohort patients).
        panel_features:      Panel-level feature suffixes (e.g. ``_max_of_AF``);
                             columns are looked up as ``<timepoint><suffix>``.
        clinical_features:   Clinical column names present in ``df``.
        timepoints:          Ordered list of timepoint labels.
        survival_time_col:   Survival time column name.
        survival_event_col:  Survival event column name (0 / 1 int).
        tensor_gene_encoded: Pre-computed latent encoder tensor,
                             shape ``(n_patients, encoder_dim, n_timepoints)``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_list: list,
        panel_features: list,
        clinical_features: list,
        timepoints: list,
        survival_time_col: str,
        survival_event_col: str,
        tensor_gene_encoded: torch.Tensor,
    ):
        self.df                  = df.reset_index(drop=True)
        self.gene_list           = gene_list
        self.panel_features      = panel_features
        self.clinical_features   = clinical_features
        self.timepoints          = timepoints
        self.time_col            = survival_time_col
        self.event_col           = survival_event_col
        self.tensor_gene_encoded = tensor_gene_encoded

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]

        # Gene-level VAFs: (n_genes, n_timepoints)
        gene_vaf = np.stack([
            row[[f"{g}_{t}" for t in self.timepoints]].to_numpy(dtype=np.float32)
            for g in self.gene_list
        ])

        # Panel-level aggregated features: (n_panel_features, n_timepoints)
        panel = np.stack([
            row[[f"{t}{f}" for t in self.timepoints]].to_numpy(dtype=np.float32)
            for f in self.panel_features
        ])

        cln = row[self.clinical_features].to_numpy(dtype=np.float32).reshape(1, -1)

        targets = (float(row[self.time_col]), int(row[self.event_col]))

        return (
            torch.tensor(gene_vaf),
            targets,
            torch.tensor(cln),
            torch.tensor(panel),
            self.tensor_gene_encoded[idx].float(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def deep_model(cfg: dict) -> None:
    """
    Full DL survival pipeline: pretraining -> feature extraction ->
    Monte-Carlo CV -> result export.

    All data loading, pretraining, and latent feature extraction are
    delegated to ``shared_utils``; this function owns only the DL
    training loop and evaluation.

    Args:
        cfg: Loaded and validated configuration dict.
    """
    # ── unpack config ─────────────────────────────────────────────────────────
    method     = cfg["method"]
    timepoints = cfg["timepoints"]
    d_channel  = len(timepoints)
    n_tp       = d_channel
    tr_hp      = cfg["training"]
    eval_cfg   = cfg["evaluation"]
    tsurv      = cfg["target_survival"]
    t_cln      = cfg["target_clinical_features"]
    pf_suf     = cfg["panel_feature_suffixes"]
    surv_time  = tsurv["time"]
    surv_evt   = tsurv["event"]
    t_min_ibs  = eval_cfg["time_min_ibs"]
    t_min_auc  = eval_cfg["time_min_auc"]
    t_max      = eval_cfg["time_max"]

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{method}_{n_tp}tp_metrics.csv"

    print(f"[INFO] Method:     {method}")
    print(f"[INFO] Timepoints: {timepoints}")
    print(f"[INFO] Results ->  {result_path}")

    # ── 1. Pretraining data preparation (shared_utils) ────────────────────────
    print("[INFO] Loading pretraining data ...")
    df_pretrain, gene_names, cln_names = prepare_pretraining_data(cfg)
    print(
        f"[INFO] Pretraining: {len(df_pretrain)} patients | "
        f"{len(gene_names)} genes | {len(cln_names)} clinical features"
    )

    # ── 2. Pretrain genomic-interaction encoder (shared_utils) ────────────────
    print("[INFO] Pretraining genomic-interaction encoder ...")
    net_SB = pretrain_encoder(df_pretrain, gene_names, cln_names, cfg)
    print("[INFO] Pretraining complete.")

    # ── 3. Load target cohort (shared_utils) ──────────────────────────────────
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

    # Validate that clinical columns expected by the Dataset exist
    missing_cln = [c for c in t_cln if c not in df_target.columns]
    if missing_cln:
        raise ValueError(
            f"Target dataset is missing clinical columns: {missing_cln}\n"
            "Update target_clinical_features in config.yaml."
        )

    # ── 4. Monte-Carlo Cross-Validation ──────────────────────────────────────
    get_target = lambda df: (df[surv_time].values, df[surv_evt].values)
    C_INDEX, AUC, IBS = [], [], []
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Running {eval_cfg['n_seeds']} Monte-Carlo CV iterations ...")
    for seed in tqdm(range(eval_cfg["n_seeds"])):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ── train / val / test split ──────────────────────────────────
        df_test      = df_target.sample(frac=tr_hp["test_frac"], random_state=seed)
        df_train_val = df_target.drop(df_test.index)
        df_val       = df_train_val.sample(frac=tr_hp["val_frac"], random_state=seed)
        df_train     = df_train_val.drop(df_val.index)

        # ── latent encoder features (shared_utils) ────────────────────
        lat_train = extract_latent_features(df_train, net_SB, gene_names, timepoints)
        lat_val   = extract_latent_features(df_val,   net_SB, gene_names, timepoints)
        lat_test  = extract_latent_features(df_test,  net_SB, gene_names, timepoints)

        # ── PyTorch Datasets and DataLoaders ──────────────────────────
        def make_dataset(df_split, lat):
            return LongitudinalSurvivalDataset(
                df_split, gene_list_restricted, pf_suf, t_cln,
                timepoints, surv_time, surv_evt, lat,
            )

        train_ds = make_dataset(df_train, lat_train)
        val_ds   = make_dataset(df_val,   lat_val)
        test_ds  = make_dataset(df_test,  lat_test)

        trainloader = DataLoader(train_ds, batch_size=len(df_train), shuffle=False)
        valloader   = DataLoader(val_ds,   batch_size=len(df_val),   shuffle=False)
        testloader  = DataLoader(test_ds,  batch_size=len(df_test),  shuffle=False)

        # Single batch for now. The .fit() method will split this into smaller batches later. 
        for inputs, targets_b, other_inputs, agg_inputs, gene_enc in trainloader:
            pass
        for vi, vt, vo, va, vge in valloader:
            val_data = tt.tuplefy(
                [vi.float(), va.float(), vo.float(), vge.float()], vt
            )

        # ── model initialisation ──────────────────────────────────────
        # Derive input dimensions directly from the first dataset sample
        # so they stay consistent with the data regardless of config changes.
        d_input = train_ds[0][0].shape[0]   # number of genes
        d_agg   = train_ds[0][3].shape[0]   # number of panel-level features
        d_other = train_ds[0][2].shape[1]   # number of clinical features

        net = VALID_METHODS[method](
            d_input=d_input,
            d_agg_input=d_agg,
            d_other_input=d_other,
            d_channel=d_channel,
            d_output=1,
            device=DEVICE,
        )
        model = CoxPH(net, tt.optim.Adam)
        model.optimizer.set_lr(tr_hp["lr"])

        # ── training ──────────────────────────────────────────────────
        model.fit(
            [inputs.float(), agg_inputs.float(),
             other_inputs.float(), gene_enc.float()],
            targets_b,
            tr_hp["batch_size"],
            tr_hp["epochs"],
            [tt.callbacks.EarlyStopping(patience=tr_hp["patience"])],
            verbose=False,
            val_data=val_data.repeat(10).cat(),
        )

        # ── inference ─────────────────────────────────────────────────
        for ti, _, to, ta, tge in testloader:
            pass

        _ = model.compute_baseline_hazards()

        # Bundle test inputs once to avoid repeating the list literal
        test_inputs = [ti.float(), ta.float(), to.float(), tge.float()]

        # ── survival targets ──────────────────────────────────────────
        dur_train, ev_train = get_target(df_train_val)
        dur_test,  ev_test  = get_target(df_test)
        y_train = sksurv.util.Surv.from_arrays(event=ev_train, time=dur_train)
        y_test  = sksurv.util.Surv.from_arrays(event=ev_test,  time=dur_test)

        risk_scores = model.predict(test_inputs).flatten()

        # ── C-index (IPCW) ────────────────────────────────────────────
        c_uno = sksurv.metrics.concordance_index_ipcw(
            y_train, y_test, estimate=risk_scores
        )[0]
        C_INDEX.append(c_uno)

        # ── cumulative dynamic AUC ────────────────────────────────────
        times_auc = np.arange(t_min_auc, t_max)
        auc = sksurv.metrics.cumulative_dynamic_auc(
            y_train, y_test, estimate=risk_scores, times=times_auc
        )[1]
        AUC.append(auc)

        # ── integrated Brier score ────────────────────────────────────
        surv_df     = model.predict_surv_df(test_inputs)
        surv_window = surv_df[
            (surv_df.index >= t_min_ibs) & (surv_df.index <= t_max)
        ]
        ibs = integrated_brier_score(
            y_train, y_test,
            estimate=np.transpose(surv_window.to_numpy()),
            times=np.array(surv_window.index),
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
        description="SHERLOC and baseline DL survival models — longitudinal ctDNA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    p.add_argument(
        "--method", choices=list(VALID_METHODS),
        help="Override method from config.yaml"
    )
    p.add_argument(
        "--timepoints", nargs="+",
        help="Override timepoints, e.g. --timepoints BL C2D1 C3D1"
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
        cfg["method"] = args.method
    if args.timepoints:
        cfg["timepoints"] = args.timepoints
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir

    # mode="dl" checks cfg["method"] against VALID_DL_METHODS in shared_utils
    validate_config(cfg, mode="dl")
    deep_model(cfg)
