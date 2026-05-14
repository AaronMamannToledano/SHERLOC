"""
utils.py — Data loading, pretraining, and latent feature extraction
utilities shared by train_DL.py and train_ML.py.

Importing this module is the only thing either training script needs to do
to get the full pretraining pipeline; no logic is duplicated between them.

Public API
----------
load_config          Load a YAML config file into a dict.
resolve_path         Substitute {n_timepoints} in path templates.
validate_config      Raise early with a clear message on bad config.
load_dataframe       Load CSV/TSV with a helpful FileNotFoundError.
PretrainingDataset   PyTorch Dataset for the genomic-interaction encoder.
prepare_pretraining_data  Full preprocessing of the pretraining cohort.
pretrain_encoder     Train MLP_SolidBiopsy on the pretraining cohort.
extract_latent_features  Apply the frozen encoder to a target-cohort DataFrame.
prepare_target_data  Load and merge the longitudinal target-cohort files.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torchtuples as tt
from pycox.models import CoxPH
from sklearn.exceptions import ConvergenceWarning
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import yaml

from sherloc.Genomic_Interaction_Encoder.MLP_SolidBiopsy import MLP_SolidBiopsy

__all__ = [
    "load_config",
    "resolve_path",
    "validate_config",
    "load_dataframe",
    "PretrainingDataset",
    "prepare_pretraining_data",
    "pretrain_encoder",
    "extract_latent_features",
    "prepare_target_data",
]

VALID_DL_METHODS = {
    "Proposed_Method",
    "Featurewise_Transformer",
    "Temporal_Transformer",
    "RNN",
    "GRU",
    "MLP",
}

VALID_ML_METHODS = {"RSF", "penalized_cox"}
VALID_ML_SUBMETHODS = {"elastic_net", "lasso", "ridge"}


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load a YAML configuration file and return it as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(template: str, n_timepoints: int) -> str:
    """Replace the ``{n_timepoints}`` placeholder in a path string."""
    return template.replace("{n_timepoints}", str(n_timepoints))


def validate_config(cfg: dict, mode: str = "dl") -> None:
    """
    Raise a clear ``ValueError`` if required config keys are missing or invalid.

    Args:
        cfg:  Loaded config dict.
        mode: ``"dl"`` to validate deep-learning fields;
              ``"ml"`` to additionally validate ML-specific fields.
    """
    required_paths = [
        "pretraining_mutations", "pretraining_cln_sample",
        "pretraining_cln_patient", "genes_shared",
        "genes_kept_restricted", "gene_ctdna", "panel_ctdna",
    ]
    for key in required_paths:
        if not cfg.get("paths", {}).get(key):
            raise ValueError(
                f"paths.{key} is not set in config.yaml"
            )

    if not cfg.get("timepoints"):
        raise ValueError("timepoints list is empty in config.yaml")

    if mode == "dl":
        method = cfg.get("method", "")
        if method not in VALID_DL_METHODS:
            raise ValueError(
                f"Unknown DL method '{method}'. "
                f"Choose from: {sorted(VALID_DL_METHODS)}"
            )

    if mode == "ml":
        ml_cfg = cfg.get("ml", {})
        method = ml_cfg.get("method", "")
        if method not in VALID_ML_METHODS:
            raise ValueError(
                f"Unknown ML method '{method}'. "
                f"Choose from: {sorted(VALID_ML_METHODS)}"
            )
        if method == "penalized_cox":
            sub = ml_cfg.get("sub_method", "")
            if sub not in VALID_ML_SUBMETHODS:
                raise ValueError(
                    f"Unknown sub_method '{sub}' for penalized_cox. "
                    f"Choose from: {sorted(VALID_ML_SUBMETHODS)}"
                )


def load_dataframe(path: str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV or TSV file, raising a helpful error if the file is missing.

    The separator is inferred from the file extension (``.txt`` / ``.tsv``
    → tab; everything else → comma).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {p}\n"
            "Check the 'paths' section of your config.yaml."
        )
    sep = "\t" if p.suffix in (".txt", ".tsv") else ","
    return pd.read_csv(p, sep=sep, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Pretraining dataset
# ─────────────────────────────────────────────────────────────────────────────

class PretrainingDataset(Dataset):
    """
    PyTorch Dataset for pretraining the genomic-interaction encoder.

    Returns per-patient gene-level VAF features, clinical features, and
    survival targets as tensors.

    Args:
        df:                 Preprocessed wide-format pretraining DataFrame.
        gene_names:         Ordered list of gene column names.
        cln_names:          Ordered list of clinical column names.
        survival_time_col:  Column name for survival time.
        survival_event_col: Column name for survival event (0/1 int).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_names: list[str],
        cln_names: list[str],
        survival_time_col: str,
        survival_event_col: str,
    ):
        self.df        = df.reset_index(drop=True)
        self.gene_names = gene_names
        self.cln_names  = cln_names
        self.time_col   = survival_time_col
        self.event_col  = survival_event_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        gene_feat = torch.tensor(
            row[self.gene_names].to_numpy(dtype=np.float32).reshape(1, -1)
        )
        cln_feat = torch.tensor(
            row[self.cln_names].to_numpy(dtype=np.float32).reshape(1, -1)
        )
        targets = (
            float(row[self.time_col]),
            int(row[self.event_col]),
        )
        return gene_feat, targets, cln_feat


# ─────────────────────────────────────────────────────────────────────────────
# Pretraining data preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_pretraining_data(
    cfg: dict,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Load and preprocess the pretraining mutation + clinical dataset.

    Steps performed:
    1. Load mutation file; compute VAF; filter silent / unknown mutations.
    2. Restrict to genes shared with the target panel.
    3. Pivot to patient × gene wide format; drop rare genes.
    4. Merge with stage, OS, and one-hot cancer-type features.

    Args:
        cfg: Loaded configuration dict.

    Returns:
        Tuple of ``(df, gene_names, cln_names)`` where ``df`` is the
        merged DataFrame ready for encoder training, ``gene_names`` is
        the ordered list of gene VAF column names, and ``cln_names`` is
        the ordered list of clinical column names.
    """
    pcols  = cfg["pretraining_columns"]
    psurv  = cfg["pretraining_survival"]
    n_tp   = len(cfg["timepoints"])
    min_prev = cfg["evaluation"].get("min_gene_prevalence", 0.02)

    # ── 1. Mutation file ─────────────────────────────────────────────
    df_mut = load_dataframe(
        resolve_path(cfg["paths"]["pretraining_mutations"], n_tp)
    )
    needed = [
        pcols["patient_id"], pcols["gene_symbol"], pcols["variant_class"],
        pcols["mutation_status"], pcols["ref_count"], pcols["alt_count"],
    ]
    missing = [c for c in needed if c not in df_mut.columns]
    if missing:
        raise ValueError(
            f"Mutation file is missing columns: {missing}\n"
            "Update 'pretraining_columns' in config.yaml."
        )

    df_mut = df_mut[needed].copy()
    df_mut["allele_frequency"] = (
        df_mut[pcols["alt_count"]]
        / (df_mut[pcols["alt_count"]] + df_mut[pcols["ref_count"]])
    )
    df_mut = df_mut[df_mut[pcols["mutation_status"]] != "UNKNOWN"]
    df_mut = df_mut[df_mut[pcols["variant_class"]] != "Silent"]

    # Trim patient ID to first two dash-separated parts
    df_mut["PATIENT_ID"] = (
        df_mut[pcols["patient_id"]].str.split("-").str[:2].str.join("-")
    )

    # ── 2. Shared gene list ──────────────────────────────────────────
    df_genes = load_dataframe(
        resolve_path(cfg["paths"]["genes_shared"], n_tp), index_col=0
    )
    if "gene" not in df_genes.columns:
        raise ValueError(
            "genes_shared file must contain a 'gene' column. "
            "Check paths.genes_shared in config.yaml."
        )
    shared_genes = (
        set(df_genes["gene"]) & set(df_mut[pcols["gene_symbol"]].unique())
    )
    df_mut = df_mut[df_mut[pcols["gene_symbol"]].isin(shared_genes)]

    # ── 3. Patient × gene wide matrix ───────────────────────────────
    df_wide = (
        df_mut.groupby(["PATIENT_ID", pcols["gene_symbol"]])["allele_frequency"]
        .max()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    gene_cols = [c for c in df_wide.columns if c != "PATIENT_ID"]
    # Drop genes mutated in fewer than min_prev of patients
    keep_genes = [
        g for g in gene_cols
        if (df_wide[g] == 0).mean() <= (1.0 - min_prev)
    ]
    df_wide = df_wide[["PATIENT_ID"] + keep_genes]

    # ── 4a. Clinical: stage + OS ──────────────────────────────────────
    df_cln_pat = load_dataframe(
        resolve_path(cfg["paths"]["pretraining_cln_patient"], n_tp),
        skiprows=4,
    )
    for col in [psurv["patient_id"], psurv["stage"], psurv["time"], psurv["event"]]:
        if col not in df_cln_pat.columns:
            raise ValueError(
                f"Patient clinical file is missing column '{col}'. "
                "Update pretraining_survival in config.yaml."
            )

    df_cln_pat = df_cln_pat[
        [psurv["patient_id"], psurv["stage"], psurv["time"], psurv["event"]]
    ].copy()
    df_cln_pat["Highest_Stage"] = np.nan
    df_cln_pat.loc[df_cln_pat[psurv["stage"]] == "Stage 1-3", "Highest_Stage"] = 0.0
    df_cln_pat.loc[df_cln_pat[psurv["stage"]] == "Stage 4",   "Highest_Stage"] = 1.0
    df_cln_pat = df_cln_pat[df_cln_pat[psurv["stage"]] != "Unknown"]

    df_cln_pat["OS_EVENT"] = df_cln_pat[psurv["event"]].map(
        {"1:DECEASED": 1, "0:LIVING": 0}
    )
    if df_cln_pat["OS_EVENT"].isna().all():
        raise ValueError(
            f"Could not parse survival event column '{psurv['event']}'. "
            "Expected values '1:DECEASED' / '0:LIVING'. "
            "Update pretraining_survival.event in config.yaml."
        )
    df_cln_pat["OS_EVENT"] = df_cln_pat["OS_EVENT"].astype(int)
    df_cln_pat = df_cln_pat.rename(columns={psurv["time"]: "OS_MONTHS"})
    df_cln_pat = df_cln_pat[
        [psurv["patient_id"], "Highest_Stage", "OS_MONTHS", "OS_EVENT"]
    ]

    # ── 4b. Clinical: cancer type + TMB ──────────────────────────────
    df_cln_smp = load_dataframe(
        resolve_path(cfg["paths"]["pretraining_cln_sample"], n_tp),
        skiprows=4,
    )
    cancer_types = cfg.get("cancer_types", [])
    if not cancer_types:
        cancer_types = df_cln_smp[pcols["cancer_type"]].dropna().unique().tolist()
        print(
            f"[INFO] No cancer_types specified — "
            f"auto-detected {len(cancer_types)} types."
        )

    df_dummies = pd.get_dummies(
        df_cln_smp[
            [psurv["patient_id"], pcols["cancer_type"], pcols["tmb"]]
        ],
        columns=[pcols["cancer_type"]]
    )
    # Normalise column names: strip prefix, spaces, hyphens
    df_dummies.columns = [
        c.replace(f"{pcols['cancer_type']}_", "")
         .replace(" ", "_")
         .replace("-", "_")
        for c in df_dummies.columns
    ]
    cancer_type_cols = [
        c.replace(" ", "_").replace("-", "_") for c in cancer_types
    ]
    present_ct_cols = [c for c in cancer_type_cols if c in df_dummies.columns]
    if not present_ct_cols:
        raise ValueError(
            "None of the cancer_types in config.yaml were found in the "
            "sample file. Check the cancer_type column name and values."
        )

    agg_dict = {c: (c, "sum") for c in present_ct_cols}
    agg_dict[pcols["tmb"]] = (pcols["tmb"], "max")
    df_dummies = (
        df_dummies.groupby(psurv["patient_id"]).agg(**agg_dict).reset_index()
    )

    # ── 4c. Merge all ────────────────────────────────────────────────
    df = df_wide.merge(df_cln_pat, on="PATIENT_ID", how="inner")
    df = df.merge(
        df_dummies, left_on="PATIENT_ID",
        right_on=psurv["patient_id"], how="inner"
    )
    df = df[df["OS_MONTHS"].notna()]
    df = df.drop(columns=["PATIENT_ID"], errors="ignore")

    cln_names = ["Highest_Stage"] + present_ct_cols + [pcols["tmb"]]
    gene_names = [
        c for c in df.columns
        if c not in ["OS_MONTHS", "OS_EVENT"] + cln_names
    ]

    return df, gene_names, cln_names


# ─────────────────────────────────────────────────────────────────────────────
# Encoder pretraining
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_encoder(
    df_pretrain: pd.DataFrame,
    gene_names: list[str],
    cln_names: list[str],
    cfg: dict,
) -> MLP_SolidBiopsy:
    """
    Pretrain the genomic-interaction encoder (``MLP_SolidBiopsy``) on the
    pretraining cohort using the Cox proportional-hazards loss.

    Args:
        df_pretrain: Preprocessed pretraining DataFrame (output of
                     :func:`prepare_pretraining_data`).
        gene_names:  Ordered list of gene VAF column names.
        cln_names:   Ordered list of clinical column names.
        cfg:         Loaded configuration dict.

    Returns:
        Trained ``MLP_SolidBiopsy`` network in eval mode.
    """
    pt_hp  = cfg["pretraining"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(1)
    torch.manual_seed(1)

    df_val   = df_pretrain.sample(frac=pt_hp["val_frac"], random_state=1)
    df_train = df_pretrain.drop(df_val.index)

    train_ds = PretrainingDataset(
        df_train, gene_names, cln_names, "OS_MONTHS", "OS_EVENT"
    )
    val_ds = PretrainingDataset(
        df_val, gene_names, cln_names, "OS_MONTHS", "OS_EVENT"
    )
    trainloader = DataLoader(train_ds, batch_size=len(df_train), shuffle=False)
    valloader   = DataLoader(val_ds,   batch_size=len(df_val),   shuffle=False)

    net = MLP_SolidBiopsy(
        d_input=len(gene_names),
        d_representations=2,
        d_output=1,
        d_other_input=len(cln_names),
        device=device,
        dropout=pt_hp["dropout"],
    )
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(pt_hp["lr"])

    # Single batch for now. The .fit() method will split this into smaller batches later. 
    for gene_feat, targets, cln_feat in trainloader:
        pass
    for v_gene, v_targets, v_cln in valloader:
        val_data = tt.tuplefy(
            [v_gene.float(), v_cln.float()], v_targets
        )

    model.fit(
        [gene_feat.float(), cln_feat.float()],
        targets,
        pt_hp["batch_size"],
        pt_hp["epochs"],
        [tt.callbacks.EarlyStopping(patience=pt_hp["patience"])],
        verbose=False,
        val_data=val_data.repeat(10).cat(),
    )

    net.eval()
    return net


# ─────────────────────────────────────────────────────────────────────────────
# Latent feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_latent_features(
    dataset: pd.DataFrame,
    net_SB: MLP_SolidBiopsy,
    gene_names: list[str],
    timepoints: list[str],
) -> torch.Tensor:
    """
    Apply the frozen genomic-interaction encoder to a target-cohort DataFrame
    and return stacked latent representations across all timepoints.

    .. note::
        The original script contained a bug where ``return`` was placed
        *inside* the ``for timepoint`` loop, so only the first timepoint's
        features were ever returned.  This function fixes that.

    Args:
        dataset:    Target-cohort DataFrame with columns
                    ``<gene>_<timepoint>`` for each gene / timepoint.
        net_SB:     Trained (frozen) ``MLP_SolidBiopsy`` encoder.
        gene_names: Ordered gene list used during pretraining.
        timepoints: Ordered list of timepoint labels (e.g. ``["BL", "C2D1"]``).

    Returns:
        Tensor of shape ``(n_patients, encoder_dim, n_timepoints)``.
    """
    net_SB.eval()
    stacked = []
    with torch.no_grad():
        for tp in timepoints:
            # Build an (n_patients × n_genes) matrix; fill missing genes with 0
            mat = np.zeros((len(dataset), len(gene_names)), dtype=np.float32)
            for i, gene in enumerate(gene_names):
                col = f"{gene}_{tp}"
                if col in dataset.columns:
                    mat[:, i] = dataset[col].to_numpy(dtype=np.float32)
            enc = net_SB.extract_encoder(torch.from_numpy(mat)).detach().cpu()
            stacked.append(enc)                   # ← return is now OUTSIDE the loop
    return torch.stack(stacked, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Target dataset preparation (shared by both DL and ML pipelines)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_target_data(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Load, merge, and filter the longitudinal target-cohort files
    (gene-level VAF + panel-level aggregated ctDNA features).

    Columns retained:
    - Aggregated panel features: ``<timepoint><suffix>`` for each
      timepoint / suffix combination defined in config.
    - Clinical features: all columns whose name starts with ``cln_``.
    - Survival columns: ``target_survival.time`` and ``target_survival.event``.
    - Gene-level VAF columns: ``<gene>_<timepoint>`` filtered to those
      matching the configured timepoints.

    Args:
        cfg: Loaded configuration dict.

    Returns:
        Tuple of ``(df_target, agg_feature_cols)`` where ``df_target`` is
        the merged DataFrame and ``agg_feature_cols`` is the ordered list of
        aggregated panel-feature column names (useful for downstream feature
        selection in ML models).
    """
    tsurv   = cfg["target_survival"]
    pf_suf  = cfg["panel_feature_suffixes"]
    pid_col = cfg["target_patient_id"]
    n_tp    = len(cfg["timepoints"])
    timepoints = cfg["timepoints"]

    agg_feature_cols = [
        f"{tp}{sf}" for tp in timepoints for sf in pf_suf
    ]

    # ── panel-level features ─────────────────────────────────────────
    df_panel = load_dataframe(
        resolve_path(cfg["paths"]["panel_ctdna"], n_tp), index_col=0
    )
    # df_panel = df_panel.reset_index(drop=False).rename(columns={"index": pid_col})

    missing_agg = [c for c in agg_feature_cols if c not in df_panel.columns]
    if missing_agg:
        raise ValueError(
            f"panel_ctdna file is missing columns: {missing_agg}\n"
            "Check timepoints and panel_feature_suffixes in config.yaml."
        )
    df_panel = df_panel[[pid_col] + agg_feature_cols]

    # ── gene-level VAF features ──────────────────────────────────────
    df_gene = load_dataframe(
        resolve_path(cfg["paths"]["gene_ctdna"], n_tp), index_col=0
    )
    df_gene = df_gene.drop(columns=[c for c in df_gene.columns if "Agg" in c])
    df_gene = df_gene.reset_index(drop=False).rename(columns={"index": pid_col})

    # ── merge ────────────────────────────────────────────────────────
    df = pd.merge(df_gene, df_panel, on=pid_col, how="inner")
    df = df.drop(columns=[pid_col])

    # ── filter columns ───────────────────────────────────────────────
    # Gene-level VAF columns: must end in _<timepoint> for a configured timepoint
    tp_suffixes = tuple(f"_{tp}" for tp in timepoints)
    gene_tp_cols = [
        c for c in df.columns
        if c.endswith(tp_suffixes)
        and "Agg" not in c
        and not c.startswith("cln_")
        and tsurv["time"] not in c
        and tsurv["event"] not in c
    ]
    # Survival + clinical columns
    survival_and_cln = [
        c for c in df.columns
        if c.startswith("cln_")
        or c == tsurv["time"]
        or c == tsurv["event"]
    ]
    keep = list(dict.fromkeys(agg_feature_cols + survival_and_cln + gene_tp_cols))
    df = df[[c for c in keep if c in df.columns]]

    return df, agg_feature_cols

# ─────────────────────────────────────────────────────────────────────────────
# Statistical utilities 
# ─────────────────────────────────────────────────────────────────────────────

def calculate_corrected_confidence_intervals(df_coefs, n_train, n_test):
    
    """
    Compute the 95% confidence intervals for model coefficients using the Nadeau and 
    Bengio correction for repeated cross-validation.
    
    Parameters:
    df_coefs (pd.DataFrame): DataFrame containing the coefficients from each iteration of 
        the repeated cross-validation. Each row corresponds to an iteration (100 iterations) 
        and each column corresponds to a coefficient of the model. The DataFrame should 
        have dimensions (100, number_of_coefficients).
    n_train (float): Proportion of the dataset used for training in each iteration.
    n_test (float): Proportion of the dataset used for testing in each iteration.
    
    Returns:
    pd.DataFrame: Résumé avec Moyenne, Borne Inférieure et Borne Supérieure.
    """
    
    # Number of iterations for repeated cross-validation
    n = len(df_coefs)  
    
    # Compute the mean and variance of the coefficients across iterations
    means = df_coefs.mean()
    variances = df_coefs.var(ddof=1)
    
    # Apply the Nadeau and Bengio correction to the variance estimates to account 
    # for the dependence between iterations in repeated cross-validation.
    # Formula : Corrected_Variance = Variance * (1/n + n_test/n_train)
    correction_factor = (1/n) + (n_test/n_train)
    corrected_variances = variances * correction_factor
    corrected_std_errors = np.sqrt(corrected_variances)
    
    # Compute the 95% confidence intervals using the corrected standard errors and 
    # the t-distribution critical value for n-1 degrees of freedom.
    t_critical = stats.t.ppf(0.975, df=n-1)
    margin_of_error = t_critical * corrected_std_errors
    lower_bound = means - margin_of_error
    upper_bound = means + margin_of_error
    
    # Calculate t-statistics and p-values for testing the null hypothesis that each coefficient is equal to zero.
    t_stats = means / corrected_std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - 1))
    
    # Create a DataFrame to summarize the results, including the mean coefficient, confidence intervals, corrected 
    # standard error, t-statistic, and p-value for each coefficient.
    results = pd.DataFrame({
        'Moyenne': means,
        'IC_Inf_95': lower_bound,
        'IC_Sup_95': upper_bound,
        'Ecart_Type_Corrige': corrected_std_errors,
        't_stat': t_stats,
        'p_value': p_values
    })
    
    return results
