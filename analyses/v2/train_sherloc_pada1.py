"""
train_sherloc_pada1.py — SHERLOC on PADA-1, landmark OS at C2.

Pipeline (mirrors SHERLOC/scripts/train_DL.py)
----------------------------------------------
1. Pretrain the genomic-interaction encoder (MLP_SolidBiopsy) on MSK-CHORD,
   restricted to the chosen genes shared by the PADA-1 panel and MSK-CHORD.
2. Freeze it and encode PADA-1 max-VAF profiles at BL and C2.
3. Monte-Carlo splits (70 % train / 30 % test, random_state = 4*seed); part of
   the training set is held out for early stopping.
4. Test metrics (sksurv): Uno's IPCW C-index, cumulative/dynamic iAUC, IBS.

Every modelling choice (features, transforms, encoder genes, optimisation) is
read from the ``model`` section of the config so that variants can be compared
with experiments.py.

Usage
-----
  python train_sherloc_pada1.py --config config.yaml              # final 100 splits
  python train_sherloc_pada1.py --config config.yaml --n_seeds 10 # quick check
"""

import argparse
import copy
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
import yaml
from pycox.models import CoxPH, LogisticHazard
from scipy import stats
from sksurv.metrics import (concordance_index_ipcw, cumulative_dynamic_auc,
                            integrated_brier_score)
from sksurv.util import Surv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sherloc_models.SHERLOC import SHERLOC                    # noqa: E402
from sherloc_models.MLP_SolidBiopsy import MLP_SolidBiopsy    # noqa: E402
from sherloc_models.Featurewise_Transformer import Featurewise_Transformer  # noqa: E402
from sherloc_models.Temporal_Transformer import Temporal_Transformer        # noqa: E402
from sherloc_models.RNN import RNN_timesteps                  # noqa: E402
from sherloc_models.GRU import GRU_timesteps                  # noqa: E402
from sherloc_models.MLP_DeepSurv import MLP_DeepSurv          # noqa: E402
from sherloc_models.SHERLOC_v2 import SHERLOC_v2              # noqa: E402
from sherloc_models.novel import SHERLOC_NAM, SHERLOC_Proto     # noqa: E402

# Same names as SHERLOC/scripts/train_DL.py
METHODS = {
    "Proposed_Method":         SHERLOC,
    "Featurewise_Transformer": Featurewise_Transformer,
    "Temporal_Transformer":    Temporal_Transformer,
    "RNN":                     RNN_timesteps,
    "GRU":                     GRU_timesteps,
    "MLP":                     MLP_DeepSurv,
    "SHERLOC_v2":              SHERLOC_v2,   # PADA-1 variant: optional linear output
    "SHERLOC_NAM":             SHERLOC_NAM,  # neural additive model (optionally monotone in C2)
    "SHERLOC_Proto":           SHERLOC_Proto,  # trajectory prototypes
}

DEVICE = torch.device("cpu")
_ENCODER_CACHE = {}


class SafeCoxPHLoss(torch.nn.Module):
    """pycox CoxPHLoss, but a mini-batch without any event contributes 0 (not 0/0 = NaN).

    With the default pycox loss, one event-free batch (e.g. the small remainder
    batch) makes the loss NaN, the weights become NaN, and early stopping silently
    falls back to the last finite epoch, so the network barely trains.
    """

    def forward(self, log_h, durations, events):
        events = events.float().view(-1)
        if events.sum() == 0:
            return log_h.sum() * 0.0
        from pycox.models.loss import cox_ph_loss
        return cox_ph_loss(log_h, durations, events)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MSK-CHORD pretraining (port of utils.prepare_pretraining_data / pretrain_encoder)
# ─────────────────────────────────────────────────────────────────────────────

_MSK_RAW = {}


def _load_msk(P):
    if "m" not in _MSK_RAW:
        m = pd.read_csv(P["msk_mutations"], sep="\t", low_memory=False,
                        usecols=["Tumor_Sample_Barcode", "Hugo_Symbol", "Variant_Classification",
                                 "Mutation_Status", "t_ref_count", "t_alt_count"])
        m["allele_frequency"] = m["t_alt_count"] / (m["t_alt_count"] + m["t_ref_count"])
        m = m[(m["Mutation_Status"] != "UNKNOWN") & (m["Variant_Classification"] != "Silent")]
        m["PATIENT_ID"] = m["Tumor_Sample_Barcode"].str.split("-").str[:2].str.join("-")
        _MSK_RAW["m"] = m
        _MSK_RAW["cp"] = pd.read_csv(P["msk_cln_patient"], sep="\t", skiprows=4)
        _MSK_RAW["cs"] = pd.read_csv(P["msk_cln_sample"], sep="\t", skiprows=4)
    return _MSK_RAW["m"], _MSK_RAW["cp"], _MSK_RAW["cs"]


MSK_COHORTS = {   # matched subtypes for encoder pretraining
    "breast_hrpos_her2neg": "PADA-1: HR+/HER2- breast cancer (patient-level HR = Yes, HER2 = No)",
    "nsclc_nonsquamous": "IMpower150: non-squamous NSCLC (detailed type not squamous / adenosquamous / "
                         "carcinoid / neuroendocrine / sarcomatoid)",
}


def msk_pfs(P, cp, min_day=30):
    """PFS from first sequencing (day 0 of the MSK-CHORD timelines, also the OS origin).
    Event = first radiology report labelled progression (NLP, PROGRESSION == 'Y') at day >= min_day, or death.
    Censoring (alive, no progression) at the last radiology report. Reports before min_day are ignored: they
    mostly describe the disease state that led to sequencing. Returns PATIENT_ID, PFS_MONTHS, PFS_EVENT."""
    if "prog" not in _MSK_RAW:
        f = P.get("msk_timeline_progression") or str(Path(P["msk_cln_patient"]).parent / "data_timeline_progression.txt")
        _MSK_RAW["prog"] = pd.read_csv(f, sep="\t", usecols=["PATIENT_ID", "START_DATE", "PROGRESSION"])
    pr = _MSK_RAW["prog"]
    pr = pr[pr["START_DATE"] >= min_day]
    first_y = pr[pr["PROGRESSION"] == "Y"].groupby("PATIENT_ID")["START_DATE"].min()
    last_scan = pr.groupby("PATIENT_ID")["START_DATE"].max()
    d = cp[["PATIENT_ID", "OS_MONTHS", "OS_STATUS"]].set_index("PATIENT_ID")
    os_days = d["OS_MONTHS"] * 30.4375
    dead = d["OS_STATUS"] == "1:DECEASED"
    fy = first_y.reindex(d.index)
    ls = last_scan.reindex(d.index)
    prog = fy.notna() & (fy <= os_days + 1)
    t = np.where(prog, fy, np.where(dead, os_days, ls))
    e = (prog | dead).astype(int)
    out = pd.DataFrame({"PATIENT_ID": d.index, "PFS_MONTHS": np.asarray(t, float) / 30.4375, "PFS_EVENT": e.values})
    return out[out["PFS_MONTHS"].notna() & (out["PFS_MONTHS"] > 0)]


def prepare_msk(P, genes_in, pt, breast_only=False, binary=False, clonal=False, endpoint="OS", cohort="pan"):
    m, cp, cs = _load_msk(P)
    if cohort == "breast_hrpos_her2neg":
        cs = cs[cs["CANCER_TYPE"] == "Breast Cancer"]
        cp = cp[(cp["HR"] == "Yes") & (cp["HER2"] == "No")]
    elif cohort == "nsclc_nonsquamous":
        cs = cs[(cs["CANCER_TYPE"] == "Non-Small Cell Lung Cancer") & ~cs["CANCER_TYPE_DETAILED"].fillna("").str.contains(
            "Squamous|Adenosquamous|Carcinoid|Neuroendocrine|Sarcomatoid|Small Cell Carcinoma", regex=True)]
    if cohort != "pan":
        m = m[m["PATIENT_ID"].isin(set(cs["PATIENT_ID"]) & set(cp["PATIENT_ID"]))]
        breast_only = True                # single cancer type: no cancer-type indicators
    if clonal:                       # VAF / patient's max VAF (all genes): purity-free clonal fraction
        m = m.copy()
        m["allele_frequency"] = m["allele_frequency"] / m.groupby("PATIENT_ID")["allele_frequency"].transform("max")
    m = m[m["Hugo_Symbol"].isin(genes_in)].copy()
    if binary:                       # mutated / not mutated: transfers across tissue and cfDNA
        m["allele_frequency"] = 1.0
    wide = (m.groupby(["PATIENT_ID", "Hugo_Symbol"])["allele_frequency"].max()
            .unstack(fill_value=0).reset_index().rename_axis(None, axis=1))

    cp = cp[cp["STAGE_HIGHEST_RECORDED"] != "Unknown"].copy()
    cp["Highest_Stage"] = cp["STAGE_HIGHEST_RECORDED"].map({"Stage 1-3": 0.0, "Stage 4": 1.0})
    cp["OS_EVENT"] = cp["OS_STATUS"].map({"1:DECEASED": 1, "0:LIVING": 0})
    cp = cp[["PATIENT_ID", "Highest_Stage", "OS_MONTHS", "OS_EVENT"]]

    if endpoint == "PFS":             # PFS takes the (internal) OS_MONTHS / OS_EVENT slots
        pf = msk_pfs(P, _load_msk(P)[1])
        cp = cp.drop(columns=["OS_MONTHS", "OS_EVENT"]).merge(
            pf.rename(columns={"PFS_MONTHS": "OS_MONTHS", "PFS_EVENT": "OS_EVENT"}), on="PATIENT_ID")

    cs = cs.copy()
    if breast_only and cohort == "pan":
        cs = cs[cs["CANCER_TYPE"] == "Breast Cancer"]
        ct_cols = []
    elif breast_only:
        ct_cols = []
    else:
        ct_cols = []
        for ct in pt["cancer_types"]:
            col = ct.replace(" ", "_").replace("-", "_")
            cs[col] = (cs["CANCER_TYPE"] == ct).astype(int)
            ct_cols.append(col)
    agg = {c: (c, "sum") for c in ct_cols}
    agg["TMB_NONSYNONYMOUS"] = ("TMB_NONSYNONYMOUS", "max")
    cs = cs.groupby("PATIENT_ID").agg(**agg).reset_index()

    # patients with no mutation in the selected genes are kept with zeros
    df = cs.merge(cp, on="PATIENT_ID").merge(wide, on="PATIENT_ID", how="left")
    gene_cols = [g for g in wide.columns if g != "PATIENT_ID"]
    df[gene_cols] = df[gene_cols].fillna(0.0)
    genes = [g for g in gene_cols if (df[g] > 0).mean() >= pt["min_gene_prevalence"]]
    df = df[df["OS_MONTHS"].notna() & df["OS_EVENT"].notna() & df["Highest_Stage"].notna()]
    df["TMB_NONSYNONYMOUS"] = df["TMB_NONSYNONYMOUS"].fillna(df["TMB_NONSYNONYMOUS"].median())
    cln = ["Highest_Stage"] + ct_cols + ["TMB_NONSYNONYMOUS"]
    return df.reset_index(drop=True), genes, cln


def pretrain_encoder(df, genes, cln, pt):
    np.random.seed(1); torch.manual_seed(1)
    df_val = df.sample(frac=pt["val_frac"], random_state=1)
    df_tr = df.drop(df_val.index)

    def arrays(d):
        x = torch.tensor(d[genes].to_numpy(np.float32)).unsqueeze(1)
        c = torch.tensor(d[cln].to_numpy(np.float32)).unsqueeze(1)
        y = (torch.tensor(d["OS_MONTHS"].to_numpy(np.float32)),
             torch.tensor(d["OS_EVENT"].to_numpy(np.int64)))
        return x, c, y

    xt, ct, yt = arrays(df_tr)
    xv, cv, yv = arrays(df_val)
    net = MLP_SolidBiopsy(d_input=len(genes), d_representations=2, d_output=1,
                          d_other_input=len(cln), device=DEVICE, dropout=pt["dropout"])
    model = CoxPH(net, tt.optim.Adam, loss=SafeCoxPHLoss())
    model.optimizer.set_lr(pt["lr"])
    model.fit([xt, ct], yt, pt["batch_size"], pt["epochs"],
              [tt.callbacks.EarlyStopping(patience=pt["patience"])], verbose=False,
              val_data=tt.tuplefy([xv, cv], yv))
    risk = model.predict([xv, cv]).flatten()
    yv_s = Surv.from_arrays(event=yv[1].numpy().astype(bool), time=yv[0].numpy())
    yt_s = Surv.from_arrays(event=yt[1].numpy().astype(bool), time=yt[0].numpy())
    c_val = concordance_index_ipcw(yt_s, yv_s, risk, tau=60)[0]
    net.eval()
    return net, c_val


def get_encoder(cfg, df, freq_rank):
    """Select encoder genes, pretrain (cached) and return (net, genes)."""
    P, pt, mc = cfg["_P"], cfg["pretraining"], cfg["model"]
    shared = pd.read_csv(Path(P["data_dir"]) / P.get("genes_shared_file", "genes_shared_PADA1_MSK.csv"),
                         index_col=0)["gene"].tolist()
    k = mc.get("encoder_n_genes")
    if k:  # the k shared genes most frequently mutated in PADA-1
        genes_in = [g for g in freq_rank if g in shared][:k]
    else:
        genes_in = shared
    key = (tuple(genes_in), bool(mc.get("encoder_breast_only", False)), bool(mc.get("encoder_binary", False)),
           bool(mc.get("encoder_clonal", False)), mc.get("encoder_endpoint", "OS"), mc.get("encoder_cohort", "pan"))
    if key not in _ENCODER_CACHE:
        d, genes, cln = prepare_msk(P, genes_in, pt, breast_only=key[1], binary=key[2], clonal=key[3],
                                    endpoint=key[4], cohort=key[5])
        net, c_val = pretrain_encoder(d, genes, cln, pt)
        print(f"[INFO] Encoder: {len(d)} MSK patients ({int(d.OS_EVENT.sum())} events), {len(genes)} genes, "
              f"breast_only={key[1]}, endpoint={key[4]}, cohort={key[5]}, MSK val Uno-C={c_val:.3f}", flush=True)
        _ENCODER_CACHE[key] = (net, genes, c_val)
    return _ENCODER_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tensors
# ─────────────────────────────────────────────────────────────────────────────

def encode(df, net_sb, enc_genes, tps, binary=False, clonal=False):
    out = []
    with torch.no_grad():
        for tp in tps:
            x = df[[f"ENC_{g}_{tp}" for g in enc_genes]].to_numpy(np.float32)
            if clonal:               # divide by the patient's max VAF at this timepoint (% -> fraction)
                mx = df[f"{tp}_max_VAF"].to_numpy(np.float32)[:, None] / 100.0
                x = np.divide(x, mx, out=np.zeros_like(x), where=mx > 0)
            x = torch.tensor((x > 0).astype(np.float32) if binary else x)
            out.append(net_sb.extract_encoder(x).cpu())
    return torch.stack(out, dim=-1)                                   # (n, 2, n_tp)


def to_tensors(df, genes, panel, cln, tps, lat, log1p, log_eps=None):
    if log_eps:                      # log(x + eps): linear temporal layers can then form log-ratios
        f = lambda a: np.log(a + log_eps)
    else:
        f = np.log1p if log1p else (lambda a: a)
    x_gene = np.stack([f(df[[f"{g}_{t}" for t in tps]].to_numpy(np.float32)) for g in genes], axis=1)
    x_pan = np.stack([f(df[[f"{t}_{p}" for t in tps]].to_numpy(np.float32)) for p in panel], axis=1)
    x_cln = df[cln].to_numpy(np.float32)[:, None, :]
    y = (torch.tensor(df["OS_rebaseline"].to_numpy(np.float32)),
         torch.tensor(df["OS_event"].to_numpy(np.int64)))
    return [torch.tensor(x_gene), torch.tensor(x_pan), torch.tensor(x_cln), lat.float()], y


# ─────────────────────────────────────────────────────────────────────────────
# 3. Monte-Carlo CV
# ─────────────────────────────────────────────────────────────────────────────

def calibration_error(surv, d_test, risk, t=24.0, n_groups=3):
    """Mean |predicted - Kaplan-Meier observed| survival at time t across risk tertiles of the test set."""
    from sksurv.nonparametric import kaplan_meier_estimator
    s_t = surv.iloc[(np.abs(surv.index - t)).argmin()].to_numpy()
    groups = pd.qcut(pd.Series(risk).rank(method="first"), n_groups, labels=False).to_numpy()
    errs = []
    for g in range(n_groups):
        m = groups == g
        tt, ss = kaplan_meier_estimator(d_test["OS_event"].to_numpy()[m].astype(bool),
                                        d_test["OS_rebaseline"].to_numpy()[m])
        obs = ss[tt <= t][-1] if (tt <= t).any() else 1.0
        errs.append(abs(s_t[m].mean() - obs))
    return float(np.mean(errs))


def nadeau_bengio_ci(x, test_frac):
    x = np.asarray(x, float); n = len(x)
    se = np.sqrt(x.var(ddof=1) * (1 / n + test_frac / (1 - test_frac)))
    t = stats.t.ppf(0.975, n - 1)
    return x.mean() - t * se, x.mean() + t * se


def run_cv(cfg, seeds, verbose=True, keep=None):
    """keep: optional list; receives (seed, model, d_test, X_te, X_tr, d_tr) for each split (explanations)."""
    P = cfg["_P"]
    tps, tr, ev, mc = cfg["timepoints"], cfg["training"], cfg["evaluation"], cfg["model"]
    data = Path(P["data_dir"])
    df = pd.read_csv(data / P.get("dataset_file", "PADA1_SHERLOC_dataset.csv"), index_col=0)
    if P.get("extra_features_file"):  # e.g. event_tokens.csv (EV_<token>_<tp>)
        extra = pd.read_csv(data / P["extra_features_file"], index_col=0)
        df = df.join(extra[[c for c in extra.columns if c not in df.columns]], how="left").fillna(
            {c: 0.0 for c in extra.columns})
    if ev.get("endpoint", "OS") != "OS":   # e.g. PFS: PFS_rebaseline / PFS_event take the internal endpoint slots
        e = ev["endpoint"]
        df = df.drop(columns=["OS_rebaseline", "OS_event"]).rename(
            columns={f"{e}_rebaseline": "OS_rebaseline", f"{e}_event": "OS_event"})
        df = df[df["OS_rebaseline"].notna() & (df["OS_rebaseline"] > 0)]
    freq_rank = pd.read_csv(data / P.get("gene_frequency_file", "gene_frequency.csv"), index_col=0).index.tolist()
    df["cln_none"] = 0.0                   # constant placeholder: clinical: [none] = model without clinical features

    if mc.get("genes_file"):         # e.g. IMpower150 genes_restricted.csv (column gene_kept_restricted)
        genes = pd.read_csv(data / mc["genes_file"], index_col=0).iloc[:, 0].tolist()
    else:
        genes = mc.get("genes") or [g for g in freq_rank if f"{g}_BL" in df.columns][: mc["n_top_genes"]]
    ev_tokens = [f"EV_{t}" for t in (mc.get("event_tokens") or [])]
    if ev_tokens and mc.get("event_static_bl"):   # control: freeze every token at its baseline value
        df = df.copy()
        for t in ev_tokens:
            for tp in tps[1:]:
                df[f"{t}_{tp}"] = df[f"{t}_{tps[0]}"]
    genes = genes + ev_tokens              # event tokens ride on the gene tensor (last rows)
    panel = mc["panel_features"]
    # clinical arm = clinical covariates + optional static ctDNA-response features (dyn_*)
    cln = [f"cln_{c}" for c in mc["clinical"]] + list(mc.get("static_features") or [])
    enc_bin = bool(mc.get("encoder_binary", False))
    enc_clonal = bool(mc.get("encoder_clonal", False))
    net_sb, enc_genes, _ = get_encoder(cfg, df, freq_rank)

    rows, preds = [], []
    t_min, t_max = ev["time_min"], ev["time_max"]
    for seed in seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        rs = ev["seed_multiplier"] * seed
        if cfg.get("_test_ids") is not None:   # fixed external train / test split (clinical_applications.py)
            d_test = df[df.index.isin(cfg["_test_ids"])]
        else:
            d_test = df.sample(frac=tr["test_frac"], random_state=rs)
        d_trv = df.drop(d_test.index)
        if tr["val_frac"] > 0:
            d_val = d_trv.sample(frac=tr["val_frac"], random_state=rs)
            d_tr = d_trv.drop(d_val.index)
        else:
            d_val, d_tr = None, d_trv

        med = d_trv[cln].median()          # clinical NA imputation: training medians
        imp = lambda d: None if d is None else d.assign(**{c: d[c].fillna(med[c]) for c in cln})
        d_tr, d_val, d_test, d_trv = imp(d_tr), imp(d_val), imp(d_test), imp(d_trv)

        mk = lambda d: to_tensors(d, genes, panel, cln, tps, encode(d, net_sb, enc_genes, tps, enc_bin, enc_clonal),
                                  mc.get("log1p", False), mc.get("log_eps"))
        X_tr, y_tr = mk(d_tr)
        X_te, _ = mk(d_test)

        X_va, y_va = mk(d_val) if d_val is not None else (None, None)
        bs = len(d_tr) if tr["batch_size"] == "full" else tr["batch_size"]
        risks, survs, n_eps = [], [], []
        for k in range(int(tr.get("n_ensemble", 1))):        # seed ensemble: average members
            if k > 0:                      # member 0 keeps the split seed (reproduces single-model runs)
                torch.manual_seed(seed * 1000 + k)
            name = mc.get("method", "Proposed_Method")
            kw = {k: mc[k] for k in ("output_sigmoid", "panel_mode", "panel_delta", "gene_delta", "use_gene_arm",
                                     "output_scale", "gene_mode", "gene_hidden", "gene_input",
                                     "use_panel_arm", "use_encoder_arm", "event_mode")
                  if k in mc} if name == "SHERLOC_v2" else {}
            kw.update(mc.get("arch_kwargs") or {})
            if ev_tokens and name == "SHERLOC_v2":
                kw["n_event_tokens"] = len(ev_tokens)
            head = mc.get("head", "cox")
            if head == "logistic_hazard":  # discrete-time hazard (non-proportional), n_cuts intervals
                labtrans = LogisticHazard.label_transform(int(mc.get("n_cuts", 15)))
                y_lh = labtrans.fit_transform(y_tr[0].numpy(), y_tr[1].numpy())
                y_fit = (torch.tensor(y_lh[0], dtype=torch.int64), torch.tensor(y_lh[1], dtype=torch.float32))
                d_out = labtrans.out_features
            else:
                y_fit, d_out = y_tr, 1
            net = METHODS[name](d_input=len(genes), d_agg_input=len(panel), d_channel=len(tps),
                                d_output=d_out, d_other_input=len(cln), device=DEVICE, **kw)
            if tr.get("dropout") is not None and hasattr(net, "dropout") and isinstance(net.dropout, torch.nn.Dropout):
                net.dropout.p = tr["dropout"]
            optim = tt.optim.AdamW(weight_decay=tr["weight_decay"]) if tr.get("weight_decay") else tt.optim.Adam
            if head == "logistic_hazard":
                model = LogisticHazard(net, optim, duration_index=labtrans.cuts)
            else:
                model = CoxPH(net, optim, loss=SafeCoxPHLoss() if tr.get("safe_cox_loss", True) else None)
            model.optimizer.set_lr(tr["lr"])
            if d_val is not None:
                y_va_fit = y_va
                if head == "logistic_hazard":
                    v = labtrans.transform(y_va[0].numpy(), y_va[1].numpy())
                    y_va_fit = (torch.tensor(v[0], dtype=torch.int64), torch.tensor(v[1], dtype=torch.float32))
                log = model.fit(X_tr, y_fit, bs, tr["epochs"],
                                [tt.callbacks.EarlyStopping(patience=tr["patience"])], verbose=False,
                                val_data=tt.tuplefy(X_va, y_va_fit).repeat(10).cat())
            else:                          # fixed number of epochs, no validation split
                log = model.fit(X_tr, y_fit, bs, tr["epochs"], verbose=False)
            if head == "logistic_hazard":
                s_df = model.interpolate(10).predict_surv_df(X_te)
                grid = s_df[s_df.index <= ev["time_max"]]
                # risk = - restricted mean survival time up to time_max
                risks.append(-np.trapezoid(grid.to_numpy(), grid.index.to_numpy(), axis=0))
                survs.append(s_df)
            else:
                _ = model.compute_baseline_hazards()
                risks.append(model.predict(X_te).flatten())
                survs.append(model.predict_surv_df(X_te))
            n_eps.append(len(log.to_pandas()))
            if keep is not None and k == 0:
                keep.append((seed, model, d_test, X_te, X_tr, d_tr))
        risk = np.mean(risks, axis=0)
        y_train = Surv.from_arrays(event=d_trv["OS_event"].astype(bool), time=d_trv["OS_rebaseline"])
        y_test = Surv.from_arrays(event=d_test["OS_event"].astype(bool), time=d_test["OS_rebaseline"])
        t_hi = min(t_max, d_test["OS_rebaseline"].max() - 1e-3, d_trv["OS_rebaseline"].max() - 1e-3)

        c_uno = concordance_index_ipcw(y_train, y_test, risk, tau=t_max)[0]
        # AUC(t) is undefined before the first test death -> start the grid there
        first_death = d_test.loc[d_test["OS_event"] == 1, "OS_rebaseline"].min()
        times = np.arange(max(t_min, np.ceil(first_death)), t_hi, 1.0)
        _, iauc = cumulative_dynamic_auc(y_train, y_test, risk, times)
        surv = sum(survs) / len(survs)
        t_lo = max(t_min, d_test["OS_rebaseline"].min())       # sksurv: grid within test follow-up
        surv = surv[(surv.index >= t_lo) & (surv.index < t_hi)]
        ibs = integrated_brier_score(y_train, y_test, surv.to_numpy().T, surv.index.to_numpy())

        n_ep = int(np.median(n_eps))
        calib = calibration_error(surv, d_test, risk, ev.get("calib_time", 24.0))
        rows.append({"seed": seed, "random_state": rs, "n_train": len(d_trv), "n_test": len(d_test),
                     "events_test": int(d_test["OS_event"].sum()), "epochs": n_ep,
                     "C_index_Uno": c_uno, "iAUC": iauc, "IBS": ibs, "calib_err_24m": calib,
                     "pred_S24_range": float(np.ptp(surv.iloc[(np.abs(surv.index - ev.get("calib_time", 24.0))).argmin()]))})
        preds.append(pd.DataFrame({"seed": seed, "Subject_Id": d_test.index, "risk": risk,
                                   "OS_rebaseline": d_test["OS_rebaseline"].values,
                                   "OS_event": d_test["OS_event"].values}))
        if verbose:
            print(f"[{seed:3d}] C-Uno={c_uno:.3f}  iAUC={iauc:.3f}  IBS={ibs:.3f}  ({n_ep} ep)", flush=True)
    return pd.DataFrame(rows), pd.concat(preds), {"genes": genes, "encoder_genes": enc_genes}


def summarise(res, test_frac):
    out = []
    for m in ["C_index_Uno", "iAUC", "IBS"]:
        lo, hi = nadeau_bengio_ci(res[m], test_frac)
        out.append({"metric": m, "mean": res[m].mean(), "sd": res[m].std(), "median": res[m].median(),
                    "CI95_NB_low": lo, "CI95_NB_high": hi, "n_splits": len(res)})
    return pd.DataFrame(out)


def load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_P"] = {k: os.path.expanduser(v) for k, v in cfg["paths"].items()}
    return cfg


def main(cfg, n_seeds=None):
    ev, tr = cfg["evaluation"], cfg["training"]
    out = Path(cfg["_P"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    tag = cfg["model"].get("name", "SHERLOC")
    t0 = time.time()
    res, preds, info = run_cv(cfg, range(n_seeds or ev["n_seeds"]))
    res.to_csv(out / f"{tag}_LandmarkOS_C2_metrics_per_split.csv", index=False)
    preds.to_csv(out / f"{tag}_test_risk_scores.csv", index=False)
    summ = summarise(res, tr["test_frac"])
    summ.to_csv(out / f"{tag}_LandmarkOS_C2_summary.csv", index=False)
    pd.Series(info["genes"], name="gene").to_csv(out / f"{tag}_gene_level_genes.csv")
    pd.Series(info["encoder_genes"], name="gene").to_csv(out / f"{tag}_encoder_genes.csv")
    with open(out / f"{tag}_config_used.yaml", "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if k != "_P"}, f, sort_keys=False)
    print(f"\n[{tag}] {time.time()-t0:.0f}s\n", summ.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n_seeds", type=int, default=None)
    ap.add_argument("--output_dir", default=None, help="override paths.output_dir")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.output_dir:
        cfg["_P"]["output_dir"] = a.output_dir
    main(cfg, a.n_seeds)
