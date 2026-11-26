"""
competitors_pada1.py — benchmark of the SHERLOC GitHub competitors on PADA-1, landmark OS at C2, with the
early-response features in longitudinal form in the panel-level input (as in
config_SHERLOC_calibrated_response_longitudinal.yaml).

Same inputs for every method:
  gene-level max VAF, 10 genes x {BL, C2}; panel-level features x {BL, C2}: mean VAF, nb drivers > 0.5 %, and the 7
  longitudinal response features; 9 clinical covariates (training-median imputation per split); frozen MSK-CHORD
  genomic-interaction encoder (16 genes, 2 dims) at BL and C2.
Methods (code from https://github.com/AaronMamannToledano/SHERLOC):
  deep  : Featurewise_Transformer, Temporal_Transformer, RNN, GRU, MLP (sherloc/ctDNA_Dynamics_Encoder, identical
          to sherloc_models/), NaN-safe Cox loss; training setting re-selected on tuning splits 100-139 among
          fixed 60 / 100 epochs and early stopping (val 20 % / 30 %).
  ml    : penalised Cox (ridge l1 1e-5, lasso, elastic net 0.5; CoxnetSurvivalAnalysis, 20-alpha path, 4-fold
          GridSearchCV) and RSF (n_estimators {50, 100} x min_samples_split {2, 6}, 4-fold) as in scripts/train_ML.py;
          FPCRF as in scripts/train_ML_FPCA.py (2 FPCs per feature, RSF permutation importance top 20,
          StandardScaler + PCA 10, elastic-net Cox with clinical + encoder features).
  Note: the repo's train_ML.py does not add the clinical covariates to the feature matrix (and does not scale
  features); here they are added so every method sees the same inputs (as described in the paper). A
  standardised variant of the penalised Cox models is also reported.
Metrics: identical code to SHERLOC (Uno C tau 36 mo; iAUC and IBS 3-36 mo; calibration at 24 mo).

Usage: python competitors_pada1.py --group deep|ml|ml_scaled|fpcrf [--n_seeds 100]
Output: results/benchmark_response_longitudinal/
"""
import argparse, copy, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
import train_sherloc_pada1 as T
from mtl_benchmark import metrics
from benchmark import with_training, TRAIN_GRID

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (scripts/)
CFG = "config_SHERLOC_calibrated_response_longitudinal.yaml"
OUT = Path("results/benchmark_response_longitudinal")
DEEP = ["Featurewise_Transformer", "Temporal_Transformer", "RNN", "GRU", "MLP"]
L1 = {"ridge": 1e-5, "lasso": 1.0, "elastic_net": 0.5}
WIN = (3.0, 36.0, 24.0)


def deep(n_seeds):
    cfg = T.load_cfg(CFG)
    cfg["model"]["output_scale"] = 1.0       # competitors keep their own (sigmoid) output
    sel = []
    for m in DEEP:
        best = None
        for s in TRAIN_GRID:
            r, _, _ = T.run_cv(with_training(cfg, m, s), range(100, 140), verbose=False)
            sel.append({"method": m, "training": s, "C_index_Uno": r.C_index_Uno.mean()})
            if best is None or r.C_index_Uno.mean() > best[1]:
                best = (s, r.C_index_Uno.mean())
        print("[select]", m, best, flush=True)
        r, p, _ = T.run_cv(with_training(cfg, m, best[0]), range(n_seeds), verbose=False)
        r.insert(0, "method", m); r["training"] = best[0]
        r.to_csv(OUT / f"{m}_metrics_per_split.csv", index=False)
        p.to_csv(OUT / f"{m}_test_risk_scores.csv", index=False)
        print("[report]", m, r[["C_index_Uno", "iAUC", "IBS", "calib_err_24m"]].mean().round(4).to_dict(), flush=True)
    pd.DataFrame(sel).to_csv(OUT / "deep_training_selection_tuning_splits.csv", index=False)


def tabular():
    """Data, gene list and frozen encoder exactly as run_cv builds them."""
    cfg = T.load_cfg(CFG)
    P, mc, tps = cfg["_P"], cfg["model"], cfg["timepoints"]
    data = Path(P["data_dir"])
    df = pd.read_csv(data / P["dataset_file"] if P.get("dataset_file") else data / "PADA1_SHERLOC_dataset.csv", index_col=0)
    extra = pd.read_csv(data / P["extra_features_file"], index_col=0)
    df = df.join(extra[[c for c in extra.columns if c not in df.columns]], how="left").fillna({c: 0.0 for c in extra.columns})
    freq = pd.read_csv(data / P.get("gene_frequency_file", "gene_frequency.csv"), index_col=0).index.tolist()
    genes = mc.get("genes") or [g for g in freq if f"{g}_BL" in df.columns][: mc["n_top_genes"]]
    net, enc_genes, _ = T.get_encoder(cfg, df, freq)
    lat = T.encode(df, net, enc_genes, tps).reshape(len(df), -1).numpy()
    lat = pd.DataFrame(lat, index=df.index, columns=[f"latent_{i}" for i in range(lat.shape[1])])
    gene_cols = [f"{g}_{tp}" for g in genes for tp in tps]
    panel_cols = [f"{tp}_{p}" for p in mc["panel_features"] for tp in tps]
    cln_cols = [f"cln_{c}" for c in mc["clinical"]]
    return cfg, df, lat, genes, gene_cols, panel_cols, cln_cols


def surv_on_grid(model, X, grid):
    """Predicted survival on a grid; non-finite values (overflow of exp(risk) in an unscaled penalised Cox fit)
    are carried forward in time and otherwise set to 1 (before any event time) — counted in n_nonfinite."""
    sf = model.predict_survival_function(X)
    S = pd.DataFrame(np.vstack([f(np.clip(grid, f.x[0], f.x[-1])) for f in sf]).T, index=grid)
    bad = int((~np.isfinite(S.to_numpy())).sum())
    S = S.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0).clip(0, 1)
    return S, bad


def fit_coxnet(X, y, l1, seed):
    """Repo procedure (20-alpha path, alpha_min_ratio 0.001, 4-fold GridSearchCV). If the path diverges
    (ArithmeticError on unscaled features), retry with alpha_min_ratio 0.01 then 0.1 (logged in FALLBACKS)."""
    for amr in (0.001, 0.01, 0.1):
        try:
            return _fit_coxnet(X, y, l1, seed, amr)
        except ArithmeticError:
            FALLBACKS.append((l1, seed, amr))
    raise ArithmeticError("coxnet failed for every alpha_min_ratio")


FALLBACKS = []


def _fit_coxnet(X, y, l1, seed, amr):
    est = CoxnetSurvivalAnalysis(l1_ratio=l1, n_alphas=20, max_iter=5000, alpha_min_ratio=amr).fit(X, y)
    cv = KFold(n_splits=4, shuffle=True, random_state=seed)
    return GridSearchCV(CoxnetSurvivalAnalysis(l1_ratio=l1, max_iter=5000, fit_baseline_model=True),
                        param_grid={"alphas": [[v] for v in est.alphas_]}, cv=cv, error_score=0.5,
                        n_jobs=1).fit(X, y).best_estimator_


def ml(n_seeds, group):
    from scripts.train_ML_FPCA import _compute_fpca_pca_features
    cfg, df, lat, genes, gene_cols, panel_cols, cln_cols = tabular()
    tps = cfg["timepoints"]
    methods = {"ml": ["ridge", "lasso", "elastic_net", "RSF"], "ml_scaled": ["ridge", "lasso", "elastic_net"],
               "fpcrf": ["FPCRF"]}[group]
    rows = {m: [] for m in methods}
    grid = np.arange(0, WIN[1] + 0.25, 0.25)
    t0 = time.time()
    for seed in range(n_seeds):
        np.random.seed(seed); torch.manual_seed(seed)
        te = df.sample(frac=0.3, random_state=4 * seed); tr = df.drop(te.index)
        med = tr[cln_cols].median()
        C_tr, C_te = tr[cln_cols].fillna(med), te[cln_cols].fillna(med)
        y_tr = Surv.from_arrays(event=tr.OS_event.astype(bool), time=tr.OS_rebaseline)
        d_tr, d_te = tr.assign(OS_t=tr.OS_rebaseline, OS_e=tr.OS_event), te.assign(OS_t=te.OS_rebaseline, OS_e=te.OS_event)
        X_tr = pd.concat([tr[gene_cols + panel_cols], C_tr, lat.loc[tr.index]], axis=1)
        X_te = pd.concat([te[gene_cols + panel_cols], C_te, lat.loc[te.index]], axis=1)
        if group == "ml_scaled":
            sc = StandardScaler().fit(X_tr)
            X_tr = pd.DataFrame(sc.transform(X_tr), index=X_tr.index, columns=X_tr.columns)
            X_te = pd.DataFrame(sc.transform(X_te), index=X_te.index, columns=X_te.columns)
        for m in methods:
            if m in L1:
                model = fit_coxnet(X_tr, y_tr, L1[m], seed)
                Xm_tr, Xm_te = X_tr, X_te
            elif m == "RSF":
                cv = KFold(n_splits=4, shuffle=True, random_state=seed)
                model = GridSearchCV(RandomSurvivalForest(random_state=1),
                                     param_grid={"n_estimators": [50, 100], "min_samples_split": [2, 6]},
                                     cv=cv, error_score=0.5, n_jobs=2).fit(X_tr, y_tr).best_estimator_
                Xm_tr, Xm_te = X_tr, X_te
            else:   # FPCRF
                to3 = lambda d, cols, n: torch.tensor(d[cols].to_numpy(np.float32).reshape(len(d), n, len(tps)))
                g_tr, g_te = to3(tr, gene_cols, len(genes)), to3(te, gene_cols, len(genes))
                pc = [f"{tp}_{p}" for p in cfg["model"]["panel_features"] for tp in tps]
                npf = len(cfg["model"]["panel_features"])
                p_tr = torch.tensor(np.stack([tr[[f"{tp}_{p}" for tp in tps]].to_numpy(np.float32) for p in cfg["model"]["panel_features"]], 1))
                p_te = torch.tensor(np.stack([te[[f"{tp}_{p}" for tp in tps]].to_numpy(np.float32) for p in cfg["model"]["panel_features"]], 1))
                fcfg = {"n_fpca_components": 2, "n_top_features": 20, "n_pca_components": 10,
                        "n_rsf_estimators_importance": 100, "timepoint_spacing_months": 0.75}
                pca_tr, pca_te = _compute_fpca_pca_features(g_tr, p_tr, g_te, p_te, y_tr, tps, fcfg, seed)
                Xm_tr = pd.concat([pd.DataFrame(pca_tr, index=tr.index).add_prefix("fpca_"), C_tr, lat.loc[tr.index]], axis=1)
                Xm_te = pd.concat([pd.DataFrame(pca_te, index=te.index).add_prefix("fpca_"), C_te, lat.loc[te.index]], axis=1)
                Xm_tr.columns = Xm_te.columns = [str(c) for c in Xm_tr.columns]
                model = fit_coxnet(Xm_tr, y_tr, 0.5, seed)
            risk = np.nan_to_num(model.predict(Xm_te), nan=0.0, posinf=1e6, neginf=-1e6)
            surv, bad = surv_on_grid(model, Xm_te, grid)
            r = metrics(d_tr, d_te, risk, surv, "OS", WIN)
            rows[m].append({"seed": seed, "C_index_Uno": r["C_index_Uno"], "iAUC": r["iAUC"], "IBS": r["IBS"],
                            "calib_err_24m": r["calib_err"], "pred_S24_range": r["pred_S_range"], "n_nonfinite": bad})
        if seed % 10 == 0:
            print(f"[{group} seed {seed}] " + " | ".join(f"{m} C={np.mean([x['C_index_Uno'] for x in rows[m]]):.4f}" for m in methods)
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    if FALLBACKS:
        pd.DataFrame(FALLBACKS, columns=["l1_ratio", "seed", "alpha_min_ratio_failed"]).to_csv(
            OUT / f"coxnet_path_fallbacks_{group}.csv", index=False)
        print(f"[{group}] coxnet path fallbacks: {len(FALLBACKS)}", flush=True)
    for m in methods:
        name = m + ("_scaled" if group == "ml_scaled" else "")
        pd.DataFrame(rows[m]).assign(method=name).to_csv(OUT / f"{name}_metrics_per_split.csv", index=False)
        print("[report]", name, pd.DataFrame(rows[m]).drop(columns="seed").mean().round(4).to_dict(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["deep", "ml", "ml_scaled", "fpcrf"])
    ap.add_argument("--n_seeds", type=int, default=100)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    deep(a.n_seeds) if a.group == "deep" else ml(a.n_seeds, a.group)
