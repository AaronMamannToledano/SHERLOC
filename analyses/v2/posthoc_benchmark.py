"""
posthoc_benchmark.py — benchmark of SHERLOC and the GitHub competitors with the same post-hoc recalibration
(5-fold out-of-fold) applied to every method, on IMpower150 (4, 3, 2 timepoints) and PADA-1 (2 timepoints), OS.

Every method (except the joint model) is trained as in the benchmark (raw), then recalibrated:
  1. its risk score eta on the patients it was trained on and on the test patients (ranking unchanged ->
     C-index / iAUC identical before and after recalibration);
  2. K = 5 out-of-fold models on the same training patients (stratified on the event; same hyper-parameters /
     training setting as the full model) -> OOF score for every training patient -> calibration slope beta
     (unpenalised 1-covariate Cox model);
  3. recalibrated test survival S(t | x) = exp(-H0(t) exp(beta * eta_te)), H0 = Breslow on the training patients
     with linear predictor beta * eta_tr.
Risk score per method: deep models and SHERLOC (scale 1, sigmoid output) -> network log-risk; penalised Cox and
FPCRF -> Cox linear predictor; RSF -> log of its predicted mortality (sum of the cumulative hazard).
FPCRF: the full feature pipeline (FPCA, RSF permutation-importance selection, scaling + PCA, alpha selection) is
re-run inside each OOF fold. Penalised Cox / RSF: scaler re-fitted per fold, hyper-parameters (alpha; n_estimators,
min_samples_split) fixed to those selected on the full training split.

Inputs per dataset are those of the existing benchmarks (same splits, same frozen MSK-CHORD encoder for every method):
  impower150_{4,3,2}tp: config_IMP150_{4,3,2}tp.yaml (paper inputs; deep models: early stopping, val 20 %,
                        the repo default and SHERLOC's setting)
  pada1:                config_SHERLOC_response_longitudinal.yaml (response features per timepoint in the panel input;
                        deep competitors: training setting selected on tuning splits 100-139, see
                        results/benchmark_response_longitudinal/deep_training_selection_tuning_splits.csv)
Usage: python posthoc_benchmark.py --dataset impower150_4tp --group deep --methods SHERLOC,RNN --seeds 0-49
Output: results/posthoc_benchmark/<dataset>/<method>_seeds<a>-<b>.csv (rows: seed x variant raw / recalibrated)
"""
import argparse, copy, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, torchtuples as tt
from pycox.models import CoxPH
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
import train_sherloc_pada1 as T
from mtl_benchmark import metrics
from benchmark import with_training
from posthoc_holdout import breslow_surv, beta_fit
from competitors_pada1 import _fit_coxnet, surv_on_grid

FALLBACKS = []


def fit_coxnet(X, y, l1, seed):
    """Repo procedure (20-alpha path, 4-fold GridSearchCV). If the path diverges (ArithmeticError), retry with a larger
    alpha_min_ratio (0.01, 0.1, 0.3); every fallback is logged (coxnet_fallbacks_<method>.csv)."""
    for amr in (0.001, 0.01, 0.1, 0.3):
        try:
            return _fit_coxnet(X, y, l1, seed, amr)
        except ArithmeticError:
            FALLBACKS.append((l1, seed, amr))
    raise ArithmeticError("coxnet failed for every alpha_min_ratio")

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (scripts/)
DATASETS = {
    "impower150_4tp": "config_IMP150_4tp.yaml",
    "impower150_3tp": "config_IMP150_3tp.yaml",
    "impower150_2tp": "config_IMP150_2tp.yaml",
    "pada1": "config_SHERLOC_response_longitudinal.yaml",
}
PADA_SEL = "results/benchmark_response_longitudinal/deep_training_selection_tuning_splits.csv"
DEEP = ["SHERLOC", "Featurewise_Transformer", "Temporal_Transformer", "RNN", "GRU", "MLP"]
L1 = {"ridge": 1e-5, "lasso": 1.0, "elastic_net": 0.5}
K = 5
OUT = Path("results/posthoc_benchmark")


def load_cfg(dataset):
    cfg = T.load_cfg(DATASETS[dataset])
    cfg["model"]["output_scale"] = 1.0
    ev = cfg["evaluation"]
    return cfg, (ev["time_min"], ev["time_max"], ev.get("calib_time", 24.0))


def load_df(cfg):
    """Same table as run_cv (OS endpoint)."""
    P = cfg["_P"]; data = Path(P["data_dir"])
    df = pd.read_csv(data / P.get("dataset_file", "PADA1_SHERLOC_dataset.csv"), index_col=0)
    if P.get("extra_features_file"):
        extra = pd.read_csv(data / P["extra_features_file"], index_col=0)
        df = df.join(extra[[c for c in extra.columns if c not in df.columns]], how="left").fillna(
            {c: 0.0 for c in extra.columns})
    return df


def recal_rows(seed, method, eta_tr, t_tr, e_tr, eta_te, oof, raw_surv, d_trv, d_te, win, extra=None):
    b = beta_fit(oof, t_tr, e_tr)
    dd_tr = d_trv.assign(OS_t=d_trv.OS_rebaseline, OS_e=d_trv.OS_event)
    dd_te = d_te.assign(OS_t=d_te.OS_rebaseline, OS_e=d_te.OS_event)
    rows = []
    for v, s in (("raw", raw_surv), ("recalibrated", breslow_surv(b * eta_tr, t_tr, e_tr, b * eta_te))):
        rows.append({"method": method, "seed": seed, "variant": v, "beta": b, **(extra or {}),
                     **metrics(dd_tr, dd_te, eta_te, s, "OS", win)})
    return rows


# ----------------------------------------------------------------------------------------------- deep models
def fit_net(cfg, X, y, dims, seed):
    """Same construction / training as run_cv (incl. inner early stopping when val_frac > 0)."""
    tr, mc, tps = cfg["training"], cfg["model"], cfg["timepoints"]
    torch.manual_seed(seed); np.random.seed(seed)
    name = mc.get("method", "SHERLOC_v2")
    kw = {k: mc[k] for k in ("output_sigmoid", "panel_mode", "panel_delta", "gene_delta", "use_gene_arm",
                             "output_scale", "gene_mode", "gene_hidden", "gene_input", "use_panel_arm",
                             "use_encoder_arm", "event_mode") if k in mc} if name == "SHERLOC_v2" else {}
    kw.update(mc.get("arch_kwargs") or {})
    net = T.METHODS[name](d_input=dims[0], d_agg_input=dims[1], d_channel=len(tps), d_output=1,
                          d_other_input=dims[2], device=T.DEVICE, **kw)
    if tr.get("dropout") is not None and hasattr(net, "dropout") and isinstance(net.dropout, torch.nn.Dropout):
        net.dropout.p = tr["dropout"]
    optim = tt.optim.AdamW(weight_decay=tr["weight_decay"]) if tr.get("weight_decay") else tt.optim.Adam
    model = CoxPH(net, optim, loss=T.SafeCoxPHLoss())
    model.optimizer.set_lr(tr["lr"])
    n = len(y[0])
    sub = lambda X, m: [x[torch.tensor(m)] for x in X]
    if tr["val_frac"] > 0:
        rng = np.random.RandomState(seed)
        va = np.zeros(n, bool); va[rng.choice(n, int(round(tr["val_frac"] * n)), replace=False)] = True
        ys = lambda m: (y[0][torch.tensor(m)], y[1][torch.tensor(m)])
        bs = int((~va).sum()) if tr["batch_size"] == "full" else tr["batch_size"]
        while int((~va).sum()) % bs == 1:    # avoid a last batch of 1 patient (batch norm fails on it)
            bs += 1
        model.fit(sub(X, ~va), ys(~va), bs, tr["epochs"], [tt.callbacks.EarlyStopping(patience=tr["patience"])],
                  verbose=False, val_data=tt.tuplefy(sub(X, va), ys(va)).repeat(10).cat())
    else:
        bs = n if tr["batch_size"] == "full" else tr["batch_size"]
        while n % bs == 1:                   # avoid a last batch of 1 patient (batch norm fails on it)
            bs += 1
        model.fit(X, y, bs, tr["epochs"], verbose=False)
    return model


def deep_method_cfg(cfg, dataset, method):
    if method == "SHERLOC":
        return copy.deepcopy(cfg), "config"
    if dataset == "pada1":
        sel = pd.read_csv(PADA_SEL)
        s = sel[sel.method == method].sort_values("C_index_Uno").iloc[-1]["training"]
        return with_training(cfg, method, s), s
    return with_training(cfg, method, None), "config (early stopping, val 0.2)"


def run_deep(dataset, method, seeds, win):
    cfg0, _ = load_cfg(dataset)
    cfg, setting = deep_method_cfg(cfg0, dataset, method)
    df = load_df(cfg)
    keep = []
    _, _, info = T.run_cv(cfg, seeds, verbose=False, keep=keep)
    mc = cfg["model"]
    dims = (len(info["genes"]), len(mc["panel_features"]), len(mc["clinical"]) + len(mc.get("static_features") or []))
    P = lambda m, X: np.asarray(m.predict(X), dtype=float).flatten()
    sub = lambda X, m: [x[torch.tensor(m)] for x in X]
    rows = []
    for (seed, model, d_test, X_te, X_tr, d_tr) in keep:
        t_tr, e_tr = d_tr["OS_rebaseline"].to_numpy(float), d_tr["OS_event"].to_numpy(int)
        eta_tr, eta_te = P(model, X_tr), P(model, X_te)
        oof = np.zeros(len(d_tr))
        for k, (i_fit, _) in enumerate(StratifiedKFold(K, shuffle=True, random_state=1000 + seed).split(oof, e_tr)):
            m = np.zeros(len(oof), bool); m[i_fit] = True
            y = (torch.tensor(t_tr[m], dtype=torch.float32), torch.tensor(e_tr[m]))
            oof[~m] = P(fit_net(cfg, sub(X_tr, m), y, dims, seed * 100 + k), sub(X_tr, ~m))
        d_trv = df.drop(d_test.index)
        rows += recal_rows(seed, method, eta_tr, t_tr, e_tr, eta_te, oof, model.predict_surv_df(X_te),
                           d_trv, d_test, win, {"training": setting})
    return rows


# ---------------------------------------------------------------------------------------- tabular methods
def tabular(dataset):
    cfg, win = load_cfg(dataset)
    P, mc, tps = cfg["_P"], cfg["model"], cfg["timepoints"]
    data = Path(P["data_dir"])
    df = load_df(cfg)
    freq = pd.read_csv(data / P.get("gene_frequency_file", "gene_frequency.csv"), index_col=0).index.tolist()
    if mc.get("genes_file"):
        genes = pd.read_csv(data / mc["genes_file"], index_col=0).iloc[:, 0].tolist()
    else:
        genes = mc.get("genes") or [g for g in freq if f"{g}_BL" in df.columns][: mc["n_top_genes"]]
    net, enc_genes, _ = T.get_encoder(cfg, df, freq)
    lat = T.encode(df, net, enc_genes, tps).reshape(len(df), -1).numpy()
    lat = pd.DataFrame(lat, index=df.index, columns=[f"latent_{i}" for i in range(lat.shape[1])])
    gene_cols = [f"{g}_{tp}" for g in genes for tp in tps]
    panel_cols = [f"{tp}_{p}" for p in mc["panel_features"] for tp in tps]
    cln_cols = [f"cln_{c}" for c in mc["clinical"]]
    return cfg, win, df, lat, genes, gene_cols, panel_cols, cln_cols


def fpcrf_features(tr, te, genes, gene_cols, cfg, y_tr, seed):
    from scripts.train_ML_FPCA import _compute_fpca_pca_features
    tps, pf = cfg["timepoints"], cfg["model"]["panel_features"]
    to3 = lambda d: torch.tensor(d[gene_cols].to_numpy(np.float32).reshape(len(d), len(genes), len(tps)))
    pan = lambda d: torch.tensor(np.stack([d[[f"{tp}_{p}" for tp in tps]].to_numpy(np.float32) for p in pf], 1))
    fcfg = {"n_fpca_components": 2, "n_top_features": 20, "n_pca_components": 10, "n_rsf_estimators_importance": 100}
    a, b = _compute_fpca_pca_features(to3(tr), pan(tr), to3(te), pan(te), y_tr, tps, fcfg, seed)
    return pd.DataFrame(a, index=tr.index).add_prefix("fpca_"), pd.DataFrame(b, index=te.index).add_prefix("fpca_")


def run_tabular(dataset, methods, seeds):
    cfg, win, df, lat, genes, gene_cols, panel_cols, cln_cols = tabular(dataset)
    tf, sm = cfg["training"]["test_frac"], cfg["evaluation"]["seed_multiplier"]
    grid = np.arange(0, win[1] + 0.25, 0.25)
    rows = []
    for seed in seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        te = df.sample(frac=tf, random_state=sm * seed); tr = df.drop(te.index)
        med = tr[cln_cols].median()
        tr = tr.assign(**{c: tr[c].fillna(med[c]) for c in cln_cols})
        te = te.assign(**{c: te[c].fillna(med[c]) for c in cln_cols})
        t_tr, e_tr = tr.OS_rebaseline.to_numpy(float), tr.OS_event.to_numpy(int)
        Y = lambda d: Surv.from_arrays(event=d.OS_event.astype(bool), time=d.OS_rebaseline)
        base = lambda d: pd.concat([d[gene_cols + panel_cols + cln_cols], lat.loc[d.index]], axis=1)
        folds = list(StratifiedKFold(K, shuffle=True, random_state=1000 + seed).split(tr, e_tr))
        for m in methods:
            oof = np.zeros(len(tr))
            if m in L1 or m == "RSF":
                def prep(a, b):                       # standardise on the fitting patients
                    sc = StandardScaler().fit(base(a))
                    f = lambda d: pd.DataFrame(sc.transform(base(d)), index=d.index, columns=base(d).columns)
                    return f(a), f(b)
                X_tr, X_te = prep(tr, te) if m in L1 else (base(tr), base(te))
                if m in L1:
                    model = fit_coxnet(X_tr, Y(tr), L1[m], seed)
                    alpha = model.alphas_[0] if hasattr(model, "alphas_") else model.alphas[0]
                    mk = lambda: CoxnetSurvivalAnalysis(l1_ratio=L1[m], alphas=[alpha], max_iter=5000)
                    score = lambda mdl, X: mdl.predict(X)
                    extra = {"alpha": alpha}
                else:
                    cv = KFold(n_splits=4, shuffle=True, random_state=seed)
                    model = GridSearchCV(RandomSurvivalForest(random_state=1),
                                         param_grid={"n_estimators": [50, 100], "min_samples_split": [2, 6]},
                                         cv=cv, error_score=0.5, n_jobs=1).fit(X_tr, Y(tr)).best_estimator_
                    bp = {"n_estimators": model.n_estimators, "min_samples_split": model.min_samples_split}
                    mk = lambda: RandomSurvivalForest(random_state=1, **bp)
                    score = lambda mdl, X: np.log(np.maximum(mdl.predict(X), 1e-6))
                    extra = bp
                for i_fit, i_out in folds:
                    a, b = tr.iloc[i_fit], tr.iloc[i_out]
                    Xa, Xb = prep(a, b) if m in L1 else (base(a), base(b))
                    try:
                        oof[i_out] = score(mk().fit(Xa, Y(a)), Xb)
                    except ArithmeticError:           # fold fit diverged: stronger penalty (logged)
                        for mult in (3, 10, 30, 100, 1000):
                            FALLBACKS.append((L1[m], seed, -mult))
                            try:
                                oof[i_out] = score(CoxnetSurvivalAnalysis(l1_ratio=L1[m], alphas=[alpha * mult],
                                                                          max_iter=5000).fit(Xa, Y(a)), Xb)
                                break
                            except ArithmeticError:
                                continue
                Xm_tr, Xm_te = X_tr, X_te
            else:                                     # FPCRF: full pipeline inside each fold
                def fpc(a, b, s):
                    fa, fb = fpcrf_features(a, b, genes, gene_cols, cfg, Y(a), s)
                    A = pd.concat([fa, a[cln_cols], lat.loc[a.index]], axis=1)
                    B = pd.concat([fb, b[cln_cols], lat.loc[b.index]], axis=1)
                    A.columns = B.columns = [str(c) for c in A.columns]
                    return A, B
                Xm_tr, Xm_te = fpc(tr, te, seed)
                model = fit_coxnet(Xm_tr, Y(tr), 0.5, seed)
                score = lambda mdl, X: mdl.predict(X)
                for k, (i_fit, i_out) in enumerate(folds):
                    a, b = tr.iloc[i_fit], tr.iloc[i_out]
                    A, B = fpc(a, b, seed * 100 + k)
                    oof[i_out] = score(fit_coxnet(A, Y(a), 0.5, seed * 100 + k), B)
                extra = {}
            eta_tr = np.nan_to_num(score(model, Xm_tr), nan=0.0, posinf=50, neginf=-50)
            eta_te = np.nan_to_num(score(model, Xm_te), nan=0.0, posinf=50, neginf=-50)
            raw, _ = surv_on_grid(model, Xm_te, grid)
            name = m + ("_scaled" if m in L1 else "")
            rows += recal_rows(seed, name, eta_tr, t_tr, e_tr, eta_te, oof, raw, tr, te, win, extra)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--group", required=True, choices=["deep", "ml", "fpcrf"])
    ap.add_argument("--methods", default=None, help="comma list (default: all of the group)")
    ap.add_argument("--seeds", default="0-99")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-")); seeds = range(lo, hi + 1)
    out = OUT / a.dataset; out.mkdir(parents=True, exist_ok=True)
    default = {"deep": DEEP, "ml": ["ridge", "lasso", "elastic_net", "RSF"], "fpcrf": ["FPCRF"]}[a.group]
    methods = a.methods.split(",") if a.methods else default
    _, win = load_cfg(a.dataset)
    for m in methods:
        f = out / f"{m}_seeds{lo}-{hi}.csv"
        if f.exists():
            print("[skip]", f); continue
        t0 = time.time()
        rows = run_deep(a.dataset, m, seeds, win) if a.group == "deep" else run_tabular(a.dataset, [m], seeds)
        r = pd.DataFrame(rows); r.to_csv(f, index=False)
        if FALLBACKS:
            pd.DataFrame(FALLBACKS, columns=["l1_ratio", "seed", "alpha_min_ratio_failed"]).to_csv(
                out / f"coxnet_fallbacks_{m}_seeds{lo}-{hi}.csv", index=False); FALLBACKS.clear()
        print(f"[{a.dataset} {m} seeds {lo}-{hi}] {time.time() - t0:.0f}s", r.groupby("variant")[
            ["C_index_Uno", "iAUC", "IBS", "calib_err"]].mean().round(4).to_dict("index"), flush=True)
