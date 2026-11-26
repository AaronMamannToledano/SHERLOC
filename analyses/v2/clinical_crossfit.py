"""
clinical_crossfit.py — clinical evaluation of SHERLOC with 10 x repeated 5-fold cross-fitting (OS from the landmark).

Models (each trained on 4/5 of the cohort, scores the held-out 1/5):
  SHERLOC        all samples up to the landmark + clinical covariates, scale 1, recalibrated with the 5-fold out-of-fold
                 slope computed inside the training folds (posthoc_benchmark.fit_net / breslow_surv)
  Clinical Cox   clinical covariates only (lifelines, penalizer 0.01)
Datasets: IMpower150 landmark C4D1 (BL, C2D1, C3D1, C4D1; 386 patients), PADA-1 landmark C2 (BL, C2; 369 patients).
Folds stratified on death (+ treatment arm for IMpower150); repeat r uses random_state 2026 + r.
After each repeat every patient has one out-of-fold score, predicted survival curve, risk group and binary prediction,
so every analysis below is run on the whole cohort, once per repeat:
  * risk groups: tertiles of the inner out-of-fold scores of the training folds (nothing from the held-out fold);
    Kaplan-Meier per group -> pointwise median curve over the 10 repeats + min-max band; numbers at risk = median;
    HR (Cox on group dummies) per repeat
  * death within the horizon (12 months IMpower150, 24 months PADA-1): predicted probability 1 - S(h) thresholded;
    threshold chosen inside the training folds only, on the inner out-of-fold probabilities, by maximising Youden's J
    (primary), F1, or the threshold reaching 80 % specificity / 80 % sensitivity (fixed operating points, to compare
    models at the same specificity or sensitivity), with inverse-probability-of-censoring weights (patients censored before h have unknown
    status: cases weighted 1/G(T-), controls 1/G(h), G = Kaplan-Meier of censoring) -> sensitivity, specificity, PPV,
    NPV, F1 of the held-out patients (IPCW-weighted)
  * discrimination / calibration: Uno C, iAUC, AUC at h, IBS, calibration error at h
Point estimate = median over the 10 repeats; 95 % CI = percentiles of a patient bootstrap (200 per repeat, pooled over
the 10 repeats, out-of-fold predictions kept fixed); paired SHERLOC - Clinical differences use the same resamples.
Usage: python clinical_crossfit.py --trial pada1|impower150|all [--repeats 10] [--boot 200]
Outputs: results/clinical_crossfit/<trial>/
"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv
import train_sherloc_pada1 as T
import posthoc_benchmark as B
from posthoc_holdout import breslow_surv, beta_fit
from clinical_applications import variant_cfg

warnings.filterwarnings("ignore")
OUT = Path("results/clinical_crossfit")
SEED = 2026
TRIALS = {"impower150": {"dataset": "impower150_4tp", "landmark": "C4D1", "horizon": 12.0, "tau": 30.0},
          "pada1": {"dataset": "pada1", "landmark": "C2", "horizon": 24.0, "tau": 36.0}}
GROUPS = ["low", "intermediate", "high"]
COL = {"low": "#2a78d6", "intermediate": "#1baf7a", "high": "#eb6834"}
INK, INK2 = "#0b0b0b", "#52514e"
MODELS = ["Clinical Cox", "SHERLOC"]
CRITS = ["youden", "f1", "spec80", "sens80"]   # threshold rules (chosen inside the training folds)


# ----------------------------------------------------------------------------------------------- helpers
def to_grid(surv, grid):
    """Step survival DataFrame (index = times) -> values on a common grid (last value carried forward)."""
    s = surv.sort_index()
    pos = np.searchsorted(s.index.to_numpy(), grid, side="right") - 1
    v = s.to_numpy()[np.clip(pos, 0, None)]
    v[pos < 0] = 1.0
    return pd.DataFrame(v, index=grid, columns=s.columns)


def censor_km(t, e):
    k = KaplanMeierFitter().fit(t, 1 - e)
    return lambda x: np.clip(k.survival_function_at_times(np.atleast_1d(x)).to_numpy(), 1e-3, 1)


def ipcw_weights(t, e, h, G):
    """Cases (death <= h): 1/G(T-); controls (T > h): 1/G(h); censored before h: 0."""
    case = (e == 1) & (t <= h); ctrl = t > h
    w = np.zeros(len(t))
    w[case] = 1 / G(np.maximum(t[case] - 1e-6, 0))
    w[ctrl] = 1 / G(h)[0]
    return case, ctrl, w


def binary_perf(flag, case, ctrl, w):
    tp = w[flag & case].sum(); fn = w[~flag & case].sum()
    fp = w[flag & ctrl].sum(); tn = w[~flag & ctrl].sum()
    sens = tp / max(tp + fn, 1e-12); spec = tn / max(tn + fp, 1e-12)
    ppv = tp / max(tp + fp, 1e-12); npv = tn / max(tn + fn, 1e-12)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-12)
    return {"sensitivity": sens, "specificity": spec, "PPV": ppv, "NPV": npv, "F1": f1,
            "flagged_fraction": float(flag.mean())}


def best_threshold(p, t, e, h, G):
    case, ctrl, w = ipcw_weights(t, e, h, G)
    cands = np.unique(np.round(p, 4))
    best = {"youden": (-1, 0.5), "f1": (-1, 0.5), "spec80": (-1, cands.max()), "sens80": (-1, cands.min())}
    for c in cands:
        r = binary_perf(p >= c, case, ctrl, w)
        j = r["sensitivity"] + r["specificity"] - 1
        if j > best["youden"][0]: best["youden"] = (j, c)
        if r["F1"] > best["f1"][0]: best["f1"] = (r["F1"], c)
        if r["specificity"] >= 0.80 and r["sensitivity"] > best["spec80"][0]: best["spec80"] = (r["sensitivity"], c)
        if r["sensitivity"] >= 0.80 and r["specificity"] > best["sens80"][0]: best["sens80"] = (r["specificity"], c)
    return {k: v[1] for k, v in best.items()}


def at(surv, h):
    return surv.iloc[int(np.abs(surv.index.to_numpy() - h).argmin())]


# ----------------------------------------------------------------------------------------------- one fold
def fold_sherloc(ds, test_ids, seed, h, grid):
    cfg = variant_cfg(ds, "full", test_ids)
    keep = []
    _, _, info = T.run_cv(cfg, [seed], verbose=False, keep=keep)
    _, model, d_te, X_te, X_tr, d_tr = keep[0]
    mc = cfg["model"]
    dims = (len(info["genes"]), len(mc["panel_features"]), len(mc["clinical"]) + len(mc.get("static_features") or []))
    P = lambda m, X: np.asarray(m.predict(X), dtype=float).flatten()
    sub = lambda X, m: [x[torch.tensor(m)] for x in X]
    t_tr, e_tr = d_tr.OS_rebaseline.to_numpy(float), d_tr.OS_event.to_numpy(int)
    eta_tr, eta_te = P(model, X_tr), P(model, X_te)
    oof = np.zeros(len(d_tr))
    for k, (i_fit, _) in enumerate(StratifiedKFold(5, shuffle=True, random_state=1000 + seed).split(oof, e_tr)):
        m = np.zeros(len(oof), bool); m[i_fit] = True
        y = (torch.tensor(t_tr[m], dtype=torch.float32), torch.tensor(e_tr[m]))
        oof[~m] = P(B.fit_net(cfg, sub(X_tr, m), y, dims, seed * 100 + k), sub(X_tr, ~m))
    b = beta_fit(oof, t_tr, e_tr)
    s_te = to_grid(breslow_surv(b * eta_tr, t_tr, e_tr, b * eta_te).set_axis(d_te.index, axis=1), grid)
    p_inner = 1 - at(breslow_surv(b * eta_tr, t_tr, e_tr, b * oof), h).to_numpy()   # inner OOF P(death by h)
    return d_tr, d_te, eta_te, s_te, oof, p_inner, b


def fold_clinical(df, cln, test_ids, seed, h, grid):
    tr, te = df[~df.index.isin(test_ids)], df[df.index.isin(test_ids)]
    med = tr[cln].median()
    X = lambda d: d[cln].fillna(med)
    fit = lambda d: CoxPHFitter(penalizer=0.01).fit(pd.concat([X(d), d[["OS_rebaseline", "OS_event"]]], axis=1),
                                                    "OS_rebaseline", "OS_event")
    cph = fit(tr)
    eta_te = np.log(cph.predict_partial_hazard(X(te))).to_numpy()
    s_te = cph.predict_survival_function(X(te), times=grid); s_te.columns = te.index
    oof, p_inner = np.zeros(len(tr)), np.zeros(len(tr))
    for i_fit, i_out in StratifiedKFold(5, shuffle=True, random_state=1000 + seed).split(tr, tr.OS_event):
        c = fit(tr.iloc[i_fit]); o = tr.iloc[i_out]
        oof[i_out] = np.log(c.predict_partial_hazard(X(o))).to_numpy()
        p_inner[i_out] = 1 - c.predict_survival_function(X(o), times=[h]).to_numpy().ravel()
    return tr, te, eta_te, s_te, oof, p_inner, np.nan


# ----------------------------------------------------------------------------------------------- one repeat
def run_repeat(trial, r, df, cln, strata, grid):
    cfgT = TRIALS[trial]; h = cfgT["horizon"]
    rows = []
    for k, (_, te_i) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(df, strata)):
        test_ids = df.index[te_i].tolist()
        for name in MODELS:
            if name == "SHERLOC":
                d_tr, d_te, eta, s_te, oof, p_in, b = fold_sherloc(cfgT["dataset"], test_ids, r * 10 + k, h, grid)
            else:
                d_tr, d_te, eta, s_te, oof, p_in, b = fold_clinical(df, cln, test_ids, r * 10 + k, h, grid)
            q = np.quantile(oof, [1 / 3, 2 / 3])                              # tertiles: training folds only
            t_tr, e_tr = d_tr.OS_rebaseline.to_numpy(float), d_tr.OS_event.to_numpy(int)
            thr = best_threshold(p_in, t_tr, e_tr, h, censor_km(t_tr, e_tr))  # threshold: training folds only
            p_te = 1 - at(s_te, h).to_numpy()
            for j, pid in enumerate(d_te.index):
                rows.append({"repeat": r, "fold": k, "model": name, "patient": pid, "eta": eta[j],
                             "group": GROUPS[int(np.searchsorted(q, eta[j], side="right"))], "p_death_h": p_te[j],
                             **{f"flag_{c}": p_te[j] >= thr[c] for c in CRITS},
                             **{f"thr_{c}": thr[c] for c in CRITS}, "beta": b,
                             **{f"S_{g:g}": v for g, v in zip(grid, s_te[pid].to_numpy())}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- metrics on the cohort
def cohort_metrics(pr, d, h, tau, grid, G_all):
    """pr: one repeat x one model, indexed like d. All metrics on the whole cohort."""
    t, e = d.OS_rebaseline.to_numpy(float), d.OS_event.to_numpy(int)
    y = Surv.from_arrays(event=e.astype(bool), time=t)
    eta = pr.eta.to_numpy()
    out = {"C_index": concordance_index_ipcw(y, y, eta, tau=tau)[0]}
    first = t[e == 1].min()
    tg = np.arange(max(3.0, np.ceil(first)), min(tau, t.max() - 1e-3), 1.0)
    out["iAUC"] = cumulative_dynamic_auc(y, y, eta, tg)[1]
    out["AUC_horizon"] = float(cumulative_dynamic_auc(y, y, eta, [h])[0][0])
    S = pr[[f"S_{g:g}" for g in grid]].to_numpy()
    m = (grid >= 3.0) & (grid < min(tau, t.max() - 1e-3))
    out["IBS"] = integrated_brier_score(y, y, S[:, m], grid[m])
    surv = pd.DataFrame(S.T, index=grid)
    out["calib_err"] = T.calibration_error(surv, d, eta, h)
    case, ctrl, w = ipcw_weights(t, e, h, G_all)
    for crit in CRITS:
        for kk, v in binary_perf(pr[f"flag_{crit}"].to_numpy().astype(bool), case, ctrl, w).items():
            out[f"{kk}_{crit}"] = v
    g = pr.group
    dm = d.assign(intermediate=(g == "intermediate").astype(float), high=(g == "high").astype(float))
    try:
        c = CoxPHFitter().fit(dm[["intermediate", "high", "OS_rebaseline", "OS_event"]], "OS_rebaseline", "OS_event")
        out["HR_intermediate_vs_low"] = float(np.exp(c.params_["intermediate"]))
        out["HR_high_vs_low"] = float(np.exp(c.params_["high"]))
    except Exception:
        out["HR_intermediate_vs_low"] = out["HR_high_vs_low"] = np.nan
    return out


def km_on_grid(t, e, grid):
    k = KaplanMeierFitter().fit(t, e)
    return k.survival_function_at_times(grid).to_numpy()


# ----------------------------------------------------------------------------------------------- main
def run_trial(trial, repeats, n_boot):
    cfgT = TRIALS[trial]; h, tau, ds = cfgT["horizon"], cfgT["tau"], cfgT["dataset"]
    out = OUT / trial; out.mkdir(parents=True, exist_ok=True)
    cfg, _ = B.load_cfg(ds)
    df = B.load_df(cfg)
    cln = [f"cln_{c}" for c in cfg["model"]["clinical"]]
    strata = df.OS_event.astype(int).astype(str)
    if "cln_Bev_CP" in df:
        strata = strata + "_" + np.where(df.cln_Bev_CP == 1, "BCP", np.where(df.cln_Atezo_CP == 1, "ACP", "ABCP"))
    grid = np.round(np.arange(0, tau + 0.25, 0.25), 2)
    f_pred = out / "oof_predictions.csv"
    done = pd.read_csv(f_pred) if f_pred.exists() else pd.DataFrame()
    t0 = time.time()
    for r in range(repeats):
        if len(done) and r in set(done.repeat):
            continue
        done = pd.concat([done, run_repeat(trial, r, df, cln, strata, grid)])
        done.to_csv(f_pred, index=False)
        print(f"[{trial}] repeat {r} done ({time.time() - t0:.0f}s)", flush=True)
    pr = done[done.repeat < repeats]
    t, e = df.OS_rebaseline.to_numpy(float), df.OS_event.to_numpy(int)
    G_all = censor_km(t, e)
    # point estimates per repeat + pooled patient bootstrap
    rng = np.random.RandomState(SEED)
    point, boot = [], []
    for r in range(repeats):
        per = {mname: pr[(pr.repeat == r) & (pr.model == mname)].set_index("patient").loc[df.index] for mname in MODELS}
        for mname in MODELS:
            point.append({"repeat": r, "model": mname, **cohort_metrics(per[mname], df, h, tau, grid, G_all)})
        for bi in range(n_boot):
            idx = rng.choice(len(df), len(df), replace=True)
            db = df.iloc[idx]
            Gb = censor_km(db.OS_rebaseline.to_numpy(float), db.OS_event.to_numpy(int))
            for mname in MODELS:
                try:
                    boot.append({"repeat": r, "b": bi, "model": mname,
                                 **cohort_metrics(per[mname].iloc[idx], db, h, tau, grid, Gb)})
                except Exception:
                    pass
    point, boot = pd.DataFrame(point), pd.DataFrame(boot)
    point.to_csv(out / "metrics_per_repeat.csv", index=False)
    mets = [c for c in point.columns if c not in ("repeat", "model")]
    summ = []
    for mname in MODELS:
        p, b = point[point.model == mname], boot[boot.model == mname]
        for mt in mets:
            summ.append({"model": mname, "metric": mt, "median_over_repeats": p[mt].median(),
                         "min_repeat": p[mt].min(), "max_repeat": p[mt].max(),
                         "ci_low": b[mt].quantile(0.025), "ci_high": b[mt].quantile(0.975)})
    bw = boot.pivot_table(index=["repeat", "b"], columns="model", values=mets)
    pw = point.set_index(["repeat", "model"]).unstack("model")
    for mt in mets:
        dlt = bw[(mt, "SHERLOC")] - bw[(mt, "Clinical Cox")]
        dp = pw[(mt, "SHERLOC")] - pw[(mt, "Clinical Cox")]
        summ.append({"model": "SHERLOC - Clinical Cox", "metric": mt, "median_over_repeats": dp.median(),
                     "min_repeat": dp.min(), "max_repeat": dp.max(), "ci_low": dlt.quantile(0.025),
                     "ci_high": dlt.quantile(0.975), "p_boot_one_sided": float((dlt <= 0).mean())})
    summ = pd.DataFrame(summ); summ.to_csv(out / "summary.csv", index=False)
    # Kaplan-Meier: median curve over repeats + min-max band, numbers at risk
    tick = np.arange(0, tau + 1e-9, 6.0)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    km_rows = []
    for ax, mname in zip(axs, MODELS):
        for gname in GROUPS:
            curves, nrisk, sizes = [], [], []
            for r in range(repeats):
                g = pr[(pr.repeat == r) & (pr.model == mname)].set_index("patient").loc[df.index, "group"]
                m = (g == gname).to_numpy()
                curves.append(km_on_grid(t[m], e[m], grid))
                nrisk.append([(t[m] >= x).sum() for x in tick]); sizes.append(m.sum())
                km_rows.append({"model": mname, "group": gname, "repeat": r, "n": int(m.sum()),
                                "deaths": int(e[m].sum()), f"KM_S{int(h)}": km_on_grid(t[m], e[m], [h])[0]})
            C = np.vstack(curves); med = np.median(C, 0)
            ax.fill_between(grid, C.min(0), C.max(0), step="post", color=COL[gname], alpha=0.18, lw=0)
            ax.step(grid, med, where="post", color=COL[gname], lw=2,
                    label=f"{gname} (median n={int(np.median(sizes))})")
            nr = np.median(np.array(nrisk), 0).astype(int)
            for x, n in zip(tick, nr):
                ax.text(x, -0.12 - 0.06 * GROUPS.index(gname), str(n), color=COL[gname], fontsize=7.5,
                        ha="center", transform=ax.get_xaxis_transform())
        hr = point[point.model == mname].HR_high_vs_low
        ax.set_title(f"{mname}: HR high vs low {hr.median():.2f} (range {hr.min():.2f}–{hr.max():.2f})",
                     fontsize=10, color=INK, loc="left")
        ax.axvline(h, color=INK2, lw=0.8, ls=":")
        ax.set_xlim(0, tau); ax.set_ylim(0, 1.02); ax.set_xticks(tick)
        ax.set_xlabel(f"Months from landmark ({cfgT['landmark']})", color=INK2, labelpad=38)
        ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    axs[0].set_ylabel("Overall survival", color=INK2)
    axs[0].text(-0.08, -0.12, "At risk", transform=axs[0].transAxes, fontsize=7.5, color=INK2, ha="right")
    fig.suptitle(f"{ {'pada1': 'PADA-1', 'impower150': 'IMpower150'}[trial]}: out-of-fold risk groups, 10 x 5-fold cross-fitting (line = median over repeats, "
                 f"band = range)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out / "km_crossfit_median_band.png", dpi=170); plt.close(fig)
    pd.DataFrame(km_rows).to_csv(out / "km_groups_per_repeat.csv", index=False)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="all"); ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    for tr in (["pada1", "impower150"] if a.trial == "all" else [a.trial]):
        s = run_trial(tr, a.repeats, a.boot)
        print(s.round(3).to_string(index=False))
