"""
clinical_applications.py — treatment-monitoring applications of SHERLOC on IMpower150 and PADA-1 (OS from the landmark),
with one fixed training / test split per trial.

Split: 70 / 30 at patient level, stratified on death (and treatment arm in IMpower150), seed 2026. IMpower150: the split
is drawn once on all patients of the three landmark cohorts (C2D1, C3D1, C4D1), so a patient is in the test set at every
landmark. Every model, cutoff and calibration slope is fitted on the training set only.
Models (training set):
  SHERLOC        all samples up to the landmark + clinical covariates                      ("full")
  SHERLOC BL     baseline sample only + clinical covariates (same patients, same endpoint)  ("bl")
  SHERLOC ctDNA  all samples up to the landmark, no clinical covariates                    ("ctdna")
  Clinical Cox   clinical covariates only (lifelines, penalizer 0.01)
SHERLOC: scale 1, recalibrated with the 5-fold out-of-fold slope (posthoc_benchmark.fit_net / breslow_surv).
Analyses on the test set:
  1 discrimination and calibration of each model (+ bootstrap 95 % CI of C and of the C difference)
  2 risk groups: tertiles of the SHERLOC score on the training set -> KM, HR, observed vs predicted survival at the horizon,
    sensitivity / specificity / PPV / NPV of the high-risk group for death by the horizon
  3 re-stratification: SHERLOC BL group vs SHERLOC (on-treatment) group
  4 SHERLOC high-risk flag vs a molecular-response rule (no >= 50 % drop of mean VAF / new detection), same proportion flagged
  5 IMpower150: landmark C2D1 / C3D1 / C4D1 on the test patients common to the three landmarks
  6 IMpower150 treatment arms (exploratory): 5-fold cross-fitted SHERLOC scores on the whole cohort; risk groups within arm,
    arm effect within risk group, interaction test
Horizon: 12 months (IMpower150), 24 months (PADA-1). Outputs: results/clinical_applications/
Usage: python clinical_applications.py [--trial impower150|pada1|all]
"""
import argparse, copy, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv
import train_sherloc_pada1 as T
import posthoc_benchmark as B
from posthoc_holdout import breslow_surv, beta_fit
from mtl_benchmark import metrics

warnings.filterwarnings("ignore")
OUT = Path("results/clinical_applications")
SEED = 2026
TRIALS = {
    "impower150": {"landmarks": {"C2D1": "impower150_2tp", "C3D1": "impower150_3tp", "C4D1": "impower150_4tp"},
                   "main": "C4D1", "horizon": 12.0, "tau": 30.0, "weeks": {"C2D1": 3, "C3D1": 6, "C4D1": 9}},
    "pada1": {"landmarks": {"C2": "pada1"}, "main": "C2", "horizon": 24.0, "tau": 36.0, "weeks": {"C2": 4}},
}
COL = {"low": "#2a78d6", "intermediate": "#1baf7a", "high": "#eb6834"}
GROUPS = ["low", "intermediate", "high"]
INK, INK2 = "#0b0b0b", "#52514e"


# ------------------------------------------------------------------------------------------------ split
def make_split(trial):
    dfs = {lm: B.load_df(B.load_cfg(ds)[0]) for lm, ds in TRIALS[trial]["landmarks"].items()}
    allp = pd.concat([d[["OS_event"] + [c for c in d.columns if c in ("cln_Bev_CP", "cln_Atezo_CP")]]
                      for d in dfs.values()])
    allp = allp[~allp.index.duplicated()]
    strata = allp.OS_event.astype(int).astype(str)
    if "cln_Bev_CP" in allp:
        arm = np.where(allp.cln_Bev_CP == 1, "BCP", np.where(allp.cln_Atezo_CP == 1, "ACP", "ABCP"))
        strata = strata + "_" + arm
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    _, te = next(sss.split(allp, strata))
    return allp.index[te].tolist(), dfs


# ------------------------------------------------------------------------------------------------ models
def variant_cfg(dataset, variant, test_ids):
    cfg, win = B.load_cfg(dataset)
    cfg = copy.deepcopy(cfg); cfg["_test_ids"] = test_ids
    mc = cfg["model"]
    if variant == "bl":
        cfg["timepoints"] = ["BL"]
        if dataset == "pada1":   # change features are 0 at BL: keep the level features
            mc["panel_features"] = [p for p in mc["panel_features"] if not p.startswith("resp_") or p == "resp_detected"]
    if variant == "ctdna":
        mc["clinical"] = ["none"]
    return cfg


def fit_sherloc(dataset, variant, test_ids, seed=0):
    """Train on the training patients, OOF-5 recalibration; returns scores and recalibrated test survival."""
    cfg = variant_cfg(dataset, variant, test_ids)
    keep = []
    _, _, info = T.run_cv(cfg, [seed], verbose=False, keep=keep)
    _, model, d_test, X_te, X_tr, d_tr = keep[0]
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
    return {"eta_tr": pd.Series(eta_tr, d_tr.index), "eta_te": pd.Series(eta_te, d_test.index), "beta": b,
            "surv_te": breslow_surv(b * eta_tr, t_tr, e_tr, b * eta_te).set_axis(d_test.index, axis=1),
            "d_tr": d_tr, "d_te": d_test}


def fit_clinical(df, test_ids, cln):
    tr, te = df[~df.index.isin(test_ids)], df[df.index.isin(test_ids)]
    med = tr[cln].median()
    X = lambda d: d[cln].fillna(med)
    cph = CoxPHFitter(penalizer=0.01).fit(pd.concat([X(tr), tr[["OS_rebaseline", "OS_event"]]], axis=1),
                                          "OS_rebaseline", "OS_event")
    grid = np.arange(0, 40.25, 0.25)
    return {"eta_tr": pd.Series(np.log(cph.predict_partial_hazard(X(tr))).to_numpy(), tr.index),
            "eta_te": pd.Series(np.log(cph.predict_partial_hazard(X(te))).to_numpy(), te.index),
            "surv_te": cph.predict_survival_function(X(te), times=grid), "d_tr": tr, "d_te": te, "beta": np.nan}


# ------------------------------------------------------------------------------------------------ evaluation
def yS(d):
    return Surv.from_arrays(event=d.OS_event.astype(bool), time=d.OS_rebaseline)


def evaluate(r, tau, h):
    dd = lambda d: d.assign(OS_t=d.OS_rebaseline, OS_e=d.OS_event)
    m = metrics(dd(r["d_tr"]), dd(r["d_te"]), r["eta_te"].to_numpy(), r["surv_te"], "OS", (3.0, tau, h))
    m["AUC_horizon"] = float(cumulative_dynamic_auc(yS(r["d_tr"]), yS(r["d_te"]), r["eta_te"].to_numpy(), [h])[0][0])
    return m


def boot_c(res, names, tau, n=1000, ref=None):
    """Bootstrap 95 % CI of Uno C on the test set, and of the paired difference vs ref."""
    rng = np.random.RandomState(SEED)
    d_te = res[names[0]]["d_te"]; ytr = yS(res[names[0]]["d_tr"])
    ids = d_te.index.to_numpy()
    out = {k: [] for k in names}; diffs = {k: [] for k in names if ref and k != ref}
    for _ in range(n):
        b = rng.choice(len(ids), len(ids), replace=True)
        bi = ids[b]; yb = yS(d_te.loc[bi])
        cs = {}
        for k in names:
            try:
                cs[k] = concordance_index_ipcw(ytr, yb, res[k]["eta_te"].loc[bi].to_numpy(), tau=tau)[0]
            except Exception:
                cs[k] = np.nan
            out[k].append(cs[k])
        for k in diffs:
            diffs[k].append(cs[k] - cs[ref])
    ci = {k: np.nanpercentile(v, [2.5, 97.5]) for k, v in out.items()}
    dci = {k: (np.nanmean(v), *np.nanpercentile(v, [2.5, 97.5]), np.nanmean(np.array(v) <= 0)) for k, v in diffs.items()}
    return ci, dci


def groups_from_train(r):
    q = np.quantile(r["eta_tr"], [1 / 3, 2 / 3])
    g = pd.cut(r["eta_te"], [-np.inf, q[0], q[1], np.inf], labels=GROUPS)
    return g, q


def km_at(d, h):
    k = KaplanMeierFitter().fit(d.OS_rebaseline, d.OS_event)
    return float(k.survival_function_at_times(h).iloc[0])


def flag_perf(flag, d, h):
    known = (d.OS_rebaseline > h) | (d.OS_event == 1)
    dead = (d.OS_event == 1) & (d.OS_rebaseline <= h)
    f, y = flag[known].astype(bool), dead[known]
    tp, fp, fn, tn = (f & y).sum(), (f & ~y).sum(), (~f & y).sum(), (~f & ~y).sum()
    return {"n_known": int(known.sum()), "deaths_by_h": int(y.sum()), "flagged": int(flag.sum()),
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "PPV": tp / max(tp + fp, 1), "NPV": tn / max(tn + fn, 1)}


def hr(d, covs):
    c = CoxPHFitter().fit(pd.concat([d[covs], d[["OS_rebaseline", "OS_event"]]], axis=1), "OS_rebaseline", "OS_event")
    s = c.summary
    return {k: (s.loc[k, "exp(coef)"], s.loc[k, "exp(coef) lower 95%"], s.loc[k, "exp(coef) upper 95%"], s.loc[k, "p"])
            for k in covs}


def fmt_hr(t):
    return f"{t[0]:.2f} ({t[1]:.2f}-{t[2]:.2f}), p={t[3]:.3g}"


# ------------------------------------------------------------------------------------------------ figures
def km_plot(ax, d, g, h, title, pred=None, xmax=30):
    for gr in GROUPS:
        dd = d[g == gr]
        if len(dd) == 0:
            continue
        k = KaplanMeierFitter().fit(dd.OS_rebaseline, dd.OS_event)
        sf = k.survival_function_
        ax.step(sf.index, sf.iloc[:, 0], where="post", color=COL[gr], lw=2,
                label=f"{gr} (n={len(dd)}, {int(dd.OS_event.sum())} deaths)")
        if pred is not None:
            p = pred.loc[:, dd.index].mean(axis=1)
            ax.plot(p.index, p.values, color=COL[gr], lw=1.2, ls="--")
    ax.axvline(h, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Months from landmark", color=INK2); ax.set_ylabel("Overall survival", color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")


# ------------------------------------------------------------------------------------------------ main
def run_trial(trial):
    cfgT = TRIALS[trial]; h, tau = cfgT["horizon"], cfgT["tau"]
    out = OUT / trial; out.mkdir(parents=True, exist_ok=True)
    test_ids, dfs = make_split(trial)
    pd.Series(test_ids, name="test_patient").to_csv(out / "test_patients.csv", index=False)
    summary = {"trial": trial, "horizon_months": h, "n_test_patients_total": len(test_ids), "landmarks": {}}
    disc_rows, grp_rows, flag_rows, recl_rows = [], [], [], []
    per_lm = {}
    for lm, ds in cfgT["landmarks"].items():
        cfg, _ = B.load_cfg(ds)
        cln = [f"cln_{c}" for c in cfg["model"]["clinical"]]
        res = {"Clinical Cox": fit_clinical(dfs[lm], test_ids, cln),
               "SHERLOC BL (+clinical)": fit_sherloc(ds, "bl", test_ids),
               "SHERLOC ctDNA only": fit_sherloc(ds, "ctdna", test_ids),
               "SHERLOC (+clinical)": fit_sherloc(ds, "full", test_ids)}
        names = list(res)
        d_te, d_tr = res["SHERLOC (+clinical)"]["d_te"], res["SHERLOC (+clinical)"]["d_tr"]
        ci, dci = boot_c(res, names, tau, ref="Clinical Cox")
        _, dci_bl = boot_c(res, ["SHERLOC BL (+clinical)", "SHERLOC (+clinical)"], tau, ref="SHERLOC BL (+clinical)")
        for k in names:
            m = evaluate(res[k], tau, h)
            row = {"trial": trial, "landmark": lm, "model": k, "n_train": len(res[k]["d_tr"]), "n_test": len(d_te),
                   "test_deaths": int(d_te.OS_event.sum()), **m, "C_lo": ci[k][0], "C_hi": ci[k][1],
                   "beta_recal": res[k]["beta"]}
            if k in dci:
                row.update({"dC_vs_clinical": dci[k][0], "dC_vs_clinical_lo": dci[k][1], "dC_vs_clinical_hi": dci[k][2],
                            "p_boot_vs_clinical": dci[k][3]})
            if k == "SHERLOC (+clinical)":
                v = dci_bl["SHERLOC (+clinical)"]
                row.update({"dC_vs_BL": v[0], "dC_vs_BL_lo": v[1], "dC_vs_BL_hi": v[2], "p_boot_vs_BL": v[3]})
            disc_rows.append(row)
        # 2 risk groups
        full = res["SHERLOC (+clinical)"]
        g, q = groups_from_train(full)
        dg = d_te.assign(group=g.astype(str))
        for gr in GROUPS:
            dd = dg[dg.group == gr]
            if len(dd):
                grp_rows.append({"trial": trial, "landmark": lm, "group": gr, "n": len(dd),
                                 "deaths": int(dd.OS_event.sum()),
                                 "median_OS": KaplanMeierFitter().fit(dd.OS_rebaseline, dd.OS_event).median_survival_time_,
                                 f"observed_S{int(h)}": km_at(dd, h),
                                 f"predicted_S{int(h)}": float(full["surv_te"].loc[:, dd.index].iloc[
                                     (np.abs(full["surv_te"].index - h)).argmin()].mean())})
        dummies = pd.get_dummies(dg.group)[["intermediate", "high"]].astype(float)
        hrs = hr(pd.concat([dg, dummies], axis=1), ["intermediate", "high"])
        trend = multivariate_logrank_test(dg.OS_rebaseline, dg.group, dg.OS_event).p_value
        high = (dg.group == "high")
        fp = flag_perf(high, dg, h)
        flag_rows.append({"trial": trial, "landmark": lm, "flag": "SHERLOC high-risk tertile", **fp,
                          "HR_flag": fmt_hr(hr(dg.assign(f=high.astype(float)), ["f"])["f"])})
        # 4 molecular-response rule
        P = cfg["model"]["panel_features"]
        mv = "mean_of_AF" if "mean_of_AF" in P else "mean_VAF"
        tps = cfg["timepoints"]
        bl, lmv = d_te[f"BL_{mv}"], d_te[f"{tps[-1]}_{mv}"]
        no_mr = ((bl > 0) & (lmv >= 0.5 * bl)) | ((bl == 0) & (lmv > 0))
        fr = flag_perf(no_mr, d_te, h)
        flag_rows.append({"trial": trial, "landmark": lm, "flag": "No molecular response (rule)", **fr,
                          "HR_flag": fmt_hr(hr(d_te.assign(f=no_mr.astype(float)), ["f"])["f"])})
        top = full["eta_te"] >= full["eta_te"].quantile(1 - no_mr.mean())
        fs = flag_perf(top, d_te, h)
        flag_rows.append({"trial": trial, "landmark": lm, "flag": "SHERLOC, same proportion as the rule", **fs,
                          "HR_flag": fmt_hr(hr(d_te.assign(f=top.astype(float)), ["f"])["f"])})
        # 3 re-stratification BL -> on-treatment
        gb, _ = groups_from_train(res["SHERLOC BL (+clinical)"])
        ct = pd.crosstab(gb.rename("BL group"), g.rename("on-treatment group"))
        mv_ = np.sign(g.cat.codes - gb.cat.codes).map({-1: "down", 0: "same", 1: "up"})
        dm = d_te.assign(move=mv_, bl_group=gb.astype(str))
        for gr in GROUPS:
            for mvv in ("down", "same", "up"):
                dd = dm[(dm.bl_group == gr) & (dm.move == mvv)]
                if len(dd):
                    recl_rows.append({"trial": trial, "landmark": lm, "BL_group": gr, "move": mvv, "n": len(dd),
                                      "deaths": int(dd.OS_event.sum()), f"observed_S{int(h)}": km_at(dd, h)})
        summary["landmarks"][lm] = {"n_test": len(d_te), "test_deaths": int(d_te.OS_event.sum()),
                                    "tertile_cutoffs_train": q.tolist(),
                                    "HR_intermediate_vs_low": fmt_hr(hrs["intermediate"]),
                                    "HR_high_vs_low": fmt_hr(hrs["high"]), "logrank_p_3groups": trend,
                                    "reclassified_fraction": float((mv_ != "same").mean()),
                                    "crosstab_BL_vs_on_treatment": ct.to_dict()}
        per_lm[lm] = {"res": res, "g": g, "dg": dg, "gb": gb}
        # figure: KM by group (observed, solid) with mean recalibrated prediction (dashed)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        km_plot(axs[0], d_te, gb, h, f"SHERLOC, baseline sample only ({trial}, test set)", xmax=tau)
        km_plot(axs[1], d_te, g, h, f"SHERLOC with on-treatment ctDNA at {lm} (dashed: predicted)", full["surv_te"], xmax=tau)
        fig.tight_layout(); fig.savefig(out / f"km_risk_groups_{lm}.png", dpi=160); plt.close(fig)
    pd.DataFrame(disc_rows).to_csv(out / "discrimination.csv", index=False)
    pd.DataFrame(grp_rows).to_csv(out / "risk_groups.csv", index=False)
    pd.DataFrame(flag_rows).to_csv(out / "flags_vs_molecular_response.csv", index=False)
    pd.DataFrame(recl_rows).to_csv(out / "restratification.csv", index=False)
    # 5 how early (IMpower150): same test patients at every landmark
    if trial == "impower150":
        common = sorted(set.intersection(*[set(per_lm[lm]["res"]["SHERLOC (+clinical)"]["d_te"].index) for lm in per_lm]))
        rows = []
        for lm in per_lm:
            r = per_lm[lm]["res"]["SHERLOC (+clinical)"]
            dc = r["d_te"].loc[common]; e = r["eta_te"].loc[common]
            c = concordance_index_ipcw(yS(r["d_tr"]), yS(dc), e.to_numpy(), tau=tau)[0]
            gg = per_lm[lm]["g"].loc[common]
            dd = dc.assign(high=(gg == "high").astype(float), inter=(gg == "intermediate").astype(float))
            hh = hr(dd, ["inter", "high"])
            rb = per_lm[lm]["res"]["SHERLOC BL (+clinical)"]
            cb = concordance_index_ipcw(yS(rb["d_tr"]), yS(dc), rb["eta_te"].loc[common].to_numpy(), tau=tau)[0]
            rows.append({"landmark": lm, "weeks_on_treatment": cfgT["weeks"][lm], "n_common": len(common),
                         "deaths": int(dc.OS_event.sum()), "C_SHERLOC": c, "C_SHERLOC_BL": cb,
                         "HR_high_vs_low": fmt_hr(hh["high"])})
        pd.DataFrame(rows).to_csv(out / "how_early_common_patients.csv", index=False)
        summary["how_early"] = rows
        summary["arms"] = arms_analysis(out, h)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=str)
    return summary


def arms_analysis(out, h, ds="impower150_4tp", lm="C4D1"):
    """Exploratory: 5-fold cross-fitted SHERLOC scores on the whole C4D1 cohort; tertiles from each fold's training set."""
    df = B.load_df(B.load_cfg(ds)[0])
    arm = pd.Series(np.where(df.cln_Bev_CP == 1, "BCP", np.where(df.cln_Atezo_CP == 1, "ACP", "ABCP")), df.index)
    strata = df.OS_event.astype(int).astype(str) + "_" + arm
    grp = pd.Series(index=df.index, dtype=object)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, strata)):
        r = fit_sherloc(ds, "full", df.index[te].tolist(), seed=k)
        g, _ = groups_from_train(r)
        grp.loc[g.index] = g.astype(str)
    d = df.assign(arm=arm, group=grp)
    rows = []
    for a in ["ABCP", "ACP", "BCP"]:
        dd = d[d.arm == a]
        dm = pd.concat([dd, pd.get_dummies(dd.group)[["intermediate", "high"]].astype(float)], axis=1)
        hh = hr(dm, ["intermediate", "high"])
        rows.append({"analysis": "risk group within arm", "arm": a, "n": len(dd), "deaths": int(dd.OS_event.sum()),
                     "HR_intermediate_vs_low": fmt_hr(hh["intermediate"]), "HR_high_vs_low": fmt_hr(hh["high"]),
                     **{f"S{int(h)}_{g}": km_at(dd[dd.group == g], h) for g in GROUPS}})
    for g in GROUPS:
        dd = d[d.group == g]
        for a in ["ABCP", "ACP"]:
            sub = dd[dd.arm.isin([a, "BCP"])].assign(trt=lambda x: (x.arm == a).astype(float))
            rows.append({"analysis": f"{a} vs BCP within risk group", "arm": g, "n": len(sub),
                         "deaths": int(sub.OS_event.sum()), "HR_treatment": fmt_hr(hr(sub, ["trt"])["trt"])})
    for a in ["ABCP", "ACP"]:
        sub = d[d.arm.isin([a, "BCP"])].assign(trt=lambda x: (x.arm == a).astype(float),
                                                 risk=lambda x: x.group.map({"low": 0, "intermediate": 1, "high": 2}))
        sub = sub.assign(trt_x_risk=sub.trt * sub.risk)
        rows.append({"analysis": f"interaction {a} x risk group (trend)", "arm": a, "n": len(sub),
                     "HR_treatment": fmt_hr(hr(sub, ["trt", "risk", "trt_x_risk"])["trt_x_risk"])})
    r = pd.DataFrame(rows); r.to_csv(out / "arms_crossfitted_C4D1.csv", index=False)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trial", default="all")
    a = ap.parse_args()
    for t in (["impower150", "pada1"] if a.trial == "all" else [a.trial]):
        s = run_trial(t)
        print(json.dumps({k: v for k, v in s.items() if k != "arms"}, indent=1, default=str)[:3000])
