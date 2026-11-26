"""
clinical_crossfit_figures.py — Kaplan-Meier figures and response-stratified analyses from the 10 x 5-fold cross-fitted
SHERLOC predictions (results/clinical_crossfit/<trial>/oof_predictions.csv; no model is refitted).

For every panel and every repeat, the out-of-fold SHERLOC risk groups (tertile cutoffs from the training folds) give one
Kaplan-Meier curve per group; the figure shows the pointwise median over the 10 repeats (line) and the range (band),
median numbers at risk, the log-rank p (3 groups) and the median OS of each group (median over repeats; NR = not reached
in >= half of the repeats), and the HR high vs low (Cox).
  fig_km_overall.png        PADA-1 (landmark C2) and IMpower150 (landmark C4D1), all patients
  fig_km_recist.png         IMpower150 within RECIST stable disease and partial response at week 6 (known at C4D1)
  fig_km_recist_pada1.png   PADA-1 within RECIST stable disease and partial response at the first assessment
Tables: recist_adjusted_hr.csv (IMpower150: SHERLOC group adjusted for week-6 RECIST), comparator_flags.csv (death by the
horizon: response rules vs SHERLOC flagging the same proportion of patients; IPCW sensitivity / specificity / PPV / NPV).
Usage: python clinical_crossfit_figures.py
"""
import os, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import posthoc_benchmark as B
from clinical_crossfit import TRIALS, GROUPS, censor_km, ipcw_weights, binary_perf
import article_style as S
COL, INK, INK2 = S.RISK, S.INK, S.INK2
LAB = {"low": "Low", "intermediate": "Intermediate", "high": "High"}

warnings.filterwarnings("ignore")
ROOT = Path("results/clinical_crossfit")
FIG = ROOT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
NAME = {"pada1": "PADA-1", "impower150": "IMpower150"}
IMP_CLN = os.environ.get("IMP150_CLINICAL", "impower150_raw/anonymized_imp150_ctDNA_patient_clinical_toShare.csv")
PADA_CLN = os.environ.get("PADA1_CLINICAL", "pada1_raw/cln/curated/cln_curated.tsv")
PADA_RECIST = os.environ.get("PADA1_RECIST", "pada1_raw/cln/curated/cln_recist_curated.tsv")


def load(trial):
    cfgT = TRIALS[trial]
    df = B.load_df(B.load_cfg(cfgT["dataset"])[0])
    pr = pd.read_csv(ROOT / trial / "oof_predictions.csv",
                     usecols=["repeat", "model", "patient", "eta", "group"])
    pr = pr[pr.model == "SHERLOC"]
    groups = {r: g.set_index("patient") for r, g in pr.groupby("repeat")}
    return cfgT, df, groups


def fmt_p(p, eq=False):
    v = "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
    return (v if v.startswith("<") else "= " + v) if eq else v


def fmt_med(v):
    return "NR" if not np.isfinite(v) else f"{v:.1f} mo"


def km_panel(ax, df, groups, ids, title, h, tau, atrisk=None, letter=None):
    """atrisk: an axis below ax for the numbers at risk (None = no table)."""
    grid = np.arange(0, tau + 0.25, 0.25); tick = np.arange(0, tau + 1e-9, 6.0)
    d = df.loc[ids]; t, e = d.OS_rebaseline.to_numpy(float), d.OS_event.to_numpy(int)
    stats = {g: {"curves": [], "n": [], "med": [], "risk": []} for g in GROUPS}
    pv, hr = [], []
    for r, gr in groups.items():
        g = gr.loc[ids, "group"].to_numpy()
        for gn in GROUPS:
            m = g == gn
            if m.sum() == 0:
                continue
            k = KaplanMeierFitter().fit(t[m], e[m])
            stats[gn]["curves"].append(k.survival_function_at_times(grid).to_numpy())
            stats[gn]["n"].append(m.sum()); stats[gn]["med"].append(k.median_survival_time_)
            stats[gn]["risk"].append([(t[m] >= x).sum() for x in tick])
        if len(set(g)) > 1:
            pv.append(multivariate_logrank_test(t, g, e).p_value)
        dm = pd.DataFrame({"T": t, "E": e, "inter": (g == "intermediate").astype(float),
                           "high": (g == "high").astype(float)})
        try:
            hr.append(np.exp(CoxPHFitter().fit(dm, "T", "E").params_["high"]))
        except Exception:
            pass
    rows = []
    present = [gn for gn in GROUPS if stats[gn]["curves"]]
    ends = []
    for gn in present:
        s = stats[gn]
        C = np.vstack(s["curves"])
        meds = np.array(s["med"], float)
        med = np.inf if np.mean(~np.isfinite(meds)) >= 0.5 else float(np.median(meds[np.isfinite(meds)]))
        ax.fill_between(grid, C.min(0), C.max(0), step="post", color=COL[gn], alpha=0.16, lw=0)
        mc = np.median(C, 0)
        ax.step(grid, mc, where="post", color=COL[gn], lw=1.4)
        ends.append((mc[-1], gn, f"{LAB[gn]}\nmedian {fmt_med(med)}"))
        if np.isfinite(med):
            ax.plot([med, med], [0, 0.5], color=COL[gn], lw=0.6, ls=(0, (2, 2)))
        if atrisk is not None:
            yy = present.index(gn)
            for x, n in zip(tick, np.median(np.array(s["risk"]), 0).astype(int)):
                atrisk.text(x, len(present) - 1 - yy, str(n), color=COL[gn], fontsize=6, ha="center", va="center")
        rows.append({"panel": title, "group": gn, "median_n": float(np.median(s["n"])), "median_OS_months": med,
                     f"S{int(h)}_median": float(np.median(C[:, int(round(h / 0.25))]))})
    ax.axhline(0.5, color=S.GRID, lw=0.6, zorder=0)
    ys = sorted(ends, reverse=True); pos = []
    for y, gn, lab in ys:                      # direct labels at the right end, de-overlapped
        y = min(y, (pos[-1] - 0.14) if pos else 1.0); y = max(y, 0.06 + 0.14 * (len(ys) - 1 - len(pos)))
        pos.append(y)
        ax.text(tau * 1.015, y, lab, color=COL[gn], fontsize=6, va="center", ha="left", linespacing=1.1,
                fontweight="semibold", clip_on=False)
    pv, hr = np.array(pv), np.array(hr)
    txt = (f"Log-rank p {fmt_p(np.median(pv), eq=True)}  ·  HR high vs low {np.median(hr):.2f} "
           f"(repeats {hr.min():.2f}–{hr.max():.2f})")
    ax.text(0.0, 1.02, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3, color=INK2)
    ax.axvline(h, color=INK2, lw=0.6, ls=":")
    ax.text(h + 0.3, 0.02, f"horizon {int(h)} mo", ha="left", va="bottom", fontsize=5.5, color=INK2)
    ax.set_xlim(0, tau); ax.set_ylim(0, 1.0); ax.set_xticks(tick)
    ax.set_title(title, fontsize=7.5, pad=13)
    S.grid(ax, "x")
    if letter:
        S.panel_label(ax, letter, x=-0.14, y=1.06)
    if atrisk is not None:
        atrisk.set_xlim(0, tau); atrisk.set_ylim(-0.6, len(present) - 0.4)
        atrisk.set_yticks(range(len(present))); atrisk.set_yticklabels([LAB[g] for g in present][::-1], fontsize=6)
        atrisk.tick_params(axis="y", pad=8)
        for tl, gn in zip(atrisk.get_yticklabels(), present[::-1]):
            tl.set_color(COL[gn])
        atrisk.set_xticks(tick); atrisk.tick_params(length=0, labelsize=6)
        for sp in atrisk.spines.values():
            sp.set_visible(False)
        atrisk.set_xlabel("Months from landmark")
        atrisk.set_title("Number at risk", fontsize=6, fontweight="normal", color=INK2, pad=2)
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Months from landmark")
    for r in rows:
        r.update({"logrank_p_median": float(np.median(pv)), "logrank_p_min": float(pv.min()),
                  "logrank_p_max": float(pv.max()), "HR_high_low_median": float(np.median(hr)),
                  "HR_high_low_min": float(hr.min()), "HR_high_low_max": float(hr.max())})
    return rows


def km_grid(n, width):
    fig = plt.figure(figsize=(width, 3.05))
    gs = fig.add_gridspec(2, n, height_ratios=[4, 1.05], hspace=0.14, wspace=0.55)
    axs = [fig.add_subplot(gs[0, i]) for i in range(n)]
    ars = [fig.add_subplot(gs[1, i]) for i in range(n)]
    return fig, axs, ars


def finish(fig, axs, path, title=None):
    for a in axs:
        a.set_ylabel("Overall survival probability" if a is axs[0] else "")
    S.save(fig, path)


def hr_ci(t, e, f):
    c = CoxPHFitter().fit(pd.DataFrame({"T": t, "E": e, "f": f.astype(float)}), "T", "E")
    lo, hi = np.exp(c.confidence_intervals_.loc["f"].to_numpy())
    return {"HR": float(np.exp(c.params_["f"])), "HR_lo": float(lo), "HR_hi": float(hi), "HR_p": float(c.summary.loc["f", "p"])}


def flags_table(trial, df, groups, h, rules):
    t, e = df.OS_rebaseline.to_numpy(float), df.OS_event.to_numpy(int)
    case, ctrl, w = ipcw_weights(t, e, h, censor_km(t, e))
    rows = []
    for name, flag in rules.items():
        f = flag.loc[df.index].to_numpy().astype(bool)
        res = {"flag": name, **binary_perf(f, case, ctrl, w)}
        res.update(hr_ci(t, e, f))
        rows.append({"trial": trial, **res})
        sher = []
        for r, gr in groups.items():
            eta = gr.loc[df.index, "eta"].to_numpy()
            fs = eta >= np.quantile(eta, 1 - f.mean())
            rr = binary_perf(fs, case, ctrl, w)
            rr.update(hr_ci(t, e, fs))
            sher.append(rr)
        sh = pd.DataFrame(sher)
        rows.append({"trial": trial, "flag": f"SHERLOC, same proportion as '{name}' (median of 10 repeats)",
                     **sh.median().to_dict(), **{f"{k}_min": v for k, v in sh.min().items()},
                     **{f"{k}_max": v for k, v in sh.max().items()}})
    return rows


def recist_adjusted(trial, df, groups, rec):
    """Cox model of OS on the SHERLOC group adjusted for the RECIST response category."""
    resp = rec.map({"PR": "PR_CR", "CR": "PR_CR", "SD": "SD", "PD": "PD_NE", "NE": "PD_NE"})
    ok = resp.notna()
    adj = []
    for r, gr in groups.items():
        g = gr.loc[df.index, "group"]
        dm = pd.DataFrame({"T": df.OS_rebaseline, "E": df.OS_event, "inter": (g == "intermediate").astype(float),
                           "high": (g == "high").astype(float), "SD": (resp == "SD").astype(float),
                           "PD_NE": (resp == "PD_NE").astype(float)})[ok]
        s_ = CoxPHFitter().fit(dm, "T", "E").summary
        adj.append({"repeat": r, **{f"HR_{k}": s_.loc[k, "exp(coef)"] for k in s_.index},
                    **{f"lo_{k}": s_.loc[k, "exp(coef) lower 95%"] for k in s_.index},
                    **{f"hi_{k}": s_.loc[k, "exp(coef) upper 95%"] for k in s_.index},
                    **{f"p_{k}": s_.loc[k, "p"] for k in s_.index}})
    adj = pd.DataFrame(adj); adj.to_csv(ROOT / trial / "recist_adjusted_hr_per_repeat.csv", index=False)
    adj.median().to_frame("median_over_repeats").to_csv(ROOT / trial / "recist_adjusted_hr.csv")
    print(trial, adj.median().round(3).to_dict())


def pada1_first_recist(index):
    """Overall response (RG) at the first RECIST assessment of the first line, PADA-1."""
    r = pd.read_csv(PADA_RECIST, sep="\t")
    r = r[(r.Assessment_Site == "RG") & (r.Assessment_Line == 1)]
    f = r.sort_values(["Subject_Id", "Assessment_Nb"]).groupby("Subject_Id").first()
    return f.reindex(index)["Assessment_Result"], pd.to_datetime(f.reindex(index)["Assessment_Date"], dayfirst=True)


def main():
    rows, flag_rows = [], []
    # overall
    fig, axs, ars = km_grid(2, S.FULL)
    for ax, ar, trial, L in zip(axs, ars, ["pada1", "impower150"], "ab"):
        cfgT, df, groups = load(trial)
        rows += km_panel(ax, df, groups, df.index,
                         f"{NAME[trial]} · landmark {cfgT['landmark']} · n = {len(df)}, {int(df.OS_event.sum())} deaths",
                         cfgT["horizon"], cfgT["tau"], atrisk=ar, letter=L)
    finish(fig, axs, FIG / "fig_km_overall.png",
           "SHERLOC out-of-fold risk groups, 10 x 5-fold cross-fitting (line: median over repeats; band: range)")
    # IMpower150 within week-6 RECIST
    cfgT, df, groups = load("impower150")
    cl = pd.read_csv(IMP_CLN).set_index("full_anonymized_patient_id")
    rec = cl.loc[df.index, "tumor_assessment_week_6"]
    fig, axs, ars = km_grid(2, S.FULL)
    for ax, ar, cat, lab, L in zip(axs, ars, ["SD", "PR"], ["Stable disease", "Partial response"], "ab"):
        ids = rec.index[rec == cat]
        rows += km_panel(ax, df, groups, ids, f"RECIST {lab.lower()} at week 6 · n = {len(ids)}, "
                                              f"{int(df.loc[ids].OS_event.sum())} deaths",
                         cfgT["horizon"], cfgT["tau"], atrisk=ar, letter=L)
    finish(fig, axs, FIG / "fig_km_recist.png",
           "SHERLOC risk groups within radiological response categories (IMpower150, landmark C4D1)")
    recist_adjusted("impower150", df, groups, rec)
    # comparator flags, IMpower150
    mv_bl, mv_lm = df["BL_mean_of_AF"], df["C4D1_mean_of_AF"]
    rules = {"No objective response at week 6 (RECIST SD/PD/NE)": ~rec.isin(["PR", "CR"]),
             "No ctDNA response (< 50 % drop of mean VAF BL->C4D1, or new detection)":
                 ((mv_bl > 0) & (mv_lm >= 0.5 * mv_bl)) | ((mv_bl == 0) & (mv_lm > 0))}
    flag_rows += flags_table("impower150", df, groups, cfgT["horizon"], rules)
    # PADA-1 within the first RECIST assessment (first line), as for IMpower150
    cfgT, df, groups = load("pada1")
    rec, rdate = pada1_first_recist(df.index)
    q = pd.read_csv(PADA_CLN, sep="\t").set_index("Subject_Id")
    c2 = pd.to_datetime(q.loc[df.index, "Blood_Collection_Date_B"], dayfirst=True, errors="coerce")
    lag = ((rdate - c2).dt.days / 30.44).describe()
    print("PADA-1 first RECIST, months after the C2 landmark:", lag.round(2).to_dict(), rec.value_counts(dropna=False).to_dict())
    fig, axs, ars = km_grid(2, S.FULL)
    for ax, ar, cat, lab, L in zip(axs, ars, ["SD", "PR"], ["Stable disease", "Partial response"], "ab"):
        ids = rec.index[rec == cat]
        rows += km_panel(ax, df, groups, ids, f"RECIST {lab.lower()}, first assessment · n = {len(ids)}, "
                                              f"{int(df.loc[ids].OS_event.sum())} deaths",
                         cfgT["horizon"], cfgT["tau"], atrisk=ar, letter=L)
    finish(fig, axs, FIG / "fig_km_recist_pada1.png", "")
    recist_adjusted("pada1", df, groups, rec)
    mv_bl, mv_lm = df["BL_mean_VAF"], df["C2_mean_VAF"]
    rules = {"No objective response at first assessment (RECIST SD/PD/NE)": ~rec.isin(["PR", "CR"]) & rec.notna(),
             "No ctDNA response (< 50 % drop of mean VAF BL->C2, or new detection)":
                 ((mv_bl > 0) & (mv_lm >= 0.5 * mv_bl)) | ((mv_bl == 0) & (mv_lm > 0))}
    flag_rows += flags_table("pada1", df, groups, cfgT["horizon"], rules)
    pd.DataFrame(rows).to_csv(ROOT / "km_panels_summary.csv", index=False)
    pd.DataFrame(flag_rows).to_csv(ROOT / "comparator_flags.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows)[["panel", "group", "median_n", "median_OS_months", "logrank_p_median",
                              "HR_high_low_median"]].round(4).to_string(index=False))
    print(pd.DataFrame(flag_rows)[["trial", "flag", "flagged_fraction", "sensitivity", "specificity", "PPV", "NPV",
                                   "F1", "HR"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
