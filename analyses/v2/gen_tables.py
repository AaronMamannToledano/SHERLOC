"""gen_tables.py — LaTeX tables of the article, generated from the result files (no hand-copied numbers)."""
import pandas as pd, numpy as np
R = "results"   # run from analyses/v2; tables are written to article/tables/
DS = [("impower150_4tp", "IMP150 4 tp"), ("impower150_3tp", "IMP150 3 tp"), ("impower150_2tp", "IMP150 2 tp"),
      ("pada1", "PADA-1")]
ORDER = ["SHERLOC", "MLP (DeepSurv)", "RNN", "GRU", "Temporal Transformer", "Feature-wise Transformer", "Ridge Cox",
         "Lasso Cox", "Elastic-net Cox", "RSF", "FPCRF"]
_j = pd.read_csv(f"{R}/jm_summary_all_datasets.csv").set_index("dataset")   # joint model v2 (run locally, 100 splits)
JM = {d: {"C_index": _j.loc[d, "C"], "iAUC": _j.loc[d, "iAUC"], "IBS": _j.loc[d, "IBS"], "calib": _j.loc[d, "calib"],
          "p_C": _j.loc[d, "p_C"]} for d in _j.index}
t = pd.read_csv(f"{R}/posthoc_benchmark/summary_all_datasets.csv")
_pp = pd.read_csv(f"{R}/posthoc_benchmark/paired_vs_sherloc.csv")
_pp = _pp[_pp.variant == "raw"].replace({"method": {"Random survival forest": "RSF"}})
P_IAUC = {(r.dataset, r.method): r.p_iAUC for r in _pp.itertuples()}


def star(p):
    return "" if pd.isna(p) else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else "$^{\\circ}$" if p < 0.1 else ""


def bench(metric, fmt, best, fname, caption, label, jm_key, extra=None):
    lines = ["\\begin{table}[t]\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}",
             "\\begin{tabular}{l" + "c" * len(DS) + "}", "\\toprule",
             "Method & " + " & ".join(n for _, n in DS) + "\\\\", "\\midrule"]
    vals = {d: t[t.dataset == d].set_index("method") for d, _ in DS}
    bests = {d: (vals[d][metric].max() if best == "max" else vals[d][metric].min()) for d, _ in DS}
    for m in ORDER:
        cells = []
        for d, _ in DS:
            v = vals[d].loc[m, metric]
            s = fmt.format(v)
            if np.isclose(v, bests[d]): s = f"\\textbf{{{s}}}"
            if metric == "C_index" and m != "SHERLOC": s += star(vals[d].loc[m, "p_C"])
            if metric == "iAUC" and m != "SHERLOC": s += star(P_IAUC.get((d, m)))
            cells.append(s)
        lines.append(m + " & " + " & ".join(cells) + "\\\\")
    lines.append("\\midrule")
    lines.append("Joint model (JMbayes2)$^{a}$ & " + " & ".join(
        (fmt.format(JM[d][jm_key]) + (star(JM[d]["p_C"]) if metric == "C_index" else star(P_IAUC.get((d, "Joint model"))) if metric == "iAUC" else "")) if d in JM else "\\pending"
        for d, _ in DS) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if extra: lines.append(extra)
    lines.append("\\end{table}")
    open(f"article/tables/{fname}", "w").write("\n".join(lines))


note = ("\\par\\vspace{2pt}\\footnotesize\\raggedright Mean over 100 random 70/30 splits (same splits for every method). "
        "Bold: best value. Paired Nadeau--Bengio corrected $t$-test, SHERLOC vs method: $^{\\circ}p<0.1$, "
        "$^{*}p<0.05$, $^{**}p<0.01$. $^{a}$Joint model: not recalibrated; IMpower150 without patient-slope terms (see Methods).")
bench("C_index", "{:.3f}", "max", "bench_c.tex",
      "Discrimination (Uno C-index) of SHERLOC and competing methods, overall survival from the landmark.",
      "tab:bench_c", "C_index", note)
bench("iAUC", "{:.3f}", "max", "bench_iauc.tex",
      "Integrated time-dependent AUC (3 to 30 months for IMpower150, 3 to 36 months for PADA-1).",
      "tab:bench_iauc", "iAUC", note)
# IBS raw -> recalibrated
lines = ["\\begin{table}[t]\\centering\\small",
         "\\caption{Integrated Brier score (IBS) of each method as trained (raw) and after the same 5-fold out-of-fold "
         "post-hoc recalibration (rec.). Lower is better.}\\label{tab:bench_ibs}",
         "\\begin{tabular}{l" + "cc" * len(DS) + "}", "\\toprule",
         "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in DS) + "\\\\",
         " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DS))),
         "Method & " + " & ".join(["raw & rec."] * len(DS)) + "\\\\", "\\midrule"]
vals = {d: t[t.dataset == d].set_index("method") for d, _ in DS}
bmin = {d: min(vals[d].IBS_raw.min(), vals[d].IBS_recal.min()) for d, _ in DS}
for m in ORDER:
    cells = []
    for d, _ in DS:
        for c in ("IBS_raw", "IBS_recal"):
            v = vals[d].loc[m, c]; s = f"{v:.3f}"
            cells.append(f"\\textbf{{{s}}}" if np.isclose(v, bmin[d]) else s)
    lines.append(m + " & " + " & ".join(cells) + "\\\\")
lines += ["\\midrule", "Joint model (JMbayes2)$^{a}$ & " + " & ".join(
    (f"{JM[d]['IBS']:.3f} & --" if d in JM else "\\pending & --") for d, _ in DS) + "\\\\", "\\bottomrule",
          "\\end{tabular}", "\\par\\vspace{2pt}\\footnotesize\\raggedright Mean over 100 splits; bold: lowest IBS "
          "of the dataset. $^{a}$Not recalibrated.", "\\end{table}"]
open("article/tables/bench_ibs.tex", "w").write("\n".join(lines))
# cross-fitting: SHERLOC vs clinical — trials side by side, grouped rows, CI in small grey type
GROUPS2 = [("Discrimination", [("C_index", "C-index"), ("iAUC", "Integrated AUC"), ("AUC_horizon", "AUC at horizon")]),
           ("Probabilistic accuracy", [("IBS", "Integrated Brier score$^{a}$"), ("calib_err", "Calibration error at horizon$^{a}$")]),
           ("Risk groups (tertiles)", [("HR_high_vs_low", "HR, high vs low"), ("HR_intermediate_vs_low", "HR, intermediate vs low")]),
           ("Threshold set at 80\\% specificity", [("sensitivity_spec80", "Sensitivity"), ("specificity_spec80", "Specificity (observed)"),
                                                   ("PPV_spec80", "PPV"), ("NPV_spec80", "NPV"), ("flagged_fraction_spec80", "Patients flagged")]),
           ("Threshold set at 80\\% sensitivity", [("sensitivity_sens80", "Sensitivity (observed)"), ("specificity_sens80", "Specificity"),
                                                   ("PPV_sens80", "PPV"), ("NPV_sens80", "NPV"), ("flagged_fraction_sens80", "Patients flagged")])]
S = {tr: pd.read_csv(f"{R}/clinical_crossfit/{tr}/summary.csv").set_index(["model", "metric"]) for tr in ("pada1", "impower150")}
def num(v, d):
    return f"{v:.{d}f}".replace("-", "$-$")
def cell(r, d, diff=False):
    sig = diff and (r.ci_low > 0 or r.ci_high < 0)
    v = num(r.median_over_repeats, d)
    if diff and not r.median_over_repeats < 0: v = "+" + v
    v = f"\\textbf{{{v}}}" if sig else v
    return f"{v} {{\\color{{inkgrey}}\\scriptsize({num(r.ci_low, d)}, {num(r.ci_high, d)})}}"
lines = ["\\begin{table*}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
         "\\caption{\\textbf{SHERLOC versus a clinical Cox model}, 10$\\times$5-fold cross-fitting on the whole cohort. "
         "SHERLOC uses all samples up to the landmark and the clinical covariates. Death within the horizon: 24 months "
         "(PADA-1) or 12 months (IMpower150) after the landmark. Values: median over the 10 repeats (95\\% CI, pooled patient "
         "bootstrap, in grey). $\\Delta$: paired difference SHERLOC $-$ clinical, in bold when its CI excludes 0. "
         "Thresholds and tertile cutoffs were chosen inside the training folds only; sensitivity, specificity, PPV and NPV "
         "are IPCW-weighted.}\\label{tab:crossfit}",
         "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l lll lll}", "\\toprule",
         " & \\multicolumn{3}{c}{PADA-1 (369 patients, 105 deaths; horizon 24 months)} & "
         "\\multicolumn{3}{c}{IMpower150 (386 patients, 261 deaths; horizon 12 months)}\\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Metric & Clinical Cox & SHERLOC & $\\Delta$ & Clinical Cox & SHERLOC & $\\Delta$\\\\", "\\midrule"]
for gi, (gname, mets) in enumerate(GROUPS2):
    if gi: lines.append("\\addlinespace[3pt]")
    lines.append(f"\\multicolumn{{7}}{{l}}{{\\textsf{{\\textbf{{\\color{{plum}}{gname}}}}}}}\\\\")
    for m, lab in mets:
        d = 2 if m.startswith("HR") or m.startswith("flagged") or m.startswith("sens") or m.startswith("spec") or m.startswith("PPV") or m.startswith("NPV") else 3
        row = [f"\\quad {lab}"]
        for tr in ("pada1", "impower150"):
            row += [cell(S[tr].loc[("Clinical Cox", m)], d), cell(S[tr].loc[("SHERLOC", m)], d),
                    "--" if m.startswith("HR") else cell(S[tr].loc[("SHERLOC - Clinical Cox", m)], d, diff=True)]
        lines.append(" & ".join(row) + "\\\\")
lines += ["\\bottomrule", "\\end{tabular}}",
          "\\par\\vspace{2pt}\\footnotesize\\raggedright $^{a}$Lower is better.", "\\end{table*}"]
open("article/tables/crossfit.tex", "w").write("\n".join(lines))
# comparator flags
c = pd.read_csv(f"{R}/clinical_crossfit/comparator_flags.csv")
lab = {"No objective response at week 6 (RECIST SD/PD/NE)": "No objective response, week 6 (RECIST)",
       "No ctDNA response (< 50 % drop of mean VAF BL->C4D1, or new detection)": "No ctDNA response (BL$\\to$C4D1)",
       "No objective response at first assessment (RECIST SD/PD/NE)": "No objective response, first assessment (RECIST)",
       "No ctDNA response (< 50 % drop of mean VAF BL->C2, or new detection)": "No ctDNA response (BL$\\to$C2)"}
lines = ["\\begin{table}[t]\\centering\\footnotesize",
         "\\caption{Response rules versus SHERLOC at the same flagging rate. Each rule flags a fraction of patients as "
         "poor responders; SHERLOC flags the same fraction of patients with the highest out-of-fold risk scores, so the "
         "two flags can be compared at equal workload. Sensitivity, specificity, PPV and NPV are for death within the "
         "horizon (12 months in IMpower150, 24 months in PADA-1), IPCW-weighted; HR (95\\% CI): Cox model of OS on the "
         "flag. SHERLOC rows: median over the 10 cross-fitting repeats.}"
         "\\label{tab:comparators}", "\\begin{tabular}{llcccccc}", "\\toprule",
         "Trial & Flag & Flagged & Sens. & Spec. & PPV & NPV & HR (95\\% CI)\\\\", "\\midrule"]
for _, r in c.iterrows():
    name = lab.get(r.flag, "\\quad SHERLOC, same proportion")
    tr = "IMpower150" if r.trial == "impower150" else "PADA-1"
    lines.append(f"{tr} & {name} & {r.flagged_fraction:.0%} & {r.sensitivity:.2f} & {r.specificity:.2f} & "
                 f"{r.PPV:.2f} & {r.NPV:.2f} & {r.HR:.2f} ({r.HR_lo:.2f}--{r.HR_hi:.2f})\\\\".replace("%", "\\%"))
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open("article/tables/comparators.tex", "w").write("\n".join(lines))
print("tables written")

# two-column layout: wide tables span both columns; narrow ones use \footnotesize
for f in ["bench_ibs.tex", "comparators.tex"]:
    s = open(f"article/tables/{f}").read().replace("\\begin{table}[t]", "\\begin{table*}[t]").replace("\\end{table}", "\\end{table*}")
    open(f"article/tables/{f}", "w").write(s)
for f in ["bench_c.tex", "bench_iauc.tex"]:
    s = open(f"article/tables/{f}").read().replace("\\centering\\small", "\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}")
    open(f"article/tables/{f}", "w").write(s)
