"""article_table1.py — Table 1: characteristics of the two landmark cohorts -> article/tables/table1.tex

Values are computed on the analysed landmark cohorts (patients alive at the landmark with ctDNA at every sample used),
from the same tables as the models. Percentages are among patients with a known value; missing counts are given.
"""
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

import posthoc_benchmark as B


def med(x, d=0):
    x = pd.Series(x).dropna(); f = f"{{:.{d}f}}"
    return f"{f.format(x.median())} ({f.format(x.quantile(.25))}--{f.format(x.quantile(.75))})"


def pct(x):
    x = pd.Series(x); n_na = int(x.isna().sum()); x = x.dropna()
    s = f"{int(x.sum())} ({100 * x.mean():.0f}\\%)"
    return s + (f" \\textcolor{{inkgrey}}{{\\scriptsize[{n_na} missing]}}" if n_na else "")


def km_os(d):
    k = KaplanMeierFitter().fit(d.OS_rebaseline, d.OS_event); m = k.median_survival_time_
    r = KaplanMeierFitter().fit(d.OS_rebaseline, 1 - d.OS_event)          # reverse KM follow-up
    return ("not reached" if not np.isfinite(m) else f"{m:.1f}"), f"{r.median_survival_time_:.1f}"


p = B.load_df(B.load_cfg("pada1")[0]); i = B.load_df(B.load_cfg("impower150_4tp")[0])
pm, pf = km_os(p); im, iff = km_os(i)
NA = "\\textcolor{inkgrey}{--}"
sections = [
 ("Cohort", [
  ("Patients / deaths after the landmark", f"{len(p)} / {int(p.OS_event.sum())}", f"{len(i)} / {int(i.OS_event.sum())}"),
  ("ctDNA samples used (landmark)", "BL, C2 ($\\approx$1 month)", "BL, C2D1, C3D1, C4D1 (9 weeks)"),
  ("Median follow-up$^{d}$, months", pf, iff),
  ("Median OS$^{d}$, months", pm, im)]),
 ("Patients", [
  ("Age, years, median (IQR)", med(p.cln_Age_At_Inclusion), med(i.cln_age)),
  ("Female sex", f"{len(p)} (100\\%)$^{{a}}$", pct(i.cln_sex_female)),
  ("Postmenopausal", pct(p.cln_menop), NA),
  ("ECOG performance status $\\geq 1$", pct(p.cln_ps_bas >= 1), pct(i.cln_baseline_ECOG >= 1))]),
 ("Disease", [
  ("Metastatic at diagnosis (de novo)$^{b}$", pct(p.cln_de_novo), NA),
  ("Number of metastatic sites, median (IQR)", med(p.cln_nb_mets), NA),
  ("$\\geq 3$ metastatic sites", pct(p.cln_nb_mets.where(p.cln_nb_mets.notna()) >= 3), pct(i.cln_nb_mets)),
  ("Visceral metastases", pct(p.cln_met_visceral), NA),
  ("Bone-only disease", pct(p.cln_bone_only), NA),
  ("Tumour grade 3 (primary tumour)", pct((p.cln_Tumor_Grade_Prim_T == 3).astype(float).where(p.cln_Tumor_Grade_Prim_T.notna())), NA),
  ("Lobular histology", pct(p.cln_lobular), NA),
  ("HER2-low (IHC 1+ or 2+, ISH-negative)", pct(p.cln_HER2_low), NA),
  ("PD-L1 high$^{c}$", NA, pct(i.cln_pdl1_high)),
  ("Baseline SLD, mm, median (IQR)", NA, med(i.cln_sld_baseline))]),
 ("Treatment", [
  ("First-line treatment", "AI + palbociclib", "ABCP {} / ACP {} / BCP {}".format(
      int(((i.cln_Bev_CP == 0) & (i.cln_Atezo_CP == 0)).sum()), int(i.cln_Atezo_CP.sum()), int(i.cln_Bev_CP.sum()))),
  ("Prior (neo)adjuvant chemotherapy", pct(p.cln_prior_chemo), NA)]),
 ("Circulating tumour DNA", [
  ("ctDNA detected at BL", pct(p.BL_mean_VAF > 0), pct(i.BL_mean_of_AF > 0)),
  ("Mean VAF at BL, \\%, median (IQR)", med(p.BL_mean_VAF, 2), med(100 * i.BL_mean_of_AF, 2)),
  ("Inputs: genes / panel / clinical", "10 / 9 / 9", "17 / 4 / 8")]),
]
lines = [r"\begin{table*}[t]", r"\centering\small", r"\renewcommand{\arraystretch}{1.12}",
         r"\caption{\textbf{Characteristics of the landmark cohorts.} Patients alive at the landmark with ctDNA at every "
         r"sample used. Percentages among patients with a known value. Follow-up by reverse Kaplan--Meier. AI, aromatase "
         r"inhibitor; ABCP, atezolizumab + bevacizumab + carboplatin--paclitaxel; ACP, atezolizumab + "
         r"carboplatin--paclitaxel; BCP, bevacizumab + carboplatin--paclitaxel; IHC, immunohistochemistry; ISH, in situ "
         r"hybridisation; IQR, interquartile range; SLD, sum of longest diameters; "
         r"VAF, variant allele frequency; --, not available or not applicable.}",
         r"\label{tab:cohorts}",
         r"\begin{tabular}{>{\raggedright\arraybackslash}p{5.9cm} >{\raggedright\arraybackslash}p{5.3cm} "
         r">{\raggedright\arraybackslash}p{5.0cm}}",
         r"\rowcolor{plum}\textcolor{white}{\textsf{\textbf{Characteristic}}} & "
         r"\textcolor{white}{\textsf{\textbf{PADA-1}} \footnotesize ER+/HER2$-$ advanced breast cancer} & "
         r"\textcolor{white}{\textsf{\textbf{IMpower150}} \footnotesize non-squamous NSCLC}\\"]
for name, rows in sections:
    lines.append(rf"\rowcolor{{boxbg}}\multicolumn{{3}}{{l}}{{\textsf{{\textbf{{\color{{plum}}{name}}}}}}}\\")
    for k, (a, b, c) in enumerate(rows):
        bg = r"\rowcolor{rowgrey}" if k % 2 else ""
        lines.append(f"{bg}\\quad {a} & {b} & {c}\\\\")
lines += [r"\arrayrulecolor{plum}\specialrule{0.8pt}{0pt}{0pt}", r"\end{tabular}",
          r"\par\vspace{2pt}\raggedright\footnotesize $^{a}$PADA-1 enrolled women only. "
          r"$^{b}$Metastasis within 6 months of the primary diagnosis. $^{c}$As coded in the shared IMpower150 clinical data. $^{d}$From the landmark.",
          r"\end{table*}"]
open("article/tables/table1.tex", "w").write("\n".join(lines))
print("\n".join(lines))
