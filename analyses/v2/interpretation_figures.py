"""
interpretation_figures.py — figures and LaTeX tables for interpretation.py outputs.

  article/figures/fig_interp_repo_<dataset>.png  repository / paper Fig. 3 layout (fc_1 refits)
  article/figures/fig_interp_ame.png             calibrated standardised marginal effects, both trials
  article/figures/fig_interp_arms.png            arm decomposition of the risk score by risk tertile
  article/tables/interp_ame.tex                  table of the AME estimates (panel + clinical)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("results/interpretation")
FIG, TAB = Path("article/figures"), Path("article/tables")
import article_style as S
INK, INK2, GRID = S.INK, S.INK2, S.GRID
ARM_COL = S.ARM
TITLE = {"pada1": "PADA-1 (BL, C2)", "impower150_4tp": "IMpower150 (BL, C2D1, C3D1, C4D1)"}



def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def load(ds):
    d = R / ds
    fc = pd.read_csv(d / "fc1_coefficients.csv", index_col=0)
    am = pd.read_csv(d / "ame_raw_inputs.csv", index_col=[0, 1, 2])
    enc = pd.read_csv(d / "encoder_gene_contribution.csv", index_col=0)
    arm = pd.read_csv(d / "arm_decomposition.csv", index_col=0)
    meta = pd.read_csv(d / "meta.csv", index_col=0).iloc[:, 0]
    return fc, am, enc, arm, meta


def hbar(ax, labels, val, lo, hi, p, col, ref=0.0, xlabel=""):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, np.asarray(val) - ref, left=ref, height=0.62, color=col, lw=0)
    ax.errorbar(val, y, xerr=[np.asarray(val) - lo, np.asarray(hi) - val], fmt="none", ecolor=INK2, lw=0.8, capsize=1.5)
    ax.axvline(ref, color=INK2, lw=0.6)
    span = np.nanmax(hi) - np.nanmin(np.minimum(lo, ref))
    for yi, h_, pp in zip(y, hi, p):
        s = stars(pp)
        if s:
            ax.text(h_ + 0.02 * span, yi, s, va="center", fontsize=6, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel(xlabel); ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_ylim(-0.7, len(labels) - 0.3)


CONST_BL = {"log-ratio mean VAF", "log-ratio max VAF", "ctDNA clearance", "Fraction mutations cleared",
            "No. new mutations", "No. drivers lost"}          # PADA-1 change features: 0 at BL by construction


def forest(ax, labels, m, lo, hi, p, cols, ref=1.0, log=True, xlabel=""):
    y = np.arange(len(labels))[::-1]
    ax.axvline(ref, color=INK2, lw=0.6, zorder=0)
    ax.hlines(y, lo, hi, color=cols, lw=1.1)
    ax.plot(m, y, "o", ms=3.2, color=INK, zorder=3, mec="white", mew=0.4)
    if log:
        ax.set_xscale("log"); ax.minorticks_off()
        lo_, hi_ = np.nanmin(lo), np.nanmax(hi)
        xt = [v for v in [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4] if lo_ * 0.9 <= v <= hi_ * 1.1]
        ax.set_xticks(xt); ax.set_xticklabels([f"{v:g}" for v in xt])
        ax.set_xlim(lo_ * 0.85, hi_ * 1.35)
    else:
        span = np.nanmax(hi) - np.nanmin(lo)
        ax.set_xlim(np.nanmin(lo) - 0.05 * span, np.nanmax(hi) + 0.18 * span)
    for yi, h_, pp in zip(y, hi, p):
        st = stars(pp)
        if st:
            ax.text(h_ * (1.06 if log else 1) + (0 if log else 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])), yi, st,
                    va="center", fontsize=6, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel(xlabel, fontsize=6.5); S.grid(ax, "x")
    return y


def fig_repo(ds):
    from matplotlib.colors import TwoSlopeNorm
    fc, _, enc, _, meta = load(ds)
    tps = [c[2:] for c in fc.columns if c.startswith("w_")]
    pan, cln = fc[fc.arm == "Panel"], fc[fc.arm == "Clinical"]
    gn, en = fc[fc.arm == "Gene"].copy(), fc[fc.arm == "Encoder"]
    npc = len(pan) + len(cln)
    h1 = 0.14 * npc + 0.6
    h2 = 0.118 * max(len(gn), len(en)) + 0.6
    h3 = 0.118 * 10 + 0.6
    fig = plt.figure(figsize=(S.FULL, h1 + h2 + h3 + 0.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[h1, h2, h3], width_ratios=[1.35, 1], hspace=0.5, wspace=0.75)
    # a: Cox-layer HR of panel representations and clinical covariates
    ax = fig.add_subplot(gs[0, 0])
    rows = pd.concat([pan, cln])
    lab = [r.replace(" [1]", ", rep. 1").replace(" [2]", ", rep. 2") for r in rows.index]
    forest(ax, lab, rows.HR, rows.HR_lo, rows.HR_hi, rows.p, [ARM_COL[a] for a in rows.arm], ref=1.0,
           xlabel="HR = exp(β) of the Cox-layer input (95% CI)")
    ax.axhline(len(cln) - 0.5, color=INK2, lw=0.5, ls=(0, (1, 2)))
    ax.set_title("Cox-layer coefficients", fontsize=7)
    S.panel_label(ax, "a", x=-0.55, y=1.0)
    # b: time-step weights of the panel filters (+ shared gene filter)
    ax = fig.add_subplot(gs[0, 1])
    W = pan[[f"w_{t}" for t in tps]].to_numpy(float)
    wg = gn[[f"w_{t}" for t in tps]].iloc[0].to_numpy(float)
    M = np.vstack([wg[None, :], W])
    feats = [r.rsplit(" [", 1)[0] for r in pan.index]
    mask = np.zeros_like(M, bool)
    for i, f in enumerate(feats):
        if f in CONST_BL and "BL" in tps:
            mask[i + 1, tps.index("BL")] = True
    m = np.nanmax(np.abs(M[~mask])) or 1
    Mm = np.ma.array(M, mask=mask)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#efefef")
    im = ax.imshow(Mm, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -m, m))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = "n/a" if mask[i, j] else f"{M[i, j]:+.2f}"
            col = "white" if (not mask[i, j] and abs(M[i, j]) > 0.6 * m) else INK
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.6, color=col)
    ylab = ["Gene filter (shared by all genes)"] + [r.replace(" [1]", ", rep. 1").replace(" [2]", ", rep. 2")
                                                     for r in pan.index]
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(ylab, fontsize=6.2)
    ax.set_xticks(range(len(tps))); ax.set_xticklabels(tps, fontsize=6.2)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for k in range(1, M.shape[0]):
        if k == 1 or (k - 1) % 2 == 0:
            ax.axhline(k - 0.5, color="white", lw=1.6 if k == 1 else 0.9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03, shrink=0.6)
    cb.ax.tick_params(labelsize=5.5); cb.set_label("Time-step weight", fontsize=6)
    ax.set_title("Time-step weights", fontsize=7, pad=14)
    S.panel_label(ax, "b", x=-1.2, y=1.07)
    # c: gene importance
    ax = fig.add_subplot(gs[1, 0])
    gn["imp"] = gn["mean"].abs(); half = (gn["hi"] - gn["lo"]) / 2
    gn = gn.sort_values("imp", ascending=False); half = half.loc[gn.index]
    forest(ax, list(gn.index), gn.imp, np.maximum(gn.imp - half, 0), gn.imp + half, gn.p, ARM_COL["Gene"], ref=0,
           log=False, xlabel="|mean β| (importance; bar: 95% CI half-width)")
    ax.set_title("Gene-level VAFs (gene arm)", fontsize=7)
    S.panel_label(ax, "c", x=-0.3, y=1.0)
    # d: encoder representations
    ax = fig.add_subplot(gs[1, 1])
    lab = [r.replace(" Representation ", ", rep. ") for r in en.index]
    forest(ax, lab, en["mean"], en.lo, en.hi, en.p, ARM_COL["Encoder"], ref=0, log=False, xlabel="β (95% CI)")
    ax.set_title("Pre-trained encoder representations", fontsize=7)
    S.panel_label(ax, "d", x=-0.5, y=1.0)
    # e, f: encoder gene contributions
    for k, r in enumerate(["Representation 1", "Representation 2"]):
        ax = fig.add_subplot(gs[2, k])
        top = enc[r].sort_values(ascending=False).head(10)
        sgn = enc.loc[top.index, f"sign_R{k + 1}"]
        y = np.arange(len(top))[::-1]
        ax.barh(y, top.values, color=[S.CRIMSON if v > 0 else S.BLUE for v in sgn], height=0.62, lw=0)
        for yi, v in zip(y, top.values):
            ax.text(v + top.max() * 0.015, yi, f"{v:.0f}%", va="center", fontsize=5.6, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels(top.index, fontsize=6.2)
        ax.set_xlim(0, top.max() * 1.2)
        ax.set_xlabel("Contribution to the representation (%)", fontsize=6.5); S.grid(ax, "x")
        ax.set_title(f"Encoder genes: {r.lower()}", fontsize=7)
        S.panel_label(ax, "ef"[k], x=-0.3 if k == 0 else -0.5, y=1.0)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=ARM_COL["Panel"], lw=1.5, label="Panel-level"),
         Line2D([], [], color=ARM_COL["Clinical"], lw=1.5, label="Clinical"),
         Patch(color=S.CRIMSON, label="Positive encoder weight (e, f)"),
         Patch(color=S.BLUE, label="Negative encoder weight (e, f)")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, -0.02))
    S.save(fig, FIG / f"fig_interp_repo_{ds}")


def ame_rows(am, k_top=6):
    am = am[(am[["mean", "lo", "hi", "e2e_lo", "e2e_hi"]].abs().sum(1) > 1e-9)].copy()   # drop constant inputs
    pc = am.loc[am.index.get_level_values(0).isin(["Panel", "Clinical"])]
    mol = am.loc[am.index.get_level_values(0).isin(["Gene", "Encoder gene"])]
    mol = mol.reindex(mol["e2e_median"].abs().sort_values(ascending=False).index).head(k_top)
    return pd.concat([pc, mol])


def fig_ame():
    dss = [d for d in TITLE if (R / d / "ame_raw_inputs.csv").exists()]
    rows = {d: ame_rows(load(d)[1]) for d in dss}
    n = max(len(r) for r in rows.values())
    fig, axs = plt.subplots(1, len(dss), figsize=(S.FULL, 0.112 * n + 0.9), gridspec_kw={"wspace": 0.75})
    axs = np.atleast_1d(axs)
    for ax, d in zip(axs, dss):
        r = rows[d]
        y = np.arange(len(r))[::-1]
        lab = [f"{f} ({t})" if a != "Clinical" else f for a, f, t in r.index]
        col = [ARM_COL[a] for a in r.index.get_level_values(0)]
        ax.hlines(y, np.exp(r.e2e_lo), np.exp(r.e2e_hi), color=col, lw=3.5, alpha=0.25)
        ax.hlines(y, np.exp(r["lo"]), np.exp(r["hi"]), color=col, lw=1.2)
        ax.plot(np.exp(r["mean"]), y, "o", ms=2.6, color=INK, zorder=3)
        ax.axvline(1, color=INK2, lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6)
        ax.set_xscale("log")
        lo_, hi_ = np.exp(np.r_[r["lo"], r.e2e_lo].min()), np.exp(np.r_[r["hi"], r.e2e_hi].max())
        xl = (min(lo_ * 0.97, 0.9), max(hi_ * 1.12, 1.1))
        xt = [0.8, 0.9, 1, 1.1, 1.25, 1.5] if xl[1] / xl[0] < 2.2 else [0.6, 0.8, 1, 1.25, 1.5, 2, 3]
        xt = [v for v in xt if xl[0] <= v <= xl[1]]
        ax.set_xticks(xt); ax.set_xticklabels([f"{v:g}" for v in xt]); ax.minorticks_off()
        ax.set_xlim(*xl)
        arms_ = list(r.index.get_level_values(0))
        for i in range(1, len(arms_)):
            if arms_[i] != arms_[i - 1]:
                ax.axhline(y[i - 1] - 0.5, color=INK2, lw=0.4, ls=(0, (1, 2)))
        S.panel_label(ax, "ab"[dss.index(d)], x=-0.62 if d == "pada1" else -0.55, y=1.0)
        for yi, s, q in zip(y, r.e2e_sign_agree, r.q_BH):
            ax.text(1.005, yi, f"{100 * s:.0f}%{' *' if q < 0.05 else ''}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=5.6, color=INK2)
        ax.text(1.005, len(r) - 0.2, "sign\nagree", transform=ax.get_yaxis_transform(), fontsize=5.6, color=INK2,
                va="bottom")
        ax.set_xlabel("Calibrated HR per 1 SD (average marginal effect)")
        ax.set_title(TITLE[d], loc="left", fontsize=7, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_ylim(-0.7, len(r) - 0.1)
    from matplotlib.lines import Line2D
    present = set().union(*[set(r.index.get_level_values(0)) for r in rows.values()])
    h = [Line2D([], [], color=ARM_COL[a], lw=3, label=a) for a in ["Panel", "Clinical", "Gene", "Encoder gene"]
         if a in present]
    h += [Line2D([], [], marker="o", ls="", color=INK, ms=3.5, label="Mean, output-layer refits")]
    h += [Line2D([], [], color=INK2, lw=1.2, label="95% CI, output-layer refits"),
          Line2D([], [], color=INK2, lw=5, alpha=0.25, label="2.5–97.5%, end-to-end refits")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    S.save(fig, FIG / "fig_interp_ame")


def fig_arms():
    dss = [d for d in TITLE if (R / d / "arm_decomposition.csv").exists()]
    fig, axs = plt.subplots(1, len(dss), figsize=(S.FULL, 1.75), gridspec_kw={"wspace": 0.6})
    axs = np.atleast_2d(axs)
    arms = ["Panel", "Encoder", "Clinical", "Gene"]
    order = ["All ctDNA on treatment", "All ctDNA at baseline", "Panel arm", "Encoder (mutation profile)",
             "Clinical arm", "Gene arm"]
    gcol = {"Panel arm": ARM_COL["Panel"], "Encoder (mutation profile)": ARM_COL["Encoder"],
            "Clinical arm": ARM_COL["Clinical"], "Gene arm": ARM_COL["Gene"]}
    for j, d in enumerate(dss):
        ax = axs[0, j]
        p = pd.read_csv(R / d / "permutation_importance.csv", index_col=0).loc[order]
        y = np.arange(len(p))[::-1]
        ax.barh(y, p["median"], color=[gcol.get(g, INK2) for g in p.index], height=0.6, lw=0)
        ax.errorbar(p["median"], y, xerr=[p["median"] - p.lo, p.hi - p["median"]], fmt="none", ecolor=INK, lw=0.8,
                    capsize=1.5)
        for yi, v, fpos in zip(y, p.hi, p.frac_pos):
            ax.text(v + 0.004, yi, f"{100 * fpos:.0f}%", va="center", fontsize=5.6, color=INK2)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels(p.index, fontsize=6)
        ax.set_xlabel("Loss of C-index when permuted (median, 2.5–97.5%)")
        S.panel_label(ax, "ab"[j], x=-0.62, y=1.02)
        ax.set_title(TITLE[d], loc="left", fontsize=7, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    S.save(fig, FIG / "fig_interp_arms")


def fmt_hr(m, lo, hi):
    return f"{np.exp(m):.2f} ({np.exp(lo):.2f}--{np.exp(hi):.2f})"


def tab_ame():
    lines = []
    for d in [d for d in TITLE if (R / d / "ame_raw_inputs.csv").exists()]:
        am = load(d)[1]
        am = am[(am[["mean", "lo", "hi"]].abs().sum(1) > 1e-9)]
        am = am.loc[am.index.get_level_values(0).isin(["Panel", "Clinical"])]
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{TITLE[d]}}}}}\\")
        for (a, f, t), r in am.iterrows():
            lab = (f if a == "Clinical" else f"{f} ({t})").replace("%", r"\%").replace(">", "$>$")
            q = r.q_BH
            qs = "$<$0.001" if q < 0.001 else f"{q:.3f}"
            lines.append(f"{lab} & {fmt_hr(r['mean'], r['lo'], r['hi'])} & {qs} & "
                         f"{fmt_hr(r.e2e_median, r.e2e_lo, r.e2e_hi)} & {100 * r.e2e_sign_agree:.0f}\\% \\\\")
        lines.append(r"\addlinespace")
    body = "\n".join(lines)
    tex = r"""\begin{table*}[t]
\centering\scriptsize
\caption{Average marginal effects of the panel-level ctDNA features (per time point) and clinical covariates on the
recalibrated log-hazard, expressed as HR per 1 SD of the raw input. Output-layer refits: mean and Nadeau--Bengio
95\% CI over 100 refits of the output layer on 80\% subsamples of the reporting split, with Benjamini--Hochberg
$q$-value. End-to-end: median and 2.5--97.5 percentiles over 100 independently trained models (benchmark splits),
and percentage of these models agreeing with the median sign.}
\label{tab:interp_ame}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lllll}
\toprule
Input & HR (95\% CI), refits & $q$ & HR (range), end-to-end & Sign agr.\\
\midrule
""" + body.replace(r"\multicolumn{6}", r"\multicolumn{5}") + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    (TAB / "interp_ame.tex").write_text(tex)


if __name__ == "__main__":
    for d in TITLE:
        if (R / d / "fc1_coefficients.csv").exists():
            fig_repo(d)
    fig_ame(); fig_arms(); tab_ame()
    print("figures written")
