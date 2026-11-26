"""article_figures.py — article figures built from the result files (style: article_style.py).

  article/figures/fig_overview.{png,pdf}            study overview: cohorts and sampling, SHERLOC, evaluation
  article/figures/fig_benchmark.{png,pdf}           C-index and IBS (raw -> recalibrated) of every method, 4 settings
  article/figures/fig_sherloc_vs_clinical.{png,pdf} 10 x 5-fold cross-fitting, SHERLOC vs clinical Cox
Usage: python article_figures.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.pyplot as plt

import article_style as S

OUT = Path("article/figures")
R = Path("results")
DS = [("impower150_4tp", "IMpower150 · 4 time points"), ("impower150_3tp", "IMpower150 · 3 time points"),
      ("impower150_2tp", "IMpower150 · 2 time points"), ("pada1", "PADA-1 · 2 time points")]
METHODS = [("SHERLOC", "SHERLOC", "SHERLOC"), ("MLP", "MLP (DeepSurv)", "deep"), ("RNN", "RNN", "deep"),
           ("GRU", "GRU", "deep"), ("Temporal_Transformer", "Temporal Transformer", "deep"),
           ("Featurewise_Transformer", "Feature-wise Transformer", "deep"), ("ridge_scaled", "Ridge Cox", "stat"),
           ("lasso_scaled", "Lasso Cox", "stat"), ("elastic_net_scaled", "Elastic-net Cox", "stat"),
           ("RSF", "Random survival forest", "stat"), ("FPCRF", "FPCRF", "stat"), ("JM", "Joint model", "jm")]
FAM = {"SHERLOC": S.CRIMSON, "deep": S.BLUE, "stat": S.AMBER, "jm": S.TEAL}
FAM_LAB = {"SHERLOC": "SHERLOC", "deep": "Deep learning", "stat": "Statistical / ensemble", "jm": "Bayesian joint model"}
JM_FILES = {"impower150_4tp": "results/jm_impower150/4tp/JointModel_v2_metrics_per_split.csv",
            "impower150_3tp": "results/jm_impower150/3tp/JointModel_v2_metrics_per_split.csv",
            "impower150_2tp": "results/jm_impower150/2tp/JointModel_v2_metrics_per_split.csv",
            "pada1": "results/benchmark_response_longitudinal/JointModel_v2_metrics_per_split.csv"}


# ─────────────────────────────────────────────────────────────────────────────
def load_bench(ds):
    fs = [f for f in glob.glob(str(R / "posthoc_benchmark" / ds / "*seeds*.csv")) if "fallback" not in f]
    d = pd.concat([pd.read_csv(f) for f in fs])
    d = d.drop_duplicates(["method", "seed", "variant"])
    j = pd.read_csv(JM_FILES[ds]).assign(method="JM")
    j = pd.concat([j.assign(variant="raw"), j.assign(variant="recalibrated")]).rename(columns={"calib_err_24m": "calib_err"})
    return pd.concat([d, j[["method", "seed", "variant", "C_index_Uno", "iAUC", "IBS"]]])


def nb_ci(x, frac=0.3):
    x = np.asarray(x, float); n = len(x)
    from scipy import stats
    se = np.sqrt(x.var(ddof=1) * (1 / n + frac / (1 - frac)))
    return x.mean(), stats.t.ppf(0.975, n - 1) * se


def fig_benchmark():
    fig, axs = plt.subplots(3, 4, figsize=(S.FULL, 7.0), sharey=True,
                            gridspec_kw={"hspace": 0.38, "wspace": 0.12})
    y = np.arange(len(METHODS))[::-1]
    for j, (ds, title) in enumerate(DS):
        d = load_bench(ds)
        raw = d[d.variant == "raw"]; rec = d[d.variant == "recalibrated"]
        for row, (col, xlab) in enumerate([("C_index_Uno", "C-index (Uno)"), ("iAUC", "Integrated AUC")]):
            sh = raw[raw.method == "SHERLOC"].set_index("seed")[col]
            ax = axs[row, j]
            for yi, (m, lab, fam) in zip(y, METHODS):
                c = raw[raw.method == m].set_index("seed")[col]
                q1, q3 = c.quantile([0.25, 0.75])
                ax.plot([q1, q3], [yi, yi], color=FAM[fam], lw=2.2, alpha=0.35, solid_capstyle="butt")
                ax.plot(c.mean(), yi, "o", ms=3.6, color=FAM[fam], mec="white", mew=0.5, zorder=3)
                if m != "SHERLOC":
                    common = c.index.intersection(sh.index)
                    win = (sh.loc[common] > c.loc[common]).mean()
                    ax.text(1.0, yi, f"{100 * win:.0f}%", transform=ax.get_yaxis_transform(), fontsize=5.3,
                            color=S.INK2, va="center", ha="right")
            ax.axvline(sh.mean(), color=S.CRIMSON, lw=0.6, ls=(0, (2, 2)), zorder=0)
            if row == 0:
                ax.set_title(title, fontsize=6.8)
            ax.set_xlabel(xlab if j == 0 else "")
            S.grid(ax, "x")
            g = raw.groupby("method")[col]
            ax.set_xlim(min(g.quantile(0.25).min(), sh.mean() - 0.06) - 0.005, g.quantile(0.75).max() + 0.03)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            if j == 0:
                S.panel_label(ax, "ab"[row], x=-0.9, y=1.02)
        # IBS raw -> recalibrated
        ax = axs[2, j]
        for yi, (m, lab, fam) in zip(y, METHODS):
            a = raw[raw.method == m].IBS.mean(); b = rec[rec.method == m].IBS.mean()
            if m != "JM":
                ax.annotate("", xy=(b, yi), xytext=(a, yi),
                            arrowprops=dict(arrowstyle="-|>", color=FAM[fam], lw=0.8, mutation_scale=5,
                                            shrinkA=1.5, shrinkB=1.5))
                ax.plot(a, yi, "o", ms=3.2, mfc="white", mec=FAM[fam], mew=0.8, zorder=3)
            ax.plot(b, yi, "o", ms=3.4, color=FAM[fam], mec="white", mew=0.4, zorder=4)
        best = min(rec.groupby("method").IBS.mean().min(), raw.groupby("method").IBS.mean().min())
        ax.axvline(best, color=S.INK2, lw=0.5, ls=":", zorder=0)
        ax.set_xlabel("Integrated Brier score (lower is better)" if j == 0 else "")
        S.grid(ax, "x")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        if j == 0:
            S.panel_label(ax, "c", x=-0.9, y=1.02)
    for ax in axs[:, 0]:
        ax.set_yticks(y); ax.set_yticklabels([lab for _, lab, _ in METHODS])
        for tl, (_, _, fam) in zip(ax.get_yticklabels(), METHODS):
            tl.set_color(FAM[fam] if fam == "SHERLOC" else S.INK)
            if fam == "SHERLOC":
                tl.set_fontweight("bold")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=c, ms=4, label=FAM_LAB[k]) for k, c in FAM.items()]
    h += [Line2D([], [], color=S.INK2, lw=2.2, alpha=0.35, label="Interquartile range over 100 splits"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK2, ms=4, label="IBS as trained"),
          Line2D([], [], marker="o", ls="", color=S.INK2, ms=4, label="IBS after out-of-fold recalibration")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.55, -0.045), fontsize=6, columnspacing=1.2)
    S.save(fig, OUT / "fig_benchmark")


# ─────────────────────────────────────────────────────────────────────────────
def fig_sherloc_vs_clinical():
    MET = [("C_index", "C-index (Uno)"), ("iAUC", "Integrated AUC"), ("AUC_horizon", "AUC at horizon"),
           ("sensitivity_spec80", "Sensitivity at 80% specificity"),
           ("specificity_sens80", "Specificity at 80% sensitivity"),
           ("PPV_spec80", "PPV at 80% specificity"), ("NPV_spec80", "NPV at 80% specificity")]
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL, 2.55), sharey=True, gridspec_kw={"wspace": 0.16})
    for ax, (tr, name), L in zip(axs, [("pada1", "PADA-1 · death within 24 months"),
                                       ("impower150", "IMpower150 · death within 12 months")], "ab"):
        s = pd.read_csv(R / "clinical_crossfit" / tr / "summary.csv").set_index(["model", "metric"])
        for j, (m, lab) in enumerate(MET):
            yb = len(MET) - 1 - j
            a, b = s.loc[("Clinical Cox", m)], s.loc[("SHERLOC", m)]
            ax.plot([a.median_over_repeats, b.median_over_repeats], [yb, yb], color=S.GRID, lw=3, zorder=1,
                    solid_capstyle="round")
            for k, (mod, r) in enumerate([("Clinical Cox", a), ("SHERLOC", b)]):
                yy = yb + (0.16 if k else -0.16)
                ax.plot([r.ci_low, r.ci_high], [yy, yy], color=S.MODEL[mod], lw=0.9, zorder=2)
                ax.plot(r.median_over_repeats, yy, "o", color=S.MODEL[mod], ms=4, mec="white", mew=0.5, zorder=3,
                        label=mod if j == 0 else None)
            dd = s.loc[("SHERLOC - Clinical Cox", m)]
            sig = (dd.ci_low > 0) or (dd.ci_high < 0)
            ax.text(1.015, yb, f"{dd.median_over_repeats:+.2f}", transform=ax.get_yaxis_transform(),
                    fontsize=6, ha="left", va="center", color=S.CRIMSON if sig else S.INK2,
                    fontweight="semibold" if sig else "normal")
        ax.set_yticks(range(len(MET))); ax.set_yticklabels([l for _, l in MET][::-1])
        ax.set_xlim(0.15, 1.0); S.grid(ax, "x")
        ax.set_title(name); ax.set_xlabel("Median over 10 repeats (95% CI, patient bootstrap)")
        S.panel_label(ax, L, x=-0.04 if L == "b" else -0.5, y=1.03)
    for a in axs:
        a.text(1.015, len(MET) - 0.45, "Δ", transform=a.get_yaxis_transform(), ha="left", fontsize=6.5, color=S.INK2)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.07))
    S.save(fig, OUT / "fig_sherloc_vs_clinical")


# ─────────────────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, text, fc, ec=None, fs=6.2, color=S.INK, weight="normal", r=0.012, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc,
                                ec=ec or fc, lw=0.7, ls=ls, transform=ax.transAxes, clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight=weight,
            transform=ax.transAxes, linespacing=1.25)


def arrow(ax, p, q, color=S.INK2, lw=0.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=6, color=color, lw=lw,
                                 transform=ax.transAxes, clip_on=False))


def grid_icon(ax, x, y, nr, nc, cell, aspect, col, vals=None, alpha_min=0.15):
    """Small heat-map icon: nr x nc cells of colour col with varying intensity (top-left at x, y)."""
    rng = np.random.RandomState(nr * 7 + nc)
    v = rng.rand(nr, nc) if vals is None else vals
    for i in range(nr):
        for j in range(nc):
            ax.add_patch(plt.Rectangle((x + j * cell, y - (i + 1) * cell * aspect), cell * 0.9, cell * aspect * 0.9,
                                       fc=col, alpha=alpha_min + (1 - alpha_min) * v[i, j], lw=0))


def fig_overview():
    W, H = S.FULL, 5.4
    asp = W / H
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_autoscale_on(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    PINKBG, BLUEBG, GREYBG = "#fbeef2", "#edf3fb", "#f4f3f1"
    C_G, C_P, C_E, C_C = S.ARM["Gene"], S.ARM["Panel"], S.ARM["Encoder"], S.ARM["Clinical"]
    # ── a: cohorts and sampling ──────────────────────────────────────────────
    ax.text(0.005, 0.985, "a", fontsize=10, fontweight="bold", va="top")
    ax.text(0.03, 0.982, "Cohorts, liquid-biopsy sampling and encoder pre-training", fontsize=7.5,
            fontweight="semibold", va="top")
    box(ax, 0.01, 0.705, 0.555, 0.245, "", GREYBG)
    rows = [("PADA-1", "ER+/HER2− advanced breast cancer\naromatase inhibitor + palbociclib\n369 patients · 105 deaths",
             [(0, "BL"), (1.0, "C2")], 1.0, S.CRIMSON, "≈1 month"),
            ("IMpower150", "Non-squamous NSCLC\nchemo-immunotherapy ± bevacizumab\n386 patients · 261 deaths",
             [(0, "BL"), (0.69, "C2D1"), (1.38, "C3D1"), (2.07, "C4D1")], 2.07, S.BLUE, "9 weeks")]
    for k, (name, sub, pts, lm, col, when) in enumerate(rows):
        y0 = 0.885 - k * 0.12
        ax.text(0.022, y0 + 0.03, name, fontsize=7, fontweight="bold", color=col, va="center")
        ax.text(0.022, y0 - 0.025, sub, fontsize=5.4, color=S.INK2, va="center", linespacing=1.2)
        x0, sc = 0.235, 0.098
        end = x0 + 2.07 * sc + 0.07
        ax.plot([x0, x0 + lm * sc], [y0, y0], color=S.INK2, lw=0.7)
        arrow(ax, (x0 + lm * sc, y0), (end, y0), color=S.INK, lw=1.0)
        ax.text(end + 0.004, y0, "OS", fontsize=5.6, va="center", color=S.INK, fontweight="semibold")
        for t, lab in pts:
            ax.plot(x0 + t * sc, y0, "o", ms=4.5, color=col, mec="white", mew=0.6, zorder=3)
            ax.text(x0 + t * sc, y0 + 0.022, lab, fontsize=5.3, ha="center", color=S.INK)
        ax.plot([x0 + lm * sc] * 2, [y0 - 0.02, y0 + 0.016], color=S.INK, lw=0.9)
        ax.text(x0 + lm * sc, y0 - 0.038, f"landmark ({when})", fontsize=5.1, ha="center", color=S.INK)
    # pre-training panel
    box(ax, 0.58, 0.705, 0.41, 0.245, "", PINKBG)
    ax.text(0.595, 0.925, "Encoder pre-training (MSK-CHORD)", fontsize=6.4, fontweight="semibold", color=S.PLUM,
            va="center")
    ax.text(0.595, 0.83, "24,947 patients · 5 cancer types\ntissue sequencing, overall survival\n"
                         "genes shared with each trial panel\n(PADA-1: 16 · IMpower150: 58)", fontsize=5.4,
            color=S.INK, va="center", linespacing=1.35)
    grid_icon(ax, 0.79, 0.895, 6, 1, 0.012, asp, C_E)
    ax.text(0.796, 0.775, "VAFs", fontsize=5, ha="center", color=S.INK2)
    arrow(ax, (0.808, 0.86), (0.822, 0.86))
    box(ax, 0.822, 0.83, 0.095, 0.06, "2 survival-aware\nrepresentations", "white", ec=C_E, fs=5)
    arrow(ax, (0.918, 0.86), (0.932, 0.86))
    box(ax, 0.932, 0.83, 0.048, 0.06, "Cox\n(OS)", "white", ec=S.INK2, fs=5)
    ax.text(0.885, 0.755, "frozen and reused at every\nctDNA time point", fontsize=5.2, color=S.PLUM, ha="center",
            va="center", style="italic")
    # ── b: SHERLOC architecture ──────────────────────────────────────────────
    ax.text(0.005, 0.675, "b", fontsize=10, fontweight="bold", va="top")
    ax.text(0.03, 0.672, "SHERLOC", fontsize=7.5, fontweight="semibold", va="top")
    ax.text(0.115, 0.672, "Shared and Exclusive Representations of Longitudinal ctDNA",
            fontsize=5.8, color=S.INK2, va="top")
    # architecture drawn as SVG (fig1b_svg.py) and placed here
    import subprocess, sys
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "fig1b_svg.py")], check=True,
                   stdout=subprocess.DEVNULL)
    img = plt.imread(OUT / "fig1b_architecture.png")
    hb = 0.99 * W * img.shape[0] / img.shape[1] / H
    axb = fig.add_axes([0.005, (0.643 + 0.225) / 2 - hb / 2, 0.99, hb]); axb.imshow(img, interpolation="lanczos")
    axb.axis("off")
    # ── c: evaluation ────────────────────────────────────────────────────────
    ax.text(0.005, 0.215, "c", fontsize=10, fontweight="bold", va="top")
    ax.text(0.03, 0.212, "Evaluation", fontsize=7.5, fontweight="semibold", va="top")
    cards = [("Benchmark", "100 identical 70/30 splits\n11 methods + Bayesian joint model\nC-index, iAUC, IBS, calibration",
              S.BLUE),
             ("Repeated cross-fitting", "10 × 5-fold: out-of-fold score\nfor every patient; thresholds and\ntertiles chosen in training folds",
              S.CRIMSON),
             ("Clinical utility", "Risk groups · death within 24/12 mo\nvs clinical Cox, RECIST and\nctDNA-response rules",
              S.CRIMSON),
             ("Interpretation", "Output-layer resampling\n+ marginal effects, end-to-end\nstability, permutation importance",
              S.AMBER)]
    w = 0.235
    for k, (t, body, col) in enumerate(cards):
        x0 = 0.01 + k * (w + 0.012)
        box(ax, x0, 0.015, w, 0.155, "", "white", ec=S.GRID)
        ax.add_patch(FancyBboxPatch((x0, 0.14), w, 0.032, boxstyle="round,pad=0,rounding_size=0.012", fc=col,
                                    ec=col))
        ax.text(x0 + 0.012, 0.156, t, fontsize=6.4, color="white", fontweight="bold", va="center")
        ax.text(x0 + 0.012, 0.075, body, fontsize=5.6, color=S.INK, va="center", linespacing=1.35)
        if k < 3:
            arrow(ax, (x0 + w + 0.001, 0.075), (x0 + w + 0.011, 0.075))
    fig.savefig(OUT / "fig_overview.png", bbox_inches=None); fig.savefig(OUT / "fig_overview.pdf", bbox_inches=None)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig_pretrain():
    """Pre-training ablation: difference (variant - pre-trained frozen encoder) per split, both trials."""
    from scipy import stats
    V = [("no_encoder", "No encoder"),
         ("scratch_trained", "Encoder trained on the trial\n(no pre-training)"),
         ("pretrained_unfrozen", "Pre-trained encoder,\nfine-tuned on the trial")]
    TR = [("pada1", "PADA-1", S.CRIMSON), ("impower150_4tp", "IMpower150", S.BLUE)]
    fig, axs = plt.subplots(1, 3, figsize=(S.FULL, 2.05), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (k, lab) in zip(axs, [("C_index_Uno", "Δ C-index"), ("iAUC", "Δ integrated AUC"), ("IBS", "Δ IBS")]):
        for j, (ds, name, col) in enumerate(TR):
            R_ = pd.concat([pd.read_csv(f) for f in glob.glob(str(R / f"pretraining_ablation/{ds}/*_seeds*.csv"))])
            R_ = R_.drop_duplicates(["variant", "seed"])
            ref = R_[R_.variant == "pretrained_frozen"].set_index("seed")[k]
            for i, (v, vl) in enumerate(V):
                x = R_[R_.variant == v].set_index("seed")[k]
                c = ref.index.intersection(x.index)
                d = (x.loc[c] - ref.loc[c]).to_numpy(); n = len(d)
                if n < 2:
                    continue
                se = np.sqrt(d.var(ddof=1) * (1 / n + 0.3 / 0.7)); h = stats.t.ppf(0.975, n - 1) * se
                pv = 2 * stats.t.sf(abs(d.mean() / se), n - 1)
                yy = len(V) - 1 - i + (0.14 if j == 0 else -0.14)
                ax.plot([d.mean() - h, d.mean() + h], [yy, yy], color=col, lw=1.1)
                ax.plot(d.mean(), yy, "o", color=col, ms=4, mec="white", mew=0.5, label=name if i == 0 else None)
                if pv < 0.05:
                    ax.text(d.mean() + h, yy, " *" if pv >= 0.01 else (" **" if pv >= 0.001 else " ***"),
                            color=col, fontsize=6, va="center")
        ax.axvline(0, color=S.INK2, lw=0.6)
        ax.set_xlabel(lab + (" (lower is better)" if k == "IBS" else " vs pre-trained, frozen"), fontsize=6.5)
        S.grid(ax, "x")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    axs[0].set_yticks(range(len(V))); axs[0].set_yticklabels([vl for _, vl in V][::-1])
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=6.5,
               bbox_to_anchor=(0.55, 1.06))
    S.save(fig, OUT / "fig_pretrain")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["overview", "benchmark", "clinical", "pretrain"]
    if "overview" in todo: fig_overview()
    if "benchmark" in todo: fig_benchmark()
    if "clinical" in todo: fig_sherloc_vs_clinical()
    if "pretrain" in todo: fig_pretrain()
    print("done")
