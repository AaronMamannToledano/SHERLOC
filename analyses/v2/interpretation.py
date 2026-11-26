"""
interpretation.py — SHERLOC interpretation for PADA-1 and IMpower150 (4 timepoints).

Part A reproduces the procedure of the SHERLOC repository (scripts/explainability.py, paper Fig. 3):
  1. train SHERLOC once on a reporting split (seed 16, 70 % train / 30 % test; 20 % of train used
     for validation when the config has a validation split);
  2. 100 x: sample 80 % of the training patients without replacement, reload the trained weights,
     re-initialise the output layer fc_1 (Xavier-uniform, zero bias), freeze every other parameter
     and refit fc_1 (Adam lr 0.005, batch 50, <= 256 epochs, early stopping patience 20 on the
     remaining 20 %);
  3. Nadeau-Bengio corrected 95 % CI and t-test p-value of every fc_1 coefficient;
  4. per-timepoint weights of the shared gene filter (filter1) and of the panel maps (mlps);
  5. gene importance |mean beta|; encoder representation coefficients per timepoint;
  6. % contribution of each encoder gene to each pre-trained representation (Fig. 3c).

Part B adds the following:
  B1  Benjamini-Hochberg q-values next to the p-values.
  B2  Standardised average marginal effects (AME) of every RAW input at every timepoint:
      d log h / d x, averaged over training patients (eval mode, through BatchNorm, the
      per-feature temporal maps, the shared gene filter, the encoder and the output sigmoid),
      times the input SD, times the out-of-fold recalibration slope -> calibrated log-HR per
      1 SD. Signed, on the hazard scale, and comparable across independently trained models.
  B3  The same AME computed in each of the 100 fc_1 refits -> Nadeau-Bengio CI (conditional on
      the frozen representations, as in A).
  B4  End-to-end stability: SHERLOC retrained from scratch on the 100 benchmark splits; AME per
      split -> median, 2.5-97.5 % range and sign agreement (unconditional on any one model).
  B5  Arm-level decomposition of the pre-sigmoid score: share of between-patient variance per
      arm (genes / panel / clinical / encoder) and mean signed contribution by risk tertile.

Usage:  python interpretation.py pada1|impower150_4tp [--n_boot 100] [--n_e2e 100]
Output: results/interpretation/<dataset>/
"""
import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxPH
from scipy import stats

import train_sherloc_pada1 as T

torch.set_num_threads(1)

DATASETS = {
    "pada1": dict(cfg="config_SHERLOC_response_longitudinal.yaml", slope=2.06,
                  slope_src="results/posthoc_benchmark/pada1"),
    "impower150_4tp": dict(cfg="config_IMP150_4tp.yaml", slope=1.57,
                           slope_src="results/posthoc_benchmark/impower150_4tp"),
}
EXP = dict(seed=16, n_boot=100, frac=0.8, lr=0.005, bs=50, epochs=256, patience=20)

PANEL_NAMES = {
    "mean_VAF": "Mean VAF", "nb_driver_VAF_above_05": "No. drivers (VAF>0.5%)",
    "resp_logratio_mean_VAF": "log-ratio mean VAF", "resp_logratio_max_VAF": "log-ratio max VAF",
    "resp_detected": "ctDNA detected", "resp_clearance": "ctDNA clearance",
    "resp_frac_mut_cleared": "Fraction mutations cleared", "resp_nb_new_mut": "No. new mutations",
    "resp_nb_driver_lost": "No. drivers lost",
    "max_of_AF": "Max VAF", "mean_of_AF": "Mean VAF", "cfDNA_conc": "cfDNA concentration", "n_kl": "No. mutations",
}
CLN_NAMES = {
    "Age_At_Inclusion": "Age", "PR_Allred_Score": "PR Allred", "ER_Allred_Score": "ER Allred",
    "Tumor_Grade_Prim_T": "Tumour grade", "ps_bas": "ECOG PS", "met_visceral": "Visceral metastases",
    "nb_mets": "No. metastatic sites", "del_tp_meta": "Time to metastasis", "HER2_low": "HER2-low",
    "pdl1_high": "PD-L1 high", "baseline_ECOG": "ECOG PS", "age": "Age", "sex_female": "Female sex",
    "sld_baseline": "Baseline SLD", "Bev_CP": "Arm: Bev+CP", "Atezo_CP": "Arm: Atezo+CP",
}


def nb(df, n_tr, n_te):
    """Nadeau-Bengio corrected mean, 95 % CI and p (identical to scripts/utils.py in the repo)."""
    n = len(df)
    m, v = df.mean(), df.var(ddof=1) * (1 / n + n_te / n_tr)
    se = np.sqrt(v)
    tc = stats.t.ppf(0.975, n - 1)
    t = m / se.replace(0, np.nan)
    p = 2 * stats.t.sf(np.abs(t), n - 1)
    return pd.DataFrame({"mean": m, "lo": m - tc * se, "hi": m + tc * se, "p": p})


def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = ~np.isnan(p)
    pv = p[ok]; o = np.argsort(pv); r = np.empty_like(o); r[o] = np.arange(1, len(pv) + 1)
    qq = pv * len(pv) / r
    qs = np.minimum.accumulate(qq[o][::-1])[::-1]; out = np.empty_like(pv); out[o] = np.minimum(qs, 1)
    q[ok] = out
    return q


class Ctx:
    """Everything needed to rebuild differentiable inputs for one trained model."""
    def __init__(self, cfg, info, net_sb, enc_genes):
        mc = cfg["model"]
        self.tps = cfg["timepoints"]
        self.genes = info["genes"]
        self.panel = mc["panel_features"]
        self.cln = [f"cln_{c}" for c in mc["clinical"]] + list(mc.get("static_features") or [])
        self.net_sb, self.enc_genes = net_sb, enc_genes

    def raw(self, d):
        f = lambda cols: torch.tensor(d[cols].to_numpy(np.float32))
        xg = torch.stack([f([f"{g}_{t}" for t in self.tps]) for g in self.genes], 1)
        xp = torch.stack([f([f"{t}_{p}" for t in self.tps]) for p in self.panel], 1)
        xc = f(self.cln)[:, None, :]
        xe = torch.stack([f([f"ENC_{g}_{t}" for g in self.enc_genes]) for t in self.tps], -1)  # (n,E,T)
        return xg, xp, xc, xe

    def names(self):
        out = []
        for g in self.genes:
            out += [("Gene", g, t) for t in self.tps]
        for p in self.panel:
            out += [("Panel", PANEL_NAMES.get(p, p), t) for t in self.tps]
        out += [("Clinical", CLN_NAMES.get(c[4:], c), "BL") for c in self.cln]
        for g in self.enc_genes:
            out += [("Encoder gene", g, t) for t in self.tps]
        return pd.MultiIndex.from_tuples(out, names=["arm", "feature", "timepoint"])

    def fc1_names(self, T_):
        pan = [f"{PANEL_NAMES.get(p, p)} [{k}]" for p in self.panel for k in (1, 2)]
        cln = [CLN_NAMES.get(c[4:], c) for c in self.cln]
        enc = [f"{t} Representation {d}" for d in (1, 2) for t in self.tps]
        arms = ["Gene"] * len(self.genes) + ["Panel"] * len(pan) + ["Clinical"] * len(cln) + ["Encoder"] * len(enc)
        return list(self.genes) + pan + cln + enc, arms


def lat_from_raw(net_sb, xe):
    """Differentiable copy of MLP_SolidBiopsy.extract_encoder, applied per timepoint -> (n,2,T)."""
    net_sb.eval()
    outs = [torch.sigmoid(net_sb.fc1(net_sb.bn_input(xe[..., t]))) for t in range(xe.shape[-1])]
    return torch.stack(outs, -1)


def ame(net, ctx, d, slope):
    """Standardised average marginal effect of every raw input on the (calibrated) log-hazard."""
    net.eval()
    xs = [x.clone().requires_grad_(True) for x in ctx.raw(d)]
    lat = lat_from_raw(ctx.net_sb, xs[3])
    g = net(xs[0], xs[1], xs[2], lat).sum()
    grads = torch.autograd.grad(g, xs)
    vals = []
    for x, gr in zip(xs, grads):
        sd = x.detach().std(0, unbiased=True)
        vals.append((gr.mean(0) * sd).flatten())       # same layout as names()
    v = torch.cat(vals)
    return slope * v.numpy()


def fc1_inputs(net, X):
    """fc_1 input vectors (eval mode) via a forward hook."""
    store = {}
    h = net.fc_1.register_forward_hook(lambda m, i, o: store.__setitem__("u", i[0].detach()))
    net.eval()
    with torch.no_grad():
        z = net(*X)
    h.remove()
    return store["u"].numpy(), z.numpy().ravel()


def safe_bs(n, bs):
    while n % bs == 1:
        bs += 1
    return bs


def refit_fc1(net, X_sub, y_sub, X_val, y_val, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    nb_ = copy.deepcopy(net)
    torch.nn.init.xavier_uniform_(nb_.fc_1.weight); torch.nn.init.zeros_(nb_.fc_1.bias)
    for p in nb_.parameters():
        p.requires_grad = False
    for p in nb_.fc_1.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, nb_.parameters()), lr=EXP["lr"])
    m = CoxPH(nb_, opt, loss=T.SafeCoxPHLoss())
    m.fit(X_sub, y_sub, safe_bs(len(y_sub[0]), EXP["bs"]), EXP["epochs"],
          [tt.callbacks.EarlyStopping(patience=EXP["patience"])], verbose=False,
          val_data=tt.tuplefy(X_val, y_val).repeat(10).cat())
    nb_.eval()
    return nb_


def take(X, idx):
    return [x[idx] for x in X]


def main(a):
    spec = DATASETS[a.dataset]
    cfg = T.load_cfg(spec["cfg"])
    out = Path("results/interpretation") / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    slope = spec["slope"]

    # ── A1: reference model (repo seed 16) ────────────────────────────────────
    keep = []
    res, _, info = T.run_cv(cfg, [EXP["seed"]], verbose=True, keep=keep)
    _, model, d_test, X_te, X_tr, d_tr = keep[0]
    net = model.net
    net_sb, enc_genes, _ = list(T._ENCODER_CACHE.values())[0]
    ctx = Ctx(cfg, info, net_sb, enc_genes)
    y_tr = (torch.tensor(d_tr["OS_rebaseline"].to_numpy(np.float32)),
            torch.tensor(d_tr["OS_event"].to_numpy(np.int64)))
    fc_names, fc_arms = ctx.fc1_names(len(ctx.tps))
    assert len(fc_names) == net.fc_1.weight.shape[1], (len(fc_names), net.fc_1.weight.shape)
    n_tr = len(d_tr)
    print(f"[INFO] {a.dataset}: reference split train={n_tr}, test={len(d_test)}, "
          f"C={res.C_index_Uno.iloc[0]:.3f}", flush=True)

    # ── A2: fc_1 refits on 80 % subsamples ───────────────────────────────────
    ck = out / "refits_checkpoint.npz"
    W, AME_b = [], []
    if ck.exists():
        z = np.load(ck); W, AME_b = list(z["W"]), list(z["A"])
        print(f"[INFO] resuming from {len(W)} refits", flush=True)
    for b in range(len(W), a.n_boot):
        rng = np.random.RandomState(10 * b)
        idx = np.sort(rng.choice(n_tr, int(round(EXP["frac"] * n_tr)), replace=False))
        rest = np.setdiff1d(np.arange(n_tr), idx)
        nb_ = refit_fc1(net, take(X_tr, idx), (y_tr[0][idx], y_tr[1][idx]),
                        take(X_tr, rest), (y_tr[0][rest], y_tr[1][rest]), seed=b)
        W.append(nb_.fc_1.weight.detach().numpy().ravel())
        AME_b.append(ame(nb_, ctx, d_tr, slope))
        if (b + 1) % 5 == 0:
            np.savez(ck, W=np.array(W), A=np.array(AME_b))
            print(f"  refit {b + 1}/{a.n_boot}", flush=True)
    n_sub = int(round(EXP["frac"] * n_tr)); n_val = n_tr - n_sub

    # A3/B1: fc_1 coefficients, repo format + BH q-values
    fc = nb(pd.DataFrame(W, columns=fc_names), n_sub, n_val)
    fc.insert(0, "arm", fc_arms)
    fc["HR"], fc["HR_lo"], fc["HR_hi"] = np.exp(fc["mean"]), np.exp(fc["lo"]), np.exp(fc["hi"])
    fc["q_BH"] = bh(fc["p"])
    fc["beta_ref"] = net.fc_1.weight.detach().numpy().ravel()
    # A4: temporal weights (single trained model, as in the repo)
    for j, t in enumerate(ctx.tps):
        fc.loc[ctx.genes, f"w_{t}"] = net.filter1.weight.detach().numpy()[0, j]
        for i, p in enumerate(ctx.panel):
            w = net.mlps[i][0].weight.detach().numpy()[:, j]
            for k in (1, 2):
                fc.loc[f"{PANEL_NAMES.get(p, p)} [{k}]", f"w_{t}"] = w[k - 1]
    fc.to_csv(out / "fc1_coefficients.csv")

    # A6: encoder gene % contribution (BatchNorm-folded weights of the pre-trained encoder)
    bn = net_sb.bn_input
    w_eff = net_sb.fc1.weight.detach().numpy() * (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
    contrib = pd.DataFrame(100 * np.abs(w_eff) / np.abs(w_eff).sum(1, keepdims=True),
                           index=["Representation 1", "Representation 2"], columns=enc_genes).T
    contrib["sign_R1"], contrib["sign_R2"] = np.sign(w_eff[0]), np.sign(w_eff[1])
    contrib.to_csv(out / "encoder_gene_contribution.csv")

    # B2/B3: AME — reference model + fc_1 refits
    names = ctx.names()
    ame_ref = ame(net, ctx, d_tr, slope)
    A = nb(pd.DataFrame(AME_b, columns=range(len(names))), n_sub, n_val)
    A.index = names
    A["ame_ref"] = ame_ref
    A["q_BH"] = bh(A["p"])
    for c in ("mean", "lo", "hi"):
        A[f"HR_{c}"] = np.exp(A[c])

    # B4: end-to-end stability over the benchmark splits
    if a.n_e2e > 0:
        E = []
        for s0 in range(0, a.n_e2e, 10):
            f = out / f"e2e_{s0}.npy"
            if not f.exists():
                keep2 = []
                T.run_cv(cfg, list(range(s0, min(s0 + 10, a.n_e2e))), verbose=False, keep=keep2)
                np.save(f, np.array([ame(m.net, ctx, d, slope) for (_, m, _, _, _, d) in keep2]))
                print(f"  end-to-end splits {s0}-{s0 + 9} done", flush=True)
            E.append(np.load(f))
        E = np.concatenate(E)
        med = np.median(E, 0)
        A["e2e_median"] = med
        A["e2e_lo"], A["e2e_hi"] = np.percentile(E, 2.5, 0), np.percentile(E, 97.5, 0)
        A["e2e_sign_agree"] = (np.sign(E) == np.sign(med)).mean(0)
        pd.DataFrame(E, columns=["|".join(n) for n in names]).to_csv(out / "ame_end_to_end_splits.csv", index=False)
    A.to_csv(out / "ame_raw_inputs.csv")

    # B5: arm decomposition of the pre-sigmoid score on the reporting split's test patients
    u, _ = fc1_inputs(net, X_te)
    u_tr, o_tr = fc1_inputs(net, X_tr)
    th = net.fc_1.weight.detach().numpy().ravel()
    C = (u - u_tr.mean(0)) * th
    arms = np.array(fc_arms)
    arm_c = pd.DataFrame({k: C[:, arms == k].sum(1) for k in ["Gene", "Panel", "Clinical", "Encoder"]})
    zc = arm_c.sum(1)
    arm_c["tertile"] = pd.qcut(zc.rank(method="first"), 3, labels=["Low", "Intermediate", "High"]).values
    share = arm_c.drop(columns="tertile").apply(lambda s: np.cov(s, zc)[0, 1] / zc.var())  # additive share
    pd.concat([arm_c.groupby("tertile", observed=True).mean().T,
               share.rename("variance_share")], axis=1).to_csv(out / "arm_decomposition.csv")

    meta = dict(dataset=a.dataset, n_train=n_tr, n_test=len(d_test), C_ref=float(res.C_index_Uno.iloc[0]),
                recal_slope=slope, n_boot=a.n_boot, n_e2e=a.n_e2e,
                mean_sigmoid_slope=float(np.mean(o_tr * (1 - o_tr))))
    pd.Series(meta).to_csv(out / "meta.csv")
    print(f"[DONE] {out}", flush=True)




# ── B6: grouped permutation importance (added after the first run; separate entry point) ──────────
def perm_groups(ctx):
    tps = ctx.tps
    g = {"Gene arm": [f"{x}_{t}" for x in ctx.genes for t in tps],
         "Panel arm": [f"{t}_{p}" for p in ctx.panel for t in tps],
         "Clinical arm": list(ctx.cln),
         "Encoder (mutation profile)": [f"ENC_{x}_{t}" for x in ctx.enc_genes for t in tps]}
    ct = lambda ts: [c for t in ts for c in [f"{x}_{t}" for x in ctx.genes] + [f"{t}_{p}" for p in ctx.panel]
                     + [f"ENC_{x}_{t}" for x in ctx.enc_genes]]
    g["All ctDNA at baseline"] = ct(tps[:1])
    g["All ctDNA on treatment"] = ct(tps[1:])
    return g


def predict_raw(net, ctx, d):
    net.eval()
    with torch.no_grad():
        xg, xp, xc, xe = ctx.raw(d)
        return net(xg, xp, xc, lat_from_raw(ctx.net_sb, xe)).numpy().ravel()


def perm_importance(dataset, n_e2e, n_perm=5):
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv
    spec = DATASETS[dataset]
    cfg = T.load_cfg(spec["cfg"])
    out = Path("results/interpretation") / dataset
    rows = []
    for s0 in range(0, n_e2e, 10):
        f = out / f"perm_{s0}.csv"
        if not f.exists():
            keep2 = []
            _, _, info = T.run_cv(cfg, list(range(s0, min(s0 + 10, n_e2e))), verbose=False, keep=keep2)
            net_sb, enc_genes, _ = list(T._ENCODER_CACHE.values())[0]
            ctx = Ctx(cfg, info, net_sb, enc_genes)
            G = perm_groups(ctx)
            tau = cfg["evaluation"]["time_max"]
            res = []
            for seed, m, d_te, _, _, d_tr in keep2:
                ytr = Surv.from_arrays(d_tr["OS_event"].astype(bool), d_tr["OS_rebaseline"])
                yte = Surv.from_arrays(d_te["OS_event"].astype(bool), d_te["OS_rebaseline"])
                cidx = lambda r: concordance_index_ipcw(ytr, yte, r, tau=min(tau, d_tr.OS_rebaseline.max() - 1e-3))[0]
                d_te = d_te.copy()
                c0 = cidx(predict_raw(m.net, ctx, d_te))
                rng = np.random.RandomState(seed)
                for name, cols in G.items():
                    dc = []
                    for _ in range(n_perm):
                        dp = d_te.copy()
                        dp[cols] = d_te[cols].to_numpy()[rng.permutation(len(d_te))]
                        dc.append(c0 - cidx(predict_raw(m.net, ctx, dp)))
                    res.append({"seed": seed, "group": name, "C_full": c0, "delta_C": float(np.mean(dc))})
            pd.DataFrame(res).to_csv(f, index=False)
            print(f"  permutation importance splits {s0}-{s0 + 9} done", flush=True)
        rows.append(pd.read_csv(f))
    P = pd.concat(rows)
    s = P.groupby("group").delta_C.agg(median="median", lo=lambda x: x.quantile(0.025),
                                        hi=lambda x: x.quantile(0.975), frac_pos=lambda x: (x > 0).mean())
    s.to_csv(out / "permutation_importance.csv")
    print(s.round(4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(DATASETS))
    ap.add_argument("--n_boot", type=int, default=100)
    ap.add_argument("--n_e2e", type=int, default=100)
    ap.add_argument("--perm_only", action="store_true", help="only B6 (grouped permutation importance)")
    a = ap.parse_args()
    perm_importance(a.dataset, a.n_e2e) if a.perm_only else main(a)
