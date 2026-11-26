"""
posthoc_holdout.py — post-hoc recalibration of SHERLOC (scale 1) with one hold-out, on a subset of splits.

Per reporting split (same splits as every analysis):
  1. full model: SHERLOC scale 1 trained on the whole training split (identical to the reported SHERLOC run)
     -> test risk eta_te: C-index and iAUC unchanged by construction
  2. hold-out model: same network trained on 80 % of the training split (stratified on the event);
     its scores on the remaining 20 % give the calibration slope beta (1-covariate Cox model, unpenalised)
  3. recalibrated survival for test patients: S(t | x) = exp(-H0(t) * exp(beta * eta_te)),
     H0 = Breslow estimator on the full training split with linear predictor beta * eta_tr
  3b. same with the pre-sigmoid score z = logit(log-risk) as covariate (undoes the sigmoid saturation; same ranking)
Reference variants on the same splits: no recalibration (pycox Breslow, = SHERLOC), beta = 1 refit (Breslow only),
beta fitted in-sample on the full model's training scores, and the existing SHERLOC calibrated (scale 2) run.
Usage: python posthoc_holdout.py --config config_SHERLOC_response_longitudinal.yaml --seeds 0-19 --tag pada1
"""
import argparse, copy, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torchtuples as tt
from pycox.models import CoxPH
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
import train_sherloc_pada1 as T
from mtl_benchmark import metrics


def fit_net(cfg, X, y, n_genes, n_panel, n_cln, seed):
    tr, mc, tps = cfg["training"], cfg["model"], cfg["timepoints"]
    torch.manual_seed(seed); np.random.seed(seed)
    kw = {k: mc[k] for k in ("output_sigmoid", "panel_mode", "panel_delta", "gene_delta", "use_gene_arm",
                             "output_scale", "gene_mode", "gene_hidden", "gene_input", "use_panel_arm",
                             "use_encoder_arm", "event_mode") if k in mc}
    kw.update(mc.get("arch_kwargs") or {})
    net = T.METHODS[mc["method"]](d_input=n_genes, d_agg_input=n_panel, d_channel=len(tps), d_output=1,
                                  d_other_input=n_cln, device=T.DEVICE, **kw)
    model = CoxPH(net, tt.optim.Adam, loss=T.SafeCoxPHLoss())
    model.optimizer.set_lr(tr["lr"])
    bs = len(y[0]) if tr["batch_size"] == "full" else tr["batch_size"]
    model.fit(X, y, bs, tr["epochs"], verbose=False)
    return model


def breslow_surv(lp_tr, t_tr, e_tr, lp_te):
    """Breslow cumulative baseline hazard from training linear predictors -> test survival DataFrame."""
    order = np.argsort(t_tr)
    c = np.max(lp_tr)                                       # shift for numerical stability (H0 rescales)
    lp_tr, lp_te = np.clip(lp_tr - c, -700, 50), np.clip(lp_te - c, -700, 50)
    t, e, r = t_tr[order], e_tr[order], np.exp(lp_tr[order])
    risk_set = np.cumsum(r[::-1])[::-1]                    # sum of exp(lp) over patients with time >= t_i
    ut = np.unique(t[e == 1])
    h = np.array([e[t == u].sum() / risk_set[np.searchsorted(t, u)] for u in ut])
    H = np.cumsum(h)
    grid = np.unique(np.concatenate([[0.0], ut]))
    H_grid = np.concatenate([[0.0], H])
    S = np.exp(-np.outer(H_grid, np.exp(lp_te)))
    return pd.DataFrame(np.nan_to_num(S, nan=1.0), index=grid)


def beta_fit(lp, t, e):
    m = CoxPHSurvivalAnalysis(alpha=0.0).fit(lp.reshape(-1, 1), Surv.from_arrays(event=e.astype(bool), time=t))
    return float(m.coef_[0])


def main(a):
    cfg = T.load_cfg(a.config)
    lo, hi = map(int, a.seeds.split("-"))
    keep = []
    t0 = time.time()
    res_raw, _, info = T.run_cv(cfg, range(lo, hi + 1), verbose=False, keep=keep)
    ev, mc = cfg["evaluation"], cfg["model"]
    win = (ev["time_min"], ev["time_max"], ev.get("calib_time", 24.0))
    n_genes, n_panel = len(info["genes"]), len(mc["panel_features"])
    n_cln = len(mc["clinical"]) + len(mc.get("static_features") or [])
    rows = []
    for (seed, model, d_test, X_te, X_tr, d_tr) in keep:
        t_tr, e_tr = d_tr["OS_rebaseline"].to_numpy(float), d_tr["OS_event"].to_numpy(int)
        P = lambda m, X: np.asarray(m.predict(X), dtype=float).flatten()
        lp_tr, lp_te = P(model, X_tr), P(model, X_te)
        # hold-out: 80 / 20 of the training split, stratified on the event
        rng = np.random.RandomState(1000 + seed)
        ho = np.zeros(len(d_tr), bool)
        for ev_val in (0, 1):
            idx = np.where(e_tr == ev_val)[0]
            ho[rng.choice(idx, int(round(0.2 * len(idx))), replace=False)] = True
        sub = lambda X, m: [x[torch.tensor(m)] for x in X]
        y_fit = (torch.tensor(t_tr[~ho], dtype=torch.float32), torch.tensor(e_tr[~ho]))
        m_ho = fit_net(cfg, sub(X_tr, ~ho), y_fit, n_genes, n_panel, n_cln, seed)
        lp_ho = P(m_ho, sub(X_tr, ho))
        b_ho = beta_fit(lp_ho, t_tr[ho], e_tr[ho])
        b_in = beta_fit(lp_tr, t_tr, e_tr)
        # pre-sigmoid score z = logit(log-risk / scale): same ranking, undoes the sigmoid saturation
        sc = float(mc.get("output_scale", 1.0))
        lg = lambda v: np.log(np.clip(v / sc, 1e-6, 1 - 1e-6) / (1 - np.clip(v / sc, 1e-6, 1 - 1e-6)))
        z_tr, z_te, z_ho = lg(lp_tr), lg(lp_te), lg(lp_ho)
        g_ho = beta_fit(z_ho, t_tr[ho], e_tr[ho])
        dd_tr = d_tr.assign(OS_t=d_tr.OS_rebaseline, OS_e=d_tr.OS_event)
        dd_te = d_test.assign(OS_t=d_test.OS_rebaseline, OS_e=d_test.OS_event)
        variants = {"no_recalibration": model.predict_surv_df(X_te),
                    "breslow_beta1": breslow_surv(lp_tr, t_tr, e_tr, lp_te),
                    "beta_in_sample": breslow_surv(b_in * lp_tr, t_tr, e_tr, b_in * lp_te),
                    "beta_holdout": breslow_surv(b_ho * lp_tr, t_tr, e_tr, b_ho * lp_te),
                    "beta_holdout_logit": breslow_surv(g_ho * z_tr, t_tr, e_tr, g_ho * z_te)}
        for v, s in variants.items():
            r = metrics(dd_tr, dd_te, lp_te, s, "OS", win)
            rows.append({"seed": seed, "variant": v, "beta_holdout": b_ho, "beta_in_sample": b_in, "beta_holdout_logit": g_ho,
                         "n_holdout": int(ho.sum()), "events_holdout": int(e_tr[ho].sum()), **r})
        print(f"[{seed}] beta hold-out {b_ho:.2f} (n={ho.sum()}, {e_tr[ho].sum()} deaths), in-sample {b_in:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    out = Path("results/posthoc") / a.tag; out.mkdir(parents=True, exist_ok=True)
    r = pd.DataFrame(rows); r.to_csv(out / f"holdout_seeds{lo}-{hi}.csv", index=False)
    print(r.groupby("variant")[["C_index_Uno", "iAUC", "IBS", "calib_err", "pred_S_range"]].mean().round(4))
    print(r.drop_duplicates("seed")[["beta_holdout", "beta_in_sample", "beta_holdout_logit"]].describe().round(3))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_SHERLOC_response_longitudinal.yaml")
    p.add_argument("--seeds", default="0-19"); p.add_argument("--tag", default="pada1")
    main(p.parse_args())
