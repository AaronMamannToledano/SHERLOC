"""
posthoc_oof.py — post-hoc recalibration of SHERLOC (scale 1) with K-fold out-of-fold (OOF) scores.

Per reporting split:
  1. full model: SHERLOC scale 1 as reported (run_cv: same split, same training, same early stopping if any)
     -> test risk eta_te (C-index / iAUC unchanged by construction)
  2. K models (default 5) trained on (K-1)/K of the full model's training patients (stratified on the event; same
     training settings, incl. an inner early-stopping split when the config uses one); each scores its held-out fold
     -> OOF scores for every training patient -> calibration slope beta (unpenalised 1-covariate Cox model)
  3. recalibrated test survival S(t | x) = exp(-H0(t) exp(beta * eta_te)); H0 = Breslow on the full model's training
     patients with linear predictor beta * eta_tr
Optional (--scale2): SHERLOC with output scale 2 on the same splits through run_cv, for a same-code comparison.
Usage: python posthoc_oof.py --config config_IMP150_4tp.yaml --seeds 0-24 --tag impower150_4tp --scale2
"""
import argparse, copy, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torchtuples as tt
from pycox.models import CoxPH
from sklearn.model_selection import StratifiedKFold
import train_sherloc_pada1 as T
from mtl_benchmark import metrics
from posthoc_holdout import breslow_surv, beta_fit


def fit_net(cfg, X, y, dims, seed):
    tr, mc, tps = cfg["training"], cfg["model"], cfg["timepoints"]
    torch.manual_seed(seed); np.random.seed(seed)
    kw = {k: mc[k] for k in ("output_sigmoid", "panel_mode", "panel_delta", "gene_delta", "use_gene_arm",
                             "output_scale", "gene_mode", "gene_hidden", "gene_input", "use_panel_arm",
                             "use_encoder_arm", "event_mode") if k in mc}
    kw.update(mc.get("arch_kwargs") or {})
    net = T.METHODS[mc["method"]](d_input=dims[0], d_agg_input=dims[1], d_channel=len(tps), d_output=1,
                                  d_other_input=dims[2], device=T.DEVICE, **kw)
    model = CoxPH(net, tt.optim.Adam, loss=T.SafeCoxPHLoss())
    model.optimizer.set_lr(tr["lr"])
    n = len(y[0])
    if tr["val_frac"] > 0:            # same inner early stopping as run_cv
        rng = np.random.RandomState(seed)
        va = np.zeros(n, bool); va[rng.choice(n, int(round(tr["val_frac"] * n)), replace=False)] = True
        sub = lambda X, m: [x[torch.tensor(m)] for x in X]
        ysub = lambda m: (y[0][torch.tensor(m)], y[1][torch.tensor(m)])
        bs = int(va.size - va.sum()) if tr["batch_size"] == "full" else tr["batch_size"]
        model.fit(sub(X, ~va), ysub(~va), bs, tr["epochs"], [tt.callbacks.EarlyStopping(patience=tr["patience"])],
                  verbose=False, val_data=tt.tuplefy(sub(X, va), ysub(va)).repeat(10).cat())
    else:
        bs = n if tr["batch_size"] == "full" else tr["batch_size"]
        model.fit(X, y, bs, tr["epochs"], verbose=False)
    return model


def main(a):
    cfg = T.load_cfg(a.config)
    cfg["model"]["output_scale"] = 1.0
    lo, hi = map(int, a.seeds.split("-")); seeds = range(lo, hi + 1)
    t0 = time.time()
    keep = []
    _, _, info = T.run_cv(cfg, seeds, verbose=False, keep=keep)
    ev, mc = cfg["evaluation"], cfg["model"]
    win = (ev["time_min"], ev["time_max"], ev.get("calib_time", 24.0))
    dims = (len(info["genes"]), len(mc["panel_features"]), len(mc["clinical"]) + len(mc.get("static_features") or []))
    P = lambda m, X: np.asarray(m.predict(X), dtype=float).flatten()
    sub = lambda X, m: [x[torch.tensor(m)] for x in X]
    rows = []
    for (seed, model, d_test, X_te, X_tr, d_tr) in keep:
        t_tr, e_tr = d_tr["OS_rebaseline"].to_numpy(float), d_tr["OS_event"].to_numpy(int)
        lp_tr, lp_te = P(model, X_tr), P(model, X_te)
        oof = np.zeros(len(d_tr))
        for k, (i_fit, i_out) in enumerate(StratifiedKFold(a.k, shuffle=True, random_state=1000 + seed).split(oof, e_tr)):
            m_fit = np.zeros(len(oof), bool); m_fit[i_fit] = True
            y_fit = (torch.tensor(t_tr[m_fit], dtype=torch.float32), torch.tensor(e_tr[m_fit]))
            mk = fit_net(cfg, sub(X_tr, m_fit), y_fit, dims, seed * 100 + k)
            oof[~m_fit] = P(mk, sub(X_tr, ~m_fit))
        b = beta_fit(oof, t_tr, e_tr)
        dd_tr = d_tr.assign(OS_t=d_tr.OS_rebaseline, OS_e=d_tr.OS_event)
        dd_te = d_test.assign(OS_t=d_test.OS_rebaseline, OS_e=d_test.OS_event)
        for v, s in {"no_recalibration": model.predict_surv_df(X_te),
                     f"beta_oof{a.k}": breslow_surv(b * lp_tr, t_tr, e_tr, b * lp_te)}.items():
            rows.append({"seed": seed, "variant": v, "beta": b, **metrics(dd_tr, dd_te, lp_te, s, "OS", win)})
        print(f"[{seed}] beta OOF{a.k} = {b:.2f} ({time.time() - t0:.0f}s)", flush=True)
    if a.scale2:
        c2 = copy.deepcopy(cfg); c2["model"]["output_scale"] = 2.0
        keep2 = []
        T.run_cv(c2, seeds, verbose=False, keep=keep2)
        for (seed, model, d_test, X_te, X_tr, d_tr) in keep2:
            dd_tr = d_tr.assign(OS_t=d_tr.OS_rebaseline, OS_e=d_tr.OS_event)
            dd_te = d_test.assign(OS_t=d_test.OS_rebaseline, OS_e=d_test.OS_event)
            rows.append({"seed": seed, "variant": "scale2_trained", "beta": np.nan,
                         **metrics(dd_tr, dd_te, P(model, X_te), model.predict_surv_df(X_te), "OS", win)})
    out = Path("results/posthoc") / a.tag; out.mkdir(parents=True, exist_ok=True)
    r = pd.DataFrame(rows); r.to_csv(out / f"oof{a.k}_seeds{lo}-{hi}.csv", index=False)
    print(r.groupby("variant")[["C_index_Uno", "iAUC", "IBS", "calib_err", "pred_S_range"]].mean().round(4))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_SHERLOC_response_longitudinal.yaml")
    p.add_argument("--seeds", default="0-19"); p.add_argument("--tag", default="pada1")
    p.add_argument("--k", type=int, default=5); p.add_argument("--scale2", action="store_true")
    main(p.parse_args())
