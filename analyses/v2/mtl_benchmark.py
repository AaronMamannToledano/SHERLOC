"""
mtl_benchmark.py — joint OS + PFS training of SHERLOC and genomic-interaction encoder pretraining variants.

Model: calibrated SHERLOC (SHERLOC_v2, output scale 2) with one shared body and two Cox heads
(fc_1 -> 2 outputs: log-risk OS, log-risk PFS). Loss = w_OS * Cox(OS) + w_PFS * Cox(PFS), NaN-safe, the PFS term
restricted to patients with a landmark PFS (progression-free at the landmark).
  OS_only  (1, 0) | PFS_only (0, 1) | joint (1, 1)
The three tasks use the same training loop, splits, inputs and network (d_output = 2), so the only difference
is the training signal. Each head's baseline hazard (Breslow) is estimated on its training patients.

Encoder (MLP_SolidBiopsy, frozen) pretrained on MSK-CHORD with
  OS / PFS (first NLP-radiology progression >= 30 days after first sequencing, or death; train_sherloc_pada1.msk_pfs)
  pan-cancer (5 cancer types, the original setting) / matched subtype (PADA-1: HR+/HER2- breast;
  IMpower150: non-squamous NSCLC).

Datasets, inputs, splits, training settings: as in the OS / PFS analyses (PADA-1: SHERLOC_response inputs,
fixed 100 epochs; IMpower150: paper features, early stopping on the summed validation loss, val 20 %).
Metrics: Uno C, iAUC, IBS, calibration error / predicted-survival spread at calib_time, per endpoint.

Usage: python mtl_benchmark.py --tag pada1|impower150_4tp|impower150_3tp
Output: results/mtl/<tag>/
"""
import argparse, copy, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxPH
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv
import train_sherloc_pada1 as T
from benchmark import nb_paired

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
DATASETS = {
    "pada1": ("config_SHERLOC_calibrated.yaml", "breast_hrpos_her2neg",
              {"OS": (3.0, 36.0, 24.0), "PFS": (3.0, 36.0, 24.0)}),
    "impower150_4tp": ("config_IMP150_4tp.yaml", "nsclc_nonsquamous",
                       {"OS": (3.0, 30.0, 12.0), "PFS": (2.0, 24.0, 12.0)}),
    "impower150_3tp": ("config_IMP150_3tp.yaml", "nsclc_nonsquamous",
                       {"OS": (3.0, 30.0, 12.0), "PFS": (2.0, 24.0, 12.0)}),
}
TASKS = {"OS_only": (1.0, 0.0), "PFS_only": (0.0, 1.0), "joint": (1.0, 1.0)}
REPORT = range(100)
LOSS = T.SafeCoxPHLoss()


class Head(torch.nn.Module):
    def __init__(self, net, k):
        super().__init__(); self.net, self.k = net, k

    def forward(self, *x):
        return self.net(*x)[:, [self.k]]


def sub(X, idx):
    return [x[idx] for x in X]


def mt_loss(net, X, y, w):
    lh = net(*X)
    t_os, e_os, t_pf, e_pf, m_pf = y
    loss = 0.0
    if w[0] > 0:
        loss = loss + w[0] * LOSS(lh[:, 0], t_os, e_os)
    if w[1] > 0 and m_pf.any():
        loss = loss + w[1] * LOSS(lh[m_pf, 1], t_pf[m_pf], e_pf[m_pf])
    return loss


def fit(net, X_tr, y_tr, X_va, y_va, w, tr):
    opt = torch.optim.Adam(net.parameters(), lr=tr["lr"])
    n, bs = len(y_tr[0]), tr["batch_size"]
    best, best_state, wait, ep_done = np.inf, None, 0, 0
    for ep in range(tr["epochs"]):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = mt_loss(net, sub(X_tr, idx), [y[idx] for y in y_tr], w)
            if torch.is_tensor(loss) and loss.requires_grad:
                opt.zero_grad(); loss.backward(); opt.step()
        ep_done = ep + 1
        if X_va is not None:
            net.eval()
            with torch.no_grad():
                v = float(mt_loss(net, X_va, y_va, w))
            if v < best - 1e-8:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait > tr["patience"]:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return ep_done


def metrics(d_trv, d_test, risk, surv, ep_name, win):
    t_min, t_max, ct = win
    tcol, ecol = f"{ep_name}_t", f"{ep_name}_e"
    y_train = Surv.from_arrays(event=d_trv[ecol].astype(bool), time=d_trv[tcol])
    y_test = Surv.from_arrays(event=d_test[ecol].astype(bool), time=d_test[tcol])
    t_hi = min(t_max, d_test[tcol].max() - 1e-3, d_trv[tcol].max() - 1e-3)
    c = concordance_index_ipcw(y_train, y_test, risk, tau=t_max)[0]
    first = d_test.loc[d_test[ecol] == 1, tcol].min()
    _, iauc = cumulative_dynamic_auc(y_train, y_test, risk, np.arange(max(t_min, np.ceil(first)), t_hi, 1.0))
    t_lo = max(t_min, d_test[tcol].min())
    s = surv[(surv.index >= t_lo) & (surv.index < t_hi)]
    ibs = integrated_brier_score(y_train, y_test, s.to_numpy().T, s.index.to_numpy())
    dd = pd.DataFrame({"OS_rebaseline": d_test[tcol].to_numpy(), "OS_event": d_test[ecol].to_numpy()})
    cal = T.calibration_error(s, dd, risk, ct)
    rng = float(np.ptp(s.iloc[(np.abs(s.index - ct)).argmin()]))
    return {"C_index_Uno": c, "iAUC": iauc, "IBS": ibs, "calib_err": cal, "pred_S_range": rng}


def run(cfg, tag, enc_endpoint, enc_cohort, task, seeds):
    P, tps, tr, mc = cfg["_P"], cfg["timepoints"], cfg["training"], cfg["model"]
    wins = DATASETS[tag][2]
    data = Path(P["data_dir"])
    df = pd.read_csv(data / P.get("dataset_file", "PADA1_SHERLOC_dataset.csv"), index_col=0)
    df["OS_t"], df["OS_e"] = df["OS_rebaseline"], df["OS_event"]
    df["PFS_m"] = df["PFS_rebaseline"].notna() & (df["PFS_rebaseline"] > 0)
    df["PFS_t"] = df["PFS_rebaseline"].where(df["PFS_m"], 1.0)
    df["PFS_e"] = df["PFS_event"].where(df["PFS_m"], 0).astype(int)
    freq_rank = pd.read_csv(data / P.get("gene_frequency_file", "gene_frequency.csv"), index_col=0).index.tolist()
    if mc.get("genes_file"):
        genes = pd.read_csv(data / mc["genes_file"], index_col=0).iloc[:, 0].tolist()
    else:
        genes = mc.get("genes") or [g for g in freq_rank if f"{g}_BL" in df.columns][: mc["n_top_genes"]]
    panel = mc["panel_features"]
    cln = [f"cln_{c}" for c in mc["clinical"]] + list(mc.get("static_features") or [])
    c2 = copy.deepcopy(cfg)
    c2["model"]["encoder_endpoint"], c2["model"]["encoder_cohort"] = enc_endpoint, enc_cohort
    net_sb, enc_genes, c_msk = T.get_encoder(c2, df, freq_rank)
    w = TASKS[task]
    eps = [e for e, wt in zip(["OS", "PFS"], w) if wt > 0]
    rows, preds = [], []
    for seed in seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        rs = cfg["evaluation"]["seed_multiplier"] * seed
        d_test = df.sample(frac=tr["test_frac"], random_state=rs)
        d_trv = df.drop(d_test.index)
        if tr["val_frac"] > 0:
            d_val = d_trv.sample(frac=tr["val_frac"], random_state=rs); d_tr = d_trv.drop(d_val.index)
        else:
            d_val, d_tr = None, d_trv
        med = d_trv[cln].median()
        imp = lambda d: None if d is None else d.assign(**{c: d[c].fillna(med[c]) for c in cln})
        d_tr, d_val, d_test, d_trv = imp(d_tr), imp(d_val), imp(d_test), imp(d_trv)
        mk = lambda d: T.to_tensors(d, genes, panel, cln, tps, T.encode(d, net_sb, enc_genes, tps),
                                    mc.get("log1p", False), mc.get("log_eps"))[0]
        ys = lambda d: [torch.tensor(d["OS_t"].to_numpy(np.float32)), torch.tensor(d["OS_e"].to_numpy(np.int64)),
                        torch.tensor(d["PFS_t"].to_numpy(np.float32)), torch.tensor(d["PFS_e"].to_numpy(np.int64)),
                        torch.tensor(d["PFS_m"].to_numpy(bool))]
        X_tr, X_te = mk(d_tr), mk(d_test)
        X_va = mk(d_val) if d_val is not None else None
        y_va = ys(d_val) if d_val is not None else None
        net = T.METHODS["SHERLOC_v2"](d_input=len(genes), d_agg_input=len(panel), d_channel=len(tps), d_output=2,
                                      d_other_input=len(cln), device=T.DEVICE, output_sigmoid=True,
                                      output_scale=float(mc.get("output_scale", 2.0)))
        n_ep = fit(net, X_tr, ys(d_tr), X_va, y_va, w, tr)
        row = {"seed": seed, "random_state": rs, "epochs": n_ep, "encoder_endpoint": enc_endpoint,
               "encoder_cohort": enc_cohort, "task": task, "msk_val_C": c_msk}
        for e in eps:
            k = 0 if e == "OS" else 1
            mtr = d_tr["PFS_m"].to_numpy() if e == "PFS" else np.ones(len(d_tr), bool)
            mte = d_test["PFS_m"].to_numpy() if e == "PFS" else np.ones(len(d_test), bool)
            mtv = d_trv["PFS_m"].to_numpy() if e == "PFS" else np.ones(len(d_trv), bool)
            model = CoxPH(Head(net, k))
            ytr = (d_tr[f"{e}_t"].to_numpy(np.float32)[mtr], d_tr[f"{e}_e"].to_numpy(np.float32)[mtr])
            model.compute_baseline_hazards(input=tt.tuplefy(sub(X_tr, torch.tensor(mtr))), target=ytr)
            Xs = tt.tuplefy(sub(X_te, torch.tensor(mte)))
            risk = model.predict(Xs).flatten()
            surv = model.predict_surv_df(Xs)
            m = metrics(d_trv[mtv], d_test[mte], risk, surv, e, wins[e])
            row.update({f"{e}_{k2}": v for k2, v in m.items()})
            preds.append(pd.DataFrame({"seed": seed, "endpoint": e, "Subject_Id": d_test.index[mte], "risk": risk,
                                       "time": d_test[f"{e}_t"].values[mte], "event": d_test[f"{e}_e"].values[mte]}))
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(preds)


def main(tag, seeds, encoders):
    f, matched, _ = DATASETS[tag]
    cfg = T.load_cfg(f)
    cfg["model"]["output_scale"] = 2.0
    out = Path("results/mtl") / tag; out.mkdir(parents=True, exist_ok=True)
    tf = cfg["training"]["test_frac"]
    encs = [("OS", "pan"), ("PFS", "pan"), ("OS", matched), ("PFS", matched)]
    if encoders:
        encs = [e for e in encs if f"{e[0]}_{'matched' if e[1] == matched else 'pan'}" in encoders]
    per = {}
    for ee, ec in encs:
        el = f"enc{ee}_{'matched' if ec == matched else 'pan'}"
        for task in TASKS:
            r, p = run(cfg, tag, ee, ec, task, seeds)
            name = f"{el}__{task}"; per[name] = r
            r.to_csv(out / f"{name}_metrics_per_split.csv", index=False)
            p.to_csv(out / f"{name}_test_risk_scores.csv", index=False)
            print("[report]", tag, name, r.filter(regex="^(OS|PFS)_").mean().round(4).to_dict(), flush=True)
    # summary
    rows = []
    for name, r in per.items():
        el, task = name.split("__")
        for e in ["OS", "PFS"]:
            if f"{e}_C_index_Uno" in r:
                rows.append({"encoder": el, "task": task, "endpoint": e, "msk_val_C": r.msk_val_C.iloc[0],
                             **{k: r[f"{e}_{k}"].mean() for k in ["C_index_Uno", "iAUC", "IBS", "calib_err", "pred_S_range"]},
                             "C_sd": r[f"{e}_C_index_Uno"].std()})
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    # paired comparisons
    comp = []
    def cmp(a, b, e, label):
        if a in per and b in per:
            for k in ["C_index_Uno", "iAUC", "IBS", "calib_err"]:
                x, y = per[a][f"{e}_{k}"], per[b][f"{e}_{k}"]
                d, p = nb_paired(x, y, tf)
                better = (x > y).mean() if k in ("C_index_Uno", "iAUC") else (x < y).mean()
                comp.append({"comparison": label, "a": a, "b": b, "endpoint": e, "metric": k,
                             "delta_a_minus_b": d, "p_NB": p, "frac_splits_a_better": better})
    for ee, ec in encs:
        el = f"enc{ee}_{'matched' if ec == matched else 'pan'}"
        cmp(f"{el}__joint", f"{el}__OS_only", "OS", "joint vs single-task")
        cmp(f"{el}__joint", f"{el}__PFS_only", "PFS", "joint vs single-task")
    ref = "encOS_pan"
    for el in ["encPFS_pan", "encOS_matched", "encPFS_matched"]:
        for task, e in [("OS_only", "OS"), ("PFS_only", "PFS"), ("joint", "OS"), ("joint", "PFS")]:
            cmp(f"{el}__{task}", f"{ref}__{task}", e, "encoder vs pan-cancer OS encoder")
    pd.DataFrame(comp).to_csv(out / "paired_comparisons.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(DATASETS))
    ap.add_argument("--n_seeds", type=int, default=100)
    ap.add_argument("--encoders", nargs="*", default=None, help="subset, e.g. OS_pan PFS_matched")
    a = ap.parse_args()
    main(a.tag, range(a.n_seeds), a.encoders)
