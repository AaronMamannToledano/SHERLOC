"""
pretraining_ablation.py — does pre-training the genomic-interaction encoder on MSK-CHORD help, in both trials?

Variants (same 100 splits, inputs, training settings and metric code as the benchmark; SHERLOC at scale 1):
  pretrained_frozen   reference: encoder pre-trained on MSK-CHORD (OS), frozen
  no_encoder          encoder arm removed
  random_frozen       same encoder architecture, random initialisation, frozen (no pre-training)
  scratch_trained     same encoder, random initialisation, trained end-to-end with SHERLOC on the trial data
  pretrained_unfrozen encoder pre-trained on MSK-CHORD, then fine-tuned end-to-end on the trial data

Usage: python pretraining_ablation.py pada1|impower150_4tp <variant> [first_seed last_seed]
Output: results/pretraining_ablation/<dataset>/<variant>_seeds<a>-<b>.csv  (+ summary via --summary)
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_sherloc_pada1 as T

torch.set_num_threads(1)
CFG = {"pada1": "config_SHERLOC_response_longitudinal.yaml", "impower150_4tp": "config_IMP150_4tp.yaml"}
VARIANTS = ["pretrained_frozen", "no_encoder", "random_frozen", "scratch_trained", "pretrained_unfrozen"]
OUT = Path("results/pretraining_ablation")


class WithEncoder(torch.nn.Module):
    """SHERLOC whose 4th input is the raw encoder-gene VAFs (n, E, T); the encoder runs inside the network."""
    def __init__(self, base, enc):
        super().__init__()
        self.base, self.bn_input, self.fc1 = base, enc.bn_input, enc.fc1
        self.fc_1, self.dropout = base.fc_1, base.dropout          # used by train_sherloc_pada1 / introspection

    def forward(self, x, x_agg, x_cln, x_enc):
        lat = torch.stack([torch.sigmoid(self.fc1(self.bn_input(x_enc[..., t]))) for t in range(x_enc.shape[-1])], -1)
        return self.base(x, x_agg, x_cln, lat)


def run(ds, variant, seeds):
    cfg = T.load_cfg(CFG[ds])
    mc = cfg["model"]
    if variant == "no_encoder":
        mc["use_encoder_arm"] = False
    orig_get, orig_encode, orig_methods = T.get_encoder, T.encode, dict(T.METHODS)

    if variant in ("random_frozen", "scratch_trained"):
        def get_encoder_random(cfg_, df, freq_rank):
            net, genes, c = orig_get(cfg_, df, freq_rank)          # same genes / architecture as the reference
            torch.manual_seed(2024)
            rnd = copy.deepcopy(net)
            for m in rnd.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
            for m in rnd.modules():                                   # BN running stats from the trial-independent
                if isinstance(m, torch.nn.BatchNorm1d):               # pre-training data are not used either
                    m.reset_running_stats()
            rnd.eval()
            return rnd, genes, c
        T.get_encoder = get_encoder_random

    if variant in ("scratch_trained", "pretrained_unfrozen"):
        def encode_raw(df, net_sb, enc_genes, tps, binary=False, clonal=False):
            return torch.stack([torch.tensor(df[[f"ENC_{g}_{tp}" for g in enc_genes]].to_numpy(np.float32))
                                for tp in tps], -1)                    # (n, E, T) raw VAFs
        T.encode = encode_raw
        name = mc.get("method", "SHERLOC_v2")
        base_cls = orig_methods[name]

        def build(**kw):
            base = base_cls(**kw)
            net_sb, _, _ = build.enc
            return WithEncoder(base, copy.deepcopy(net_sb))
        # get_encoder is called inside run_cv before the networks are built: capture its output
        inner_get = T.get_encoder

        def get_and_store(cfg_, df, freq_rank):
            r = inner_get(cfg_, df, freq_rank)
            build.enc = r
            return r
        T.get_encoder = get_and_store
        T.METHODS[name] = build
    try:
        res, _, _ = T.run_cv(cfg, seeds, verbose=False)
    finally:
        T.get_encoder, T.encode = orig_get, orig_encode
        T.METHODS.clear(); T.METHODS.update(orig_methods)
    return res.assign(variant=variant, dataset=ds)


def summary():
    from scipy import stats
    rows = []
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        R = pd.concat([pd.read_csv(f) for f in d.glob("*_seeds*.csv")]).drop_duplicates(["variant", "seed"])
        ref = R[R.variant == "pretrained_frozen"].set_index("seed")
        for v, g in R.groupby("variant"):
            g = g.set_index("seed"); common = ref.index.intersection(g.index)
            r = {"dataset": d.name, "variant": v, "n_splits": len(common)}
            for k in ("C_index_Uno", "iAUC", "IBS"):
                r[k] = g.loc[common, k].mean()
                if v != "pretrained_frozen":
                    x = (ref.loc[common, k] - g.loc[common, k]).to_numpy(); n = len(x)
                    se = np.sqrt(x.var(ddof=1) * (1 / n + 0.3 / 0.7))
                    r[f"delta_ref_minus_variant_{k}"] = x.mean()
                    r[f"p_{k}"] = 2 * stats.t.sf(abs(x.mean() / se), n - 1)
                    r[f"ref_better_{k}"] = float((x < 0).mean() if k == "IBS" else (x > 0).mean())
            rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(); sys.exit()
    ds, variant = sys.argv[1], sys.argv[2]
    a, b = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 99)
    (OUT / ds).mkdir(parents=True, exist_ok=True)
    for s0 in range(a, b + 1, 10):
        f = OUT / ds / f"{variant}_seeds{s0}-{min(s0 + 9, b)}.csv"
        if f.exists():
            continue
        run(ds, variant, list(range(s0, min(s0 + 9, b) + 1))).to_csv(f, index=False)
        print(f"{ds} {variant} seeds {s0}-{min(s0 + 9, b)} done", flush=True)
