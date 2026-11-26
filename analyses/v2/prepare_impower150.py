"""
prepare_impower150.py — Python port of SHERLOC/scripts/data/analysis_main.R (IMpower150, EGAS00001006703).

For each landmark (n_cycles = 2, 3 or 4; OS from the last on-treatment ctDNA sample):
  - mutations: drop synonymous / missing driver status, max VAF per patient x timepoint x gene (C8D1 dropped),
    VAF x 100 (%);
  - genes_kept (>= 5 patients mutated, used for the genomic-interaction encoder) and
    genes_restricted (>= 20 patients, ~5 % of the cohort, used for the gene-level ctDNA-dynamics arm);
  - clinical: pdl1_high, nb_mets (>= 3 sites), baseline_ECOG, age, sex_female, sld_baseline, Bev_CP, Atezo_CP;
  - panel-level features from the ctDNAfeatures table (paper: mean VAF, max VAF, cfDNA conc., n drivers);
  - missing values -> 0 (as in the R script).
Output (one folder per landmark): IMP150_SHERLOC_dataset.csv with OS_rebaseline, OS_event, cln_*, <gene>_<tp>,
<tp>_<panel>, ENC_<gene>_<tp> (encoder input, same % scale as the original pipeline), plus gene lists.

Usage:  python prepare_impower150.py --input_dir <Impower150_data_copy> --out_dir data_impower150
"""
import argparse
from pathlib import Path

import pandas as pd

PANEL = ["mean_of_AF", "max_of_AF", "cfDNA_conc", "n_kl"]
TPS_ALL = ["BL", "C2D1", "C3D1", "C4D1"]


def find(input_dir, pattern):
    hits = sorted(Path(input_dir).rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {input_dir}")
    return hits[0]


def main(input_dir, out_dir, n_cycles, freq_restricted=20, freq_minimal=5):
    mut = pd.read_csv(find(input_dir, "*ctDNA_calls_toShare*.csv"))
    cln_raw = pd.read_csv(find(input_dir, "*patient_clinical_toShare*.csv"))
    agg = pd.read_csv(find(input_dir, "*ctDNAfeatures_toShare*.csv"))

    mut = mut[~mut["driver_mutation_status"].isin(["synonymous"]) & mut["driver_mutation_status"].notna()]
    mut_s = (mut.groupby(["full_anonymized_patient_id", "TIMEPOINT", "gene"])["variant_allele_frequency"]
             .max().rename("max_VAF_gene").reset_index())
    mut_s = mut_s[mut_s["TIMEPOINT"] != "C8D1"]

    endpoint = {2: "TTEOS_rebaseline_C2D1", 3: "TTEOS_rebaseline_C3D1", 4: "TTEOS_rebaseline_C4D1"}[n_cycles]
    cln = cln_raw.rename(columns={endpoint: "OS_rebaseline"})
    cln = cln[cln["OS_rebaseline"].notna() & (cln["OS_rebaseline"] > 0)].copy()
    cln["Bev_CP"] = (cln["ARM1"] == "Bev+CP").astype(int)
    cln["Atezo_CP"] = (cln["ARM1"] == "Atezo+CP").astype(int)
    cln["nb_mets"] = (cln["number_metastatic_sites"] >= 3).astype(int)
    pts = set(cln["full_anonymized_patient_id"])

    cnt = (mut_s[mut_s["full_anonymized_patient_id"].isin(pts) & (mut_s["max_VAF_gene"] > 0)]
           .groupby("gene")["full_anonymized_patient_id"].nunique().sort_values(ascending=False))
    genes_restricted = cnt[cnt >= freq_restricted].index.tolist()
    genes_kept = cnt[cnt >= freq_minimal].index.tolist()

    wide = mut_s[mut_s["gene"].isin(genes_kept)].assign(
        v=lambda d: d["max_VAF_gene"] * 100, col=lambda d: d["gene"] + "_" + d["TIMEPOINT"])
    wide = wide.pivot_table(index="full_anonymized_patient_id", columns="col", values="v", aggfunc="max")

    cvars = ["pdl1_high", "nb_mets", "baseline_ECOG", "age", "sex_female", "sld_baseline", "Bev_CP", "Atezo_CP"]
    df = cln.set_index("full_anonymized_patient_id")[["OS_event", "OS_rebaseline"] + cvars]
    df = df.rename(columns={c: f"cln_{c}" for c in cvars}).join(wide, how="left")
    missing = [f"{g}_{tp}" for g in genes_kept for tp in TPS_ALL if f"{g}_{tp}" not in df]
    df = pd.concat([df, pd.DataFrame(0.0, index=df.index, columns=missing)], axis=1)  # complete gene x tp grid
    pan = agg.set_index("full_anonymized_patient_id")[[f"{tp}_{p}" for tp in TPS_ALL for p in PANEL]]
    df = df.join(pan, how="left")
    enc = df[[f"{g}_{tp}" for g in genes_kept for tp in TPS_ALL]].copy()
    enc.columns = [f"ENC_{c}" for c in enc.columns]  # encoder input (same scale as the original pipeline)
    df = pd.concat([df, enc], axis=1).fillna(0.0)

    out = Path(out_dir) / f"{n_cycles}tp"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "IMP150_SHERLOC_dataset.csv")
    pd.DataFrame({"gene_kept_restricted": genes_restricted}).to_csv(out / "genes_restricted.csv")
    pd.DataFrame({"gene": genes_kept}).to_csv(out / "genes_kept.csv")
    cnt.rename("n_patients").to_frame().to_csv(out / "gene_frequency.csv")
    print(f"[{n_cycles}tp] {len(df)} patients, {int(df.OS_event.sum())} deaths | genes_kept={len(genes_kept)} "
          f"| genes_restricted={len(genes_restricted)}: {genes_restricted}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", default="data_impower150")
    ap.add_argument("--n_cycles", type=int, nargs="+", default=[3, 4])
    a = ap.parse_args()
    for n in a.n_cycles:
        main(a.input_dir, a.out_dir, n)
