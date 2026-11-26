"""
preprocess_pada1.py — Build the SHERLOC input tables for PADA-1.

Steps
-----
1. ``create_mol_data_filtered_LoD``: Python port of the R function of the same
   name (scripts/survival_analysis/functions/build_datasets.R). Somatic,
   non-silent cfDNA variants are kept if VAF >= LoD at baseline (A) or at C2 (B),
   where the LoD depends on variant type, driver status and cfDNA input (ng).
2. Landmark OS at C2 (sample B): OS_rebaseline = Survival_Time - months(BL→C2);
   patients who died / were censored before the C2 sample are excluded.
3. Longitudinal features at BL and C2:
   - Gene-level   : max VAF over mutations of the same gene (top-N mutated genes).
   - Panel-level  : mean VAF, and number of driver mutations with VAF > 0.5 %.
   - Encoder input: max VAF per gene for every gene shared by the PADA-1 panel
                    and MSK-CHORD (fed to the pretrained genomic-interaction
                    encoder, converted to the 0–1 scale used in MSK-CHORD).
   - Clinical     : age, PR Allred score, tumour grade, ECOG PS, liver mets,
                    number of metastatic sites (NAs imputed per split later).

Patients in the cohort with no mutation surviving the filters get 0 everywhere.

Usage
-----
  python preprocess_pada1.py --config config.yaml
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# LoD table (port of LoDs_function, 1-ng bins between 5 and 30 ng)
# ─────────────────────────────────────────────────────────────────────────────

def lods_function(lod_5ng, lod_30ng, cfdna_ng_name, col_lod):
    n_steps = len(range(6, 31))                      # 25 bins of 1 ng
    rows = [("0-5 ng", lod_5ng)]
    for j, i in enumerate(range(6, 31), start=1):
        rows.append((f"{i-1}-{i} ng", lod_5ng - j * (lod_5ng - lod_30ng) / n_steps))
    rows.append((">30 ng", lod_30ng))
    return pd.DataFrame(rows, columns=[cfdna_ng_name, col_lod])


def cfdna_category(ng: pd.Series) -> pd.Series:
    """'0-5 ng', '5-6 ng', ..., '29-30 ng', '>30 ng' (right-closed bins)."""
    out = pd.Series(np.nan, index=ng.index, dtype=object)
    out[ng <= 5] = "0-5 ng"
    for i in range(6, 31):
        out[(ng > i - 1) & (ng <= i)] = f"{i-1}-{i} ng"
    out[ng > 30] = ">30 ng"
    return out


# LoD (% VAF) at 5 ng and 30 ng, per variant type and driver status
LOD_SPEC = {
    ("DEL", "Other"): (0.7, 0.43), ("DEL", "Driver"): (0.2, 0.11),
    ("INS", "Other"): (0.7, 0.43), ("INS", "Driver"): (0.2, 0.11),
    ("SNP", "Other"): (0.4, 0.34), ("SNP", "Driver"): (0.3, 0.2),
    ("DNP", "Other"): (0.4, 0.34), ("DNP", "Driver"): (0.3, 0.2),
    ("TNP", "Other"): (0.4, 0.34), ("TNP", "Driver"): (0.3, 0.2),
}


def build_lod_tables():
    parts = []
    for (vt, ds), (l5, l30) in LOD_SPEC.items():
        t = lods_function(l5, l30, "category_cfDNA_ng_A", "LoD_A")
        t["Variant_Type"], t["driver_status"] = vt, ds
        parts.append(t)
    lod_a = pd.concat(parts, ignore_index=True)
    lod_b = lod_a.rename(columns={"category_cfDNA_ng_A": "category_cfDNA_ng_B", "LoD_A": "LoD_B"})
    return lod_a, lod_b


# ─────────────────────────────────────────────────────────────────────────────
# Port of create_mol_data_filtered_LoD
# ─────────────────────────────────────────────────────────────────────────────

def read_table_skip_header(path, prefix="##", **kw):
    """Equivalent of rpada1::load_table(header_prefix='##')."""
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    n = 0
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith(prefix):
                n += 1
            else:
                break
    sep = "," if ".csv" in str(path) else "\t"
    return pd.read_csv(path, sep=sep, skiprows=n, low_memory=False, **kw)


def create_mol_data_filtered_LoD(df_cln, df_mut, df_review_driver,
                                 col_oncokb="OncoKB_Annotated", col_civic="CIViC_Annotated",
                                 col_variant_classif="Variant_Classification",
                                 col_status_a="Status_A", col_status_b="Status_B",
                                 keep_decision_column_filter=False):
    df_mut = df_mut[~df_mut["Subject_Id"].isin(["250001-068", "250012-005"])]
    df_mut = df_mut[~df_mut["Variant_Classification"].isin(["5'Flank", "5'UTR", "Intron"])]

    df_review_driver = df_review_driver[["Row_Key", "REVIEW_DRIVER_EB", "REVIEW_DRIVER_EB_NOTES"]]
    df_mut = df_mut.merge(df_review_driver, on="Row_Key", how="left")

    lod_a, lod_b = build_lod_tables()

    df_mut = df_mut.merge(df_cln[["Subject_Id", "Cfdna_Ng_A", "Cfdna_Ng_B"]], on="Subject_Id", how="inner")
    df_mut["category_cfDNA_ng_A"] = cfdna_category(df_mut["Cfdna_Ng_A"])
    df_mut["category_cfDNA_ng_B"] = cfdna_category(df_mut["Cfdna_Ng_B"])
    df_mut = df_mut.dropna(subset=["category_cfDNA_ng_A", "category_cfDNA_ng_B"])

    df_mut["VAF_A"] = df_mut["VAF_A"].fillna(0)
    df_mut["VAF_B"] = df_mut["VAF_B"].fillna(0)

    df_mut = df_mut[df_mut[col_variant_classif] != "Silent"].copy()
    df_mut.loc[df_mut["Status_A_B"] == "NA_somatic", col_status_a] = "somatic"
    df_mut.loc[df_mut["Status_A_B"] == "somatic_NA", col_status_b] = "somatic"
    df_mut = df_mut[df_mut[col_status_a] == "somatic"].copy()

    df_mut["driver_status"] = "Other"
    is_driver = (df_mut[col_civic] == "Yes") | (df_mut[col_oncokb] == "Yes")
    df_mut.loc[is_driver, "driver_status"] = "Driver"
    df_mut.loc[df_mut["REVIEW_DRIVER_EB"] == "No", "driver_status"] = "Other"

    df_mut = df_mut.merge(lod_a, on=["category_cfDNA_ng_A", "Variant_Type", "driver_status"], how="inner")
    df_mut = df_mut.merge(lod_b, on=["category_cfDNA_ng_B", "Variant_Type", "driver_status"], how="inner")

    above = set(df_mut.loc[df_mut["VAF_A"] >= df_mut["LoD_A"], "Row_Key"]) | \
            set(df_mut.loc[df_mut["VAF_B"] >= df_mut["LoD_B"], "Row_Key"])

    if keep_decision_column_filter:
        df_mut["Keep_LoD"] = np.where(df_mut["Row_Key"].isin(above), "kept", "removed")
        return df_mut
    return df_mut[df_mut["Row_Key"].isin(above)].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Clinical / landmark OS
# ─────────────────────────────────────────────────────────────────────────────

def rename_clinical_dataframe(df_cln):
    df_cln = df_cln.copy()
    df_cln.columns = [c.replace("(Prim_T)", "_Prim_T").replace("(mm)", "") for c in df_cln.columns]
    death_no_prog = (df_cln["Survival_Status"] == "Deceased") & (df_cln["Progression_Status"] == "Not progressed")
    df_cln.loc[death_no_prog, "Progression_Status"] = "Progressed"
    return df_cln


def landmark_os_c2(df_cln):
    """Port of create_dataframe_OS_From_Blood_SampleB (landmark at the C2 sample)."""
    d = df_cln[df_cln["Survival_Time"].notna()].copy()
    d["OS_event"] = (d["Survival_Status"] == "Deceased").astype(int)
    d["Months_BL_C2"] = d["Days_Between_Samples"] * (12 / 365)
    d = d[d["Survival_Time"] >= d["Months_BL_C2"]]
    d = d[d["Subject_Id"] != "250012-005"]
    d["OS_rebaseline"] = d["Survival_Time"] - d["Months_BL_C2"]
    return d[["Subject_Id", "OS_rebaseline", "OS_event"]]


def clinical_features(df_cln, path_extra):
    c = df_cln[["Subject_Id", "Age_At_Inclusion", "PR_Allred_Score", "Tumor_Grade_Prim_T"]].copy()
    c["PR_Allred_Score"] = pd.to_numeric(c["PR_Allred_Score"], errors="coerce")       # 'K','N','D' -> NA
    c["Tumor_Grade_Prim_T"] = c["Tumor_Grade_Prim_T"].map({"GRADE I": 1, "GRADE II": 2, "GRADE III": 3})

    x = pd.read_csv(path_extra, sep=" ", quotechar='"')
    x = x.rename(columns={"subjid": "Subject_Id"})
    x["ps_bas"] = x["ps_bas.f"].str.extract(r"(\d)").astype(float)
    # Metastatic sites used for nb_mets (same set as build_datasets.R)
    met_cols = ["met_foie_bas.f", "met_gg_bas.f", "met_os_bas.f", "met_ovair_bas.f",
                "met_peau_bas.f", "met_perit_bas.f", "met_plev_bas.f", "met_poum_bas.f",
                "met_sein_bas.f", "met_snc_bas.f"]
    for col in met_cols:
        x[col] = x[col].astype(str).str.endswith(":YES").astype(float).where(x[col].notna())
    x["nb_mets"] = x[met_cols].sum(axis=1, min_count=1)
    x["met_foie_bas"] = x["met_foie_bas.f"]
    x["met_os_bas"] = x["met_os_bas.f"]
    x["met_poum_bas"] = x["met_poum_bas.f"]
    x["met_visceral"] = x[["met_foie_bas.f", "met_poum_bas.f", "met_plev_bas.f",
                           "met_perit_bas.f", "met_snc_bas.f"]].max(axis=1)
    x["bone_only"] = ((x["met_os_bas.f"] == 1) & (x["nb_mets"] == 1)).astype(float)
    x["menop"] = (x["menop.f"] == "Post-menopause").astype(float).where(x["menop.f"].notna())
    x["del_tp_meta"] = pd.to_numeric(x["del_tp_meta.m"], errors="coerce").clip(lower=0)
    x["de_novo"] = (x["del_tp_meta"] < 6).astype(float).where(x["del_tp_meta"].notna())
    c = c.merge(x[["Subject_Id", "ps_bas", "met_foie_bas", "nb_mets", "met_os_bas", "met_poum_bas",
                   "met_visceral", "bone_only", "menop", "del_tp_meta", "de_novo"]],
                on="Subject_Id", how="left")
    extra = df_cln.set_index("Subject_Id")
    c = c.set_index("Subject_Id")
    c["ER_Allred_Score"] = pd.to_numeric(extra["ER_Allred_Score"], errors="coerce")
    c["Ki67_Pct"] = extra["Ki67_Pct_Prim_T"]
    c["lobular"] = (extra["Histological_Type_Prim_T"] == "LOBULAR").astype(float).where(
        extra["Histological_Type_Prim_T"].notna())
    c["HER2_low"] = extra["HER2_Ihc"].isin(["1+", "2+"]).astype(float).where(extra["HER2_Ihc"].notna())
    c["prior_chemo"] = (extra["Chemotherapy_Prim_T"] == "YES").astype(float).where(
        extra["Chemotherapy_Prim_T"].notna())
    c = c.reset_index()
    c.columns = ["Subject_Id"] + [f"cln_{k}" for k in c.columns[1:]]
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Feature construction
# ─────────────────────────────────────────────────────────────────────────────

def gene_max_vaf(df_mut, genes, patients, tp_cols):
    """Wide table <gene>_<tp> = max VAF over mutations of that gene."""
    out = pd.DataFrame(index=pd.Index(patients, name="Subject_Id"))
    for tp, vaf_col in tp_cols.items():
        w = (df_mut[df_mut["Hugo_Symbol"].isin(genes)]
             .groupby(["Subject_Id", "Hugo_Symbol"])[vaf_col].max().unstack())
        w = w.reindex(index=patients, columns=genes).fillna(0.0)
        w.columns = [f"{g}_{tp}" for g in w.columns]
        out = out.join(w)
    return out


def panel_features(df_mut, df_cln, patients, tp_cols, thr):
    """Candidate panel-level (aggregated) ctDNA features, named <tp>_<feature>."""
    out = pd.DataFrame(index=pd.Index(patients, name="Subject_Id"))
    ng = df_cln.set_index("Subject_Id")
    for (tp, vaf_col), ng_col in zip(tp_cols.items(), ["Cfdna_Ng_A", "Cfdna_Ng_B"]):
        detected = df_mut[df_mut[vaf_col] > 0]
        g = detected.groupby("Subject_Id")[vaf_col]
        out[f"{tp}_mean_VAF"] = g.mean()
        out[f"{tp}_max_VAF"] = g.max()
        out[f"{tp}_sum_VAF"] = g.sum()
        out[f"{tp}_nb_mut"] = g.size()
        drv = df_mut[(df_mut["driver_status"] == "Driver") & (df_mut[vaf_col] > thr)]
        out[f"{tp}_nb_driver_VAF_above_05"] = drv.groupby("Subject_Id").size()
        drv_any = detected[detected["driver_status"] == "Driver"]
        out[f"{tp}_nb_driver"] = drv_any.groupby("Subject_Id").size()
        out[f"{tp}_max_driver_VAF"] = drv_any.groupby("Subject_Id")[vaf_col].max()
        out[f"{tp}_nb_mut_VAF_above_1"] = detected[detected[vaf_col] > 1].groupby("Subject_Id").size()
        out[f"{tp}_cfDNA_ng"] = ng[ng_col]
        # tumour-derived cfDNA mass (ng) ≈ cfDNA input × mean VAF
        out[f"{tp}_ctDNA_ng"] = ng[ng_col] * out[f"{tp}_mean_VAF"].fillna(0) / 100
    out = out.fillna(0.0)

    # Early molecular response BL -> C2 (static features, prefix dyn_)
    bl, c2 = list(tp_cols)[:2]
    eps = 0.05                                                    # % VAF pseudo-count
    out["dyn_log_ratio_mean_VAF"] = np.log((out[f"{c2}_mean_VAF"] + eps) / (out[f"{bl}_mean_VAF"] + eps))
    out["dyn_log_ratio_max_VAF"] = np.log((out[f"{c2}_max_VAF"] + eps) / (out[f"{bl}_max_VAF"] + eps))
    out["dyn_C2_detected"] = (out[f"{c2}_nb_mut"] > 0).astype(float)
    out["dyn_clearance"] = ((out[f"{bl}_nb_mut"] > 0) & (out[f"{c2}_nb_mut"] == 0)).astype(float)
    bl_det = df_mut[df_mut["VAF_A"] > 0]
    frac = (bl_det.assign(cleared=bl_det["VAF_B"] == 0).groupby("Subject_Id")["cleared"].mean())
    out["dyn_frac_mut_cleared"] = frac.reindex(out.index).fillna(0.0)
    new = df_mut[(df_mut["VAF_A"] == 0) & (df_mut["VAF_B"] > 0)].groupby("Subject_Id").size()
    out["dyn_nb_new_mut_C2"] = new.reindex(out.index).fillna(0.0)
    out["dyn_nb_driver_lost"] = (out[f"{bl}_nb_driver"] - out[f"{c2}_nb_driver"]).clip(lower=0)
    return out


def main(cfg):
    P = {k: os.path.expanduser(v) for k, v in cfg["paths"].items()}
    data_dir = Path(P["data_dir"]); data_dir.mkdir(parents=True, exist_ok=True)
    tp_cols = dict(zip(cfg["timepoints"], ["VAF_A", "VAF_B"]))

    df_cln = rename_clinical_dataframe(pd.read_csv(P["cln_curated"], sep="\t"))
    df_mut_raw = read_table_skip_header(P["mol_calls"])
    df_review = pd.read_csv(P["review_driver"], index_col=0, low_memory=False)

    # 1. LoD filtering ---------------------------------------------------------
    df_mut = create_mol_data_filtered_LoD(df_cln, df_mut_raw, df_review)
    df_mut.to_csv(data_dir / "cfdna_calls_filtered_LoD.csv", index=False)
    print(f"[INFO] Mutations after LoD filtering: {len(df_mut)} "
          f"({df_mut.Subject_Id.nunique()} patients)")

    # 2. Cohort: patients with sequenced cfDNA (QC pass) + landmark OS ----------
    seq_pts = set(df_mut_raw["Subject_Id"]) - {"250001-068", "250012-005"}
    df_os = landmark_os_c2(df_cln)
    df_os = df_os[df_os["Subject_Id"].isin(seq_pts)]
    patients = sorted(df_os["Subject_Id"])
    df_mut = df_mut[df_mut["Subject_Id"].isin(patients)]
    print(f"[INFO] Landmark-OS cohort: {len(patients)} patients, "
          f"{int(df_os.OS_event.sum())} deaths")

    # 3a. Top-N most frequently mutated genes (patients with >=1 kept mutation)
    freq = (df_mut.groupby("Hugo_Symbol")["Subject_Id"].nunique()
            .sort_values(ascending=False))
    top_genes = freq.index[: cfg["model"]["n_top_genes"]].tolist()
    freq.rename("n_patients").to_frame().assign(
        pct=lambda d: 100 * d.n_patients / len(patients),
        in_top=lambda d: d.index.isin(top_genes)).to_csv(data_dir / "gene_frequency.csv")
    pd.DataFrame({"gene_kept_restricted": top_genes}).to_csv(data_dir / "genes_top.csv")
    print(f"[INFO] Top {len(top_genes)} genes: {top_genes}")

    # 3b. Genes shared by the PADA-1 panel and MSK-CHORD (encoder arm)
    pada_panel = set(df_mut_raw["Hugo_Symbol"].dropna())
    msk_genes = set(pd.read_csv(P["msk_mutations"], sep="\t", usecols=["Hugo_Symbol"],
                                low_memory=False)["Hugo_Symbol"].dropna())
    shared = sorted(pada_panel & msk_genes)
    pd.DataFrame({"gene": shared}).to_csv(data_dir / "genes_shared_PADA1_MSK.csv")
    print(f"[INFO] PADA-1 panel genes: {len(pada_panel)} | shared with MSK-CHORD: {len(shared)}")

    # 3c. Feature tables
    # gene-level max VAF for every gene mutated in >= 2 % of the cohort (top-N chosen at training time)
    cand_genes = freq[freq >= 0.02 * len(patients)].index.tolist()
    gene_feat = gene_max_vaf(df_mut, cand_genes, patients, tp_cols)
    panel_feat = panel_features(df_mut, df_cln, patients, tp_cols, cfg["features"]["driver_vaf_threshold"])
    enc_feat = gene_max_vaf(df_mut, shared, patients, tp_cols) / 100.0     # % -> fraction (MSK scale)
    enc_feat.columns = [f"ENC_{c}" for c in enc_feat.columns]
    cln_feat = clinical_features(df_cln, P["cln_extra"]).set_index("Subject_Id")

    df = (df_os.set_index("Subject_Id")
          .join(cln_feat).join(panel_feat).join(gene_feat).join(enc_feat))
    df.to_csv(data_dir / "PADA1_SHERLOC_dataset.csv")
    print(f"[INFO] Dataset: {df.shape} -> {data_dir / 'PADA1_SHERLOC_dataset.csv'}")
    print("[INFO] Clinical NA counts:", df.filter(like="cln_").isna().sum().to_dict())
    print(df.filter(regex="mean_VAF|nb_driver").describe().round(2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    with open(ap.parse_args().config) as f:
        main(yaml.safe_load(f))
