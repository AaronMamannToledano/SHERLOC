# =============================================================================
# ImPower150 ctDNA Analysis — Main Script
# =============================================================================
# Usage: set parameters in config.R, then source this script.
# =============================================================================

source("config.R")

# --- Packages ----------------------------------------------------------------
suppressPackageStartupMessages({
  library(labelled)
  library(forestmodel)
  library(survminer)
  library(survival)
  library(Hmisc)
  library(data.table)
  library(glmnet)
  library(ggpubr)
  library(ggsci)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(readr)
  library(writexl)
  library(plotly)
  library(rjson)
})

# --- Helpers -----------------------------------------------------------------

#' Build a tagged output file path, e.g. "ImPower150_OS_3.csv"
out_path <- function(tag, ext = "csv") {
  file.path(OUTPUT_DIR, paste0("ImPower150_", tag, "_", NB_CYCLES, ".", ext))
}

#' Write a data frame to CSV and print a confirmation message
save_csv <- function(df, tag, row_names = TRUE) {
  path <- out_path(tag)
  write.csv(df, file = path, row.names = row_names)
  message("Saved: ", path)
  invisible(path)
}

# =============================================================================
# 1. Load & filter mutation data
# =============================================================================
message("Loading mutation data...")
df_mut <- read.csv(PATH_MUTATIONS)

df_mut <- df_mut %>%
  filter(!driver_mutation_status %in% DRIVER_STATUS_EXCLUDE)

# Summarise to max VAF per patient / timepoint / gene; drop excluded timepoint
df_mut_s <- df_mut %>%
  group_by(full_anonymized_patient_id, TIMEPOINT, gene) %>%
  summarise(max_VAF_gene = max(variant_allele_frequency), .groups = "drop") %>%
  filter(TIMEPOINT != TIMEPOINT_EXCLUDE)

# =============================================================================
# 2. Load & prepare clinical data
# =============================================================================
message("Loading clinical data...")
df_cln_raw <- read.csv(PATH_CLINICAL)

# Select the right OS endpoint based on NB_CYCLES
endpoint_col <- switch(
  as.character(NB_CYCLES),
  "2" = "TTEOS_rebaseline_C2D1",
  "3" = "TTEOS_rebaseline_C3D1",
  "4" = "TTEOS_rebaseline_C4D1",
  stop("NB_CYCLES must be 2, 3, or 4.")
)

df_cln <- df_cln_raw %>%
  rename(OS_rebaseline = all_of(endpoint_col)) %>%
  filter(OS_rebaseline > 0, !is.na(OS_rebaseline)) %>%
  mutate(
    ctDNA_pos  = as.integer(ctDNA_status == "ctDNA_positive"),
    Bev_CP     = as.integer(ARM1 == ARM_BEV_CP),
    Atezo_CP   = as.integer(ARM1 == ARM_ATEZO_CP),
    Age_Group  = as.integer(AGEGRP == ">65"),
    nb_mets    = as.integer(number_metastatic_sites >= 3)
  )

pts_cln <- unique(df_cln$full_anonymized_patient_id)

# =============================================================================
# 3. Gene filtering
# =============================================================================
message("Filtering genes by frequency...")

df_count_gene <- df_mut_s %>%
  filter(full_anonymized_patient_id %in% pts_cln, max_VAF_gene > 0) %>%
  group_by(full_anonymized_patient_id, gene) %>%
  summarise(n = n(), .groups = "drop") %>%
  filter(n != 0) %>%
  count(gene, name = "Freq")

gene_restricted      <- df_count_gene %>% filter(Freq >= FREQ_RESTRICTED)      %>% pull(gene)
genes_kept           <- df_count_gene %>% filter(Freq >= FREQ_MINIMAL)         %>% pull(gene)

message(sprintf(
  "Genes kept: %d (>=%d) | restricted: %d (>=%d)",
  length(genes_kept), FREQ_MINIMAL,
  length(gene_restricted), FREQ_RESTRICTED
))

# Save gene lists 
# For the ctDNA-dynamics encoder, we will use the "restricted" set to focus 
# on the dynamics of more prevalent mutations and reduce noise. 
# For the genomic-interaction encoder, we will use the "kept" set to capture 
# a broader range of mutations.
save_csv(data.frame(gene_kept_restricted = gene_restricted), "genes_restricted") 
save_csv(data.frame(gene = genes_kept),                            "genes_kept")

# =============================================================================
# 4. Pivot mutation data to wide format (one column per gene × timepoint)
# =============================================================================
df_mut_wide <- df_mut_s %>%
  filter(gene %in% genes_kept) %>%
  mutate(
    max_VAF_gene    = max_VAF_gene * 100,
    gene_timepoint  = paste0(gene, "_", TIMEPOINT)
  ) %>%
  select(full_anonymized_patient_id, gene_timepoint, max_VAF_gene) %>%
  pivot_wider(names_from = gene_timepoint, values_from = max_VAF_gene, values_fill = 0)

# =============================================================================
# 5. Assemble modelling data frame
# =============================================================================
df_model_cln <- df_cln %>%
  select(
    full_anonymized_patient_id,
    OS_event, OS_rebaseline,
    pdl1_high, nb_mets, baseline_ECOG,
    age, sex_female, sld_baseline,
    Bev_CP, Atezo_CP
  )

df <- left_join(df_model_cln, df_mut_wide, by = "full_anonymized_patient_id")
df[is.na(df)] <- 0

# --- Rename variable groups with prefixes ------------------------------------
df <- df %>%
  rename_with(~ paste0("cln_", .), all_of(CLN_VARS))

# Set patient ID as row name and drop the column
rownames(df) <- df$full_anonymized_patient_id
df <- df %>% select(-full_anonymized_patient_id)

# Save main modelling data frame
save_csv(df, "OS", row_names = TRUE)

# =============================================================================
# 6. Save aggregated features for all clinical patients
# =============================================================================
message("Saving full aggregated features table...")

df_cln_full <- read.csv(PATH_CLINICAL) %>%
  rename(OS_rebaseline = all_of(endpoint_col)) %>%
  filter(OS_rebaseline > 0, !is.na(OS_rebaseline))

df_agg_full <- left_join(
  df_cln_full %>% select(full_anonymized_patient_id),
  read.csv(PATH_AGG_FEATURES),
  by = "full_anonymized_patient_id"
)
df_agg_full[is.na(df_agg_full)] <- 0

save_csv(df_agg_full, "AggFeatures", row_names = TRUE)

message("Done. All outputs written to: ", OUTPUT_DIR)