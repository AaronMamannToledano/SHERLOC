# =============================================================================
# config.R — ImPower150 ctDNA Analysis Configuration
# =============================================================================

# --- Analysis parameters -----------------------------------------------------
NB_CYCLES <- 4          # Number of cycles before rebaseline: 2, 3, or 4

# Gene frequency thresholds (minimum number of patients with mutation)
FREQ_RESTRICTED      <- 20
FREQ_MINIMAL         <- 5   # Genes used in the model

# Timepoint to exclude from mutation data
TIMEPOINT_EXCLUDE <- "C8D1"

# --- Input file paths ---------------------------------------------------------
INPUT_DIR <- "target"   # Change if input files are located elsewhere

PATH_MUTATIONS <- file.path(
  INPUT_DIR, "EGAF00007700270",
  "anonymized_imp150_ctDNA_calls_toShare.ctDNApositiveSamples.csv"
)

PATH_CLINICAL <- file.path(
  INPUT_DIR, "EGAF00007700272",
  "anonymized_imp150_ctDNA_patient_clinical_toShare.csv"
)

PATH_AGG_FEATURES <- file.path(
  INPUT_DIR, "EGAF00007700271",
  "anonymized_imp150_ctDNA_ctDNAfeatures_toShare.csv"
)

# --- Output directory ---------------------------------------------------------
OUTPUT_DIR <- file.path(INPUT_DIR)   # Change to a dedicated folder if preferred

# --- Clinical variables to include in the model ------------------------------
CLN_VARS <- c(
  "pdl1_high",
  "age",
  "nb_mets",
  "baseline_ECOG",
  "sex_female",
  "Bev_CP",
  "Atezo_CP",
  "sld_baseline"
)

# --- Treatment arm labels (as they appear in ARM1 column) --------------------
ARM_BEV_CP  <- "Bev+CP"
ARM_ATEZO_CP <- "Atezo+CP"

# --- Driver mutation statuses to exclude -------------------------------------
DRIVER_STATUS_EXCLUDE <- c("synonymous", "")
