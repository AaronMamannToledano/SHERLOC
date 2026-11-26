# train_jm.R — Joint model for longitudinal and survival data (JMbayes2)
#
# Pipeline
# --------
#  1. Read settings from config.yaml (paths, column names, timepoint mapping,
#     MCMC hyperparameters, SLURM options).
#  2. Load the longitudinal + survival dataset and the pre-computed train/test
#     index splits (produced by the Python pipeline).
#  3. For each of the 100 (configurable) train/test experiments, call
#     joint_model():
#       a. Map categorical timepoint labels to numeric values.
#       b. Filter features whose prevalence is below the configured threshold.
#       c. Scale features using training-set statistics (applied to test set).
#       d. Fit a Cox PH survival submodel and LME longitudinal submodels.
#       e. Fit the joint model (JMbayes2).
#       f. Evaluate with integrated Brier score and mean time-dependent AUC.
#  4. Collect results and write a CSV with IBS and AUC per experiment.
#
# Usage
# -----
#   Rscript train_jm.R                              # reads config.yaml
#   Rscript train_jm.R --config my_cfg.yaml
#   Rscript train_jm.R --n_timepoints 2
#   Rscript train_jm.R --no_slurm                  # sequential run, no SLURM
#
# Bugs fixed vs original script
# ------------------------------
#  1. AUC loop: `for (time in length(seq(times)))` iterated over a single
#     integer (the length of the vector) instead of the time values themselves.
#     Fixed to `for (time in times)`.
#  2. Test-set scaling: the original scaled train and test sets independently,
#     introducing data leakage. Fixed by saving training-set center/scale and
#     applying them to the test set.
#  3. Predictions: the original evaluated the joint model on the test patients'
#     baseline rows only, so the on-treatment samples were never used. Test
#     patients are now evaluated with all their samples observed up to Tstart
#     (joint_model.use_longitudinal_history, default true); JMbayes2 conditions
#     the prediction on survival up to Tstart. Tstart should be at or after the
#     last sample used (the landmark).
#  4. tvBrier now uses joint_model.ibs_tstart (it was set in config.yaml but
#     unused; eval_time_start was passed instead).

suppressPackageStartupMessages({
  library(yaml)
  library(optparse)
  library(nlme)
  library(data.table)
  library(dplyr)
  library(survival)
  library(JMbayes2)
  library(GLMMadaptive)
})


`%||%` <- function(a, b) if (is.null(a)) b else a

# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

#' Load a YAML config file into an R list.
#'
#' @param path Path to config.yaml.
#' @return Named list of configuration values.
load_config <- function(path) {
  if (!file.exists(path)) {
    stop(sprintf(
      "[ERROR] Config file not found: %s\n",
      "Copy config.yaml to your working directory and edit the paths section.",
      path
    ))
  }
  yaml::read_yaml(path)
}


#' Replace the {n_timepoints} placeholder in a path template.
#'
#' @param template String containing an optional {n_timepoints} placeholder.
#' @param n_timepoints Integer; number of configured timepoints.
#' @return Resolved path string.
resolve_path <- function(template, n_timepoints) {
  gsub("\\{n_timepoints\\}", as.character(n_timepoints), template)
}


#' Validate that all required joint-model config fields are present.
#'
#' Raises an informative error for the first missing or mis-specified field
#' so users know exactly what to fix in config.yaml.
#'
#' @param cfg Top-level config list (output of load_config).
validate_config <- function(cfg) {
  jm <- cfg$joint_model
  if (is.null(jm)) {
    stop("[ERROR] 'joint_model' section is missing from config.yaml.")
  }

  required_fields <- c(
    "survival_time_col", "survival_event_col",
    "patient_id_col", "timepoint_col",
    "timepoint_numeric",
    "clinical_prefix", "encoder_prefix",
    "threshold_feature_frequency",
    "n_iter", "n_burnin",
    "eval_time_start", "eval_time_end", 
    "eval_time_step", "auc_tstart", "ibs_tstart",
    "seed", "n_experiments",
    "data_file", "indices_file", "output_file"
  )
  for (field in required_fields) {
    if (is.null(jm[[field]])) {
      stop(sprintf(
        "[ERROR] joint_model.%s is missing from config.yaml.", field
      ))
    }
  }

  required_paths <- c("data_file", "indices_file")
  n_tp <- length(cfg$timepoints)
  for (key in required_paths) {
    resolved <- resolve_path(jm[[key]], n_tp)
    if (!file.exists(resolved)) {
      stop(sprintf(
        "[ERROR] File not found: %s\nCheck joint_model.%s in config.yaml.",
        resolved, key
      ))
    }
  }

  if (length(cfg$timepoints) == 0) {
    stop("[ERROR] timepoints list is empty in config.yaml.")
  }

  tp_map <- jm$timepoint_numeric
  missing_tp <- setdiff(cfg$timepoints, names(tp_map))
  if (length(missing_tp) > 0) {
    stop(sprintf(
      "[ERROR] timepoints %s are in the timepoints list but have no numeric ",
      "mapping in joint_model.timepoint_numeric. Add them to config.yaml.",
      paste(missing_tp, collapse = ", ")
    ))
  }
}


# ─────────────────────────────────────────────────────────────────────────────
# Joint model function
# ─────────────────────────────────────────────────────────────────────────────

#' Fit and evaluate a Bayesian joint model for one train/test split.
#'
#' @param df              Long-format DataFrame with longitudinal and survival
#'                        data.
#' @param df_indices      DataFrame whose rows are test-set patient indices for
#'                        each of the 100 (or n_experiments) splits.
#' @param experiment_number Integer (1 … n_experiments) selecting the split.
#' @param cfg             Full config list (output of load_config).
#'
#' @return Named list with elements \code{ibs} (integrated Brier score) and
#'         \code{auc} (mean time-dependent AUC).
joint_model <- function(df, df_indices, experiment_number, cfg) {

  jm       <- cfg$joint_model
  timepoints <- cfg$timepoints
  n_tp     <- length(timepoints)

  # ── Unpack column-name config ────────────────────────────────────
  time_col    <- jm$survival_time_col
  event_col   <- jm$survival_event_col
  id_col      <- jm$patient_id_col
  tp_col      <- jm$timepoint_col
  cln_prefix  <- jm$clinical_prefix
  enc_prefix  <- jm$encoder_prefix
  tp_map      <- jm$timepoint_numeric

  # Validate that mandatory columns exist in the data
  required_cols <- c(time_col, event_col, id_col, tp_col)
  missing_cols  <- setdiff(required_cols, colnames(df))
  if (length(missing_cols) > 0) {
    stop(sprintf(
      "[ERROR] Dataset is missing columns: %s\n",
      "Update joint_model column-name settings in config.yaml.",
      paste(missing_cols, collapse = ", ")
    ))
  }

  # ── Evaluation time grid ─────────────────────────────────────────
  times <- seq(jm$eval_time_start, jm$eval_time_end, by = jm$eval_time_step)

  # ── Timepoint label → numeric conversion ─────────────────────────
  # Replaces categorical labels with the numeric values in timepoint_numeric,
  # making the time variable suitable for LME submodel fitting.
  for (tp_label in names(tp_map)) {
    df[df[[tp_col]] == tp_label, tp_col] <- tp_map[[tp_label]]
  }
  df[[tp_col]] <- as.numeric(df[[tp_col]])

  # ── Build survival submodel formula ──────────────────────────────
  # Includes all latent encoder features (columns starting with enc_prefix)
  # and all clinical covariates (columns starting with cln_prefix).
  enc_cols <- grep(paste0("^", enc_prefix), colnames(df), value = TRUE)
  cln_cols <- grep(paste0("^", cln_prefix), colnames(df), value = TRUE)

  if (length(enc_cols) == 0) {
    warning(sprintf(
      "[WARN] No encoder columns found with prefix '%s'. ",
      "Check joint_model.encoder_prefix in config.yaml.", enc_prefix
    ))
  }
  if (length(cln_cols) == 0) {
    warning(sprintf(
      "[WARN] No clinical columns found with prefix '%s'. ",
      "Check joint_model.clinical_prefix in config.yaml.", cln_prefix
    ))
  }

  surv_covariates <- paste(c(enc_cols, cln_cols), collapse = " + ")
  formula_surv <- as.formula(sprintf(
    "Surv(%s, %s) ~ %s", time_col, event_col, surv_covariates
  ))

  # ── Train / test split ────────────────────────────────────────────
  test_indices  <- as.numeric(df_indices[experiment_number, ])
  train_indices <- setdiff(unique(df[[id_col]]), test_indices)
  df_train      <- df %>% filter(.data[[id_col]] %in% train_indices)
  df_test       <- df %>% filter(.data[[id_col]] %in% test_indices)

  # ── Feature selection by prevalence in the training set ───────────
  # Columns excluded from feature selection: survival columns, patient ID,
  # timepoint, clinical covariates, and latent encoder features.
  fixed_cols <- c(time_col, event_col, id_col, tp_col, cln_cols, enc_cols)
  candidate_cols <- setdiff(colnames(df), fixed_cols)

  freq_threshold <- jm$threshold_feature_frequency
  features_list <- Filter(function(feat) {
    n_patients_total    <- length(unique(df_train[[id_col]]))
    n_patients_nonzero  <- length(unique(
      df_train[df_train[[feat]] > 0, id_col]
    ))
    (n_patients_nonzero / n_patients_total) > freq_threshold
  }, candidate_cols)

  if (length(features_list) == 0) {
    warning(sprintf(
      "[WARN] No features passed the %.0f%% prevalence threshold for ",
      "experiment %d. Consider lowering threshold_feature_frequency.",
      freq_threshold * 100, experiment_number
    ))
    return(list(ibs = NA_real_, auc = NA_real_))
  }

  # ── Feature scaling: fit on train, apply to test ──────────────────
  # Bug fix: the original script scaled train and test independently,
  # leaking test-set statistics into the scaling. Here we save the
  # training-set center and scale, then apply them to the test set.
  train_scale   <- scale(df_train[features_list])
  scale_center  <- attr(train_scale, "scaled:center")
  scale_sd      <- attr(train_scale, "scaled:scale")

  df_train[features_list] <- as.data.frame(train_scale)
  df_test[features_list]  <- as.data.frame(
    scale(df_test[features_list], center = scale_center, scale = scale_sd)
  )

  # ── Set seed ──────────────────────────────────────────────────────
  set.seed(jm$seed)

  # ── Fit survival submodel (baseline timepoint only) ───────────────
  df_train_baseline <- df_train %>% filter(.data[[tp_col]] == 0)
  cox_fit <- coxph(formula_surv, data = df_train_baseline)

  # ── Fit LME longitudinal submodels (one per selected feature) ────
  lme_list <- lapply(features_list, function(feat) {
    lme(
      as.formula(sprintf("%s ~ %s", feat, tp_col)),
      random = as.formula(sprintf("~ 1 | %s", id_col)),
      data   = df_train
    )
  })
  names(lme_list) <- features_list

  # ── Fit the Bayesian joint model ──────────────────────────────────
  joint_fit <- jm(
    cox_fit, lme_list,
    time_var = tp_col,
    n_iter   = jm$n_iter,
    n_burnin = jm$n_burnin
  )

  # ── Evaluate on the test set ──────────────────────────────────────
  # Longitudinal history of each test patient: every sample observed up to
  # Tstart (or the baseline row only if use_longitudinal_history is false).
  use_hist <- isTRUE(jm$use_longitudinal_history %||% TRUE)
  ibs_tstart <- jm$ibs_tstart %||% jm$eval_time_start
  history_until <- function(t0) {
    if (use_hist) df_test %>% filter(.data[[tp_col]] <= t0)
    else df_test %>% filter(.data[[tp_col]] == 0)
  }
  if (use_hist && max(df_test[[tp_col]]) > min(ibs_tstart, jm$auc_tstart)) {
    warning("[WARN] Some samples are after Tstart and will not be used; ",
            "set ibs_tstart / auc_tstart at or after the last sample (landmark).")
  }

  # Integrated Brier score
  ibs_result <- tvBrier(
    joint_fit,
    newdata       = history_until(ibs_tstart),
    Tstart        = ibs_tstart,
    Thoriz        = jm$eval_time_end,
    integrated    = TRUE,
    cores         = 1L,
    type_weights  = "IPCW"
  )$Brier

  # Time-dependent AUC at each evaluation timepoint
  auc_values <- vapply(times, function(t_horiz) {
    tvAUC(
      joint_fit,
      newdata      = history_until(jm$auc_tstart),
      Tstart       = jm$auc_tstart,
      Thoriz       = t_horiz,
      cores        = 1L,
      type_weights = "IPCW"
    )$auc
  }, numeric(1))

  auc_mean <- mean(auc_values, na.rm = TRUE)

  return(list(ibs = ibs_result, auc = auc_mean))
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

get_args <- function() {
  option_list <- list(
    make_option(
      "--config",
      type    = "character",
      default = "config.yaml",
      help    = "Path to YAML config file [default: config.yaml]"
    ),
    make_option(
      "--n_timepoints",
      type    = "integer",
      default = NULL,
      help    = paste(
        "Override the number of timepoints (selects the matching data file).",
        "If omitted, uses length(timepoints) from config.yaml."
      )
    ),
    make_option(
      "--no_slurm",
      action  = "store_true",
      default = FALSE,
      help    = "Run experiments sequentially without SLURM [default: FALSE]"
    ),
    make_option(
      "--output_dir",
      type    = "character",
      default = NULL,
      help    = "Override the output directory from config.yaml"
    )
  )
  optparse::parse_args(optparse::OptionParser(
    usage       = "Rscript train_jm.R [options]",
    option_list = option_list,
    description = paste(
      "Bayesian joint model for longitudinal ctDNA and survival data.",
      "All settings are read from config.yaml; CLI flags override specific fields."
    )
  ))
}


# ─────────────────────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────────────────────

main <- function() {
  args <- get_args()

  # ── Load and patch config ─────────────────────────────────────────
  cfg <- load_config(args$config)

  if (!is.null(args$n_timepoints)) {
    # Trim or pad the timepoints list to the requested length
    tp_all <- names(cfg$joint_model$timepoint_numeric)
    if (args$n_timepoints > length(tp_all)) {
      stop(sprintf(
        "[ERROR] --n_timepoints %d exceeds the %d timepoints defined in ",
        "joint_model.timepoint_numeric. Add more entries to config.yaml.",
        args$n_timepoints, length(tp_all)
      ))
    }
    cfg$timepoints <- tp_all[seq_len(args$n_timepoints)]
    message(sprintf("[INFO] Using %d timepoints: %s",
                    args$n_timepoints,
                    paste(cfg$timepoints, collapse = ", ")))
  }

  if (!is.null(args$output_dir)) {
    cfg$paths$output_dir <- args$output_dir
  }

  if (isTRUE(args$no_slurm)) {
    cfg$joint_model$use_slurm <- FALSE
  }

  validate_config(cfg)

  jm   <- cfg$joint_model
  n_tp <- length(cfg$timepoints)

  # ── Load data ─────────────────────────────────────────────────────
  data_path    <- resolve_path(jm$data_file,    n_tp)
  indices_path <- resolve_path(jm$indices_file, n_tp)
  output_path  <- resolve_path(jm$output_file,  n_tp)

  # Override output directory if specified
  if (!is.null(cfg$paths$output_dir)) {
    dir.create(cfg$paths$output_dir, showWarnings = FALSE, recursive = TRUE)
    output_path <- file.path(
      cfg$paths$output_dir,
      basename(output_path)
    )
  }

  message(sprintf("[INFO] Loading data from:    %s", data_path))
  message(sprintf("[INFO] Loading indices from: %s", indices_path))
  message(sprintf("[INFO] Results will be saved to: %s", output_path))

  df         <- read.csv(data_path,    check.names = FALSE)
  df_indices <- read.csv(indices_path, check.names = FALSE)

  n_experiments <- jm$n_experiments
  experiment_seq <- seq_len(n_experiments)

  # ── Run experiments ───────────────────────────────────────────────
  if (isTRUE(jm$use_slurm)) {

    # ── SLURM parallel execution ────────────────────────────────────
    library(rslurm)

    slurm_opts <- list(
      time                = jm$slurm$time,
      ntasks              = jm$slurm$ntasks,
      `cpus-per-task`     = jm$slurm$cpus_per_task,
      partition           = jm$slurm$partition
    )

    # Objects that each SLURM worker needs
    global_objects <- c("joint_model", "df", "df_indices", "cfg")

    slurm_job <- slurm_apply(
      function(experiment_number) {
        joint_model(
          df                = df,
          df_indices        = df_indices,
          experiment_number = experiment_number,
          cfg               = cfg
        )
      },
      params          = data.frame(experiment_number = experiment_seq),
      nodes           = n_experiments,
      cpus_per_node   = 1,
      pkgs = c("JMbayes2", "dplyr", "survival", "nlme", "GLMMadaptive"),
      global_objects  = global_objects,
      slurm_options   = slurm_opts
    )

    results_raw <- get_slurm_out(slurm_job, outtype = "raw")

  } else {

    # ── Sequential execution (no SLURM) ────────────────────────────
    message(sprintf(
      "[INFO] Running %d experiments sequentially (no SLURM) ...",
      n_experiments
    ))
    results_raw <- lapply(experiment_seq, function(exp_num) {
      message(sprintf("[INFO]   Experiment %d / %d", exp_num, n_experiments))
      joint_model(
        df                = df,
        df_indices        = df_indices,
        experiment_number = exp_num,
        cfg               = cfg
      )
    })

  }

  # ── Collect and export results ────────────────────────────────────
  df_results <- do.call(rbind, lapply(results_raw, function(x) {
    data.frame(IBS = x$ibs, AUC = x$auc)
  }))

  message("\n[INFO] Summary across experiments:")
  print(summary(df_results))

  write.csv(df_results, file = output_path, row.names = TRUE)
  message(sprintf("\n[INFO] Results saved to %s", output_path))
}

main()
