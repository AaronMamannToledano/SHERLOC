# train_jm_impower150.R — joint model v2 (JMbayes2) on IMpower150 OS, 4 / 3 / 2 timepoints.
# Same design as train_jm_pada1_v2.R (PADA-1); input from prepare_jm_impower150.py:
#  * 7 markers: 3 most frequent genes (TP53, KRAS, SPTA1; log(VAF % + 0.05)) -> hazard: value
#               mean VAF (log(% + 0.05)) and number of drivers (log1p)     -> value
#    (PADA-1 also has a patient-slope term for these two; on IMpower150 every sample is on the same nominal 21-day
#     grid and the slope terms never converged on split 0 (Rhat 1.9-3.7 with the noise prior, without it, with a ridge
#     penalty, or with mean VAF only); value-only converges (Rhat <= 1.03). JM_KINETIC=log_meanVAF,nb_driver restores them)
#               the 2 dimensions of the frozen MSK-CHORD encoder           -> value
#  * Cox sub-model: the 8 clinical covariates (standardised)
#  * trajectories frozen after the landmark sample (IE_time = t_LM); time centred at the landmark (intercept = level
#    at the landmark); nominal times (21-day cycles); random intercept + slope per marker, markers independent
#  * prediction conditional on survival to the landmark (Tstart = t_LM)
#  * JM_SIGMA (default 0.3): informative prior on residual SDs, as in PADA-1; 0 = default JMbayes2 prior
# Usage: Rscript train_jm_impower150.R <4tp|3tp|2tp> <first_split> <last_split> [n_iter] [n_burnin] [n_chains] [cores]
suppressPackageStartupMessages({
  library(nlme); library(survival); library(dplyr); library(GLMMadaptive); library(JMbayes2)
})
if (!"IE_time" %in% names(formals(JMbayes2::value)))
  stop("This JMbayes2 (", packageVersion("JMbayes2"), ") has no IE_time option: install the GitHub version, ",
       "remotes::install_github('drizopoulos/JMbayes2')")
# JMbayes2 0.6.x bug: coefs(..., zero_ind) runs `m[, ind, drop = FALSE] <- 0`, which R rejects
# ("incorrect number of subscripts"). Patch the function in the namespace (no reinstall needed).
local({
  fn <- get("design_matrices_functional_forms", envir = asNamespace("JMbayes2"))
  src <- deparse(fn)
  src <- gsub("m[, ind, drop = FALSE] <- 0", "m[, ind] <- 0", src, fixed = TRUE)
  src <- gsub("m[, -ind, drop = FALSE] <- 1", "m[, -ind] <- 1", src, fixed = TRUE)
  fn2 <- eval(parse(text = src)); environment(fn2) <- asNamespace("JMbayes2")
  utils::assignInNamespace("design_matrices_functional_forms", fn2, ns = "JMbayes2")
})
args <- commandArgs(trailingOnly = TRUE)
tp <- args[1]; first <- as.integer(args[2]); last <- as.integer(args[3])
n_iter   <- ifelse(length(args) >= 4, as.integer(args[4]), 10000L)
n_burnin <- ifelse(length(args) >= 5, as.integer(args[5]), 3000L)
n_chains <- ifelse(length(args) >= 6, as.integer(args[6]), 3L)
n_cores  <- ifelse(length(args) >= 7, as.integer(args[7]), n_chains)
src <- file.path("results/jm_impower150", tp)
out <- Sys.getenv("JM_OUT", file.path(src, "jm_v2"))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
raw <- read.csv(file.path(src, "IMP150_JM_long.csv"), check.names = FALSE)
idx <- read.csv(file.path(src, "IMP150_JM_test_indices.csv"))
Tstart <- raw$t_LM[1]; grid <- seq(Tstart, Tstart + 36, by = 0.5)
T_CENTRE <- Tstart
EPS <- 0.05
cln_cols <- make.names(grep("^cln_", colnames(raw), value = TRUE))
names(raw) <- make.names(names(raw))

genes <- grep("^gene_", names(raw), value = TRUE)
df <- raw %>% transmute(index, timepoint, time, OS_event, OS_time_BL, t_LM, OS_landmark,
                        log_meanVAF = log(mean_VAF + EPS), nb_driver = as.integer(round(nb_driver)), enc0, enc1)
for (g in genes) df[[g]] <- log(raw[[g]] + EPS)
df <- cbind(df, raw[cln_cols])
SIGMA <- as.numeric(Sys.getenv("JM_SIGMA", "0.3"))                 # >0: informative prior on residual SDs
PRED_SAMPLES <- as.integer(Sys.getenv("JM_PRED_SAMPLES", "200"))  # Monte Carlo samples per prediction (JMbayes2 default 200)
PENALTY <- Sys.getenv("JM_PENALTY", "none")                       # shrinkage on associations: none / ridge / horseshoe
DRIVER_FAMILY <- Sys.getenv("JM_DRIVER_FAMILY", "gaussian")        # "poisson" or "gaussian" (log1p count)
if (DRIVER_FAMILY == "gaussian") df$nb_driver <- log1p(df$nb_driver)
gauss <- c(genes, "log_meanVAF", "enc0", "enc1", if (DRIVER_FAMILY == "gaussian") "nb_driver")
kinetic <- strsplit(Sys.getenv("JM_KINETIC", "none"), ",")[[1]]   # markers with a slope term
markers <- c(genes, "log_meanVAF", "nb_driver", "enc0", "enc1")

ctrl <- lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200, returnObject = TRUE)
fit_gauss <- function(feat, d) {
  f <- as.formula(sprintf("%s ~ I(time - %s)", feat, T_CENTRE))
  m <- tryCatch(lme(f, random = as.formula(sprintf("~ I(time - %s) | index", T_CENTRE)), data = d, control = ctrl),
                error = function(e) NULL)
  if (!is.null(m)) return(list(m = m, re = "unstructured"))
  m <- tryCatch(lme(f, random = list(index = pdDiag(as.formula(sprintf("~ I(time - %s)", T_CENTRE)))),
                    data = d, control = ctrl), error = function(e) NULL)
  if (!is.null(m)) return(list(m = m, re = "diagonal"))
  list(m = lme(f, random = ~ 1 | index, data = d, control = ctrl), re = "intercept only")
}
fit_pois <- function(feat, d) {
  f <- as.formula(sprintf("%s ~ I(time - %s)", feat, T_CENTRE))
  m <- tryCatch(mixed_model(f, random = as.formula(sprintf("~ I(time - %s) | index", T_CENTRE)), data = d,
                            family = poisson()), error = function(e) NULL)
  if (!is.null(m)) return(list(m = m, re = "unstructured"))
  m <- tryCatch(mixed_model(f, random = as.formula(sprintf("~ I(time - %s) || index", T_CENTRE)), data = d,
                            family = poisson()), error = function(e) NULL)
  if (!is.null(m)) return(list(m = m, re = "diagonal"))
  list(m = mixed_model(f, random = ~ 1 | index, data = d, family = poisson()), re = "intercept only")
}
ff_term <- function(feat, re) {
  v <- sprintf("value(%s, IE_time = 't_LM')", feat)
  if (feat %in% kinetic && re != "intercept only")
    v <- paste(v, sprintf("coefs(%s, zero_ind = list(X = 1, Z = 1), IE_time = 't_LM')", feat), sep = " + ")
  v
}

for (s in first:last) {
  t0 <- Sys.time()
  test_ids <- as.numeric(idx[s + 1, ])
  tr <- df %>% filter(!(index %in% test_ids)); te <- df %>% filter(index %in% test_ids)
  for (c in cln_cols) {
    med <- median(tr[[c]], na.rm = TRUE); tr[[c]][is.na(tr[[c]])] <- med; te[[c]][is.na(te[[c]])] <- med
    mu <- mean(tr[[c]]); sc <- sd(tr[[c]]); if (sc == 0) sc <- 1      # standardised: better MCMC mixing
    tr[[c]] <- (tr[[c]] - mu) / sc; te[[c]] <- (te[[c]] - mu) / sc
  }
  ce <- sapply(tr[gauss], mean); sdv <- sapply(tr[gauss], sd); sdv[sdv == 0] <- 1
  for (g in gauss) { tr[[g]] <- (tr[[g]] - ce[g]) / sdv[g]; te[[g]] <- (te[[g]] - ce[g]) / sdv[g] }
  set.seed(3)
  base_tr <- tr %>% filter(timepoint == "BL")
  cox_fit <- coxph(as.formula(paste("Surv(OS_time_BL, OS_event) ~", paste(cln_cols, collapse = " + "))),
                   data = base_tr)
  fits <- lapply(markers, function(f) if (f %in% gauss) fit_gauss(f, tr) else fit_pois(f, tr))
  names(fits) <- markers
  re_used <- sapply(fits, `[[`, "re")
  ff <- as.formula(paste("~", paste(mapply(ff_term, markers, re_used), collapse = " + ")))
  prs <- list()
  if (PENALTY != "none") prs$penalty_alphas <- PENALTY
  if (SIGMA > 0) {   # pin the residual SD (standardised units): with every C2 sample at ~1 month, residual
    # variance and random-slope variance are not separately identified; an informative prior fixes the scale
    n_sig <- sum(markers %in% gauss)
    prs$sigmas_mean <- rep(SIGMA, n_sig); prs$sigmas_shape <- 100
  }
  if (!length(prs)) prs <- NULL
  t1 <- Sys.time()
  jfit <- jm(cox_fit, lapply(fits, `[[`, "m"), time_var = "time", functional_forms = ff,
             which_independent = "all", priors = prs,
             n_iter = n_iter, n_burnin = n_burnin, n_chains = n_chains, cores = n_cores)
  t2 <- Sys.time()
  sm <- summary(jfit)
  al <- sm$Survival
  al$term <- rownames(al)
  write.csv(al, file.path(out, sprintf("survival_coefs_split%03d.csv", s)), row.names = FALSE)
  te_lm <- te; te_lm$OS_time_BL <- Tstart; te_lm$OS_event <- 0
  pr <- predict(jfit, newdata = te_lm, process = "event", times = grid, return_newdata = TRUE,
                n_samples = PRED_SAMPLES, cores = n_cores)
  pr <- pr[, c("index", "time", "pred_CIF")]
  write.csv(pr, file.path(out, sprintf("pred_split%03d.csv", s)), row.names = FALSE)
  rh <- al$Rhat[!al$term %in% cln_cols]
  res <- data.frame(seed = s, n_markers = length(markers),
                    re_structures = paste(sprintf("%s:%s", markers, re_used), collapse = ";"),
                    max_Rhat_assoc = max(rh, na.rm = TRUE), max_Rhat_all = max(al$Rhat, na.rm = TRUE),
                    min_lme = as.numeric(difftime(t1, t0, units = "mins")),
                    min_mcmc = as.numeric(difftime(t2, t1, units = "mins")),
                    minutes = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  write.csv(res, file.path(out, sprintf("res_split%03d.csv", s)), row.names = FALSE)
  print(res); print(al[, c("term", "Mean", "2.5%", "97.5%", "Rhat")], digits = 3); flush.console()
}
