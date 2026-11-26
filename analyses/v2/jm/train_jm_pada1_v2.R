# train_jm_pada1_v2.R — leaner Bayesian joint model (JMbayes2) on PADA-1.
# Same patients, splits and clinical covariates as train_jm_pada1.R (input: prepare_jm_pada1.py output).
# Changes vs v1:
#  * 7 longitudinal markers instead of 19:
#      PIK3CA, TP53, CDH1 (log(VAF + 0.05), Gaussian)         -> hazard: value
#      mean VAF (log(VAF + 0.05), Gaussian)                     -> hazard: value + patient BL->C2 slope
#      nb drivers VAF > 0.5 % (count, Poisson GLMM)             -> hazard: value + patient BL->C2 slope
#      frozen MSK-CHORD encoding dim 0 and dim 1 (Gaussian)     -> hazard: value
#    the 7 change-type response features are dropped (0 at BL by construction; the slope replaces them)
#  * the encoder output is a longitudinal marker (BL and C2), no longer a covariate of the baseline Cox model;
#    the Cox sub-model holds the 9 clinical covariates only (no C2 information at time 0)
#  * trajectories frozen after each patient's own C2 sample in the hazard (IE_time = t_C2): no linear
#    extrapolation of the BL->C2 kinetics to 36 months; the slope enters as a time-invariant term (coefs())
#  * time centred (time - 0.49 months) to decorrelate random intercepts and slopes
#  * random effects of different markers independent (which_independent = "all")
#  * clinical covariates standardised (training mean / SD) for mixing; prediction grid every 0.5 month
#  * MCMC chains run in parallel; Rhat of the association parameters exported per split
# Usage: Rscript train_jm_pada1_v2.R <first_split> <last_split> [n_iter] [n_burnin] [n_chains] [cores]
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
first <- as.integer(args[1]); last <- as.integer(args[2])
n_iter   <- ifelse(length(args) >= 3, as.integer(args[3]), 3000L)
n_burnin <- ifelse(length(args) >= 4, as.integer(args[4]), 1000L)
n_chains <- ifelse(length(args) >= 5, as.integer(args[5]), 3L)
n_cores  <- ifelse(length(args) >= 6, as.integer(args[6]), n_chains)
src <- "results/benchmark_response_longitudinal/jm"
out <- Sys.getenv("JM_OUT", "results/benchmark_response_longitudinal/jm_v2")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
raw <- read.csv(file.path(src, "PADA1_JM_long.csv"), check.names = FALSE)
idx <- read.csv(file.path(src, "PADA1_JM_test_indices.csv"))
Tstart <- 1.7; grid <- seq(Tstart, 40, by = 0.5)
T_CENTRE <- as.numeric(Sys.getenv("JM_T_CENTRE", "0.92"))
EPS <- 0.05
cln_cols <- make.names(grep("^cln_", colnames(raw), value = TRUE))
names(raw) <- make.names(names(raw))

bl <- raw$timepoint == "BL"
df <- raw %>% transmute(index, timepoint, time, OS_event, OS_time_BL, t_C2, OS_landmark,
                        PIK3CA = log(PIK3CA + EPS), TP53 = log(TP53 + EPS), CDH1 = log(CDH1 + EPS),
                        log_meanVAF = log(mean_VAF + EPS),
                        nb_driver = as.integer(round(nb_driver_VAF_above_05)),
                        enc0 = ifelse(bl, Encoding_0, Encoding_1),   # encoder output is (dim, timepoint)
                        enc1 = ifelse(bl, Encoding_2, Encoding_3))
df <- cbind(df, raw[cln_cols])
SIGMA <- as.numeric(Sys.getenv("JM_SIGMA", "0.3"))                 # >0: informative prior on residual SDs
PRED_SAMPLES <- as.integer(Sys.getenv("JM_PRED_SAMPLES", "200"))  # Monte Carlo samples per prediction (JMbayes2 default 200)
PENALTY <- Sys.getenv("JM_PENALTY", "none")                       # shrinkage on associations: none / ridge / horseshoe
DRIVER_FAMILY <- Sys.getenv("JM_DRIVER_FAMILY", "gaussian")        # "poisson" or "gaussian" (log1p count)
if (DRIVER_FAMILY == "gaussian") df$nb_driver <- log1p(df$nb_driver)
gauss <- c("PIK3CA", "TP53", "CDH1", "log_meanVAF", "enc0", "enc1", if (DRIVER_FAMILY == "gaussian") "nb_driver")
kinetic <- c("log_meanVAF", "nb_driver")
markers <- c("PIK3CA", "TP53", "CDH1", "log_meanVAF", "nb_driver", "enc0", "enc1")

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
  v <- sprintf("value(%s, IE_time = 't_C2')", feat)
  if (feat %in% kinetic && re != "intercept only")
    v <- paste(v, sprintf("coefs(%s, zero_ind = list(X = 1, Z = 1), IE_time = 't_C2')", feat), sep = " + ")
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
