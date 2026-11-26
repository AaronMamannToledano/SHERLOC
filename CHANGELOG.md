# Changelog

## 2.0.0 — V2 of the article (PADA-1 and IMpower150)

### Fixed
- **Cox loss on event-free mini-batches** (`sherloc/losses.py`): pycox's `CoxPHLoss` returns NaN on a mini-batch
  without events; the weights became NaN and early stopping silently restored an early epoch. `SafeCoxPHLoss` makes
  such batches contribute zero; it is the default in `scripts/train_DL.py` (`training.safe_cox_loss`), in the
  encoder pre-training and in `scripts/explainability.py`.
- **`scripts/train_ML.py`**: clinical covariates are now included in the feature matrix (as described in the paper);
  features are standardised inside the Coxnet pipeline; a diverging Coxnet path is refitted with a larger
  `alpha_min_ratio`.
- **`scripts/train_jm.R`**: test patients are evaluated with all samples observed up to `Tstart` instead of their
  baseline row only; `tvBrier` uses `ibs_tstart`.

### Added
- `sherloc/recalibration.py`: out-of-fold recalibration (calibration slope and Breslow baseline) for any risk score.
- `analyses/v2/`: the analyses of the V2 article — PADA-1 preprocessing, benchmark against ten competitors and a
  Bayesian joint model on 100 splits (raw and recalibrated, paired Nadeau–Bengio tests), pre-training ablation,
  10 × 5-fold cross-fitting against a clinical Cox model, RECIST-stratified analyses and response rules,
  interpretation (average marginal effects, end-to-end stability, grouped permutation importance).
- Tests for the loss and the recalibration; architecture figure in `docs/`.

### Unchanged
- The SHERLOC network (`sherloc/ctDNA_Dynamics_Encoder/SHERLOC.py`) and the pre-trained encoder.

## 0.1.0 — V1 of the article (IMpower150)
