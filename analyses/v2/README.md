# SHERLOC V2 analyses (PADA-1 and IMpower150)

Code of the analyses in the V2 of the SHERLOC article (bioRxiv), applied to **PADA-1** (ER+/HER2− advanced breast
cancer, landmark at cycle 2) and **IMpower150** (non-squamous NSCLC, landmarks C2D1, C3D1 and C4D1). The scripts are
the ones that produced the results of the article. No data and no patient-level output are included.

The network is the one of the package: `sherloc_models/SHERLOC_v2.py` with `output_scale: 1` gives a forward pass
identical to `sherloc/ctDNA_Dynamics_Encoder/SHERLOC.py`. `sherloc_models/` is a copy of the encoders with options
used during development (alternative gene-level encoders, output scale); the defaults reproduce SHERLOC.

Run every command from this folder (`analyses/v2`). Results are written to `results/`, article figures and tables to
`article/figures/` and `article/tables/`.

## Data

| Data | Access | Expected location |
|---|---|---|
| MSK-CHORD (pre-training of the encoder) | cBioPortal, `msk_chord_2024` | `data/msk_chord_2024/` |
| IMpower150 ctDNA and clinical data | EGA, EGAS00001006703 (controlled access) | folder given to `prepare_impower150.py --input_dir` |
| PADA-1 ctDNA and clinical data | Unicancer, on request (pada-1@unicancer.fr) | `pada1_raw/` (paths in the PADA-1 configs) |

Paths can be changed in the `paths` section of each config. `clinical_crossfit_figures.py` and
`jm/prepare_jm_pada1.py` also read the environment variables `IMP150_CLINICAL`, `PADA1_CLINICAL` and `PADA1_RECIST`.

## Configurations

| Config | Use |
|---|---|
| `config.yaml` | PADA-1 preprocessing and base settings |
| `config_SHERLOC_response_longitudinal.yaml` | PADA-1, final SHERLOC (response features per time point in the panel arm) |
| `config_SHERLOC_calibrated*.yaml` | PADA-1 with output scale 2 (sensitivity analysis) |
| `config_IMP150_{4,3,2}tp.yaml` | IMpower150, landmarks C4D1 (main), C3D1, C2D1 |

## Pipeline and correspondence with the article

| Step | Command | Article |
|---|---|---|
| PADA-1 preprocessing (limit-of-detection filtering, features) | `python preprocess_pada1.py --config config.yaml` then `python build_response_longitudinal.py` | Methods |
| IMpower150 preprocessing | `python prepare_impower150.py --input_dir <EGA folder> --n_cycles 2 3 4` | Methods |
| SHERLOC, 100 splits | `python train_sherloc_pada1.py --config <config>` | — |
| Benchmark: SHERLOC and 10 competitors, raw and recalibrated | `python posthoc_benchmark.py --dataset <ds> --group deep|ml|fpcrf` (`--methods SHERLOC` for SHERLOC alone), then `python posthoc_benchmark_summary.py` | Fig. 2, Tables S1–S3 |
| Paired tests and ranks | `python benchmark_paired.py` | Fig. 2, Results |
| Out-of-fold recalibration of SHERLOC | `python posthoc_oof.py --config <config> --tag <ds> --seeds 0-99` | Results, recalibration |
| Bayesian joint model (JMbayes2) | `python jm/prepare_jm_pada1.py`, `python jm/prepare_jm_impower150.py --tp 4tp` (and `3tp`, `2tp`); `bash jm/run_jm_v2_local.sh`, `bash jm/run_jm_impower150_local.sh all`; `python jm/jm_summary.py` | Fig. 2, Tables S1–S3 |
| Pre-training ablation | `bash run_pretraining_ablation.sh`, then `python pretraining_ablation.py --summary` | Fig. S1 |
| 10 × 5-fold cross-fitting, SHERLOC vs clinical Cox, risk groups, thresholds | `python clinical_crossfit.py` (both trials; `--trial` to run one) | Table 2, Fig. 4 |
| Kaplan–Meier, RECIST-stratified analyses, response rules | `python clinical_crossfit_figures.py` | Fig. 3, Fig. 5, Fig. S2, Table 3 |
| Interpretation (output-layer refits, marginal effects, end-to-end stability, permutation importance) | `python interpretation.py pada1` and `impower150_4tp`, then `--perm_only`; `python interpretation_figures.py` | Fig. 6, Fig. 7, Figs. S3–S4, Table S4 |
| Article figures and tables | `python article_figures.py overview|benchmark|clinical|pretrain`, `python fig1b_svg.py`, `python article_table1.py`, `python gen_tables.py` | Fig. 1, Table 1, tables |

`<ds>` is one of `pada1`, `impower150_4tp`, `impower150_3tp`, `impower150_2tp`.

## Settings used in the article

- 100 random 70/30 splits (seeds 0–99, `random_state = 4 · seed`); PADA-1 tuning on 40 disjoint splits (seeds 100–139).
- Uno C-index (τ = 36 months for PADA-1, 30 for IMpower150), cumulative/dynamic AUC and IBS from 3 months to τ,
  calibration error at 24 months; Nadeau–Bengio corrected paired t-test.
- Cox loss with event-free mini-batches contributing zero (`sherloc/losses.py`).
- Recalibration: 5-fold out-of-fold slope and Breslow baseline (`sherloc/recalibration.py`).

Exploratory analyses that are not in the article (alternative gene-level encoders, event tokens, neural additive and
prototype models, multi-task OS/PFS training, read-level kinetic model, PFS benchmark) are not included.
