# SHERLOC

<img width="300" alt="SHERLOC_Logo" src="https://github.com/user-attachments/assets/df1d6ac3-1371-47b8-a50d-7705dfcc757a" />

SHERLOC: An interpretable deep learning model for longitudinal ctDNA data in survival analysis

Longitudinal measurements of circulating tumor DNA (ctDNA) provide a noninvasive way to track treatment response, yet clinical trial data pose significant methodological challenges due to short, high-dimensional longitudinal sequences and limited sample sizes. We present SHERLOC, a deep learning framework tailored for survival analysis with on-treatment ctDNA data. It combines shared temporal representations of gene-level variant allele frequencies, feature-specific temporal patterns from panel-level ctDNA measurements, and survival-informed genomic representations pre-trained on large tissue-biopsy datasets, all within an interpretable Cox proportional hazards framework.

Evaluated on a non–small-cell lung cancer cohort from the phase III IMpower150 trial, SHERLOC outperformed a range of statistical, ensemble, and deep learning methods. It demonstrated improved survival discrimination, better calibration, and greater robustness to reduced longitudinal sampling, while generating a ctDNA-based risk score that adds prognostic value beyond standard imaging-based response assessments.
