# SHERLOC

## 🎬 Interactive Overview
👉 [aaronmamanntoledano.github.io/SHERLOC](https://aaronmamanntoledano.github.io/SHERLOC/)

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**SHERLOC: An interpretable deep learning model for longitudinal ctDNA data in survival analysis**

SHERLOC is a deep learning framework specifically designed for survival analysis within the constraints of clinical trials, characterized by short longitudinal trajectories, high-dimensional genomic data, and limited sample sizes.

---

## 📖 Overview

In this study, we address the challenge of extracting robust and clinically actionable prognostic information from high-dimensional ctDNA measurements. SHERLOC integrates:
*   **Shared Temporal Representations:** Learned across gene-level variant allele frequencies (VAFs).
*   **Feature-Specific Patterns:** For panel-level biomarkers (mean VAF, cfDNA concentration, etc.).
*   **Pre-trained Genomic Encoders:** Leveraging survival-informed representations from large tissue-biopsy datasets (MSK-CHORD).

### Key Findings
*   **Superior Performance:** SHERLOC consistently outperforms classical statistical methods, ensemble models, and state-of-the-art deep learning models in survival discrimination and calibration.
*   **Clinical Utility:** The ctDNA-based risk score remains a strong independent predictor of overall survival even after adjusting for clinical covariates and standard radiographic response (RECIST).
*   **Interpretability:** By design, SHERLOC links longitudinal ctDNA evolution to prognosis through a transparent Cox proportional hazards layer, identifying key drivers.

---

## 🛠 Technical Setup & Installation

To ensure the DL_models package is recognized correctly across the project and to avoid ModuleNotFoundError, follow these installation steps:

### 1. Clone & Environment

```bash
git clone https://github.com/AaronMamannToledano/SHERLOC.git
cd SHERLOC
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install SHERLOC Package

Install the repository in editable mode. This allows the scripts in the `scripts/` folder to import from `DL_models` seamlessly:

```bash
pip install -e .
```

## 💻 Usage

To run one of the proposed deep learning models, use the module flag from the project root. This ensures all internal paths are resolved correctly:

```bash
python -m scripts.train_DL --config path/to/your_config.yaml
```

To train the classical baselines or joint models instead:

```bash
python -m scripts.train_ML       --config scripts/config.yaml   # Penalized Cox + RSF
python -m scripts.train_ML_FPCA  --config scripts/config.yaml   # FPCA + penalized Cox
Rscript scripts/train_jm.R                                      # Joint models
```

---

## 🔬 Reproducing the paper

| Figure / table | Script |
| --- | --- |
| Deep-learning benchmark (iAUC, C-index, IBS) | `scripts/train_DL.py` |
| Penalized Cox / RSF baselines         | `scripts/train_ML.py` |
| FPCA + penalized Cox baselines        | `scripts/train_ML_FPCA.py` |
| Joint-model baselines                 | `scripts/train_jm.R` |
| SHERLOC interpretability plots        | `scripts/explainability.py` |

All scripts read hyperparameters from `scripts/config.yaml`; copy it and pass `--config`
to vary settings without editing the default file.

---

## 📊 Data

SHERLOC was developed on two cohorts. Neither is redistributed with this repository:

- **MSK-CHORD** — used to pre-train the Genomic Interaction Encoder. Available from
  [cBioPortal](https://www.cbioportal.org/study/summary?id=msk_chord_2024) under their
  data-use terms. Place the unpacked files under `scripts/data/msk_chord/`.
- **IMpower150 (target cohort)** — longitudinal ctDNA + survival data. Access is
  controlled by the original study sponsor; see `scripts/data/target/README.md` for
  the expected file layout.

---

## 🧪 Development

```bash
pip install -e ".[dev]"
ruff check .
black --check .
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.

---

## 📄 Citation

The SHERLOC manuscript is currently **under peer review**. Until the paper is
out, please cite the software directly — GitHub's "Cite this repository"
button reads from [`CITATION.cff`](CITATION.cff). A BibTeX entry is also
provided below; replace it with the published reference once the article is
accepted.

```bibtex

@article {mamann_sherloc_2026,
	author = {Mamann, Aaron and Das, Jyotishka and Benkirane, Hakim and Bugiotti, Francesca and Bernard, Elsa and Besse, Benjamin and Michiels, Stefan and Cournede, Paul-Henry},
	title = {SHERLOC: An interpretable deep learning model for longitudinal circulating tumor DNA data in survival analysis},
	elocation-id = {2026.06.04.730097},
	year = {2026},
	doi = {10.64898/2026.06.04.730097},
	URL = {https://www.biorxiv.org/content/early/2026/06/09/2026.06.04.730097},
	eprint = {https://www.biorxiv.org/content/early/2026/06/09/2026.06.04.730097.full.pdf},
	journal = {bioRxiv}
}


```

## 📁 Project Structure
```text
SHERLOC/
├── sherloc/                              # Core architecture and encoders
│   ├── __init__.py
│   ├── ctDNA_Dynamics_Encoder/
│   │   ├── __init__.py
│   │   ├── SHERLOC.py                    # Proposed Method
│   │   ├── Featurewise_Transformer.py    # Feature-wise Transformer
│   │   ├── Temporal_Transformer.py       # Temporal Transformer
│   │   ├── GRU.py                        # Gated Recurrent Unit
│   │   ├── RNN.py                        # Recurrent Neural Network
│   │   └── MLP_DeepSurv.py               # Multi-Layer Perceptron
│   ├── Genomic_Interaction_Encoder/
│   │   ├── __init__.py
│   │   └── MLP_SolidBiopsy.py            # Pre-trained tissue encoder
│   └── utils.py                          # Attention-based architecture
│
├── scripts/
│   ├── __init__.py
│   ├── config.yaml                       # Configuration file for all scripts
│   ├── train_DL.py                       # Training & evaluation — Deep-Learning models
│   ├── train_ML.py                       # Training & evaluation — penalized Cox / RSF
│   ├── train_ML_FPCA.py                  # Training & evaluation — FPCA + penalized Cox
│   ├── train_jm.R                        # Training & evaluation — joint models
│   ├── prepare_jm_data.py                # Data preparation for joint-model training
│   ├── explainability.py                 # Explainability / interpretability for SHERLOC
│   ├── utils.py                          # Data loading, pretraining, latent feature extraction
│   ├── results/                          # Cross-validated metrics (CSV outputs)
│   └── data/
│       ├── analysis_main.R               # R analysis entry point
│       ├── config.R                      # R configuration
│       ├── msk_chord/                    # Pre-training cohort (MSK-CHORD)
│       │   └── README.md
│       └── target/                       # Target cohort with longitudinal ctDNA (e.g., IMpower150)
│           └── README.md
│
├── tests/                                # Smoke tests run in CI
│   ├── __init__.py
│   └── test_smoke.py
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml                        # Lint + pytest on push / PR
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   ├── feature_request.yml
│   │   └── config.yml
│   └── pull_request_template.md
│
├── pyproject.toml                        # Build, dependencies, ruff / black / pytest config
├── requirements.txt                      # Pinned runtime dependencies
├── setup.py                              # Compatibility stub (config lives in pyproject.toml)
├── CITATION.cff                          # "Cite this repository" metadata
├── CONTRIBUTING.md                       # Contributor guide
├── REPO_CLEANUP.md                       # One-time housekeeping checklist
├── LICENSE
├── .gitignore
└── README.md
```

---

## 📜 License

Released under the [MIT License](LICENSE).
