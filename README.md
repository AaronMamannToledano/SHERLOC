# SHERLOC

<img width="500" alt="SHERLOC_Logo" src="https://github.com/user-attachments/assets/df1d6ac3-1371-47b8-a50d-7705dfcc757a" />

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

## 📁 Project Structure
```text
SHERLOC/
├── sherloc/                            # Core architecture and encoders
│   ├── __init__.py
│   ├── ctDNA_Dynamics_Encoder/
│   │   ├── __init__.py                  
│   │   ├── SHERLOC.py                  # Proposed Method
│   │   ├── Featurewise_Transformer.py  # Feature-wise Transformer 
│   │   ├── GRU.py                      # Gated Recurrent Unit
│   │   ├── MLP_DeepSurv.py             # Multi-Layer Perceptron
│   │   ├── RNN.py                      # Recurrent Neural Network
│   │   └── Temporal_Transformer.py.    # Temporal Transformer
│   ├── Genomic_Interaction_Encoder/
│   │   ├── __init__.py
│   │   └── MLP_SolidBiopsy.py          # Pre-trained tissue encoder
│   └── utils.py                        # Attention-based architecture 
│
├── scripts/                        
│   ├── __init__.py
│   ├── data/
│   │    ├── msk_chord/                # Pre-training cohort (MSK-CHORD)
│   │    │    └── README.md
│   │    ├── target/                   # Cohort with longitudinal ctDNA data (e.g., IMpower150)
│   │    └──  └── README.md       
│   ├── config.yaml                    # Configuration file for all scripts
│   ├── explainability.py              # Explainability script for SHERLOC
│   ├── train_DL.py                    # Training and evaluation scripts for Deep-Learning models
│   ├── train_ML.py                    # Training and evaluation scripts for penalized Cox models and RSF 
│   ├── train_jm.R                     # Training and evaluation scripts for joint models
│   └── utils.py                       # Data loading, pretraining, and latent feature extraction utilities 
├── LICENSE      
├── setup.py                           # Project installation script
└── README.md
```

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


