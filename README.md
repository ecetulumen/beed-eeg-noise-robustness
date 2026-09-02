# BEED EEG Classification Robustness under AWGN

This project evaluates the clean-data performance and additive white Gaussian noise (AWGN) robustness of machine-learning and deep-learning classifiers on the Bangalore EEG Epilepsy Dataset (BEED).

The experiment compares two training strategies:

- **Clean training:** models are trained only on the clean training split.
- **Robust training:** the clean training split is augmented with noisy copies at 30, 20, and 10 dB SNR.

Both model groups are evaluated on clean data and on independently generated noisy test sets. The noisy evaluation is repeated five times at each SNR level.

## Models

- Support Vector Machine (RBF kernel)
- Random Forest
- XGBoost
- Multilayer Perceptron (TensorFlow/Keras)

## Dataset

The included cleaned dataset contains 7,729 rows, 16 EEG input channels (`X1`–`X16`), and one target column (`y`). It has no missing or duplicate rows.

| Label | Class | Rows in cleaned file |
|---:|---|---:|
| 0 | Healthy subjects | 1,762 |
| 1 | Generalized seizures | 1,999 |
| 2 | Focal seizures | 1,983 |
| 3 | Seizure events with associated activities | 1,985 |

Source: [UCI Machine Learning Repository — BEED: Bangalore EEG Epilepsy Dataset](https://archive.ics.uci.edu/dataset/1134/beed%3A%2Bbangalore%2Beeg%2Bepilepsy%2Bdataset)

DOI: [10.24432/C5K33B](https://doi.org/10.24432/C5K33B)

The original dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The copy in `data/BEED_Data_clean.csv` is a cleaned version of that dataset. See [`data/README.md`](data/README.md) for attribution and details.

## Method

1. Perform a stratified 80/20 train-test split with random seed 42.
2. Fit a `RobustScaler` on the training split only and transform the test split with the fitted scaler.
3. Clip scaled values to the interval `[-4, 4]`.
4. Train baseline models on clean training data.
5. Generate robust training data by adding AWGN copies at 30, 20, and 10 dB.
6. Train the same model families on the augmented training data.
7. Evaluate all models on the clean test split and on noisy test sets at 30, 20, and 10 dB.
8. Report accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, specificity, Cohen's kappa, MCC, ROC-AUC, and PR-AUC.

The script saves CSV reports, confusion matrices, robustness curves, MLP learning curves, ROC/PR curves, and an automatically generated summary under `results/`.

## Repository structure

```text
.
├── data/
│   ├── BEED_Data_clean.csv
│   └── README.md
├── results/
│   └── .gitkeep
├── src/
│   └── beed_robustness_analysis.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Python 3.10–3.12 is recommended.

```bash
git clone https://github.com/ecetulumen/beed-eeg-noise-robustness.git
cd beed-eeg-noise-robustness
python -m venv .venv
```

Activate the virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

From the repository root, run:

```bash
python src/beed_robustness_analysis.py
```

By default, the script reads `data/BEED_Data_clean.csv` and writes all outputs to `results/`. Alternative locations can be provided through environment variables:

```bash
BEED_DATA_PATH=/path/to/data.csv BEED_OUTPUT_DIR=/path/to/results python src/beed_robustness_analysis.py
```

## Important limitation

The cleaned CSV does not include subject identifiers. Therefore, the current stratified split is row-based rather than subject-independent. Samples originating from the same participant may occur in both the training and test sets, so the reported scores should be interpreted as exploratory sample-level performance rather than evidence of subject-level clinical generalization.

This repository is intended for research and educational use and is not a medical diagnostic system.

## Dataset citation

> Najmusseher, & Banu P. K., N. (2024). *BEED: Bangalore EEG Epilepsy Dataset* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K33B
