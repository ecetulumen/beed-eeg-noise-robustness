"""BEED clean-vs-robust training and AWGN robustness analysis.

Run this file from the repository root after installing ``requirements.txt``.
The default input and output locations can be overridden with the
``BEED_DATA_PATH`` and ``BEED_OUTPUT_DIR`` environment variables.
"""

import os
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc
)

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

try:
    from IPython.display import display
except ImportError:
    def display(value):
        """Print tabular output when IPython is unavailable."""
        print(value)


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(
    os.environ.get("BEED_DATA_PATH", PROJECT_ROOT / "data" / "BEED_Data_clean.csv")
)
OUTPUT_DIR = Path(
    os.environ.get("BEED_OUTPUT_DIR", PROJECT_ROOT / "results")
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("DATA_PATH:", DATA_PATH)
print("Data file found?:", DATA_PATH.exists())
print("OUTPUT_DIR:", OUTPUT_DIR)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. Set BEED_DATA_PATH to use another file."
    )

RANDOM_STATE = 42
TEST_SIZE = 0.20

SCALER_TYPE = "robust"   # "robust" veya "standard"
USE_FEATURE_CLIPPING = True
CLIP_VALUE = 4.0

# Sadece istediğin AWGN senaryoları
SNR_LEVELS = [30, 20, 10]
N_NOISE_REPEATS = 5

# Robust train için eğitime eklenecek gürültülü kopyalar
ROBUST_TRAIN_SNR_LEVELS = [30, 20, 10]
ROBUST_REPEAT_MAP = {
    30: 1,
    20: 2,
    10: 3
}

MLP_EPOCHS = 150
MLP_BATCH_SIZE = 32
MLP_VALIDATION_SPLIT = 0.15

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(RANDOM_STATE)


def print_section(title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def save_df(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_fig(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {path}")


def display_and_save(df, title, filename):
    print_section(title)
    display(df)
    save_df(df, filename)


def apply_clipping_if_needed(X):
    if USE_FEATURE_CLIPPING:
        return np.clip(X, -CLIP_VALUE, CLIP_VALUE)
    return X


def create_scaler():
    if SCALER_TYPE.lower() == "robust":
        print("Scaler: RobustScaler")
        return RobustScaler()
    else:
        print("Scaler: StandardScaler")
        return StandardScaler()


# ============================================================
# 3. NOISE FUNCTIONS
# ============================================================

def add_awgn_noise(X, snr_db, random_state=42):
    rng = np.random.default_rng(random_state)

    X_noisy = X.copy()
    signal_power = np.mean(X_noisy ** 2)

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = rng.normal(
        loc=0,
        scale=np.sqrt(noise_power),
        size=X_noisy.shape
    )

    X_noisy = X_noisy + noise
    X_noisy = apply_clipping_if_needed(X_noisy)

    return X_noisy


def create_robust_training_data(X_train, y_train):
    X_parts = [X_train]
    y_parts = [y_train]

    for snr in ROBUST_TRAIN_SNR_LEVELS:
        repeat_count = ROBUST_REPEAT_MAP.get(snr, 1)

        for rep in range(repeat_count):
            seed = RANDOM_STATE + snr * 100 + rep
            X_noisy = add_awgn_noise(X_train, snr_db=snr, random_state=seed)

            X_parts.append(X_noisy)
            y_parts.append(y_train)

    X_robust = np.vstack(X_parts)
    y_robust = np.concatenate(y_parts)

    return X_robust, y_robust


# ============================================================
# 4. METRIC FUNCTIONS
# ============================================================

def get_specificity_per_class(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    specificities = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP

        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificities.append(specificity)

    return specificities


def safe_roc_auc(y_true, y_prob, num_classes):
    try:
        if y_prob is None:
            return np.nan

        if num_classes == 2:
            return roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            return roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
    except:
        return np.nan


def safe_pr_auc(y_true, y_prob, num_classes):
    try:
        if y_prob is None:
            return np.nan

        if num_classes == 2:
            return average_precision_score(y_true, y_prob[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            return average_precision_score(y_true_bin, y_prob, average="macro")
    except:
        return np.nan


def calculate_metrics(y_true, y_pred, y_prob, num_classes):
    specificity_per_class = get_specificity_per_class(y_true, y_pred, num_classes)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall / Sensitivity": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Mean Specificity": np.mean(specificity_per_class),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC AUC": safe_roc_auc(y_true, y_prob, num_classes),
        "PR AUC": safe_pr_auc(y_true, y_prob, num_classes)
    }

    return metrics


def predict_model(model, X_data, model_type):
    if model_type == "mlp":
        y_prob = model.predict(X_data, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred, y_prob

    y_pred = model.predict(X_data)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_data)
    else:
        y_prob = None

    return y_pred, y_prob


# ============================================================
# 5. MODEL DEFINITIONS
# ============================================================

def create_ml_models(num_classes):
    models = {}

    models["SVM"] = SVC(
        kernel="rbf",
        C=80.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE
    )

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    if num_classes == 2:
        models["XGBoost"] = XGBClassifier(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=1,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.5,
            reg_alpha=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        models["XGBoost"] = XGBClassifier(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=1,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.5,
            reg_alpha=0.05,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=num_classes,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    return models


def build_mlp(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.35),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.30),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.20),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 6. TRAINING FUNCTIONS
# ============================================================

def train_ml_models(X_train, y_train, training_type, num_classes):
    print_section(f"{training_type} - MACHINE LEARNING TRAINING")

    models = create_ml_models(num_classes)
    trained_models = {}

    for model_name, model in models.items():
        print(f"\n{model_name} training started...")
        model.fit(X_train, y_train)

        trained_models[model_name] = {
            "model": model,
            "type": "ml",
            "training_type": training_type
        }

        print(f"{model_name} training completed.")

    return trained_models


def train_mlp(X_train, y_train, input_dim, num_classes, training_type):
    print_section(f"{training_type} - MLP TRAINING")

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    model = build_mlp(input_dim, num_classes)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-5
        )
    ]

    history = model.fit(
        X_train,
        y_train_cat,
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        validation_split=MLP_VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    return {
        "model": model,
        "type": "mlp",
        "training_type": training_type
    }, history


# ============================================================
# 7. LOAD DATA
# ============================================================

print_section("DATASET INFORMATION")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

if "y" not in df.columns:
    raise ValueError("Target column must be named 'y'.")

df["y"] = df["y"].astype(int)

print("\nClass distribution:")
display(df["y"].value_counts().sort_index().to_frame("Count"))

X = df.drop("y", axis=1)
y = df["y"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(np.unique(y_encoded))
input_dim = X.shape[1]

print("\nNumber of features:", input_dim)
print("Number of classes:", num_classes)
print("Class labels:", label_encoder.classes_)


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)


# ============================================================
# 9. SCALING WITHOUT DATA LEAKAGE
# ============================================================

scaler = create_scaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = apply_clipping_if_needed(X_train_scaled)
X_test_scaled = apply_clipping_if_needed(X_test_scaled)

print("\nScaling completed without data leakage.")
print("Feature clipping:", USE_FEATURE_CLIPPING)


# ============================================================
# 10. CREATE ROBUST TRAINING DATA
# ============================================================

X_train_robust, y_train_robust = create_robust_training_data(
    X_train_scaled,
    y_train
)

print_section("ROBUST TRAINING DATA INFORMATION")
print("Clean train shape :", X_train_scaled.shape)
print("Robust train shape:", X_train_robust.shape)

print("\nRobust training includes:")
print("- Original clean train data")
for snr in ROBUST_TRAIN_SNR_LEVELS:
    print(f"- AWGN {snr} dB copies x {ROBUST_REPEAT_MAP[snr]}")


# ============================================================
# 11. TRAIN MODELS
# ============================================================

clean_models = train_ml_models(
    X_train_scaled,
    y_train,
    training_type="Clean Train",
    num_classes=num_classes
)

mlp_clean, hist_clean = train_mlp(
    X_train_scaled,
    y_train,
    input_dim,
    num_classes,
    training_type="Clean Train"
)

clean_models["MLP"] = mlp_clean

robust_models = train_ml_models(
    X_train_robust,
    y_train_robust,
    training_type="Robust Train",
    num_classes=num_classes
)

mlp_robust, hist_robust = train_mlp(
    X_train_robust,
    y_train_robust,
    input_dim,
    num_classes,
    training_type="Robust Train"
)

robust_models["MLP"] = mlp_robust


# ============================================================
# 12. EVALUATION FUNCTIONS
# ============================================================

def evaluate_model_group(model_group, X_eval, y_eval, test_type, condition, snr_level=None, repeat=None):
    rows = []
    predictions = {}
    probabilities = {}

    for base_model_name, item in model_group.items():
        model = item["model"]
        model_type = item["type"]
        training_type = item["training_type"]

        y_pred, y_prob = predict_model(model, X_eval, model_type)

        metrics = calculate_metrics(
            y_true=y_eval,
            y_pred=y_pred,
            y_prob=y_prob,
            num_classes=num_classes
        )

        row = {
            "Model": base_model_name,
            "Training Type": training_type,
            "Test Type": test_type,
            "Condition": condition,
            "SNR": snr_level,
            "Repeat": repeat
        }

        row.update(metrics)
        rows.append(row)

        full_name = f"{base_model_name} | {training_type} | {condition}"
        predictions[full_name] = y_pred
        probabilities[full_name] = y_prob

    return pd.DataFrame(rows), predictions, probabilities


def evaluate_clean_test():
    all_rows = []
    all_preds = {}
    all_probs = {}

    df_clean, preds_clean, probs_clean = evaluate_model_group(
        clean_models,
        X_test_scaled,
        y_test,
        test_type="Clean Test",
        condition="Clean Train -> Clean Test"
    )

    df_robust, preds_robust, probs_robust = evaluate_model_group(
        robust_models,
        X_test_scaled,
        y_test,
        test_type="Clean Test",
        condition="Robust Train -> Clean Test"
    )

    all_rows.append(df_clean)
    all_rows.append(df_robust)

    all_preds.update(preds_clean)
    all_preds.update(preds_robust)

    all_probs.update(probs_clean)
    all_probs.update(probs_robust)

    return pd.concat(all_rows, ignore_index=True), all_preds, all_probs


def evaluate_noisy_test():
    all_rows = []

    for snr in SNR_LEVELS:
        for repeat in range(1, N_NOISE_REPEATS + 1):
            seed = RANDOM_STATE + snr * 100 + repeat
            X_test_noisy = add_awgn_noise(X_test_scaled, snr_db=snr, random_state=seed)

            df_clean_noisy, _, _ = evaluate_model_group(
                clean_models,
                X_test_noisy,
                y_test,
                test_type="Noisy Test",
                condition="Clean Train -> Noisy Test",
                snr_level=snr,
                repeat=repeat
            )

            df_robust_noisy, _, _ = evaluate_model_group(
                robust_models,
                X_test_noisy,
                y_test,
                test_type="Noisy Test",
                condition="Robust Train -> Noisy Test",
                snr_level=snr,
                repeat=repeat
            )

            all_rows.append(df_clean_noisy)
            all_rows.append(df_robust_noisy)

    return pd.concat(all_rows, ignore_index=True)


# ============================================================
# 13. RUN EVALUATIONS
# ============================================================

clean_test_results_df, clean_test_predictions, clean_test_probabilities = evaluate_clean_test()
noisy_test_detailed_df = evaluate_noisy_test()

display_and_save(
    clean_test_results_df.sort_values(["Condition", "Macro F1"], ascending=[True, False]),
    "CLEAN TEST RESULTS",
    "01_clean_test_results.csv"
)

display_and_save(
    noisy_test_detailed_df.head(),
    "NOISY TEST DETAILED RESULTS - FIRST ROWS",
    "02_noisy_test_detailed_results.csv"
)


# ============================================================
# 14. NOISY TEST SUMMARY
# ============================================================

noisy_summary_df = noisy_test_detailed_df.groupby(
    ["Model", "Training Type", "Test Type", "Condition", "SNR"],
    as_index=False
).agg(
    Accuracy_Mean=("Accuracy", "mean"),
    Accuracy_Std=("Accuracy", "std"),
    Balanced_Accuracy_Mean=("Balanced Accuracy", "mean"),
    Balanced_Accuracy_Std=("Balanced Accuracy", "std"),
    Macro_Precision_Mean=("Macro Precision", "mean"),
    Macro_Recall_Mean=("Macro Recall / Sensitivity", "mean"),
    Macro_F1_Mean=("Macro F1", "mean"),
    Macro_F1_Std=("Macro F1", "std"),
    Weighted_F1_Mean=("Weighted F1", "mean"),
    Weighted_F1_Std=("Weighted F1", "std"),
    Mean_Specificity_Mean=("Mean Specificity", "mean"),
    Cohen_Kappa_Mean=("Cohen Kappa", "mean"),
    MCC_Mean=("MCC", "mean"),
    ROC_AUC_Mean=("ROC AUC", "mean"),
    PR_AUC_Mean=("PR AUC", "mean")
)

noisy_summary_df["Macro_F1_95CI"] = 1.96 * noisy_summary_df["Macro_F1_Std"] / np.sqrt(N_NOISE_REPEATS)
noisy_summary_df["Accuracy_95CI"] = 1.96 * noisy_summary_df["Accuracy_Std"] / np.sqrt(N_NOISE_REPEATS)

display_and_save(
    noisy_summary_df.sort_values(["SNR", "Macro_F1_Mean"], ascending=[False, False]),
    "NOISY TEST SUMMARY",
    "03_noisy_test_summary.csv"
)


# ============================================================
# 15. SNR-BASED BEST MODELS
# ============================================================

snr_best_rows = []

for snr in SNR_LEVELS:
    temp = noisy_summary_df[noisy_summary_df["SNR"] == snr].copy()
    best_row = temp.sort_values("Macro_F1_Mean", ascending=False).iloc[0]

    snr_best_rows.append({
        "SNR (dB)": snr,
        "Best Model": best_row["Model"],
        "Training Type": best_row["Training Type"],
        "Condition": best_row["Condition"],
        "Accuracy Mean": best_row["Accuracy_Mean"],
        "Balanced Accuracy Mean": best_row["Balanced_Accuracy_Mean"],
        "Macro Precision Mean": best_row["Macro_Precision_Mean"],
        "Macro Recall Mean": best_row["Macro_Recall_Mean"],
        "Macro F1 Mean": best_row["Macro_F1_Mean"],
        "Macro F1 95% CI": best_row["Macro_F1_95CI"],
        "Weighted F1 Mean": best_row["Weighted_F1_Mean"],
        "Mean Specificity": best_row["Mean_Specificity_Mean"],
        "Cohen Kappa Mean": best_row["Cohen_Kappa_Mean"],
        "MCC Mean": best_row["MCC_Mean"],
        "ROC AUC Mean": best_row["ROC_AUC_Mean"],
        "PR AUC Mean": best_row["PR_AUC_Mean"]
    })

snr_best_df = pd.DataFrame(snr_best_rows)

display_and_save(
    snr_best_df.sort_values("SNR (dB)", ascending=False),
    "BEST MODEL FOR EACH SNR LEVEL",
    "04_best_model_for_each_snr.csv"
)


# ============================================================
# 16. ROBUST TRAINING GAIN PER SNR
# ============================================================

snr_gain_rows = []

for snr in SNR_LEVELS:
    for model_name in noisy_summary_df["Model"].unique():

        clean_row = noisy_summary_df[
            (noisy_summary_df["SNR"] == snr) &
            (noisy_summary_df["Model"] == model_name) &
            (noisy_summary_df["Condition"] == "Clean Train -> Noisy Test")
        ]

        robust_row = noisy_summary_df[
            (noisy_summary_df["SNR"] == snr) &
            (noisy_summary_df["Model"] == model_name) &
            (noisy_summary_df["Condition"] == "Robust Train -> Noisy Test")
        ]

        if len(clean_row) == 0 or len(robust_row) == 0:
            continue

        clean_row = clean_row.iloc[0]
        robust_row = robust_row.iloc[0]

        snr_gain_rows.append({
            "SNR (dB)": snr,
            "Model": model_name,
            "Clean Train Macro F1": clean_row["Macro_F1_Mean"],
            "Robust Train Macro F1": robust_row["Macro_F1_Mean"],
            "Macro F1 Gain": robust_row["Macro_F1_Mean"] - clean_row["Macro_F1_Mean"],
            "Clean Train Accuracy": clean_row["Accuracy_Mean"],
            "Robust Train Accuracy": robust_row["Accuracy_Mean"],
            "Accuracy Gain": robust_row["Accuracy_Mean"] - clean_row["Accuracy_Mean"],
            "Clean Train MCC": clean_row["MCC_Mean"],
            "Robust Train MCC": robust_row["MCC_Mean"],
            "MCC Gain": robust_row["MCC_Mean"] - clean_row["MCC_Mean"]
        })

snr_gain_df = pd.DataFrame(snr_gain_rows)

display_and_save(
    snr_gain_df.sort_values(["SNR (dB)", "Macro F1 Gain"], ascending=[False, False]),
    "ROBUST TRAINING GAIN BY SNR",
    "05_robust_training_gain_by_snr.csv"
)


# ============================================================
# 17. OVERALL NOISE ROBUSTNESS RANKING
# ============================================================

noise_robustness_rows = []

for model_name in noisy_summary_df["Model"].unique():
    for training_type in noisy_summary_df["Training Type"].unique():

        temp = noisy_summary_df[
            (noisy_summary_df["Model"] == model_name) &
            (noisy_summary_df["Training Type"] == training_type)
        ].copy()

        if len(temp) == 0:
            continue

        mean_noisy_f1 = temp["Macro_F1_Mean"].mean()
        worst_noisy_f1 = temp["Macro_F1_Mean"].min()
        mean_noisy_acc = temp["Accuracy_Mean"].mean()
        worst_noisy_acc = temp["Accuracy_Mean"].min()

        noise_robustness_score = 0.60 * mean_noisy_f1 + 0.40 * worst_noisy_f1

        noise_robustness_rows.append({
            "Model": model_name,
            "Training Type": training_type,
            "Mean Noisy Macro F1": mean_noisy_f1,
            "Worst Noisy Macro F1": worst_noisy_f1,
            "Mean Noisy Accuracy": mean_noisy_acc,
            "Worst Noisy Accuracy": worst_noisy_acc,
            "Noise Robustness Score": noise_robustness_score
        })

noise_robustness_df = pd.DataFrame(noise_robustness_rows)

display_and_save(
    noise_robustness_df.sort_values("Noise Robustness Score", ascending=False),
    "OVERALL NOISE ROBUSTNESS RANKING",
    "06_overall_noise_robustness_ranking.csv"
)


# ============================================================
# 18. CLEAN BASELINE + FINAL BEST MODEL DECISION
# ============================================================

clean_baseline_df = clean_test_results_df[
    clean_test_results_df["Condition"] == "Clean Train -> Clean Test"
].copy()

best_clean_baseline_row = clean_baseline_df.sort_values("Macro F1", ascending=False).iloc[0]

best_noise_model = noise_robustness_df.sort_values(
    "Noise Robustness Score",
    ascending=False
).iloc[0]

print_section("FINAL DECISION")

print(f"Best clean baseline model: {best_clean_baseline_row['Model']}")
print(f"Clean Train -> Clean Test Macro F1: {best_clean_baseline_row['Macro F1']:.4f}")

print("\nNoise robustness açısından en iyi model:")
print(f"Model          : {best_noise_model['Model']}")
print(f"Training Type  : {best_noise_model['Training Type']}")
print(f"Mean Noisy Macro F1    : {best_noise_model['Mean Noisy Macro F1']:.4f}")
print(f"Worst Noisy Macro F1   : {best_noise_model['Worst Noisy Macro F1']:.4f}")
print(f"Noise Robustness Score : {best_noise_model['Noise Robustness Score']:.4f}")


# ============================================================
# 19. CONFUSION MATRIX PLOTTING
# İSTEDİĞİN FORMAT
# ============================================================

def plot_clean_test_confusion_grid_only_clean_models(y_true, predictions, title, filename):
    model_order = [
        "SVM | Clean Train | Clean Train -> Clean Test",
        "Random Forest | Clean Train | Clean Train -> Clean Test",
        "XGBoost | Clean Train | Clean Train -> Clean Test",
        "MLP | Clean Train | Clean Train -> Clean Test"
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8), constrained_layout=True)
    classes = np.arange(num_classes)

    for idx, model_name in enumerate(model_order):
        ax = axes[idx]

        if model_name not in predictions:
            ax.axis("off")
            ax.set_title("Prediction not found", fontsize=11)
            continue

        y_pred = predictions[model_name]

        cm = confusion_matrix(y_true, y_pred, labels=classes)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        im = ax.imshow(cm_norm, vmin=0, vmax=1)

        short_title = model_name.split("|")[0].strip()
        acc = accuracy_score(y_true, y_pred)

        ax.set_title(f"{short_title}\nAccuracy = {acc*100:.2f}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_xticks(classes)
        ax.set_yticks(classes)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                value = cm_norm[i, j]
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center",
                    color="white" if value > 0.55 else "black",
                    fontsize=9,
                    fontweight="bold" if value > 0.50 else "normal"
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=18, fontweight="bold")
    save_fig(filename)


def plot_awgn_clean_vs_robust_grid(y_true, predictions, snr, filename):
    top_row = [
        "SVM | Clean Train | Clean Train -> Noisy Test",
        "Random Forest | Clean Train | Clean Train -> Noisy Test",
        "XGBoost | Clean Train | Clean Train -> Noisy Test",
        "MLP | Clean Train | Clean Train -> Noisy Test"
    ]

    bottom_row = [
        "SVM | Robust Train | Robust Train -> Noisy Test",
        "Random Forest | Robust Train | Robust Train -> Noisy Test",
        "XGBoost | Robust Train | Robust Train -> Noisy Test",
        "MLP | Robust Train | Robust Train -> Noisy Test"
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True)
    classes = np.arange(num_classes)

    all_rows = [top_row, bottom_row]

    for row_idx, row_models in enumerate(all_rows):
        for col_idx, model_name in enumerate(row_models):
            ax = axes[row_idx, col_idx]

            if model_name not in predictions:
                ax.axis("off")
                ax.set_title("Prediction not found", fontsize=11)
                continue

            y_pred = predictions[model_name]

            cm = confusion_matrix(y_true, y_pred, labels=classes)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

            im = ax.imshow(cm_norm, vmin=0, vmax=1)

            short_title = model_name.split("|")[0].strip()
            acc = accuracy_score(y_true, y_pred)

            if row_idx == 0:
                title_prefix = f"{short_title} Clean"
            else:
                title_prefix = f"{short_title} Robust"

            ax.set_title(f"{title_prefix}\nAccuracy = {acc*100:.2f}%", fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("True Class")
            ax.set_xticks(classes)
            ax.set_yticks(classes)
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)

            for i in range(cm_norm.shape[0]):
                for j in range(cm_norm.shape[1]):
                    value = cm_norm[i, j]
                    ax.text(
                        j, i, f"{value:.2f}",
                        ha="center", va="center",
                        color="white" if value > 0.55 else "black",
                        fontsize=9,
                        fontweight="bold" if value > 0.50 else "normal"
                    )

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"GÜRÜLTÜLÜ SONUÇLAR {snr} dB\nRobustness Test - AWGN {snr} dB",
                 fontsize=18, fontweight="bold")
    save_fig(filename)


def get_clean_test_predictions_for_clean_models_only(clean_models, X_data):
    preds = {}

    for base_model_name, item in clean_models.items():
        model = item["model"]
        model_type = item["type"]

        y_pred, _ = predict_model(model, X_data, model_type)
        key = f"{base_model_name} | Clean Train | Clean Train -> Clean Test"
        preds[key] = y_pred

    return preds


def get_awgn_predictions_for_grid(clean_models, robust_models, X_test_scaled, snr):
    X_test_noisy = add_awgn_noise(X_test_scaled, snr_db=snr, random_state=RANDOM_STATE)
    preds = {}

    for base_model_name, item in clean_models.items():
        model = item["model"]
        model_type = item["type"]

        y_pred, _ = predict_model(model, X_test_noisy, model_type)
        key = f"{base_model_name} | Clean Train | Clean Train -> Noisy Test"
        preds[key] = y_pred

    for base_model_name, item in robust_models.items():
        model = item["model"]
        model_type = item["type"]

        y_pred, _ = predict_model(model, X_test_noisy, model_type)
        key = f"{base_model_name} | Robust Train | Robust Train -> Noisy Test"
        preds[key] = y_pred

    return preds


# ============================================================
# 20. OTHER PLOTS
# ============================================================

def plot_mlp_history(history, title_prefix, filename_prefix):
    hist = history.history

    plt.figure(figsize=(8, 4))
    plt.plot(hist["accuracy"], label="Train Accuracy")
    plt.plot(hist["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy")
    plt.grid(alpha=0.4)
    plt.legend()
    save_fig(f"{filename_prefix}_accuracy.png")

    plt.figure(figsize=(8, 4))
    plt.plot(hist["loss"], label="Train Loss")
    plt.plot(hist["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.grid(alpha=0.4)
    plt.legend()
    save_fig(f"{filename_prefix}_loss.png")


def plot_snr_bar(noisy_summary_df, snr):
    temp = noisy_summary_df[noisy_summary_df["SNR"] == snr].copy()
    temp["Model Label"] = temp["Model"] + "\n" + temp["Training Type"]
    temp = temp.sort_values("Macro_F1_Mean", ascending=False)

    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        temp["Model Label"],
        temp["Macro_F1_Mean"],
        yerr=temp["Macro_F1_95CI"],
        capsize=4
    )

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Macro F1 Mean ± 95% CI")
    plt.ylim(0, 1.05)
    plt.title(f"AWGN {snr} dB - Model Robustness Comparison")
    plt.grid(axis="y", alpha=0.4)

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    save_fig(f"awgn_{snr}db_model_comparison_bar.png")


def plot_all_models_noise_curve(noisy_summary_df):
    plt.figure(figsize=(13, 7))

    for model_name in noisy_summary_df["Model"].unique():
        for condition in ["Clean Train -> Noisy Test", "Robust Train -> Noisy Test"]:
            temp = noisy_summary_df[
                (noisy_summary_df["Model"] == model_name) &
                (noisy_summary_df["Condition"] == condition)
            ].copy()

            temp = temp.sort_values("SNR", ascending=False)

            label = f"{model_name} - {'Robust' if 'Robust' in condition else 'Clean'}"

            plt.plot(
                temp["SNR"],
                temp["Macro_F1_Mean"],
                marker="o",
                linewidth=2,
                label=label
            )

    plt.gca().invert_xaxis()
    plt.xlabel("SNR Level (dB)")
    plt.ylabel("Macro F1 Mean")
    plt.ylim(0, 1.05)
    plt.title("Noise Robustness Trend Across 30 dB, 20 dB and 10 dB")
    plt.grid(alpha=0.4)
    plt.legend(fontsize=8, ncol=2)

    save_fig("all_models_30_20_10db_noise_trend.png")


def plot_noise_robustness_ranking(noise_robustness_df):
    temp = noise_robustness_df.sort_values("Noise Robustness Score", ascending=False).copy()
    temp["Model Label"] = temp["Model"] + "\n" + temp["Training Type"]

    plt.figure(figsize=(12, 6))

    bars = plt.bar(temp["Model Label"], temp["Noise Robustness Score"])

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Noise Robustness Score")
    plt.ylim(0, 1.05)
    plt.title("Overall Noise Robustness Ranking")
    plt.grid(axis="y", alpha=0.4)

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    save_fig("overall_noise_robustness_ranking.png")


def save_table_as_png(df, title, filename, max_rows=20):
    temp = df.copy().head(max_rows)

    numeric_cols = temp.select_dtypes(include=[np.number]).columns
    temp[numeric_cols] = temp[numeric_cols].round(4)

    fig, ax = plt.subplots(figsize=(min(24, 2.2 * len(temp.columns)), 0.7 * len(temp) + 2.5))
    ax.axis("off")

    table = ax.table(
        cellText=temp.values,
        colLabels=temp.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.35)

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    save_fig(filename)


def plot_best_model_roc_pr(y_true, y_prob, title_prefix, filename_prefix):
    if y_prob is None:
        print("Probability output yok, ROC/PR çizilemedi.")
        return

    if num_classes == 2:
        y_score = y_prob[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc_value = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc_value = auc(recall, precision)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_value:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{title_prefix} ROC Curve")
        plt.grid(alpha=0.4)
        plt.legend()
        save_fig(f"{filename_prefix}_roc_curve.png")

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"PR-AUC = {pr_auc_value:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{title_prefix} Precision-Recall Curve")
        plt.grid(alpha=0.4)
        plt.legend()
        save_fig(f"{filename_prefix}_pr_curve.png")

    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc_value = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
        pr_auc_value = auc(recall, precision)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"Micro AUC = {roc_auc_value:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{title_prefix} Micro-Average ROC Curve")
        plt.grid(alpha=0.4)
        plt.legend()
        save_fig(f"{filename_prefix}_micro_roc_curve.png")

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"Micro PR-AUC = {pr_auc_value:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{title_prefix} Micro-Average Precision-Recall Curve")
        plt.grid(alpha=0.4)
        plt.legend()
        save_fig(f"{filename_prefix}_micro_pr_curve.png")


# ============================================================
# 21. GENERATE FIGURES
# ============================================================

plot_mlp_history(hist_clean, "Clean Train MLP", "clean_train_mlp")
plot_mlp_history(hist_robust, "Robust Train MLP", "robust_train_mlp")

# Clean train -> clean test (1x4)
clean_only_predictions = get_clean_test_predictions_for_clean_models_only(
    clean_models=clean_models,
    X_data=X_test_scaled
)

plot_clean_test_confusion_grid_only_clean_models(
    y_true=y_test,
    predictions=clean_only_predictions,
    title="CLEAN TRAIN - CLEAN TEST SONUÇLARI",
    filename="clean_train_clean_test_confusion_grid.png"
)

# 30-20-10 dB (üst clean / alt robust)
for snr in [30, 20, 10]:
    awgn_preds = get_awgn_predictions_for_grid(
        clean_models=clean_models,
        robust_models=robust_models,
        X_test_scaled=X_test_scaled,
        snr=snr
    )

    plot_awgn_clean_vs_robust_grid(
        y_true=y_test,
        predictions=awgn_preds,
        snr=snr,
        filename=f"awgn_{snr}db_clean_vs_robust_confusion_grid.png"
    )

# Diğer grafikler
for snr in SNR_LEVELS:
    plot_snr_bar(noisy_summary_df, snr)

plot_all_models_noise_curve(noisy_summary_df)
plot_noise_robustness_ranking(noise_robustness_df)

# Tabloları png olarak da kaydet
save_table_as_png(
    clean_test_results_df.sort_values(["Condition", "Macro F1"], ascending=[True, False]),
    "Clean Test Results",
    "clean_test_results_table.png"
)

save_table_as_png(
    noisy_summary_df.sort_values(["SNR", "Macro_F1_Mean"], ascending=[False, False]),
    "Noisy Test Summary",
    "noisy_test_summary_table.png"
)

save_table_as_png(
    snr_best_df.sort_values("SNR (dB)", ascending=False),
    "Best Model for Each SNR",
    "best_model_for_each_snr_table.png"
)

save_table_as_png(
    noise_robustness_df.sort_values("Noise Robustness Score", ascending=False),
    "Overall Noise Robustness Ranking",
    "overall_noise_robustness_ranking_table.png"
)


# ============================================================
# 22. BEST MODEL REPORTS + ERROR ANALYSIS
# ============================================================

best_model_name = best_noise_model["Model"]
best_training_type = best_noise_model["Training Type"]

if best_training_type == "Robust Train":
    best_model_item = robust_models[best_model_name]
else:
    best_model_item = clean_models[best_model_name]

best_model = best_model_item["model"]
best_model_type = best_model_item["type"]

# Clean testte en iyi robust/noise model raporu
best_clean_pred, best_clean_prob = predict_model(best_model, X_test_scaled, best_model_type)

print_section(f"CLASSIFICATION REPORT - BEST NOISE ROBUST MODEL ON CLEAN TEST: {best_model_name} ({best_training_type})")
clean_report = classification_report(y_test, best_clean_pred, zero_division=0)
print(clean_report)

with open(os.path.join(OUTPUT_DIR, "best_noise_model_clean_test_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(clean_report)

# 10 dB test
X_test_noisy_10db = add_awgn_noise(X_test_scaled, snr_db=10, random_state=RANDOM_STATE)
best_noisy_pred, best_noisy_prob = predict_model(best_model, X_test_noisy_10db, best_model_type)

print_section(f"CLASSIFICATION REPORT - BEST NOISE ROBUST MODEL ON 10 dB: {best_model_name} ({best_training_type})")
noisy_report = classification_report(y_test, best_noisy_pred, zero_division=0)
print(noisy_report)

with open(os.path.join(OUTPUT_DIR, "best_noise_model_10db_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(noisy_report)


def error_analysis_df(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    rows = []

    for cls in range(num_classes):
        total = cm[cls, :].sum()
        correct = cm[cls, cls]
        wrong = total - correct

        rows.append({
            "Class": cls,
            "Total Samples": total,
            "Correct": correct,
            "Wrong": wrong,
            "Class Accuracy": correct / total if total > 0 else 0
        })

    df_error = pd.DataFrame(rows)

    print_section(title)
    display(df_error)

    return df_error


error_clean_df = error_analysis_df(
    y_test,
    best_clean_pred,
    "ERROR ANALYSIS - BEST NOISE ROBUST MODEL ON CLEAN TEST"
)

error_noisy_df = error_analysis_df(
    y_test,
    best_noisy_pred,
    "ERROR ANALYSIS - BEST NOISE ROBUST MODEL ON 10 dB"
)

save_df(error_clean_df, "07_error_analysis_best_noise_model_clean_test.csv")
save_df(error_noisy_df, "08_error_analysis_best_noise_model_10db.csv")

save_table_as_png(
    error_clean_df,
    "Error Analysis - Best Noise Robust Model on Clean Test",
    "error_analysis_best_noise_model_clean_test.png"
)

save_table_as_png(
    error_noisy_df,
    "Error Analysis - Best Noise Robust Model on 10 dB",
    "error_analysis_best_noise_model_10db.png"
)

plot_best_model_roc_pr(
    y_test,
    best_clean_prob,
    f"Best Noise Robust Model Clean Test - {best_model_name}",
    "best_noise_model_clean_test"
)

plot_best_model_roc_pr(
    y_test,
    best_noisy_prob,
    f"Best Noise Robust Model 10 dB - {best_model_name}",
    "best_noise_model_10db"
)


# ============================================================
# 23. AUTOMATIC REPORT TEXT
# ============================================================

report_text = f"""
AUTOMATIC RESULT INTERPRETATION

In this project, the classification performance and noise robustness of four models were evaluated:
SVM, Random Forest, XGBoost, and MLP.

The experiment was designed in three main stages.

1) Clean Train -> Clean Test:
All models were first trained using only clean training data and evaluated on the clean test set.
This stage represents the baseline classification performance.

2) Clean Train -> Noisy Test:
The clean-trained models were then evaluated on noisy versions of the test set.
AWGN noise was added at 30 dB, 20 dB, and 10 dB levels.
This stage measured how strongly the models were affected by increasing noise.

3) Robust Train -> Noisy Test:
The training data were augmented with noisy copies and the models were retrained.
These robust-trained models were evaluated under the same 30 dB, 20 dB, and 10 dB noisy conditions.

The best clean baseline model was {best_clean_baseline_row['Model']}, with a Clean Train -> Clean Test Macro F1 score of {best_clean_baseline_row['Macro F1']:.4f}.

The best model at 30 dB was {snr_best_df[snr_best_df['SNR (dB)'] == 30].iloc[0]['Best Model']} ({snr_best_df[snr_best_df['SNR (dB)'] == 30].iloc[0]['Training Type']}).
The best model at 20 dB was {snr_best_df[snr_best_df['SNR (dB)'] == 20].iloc[0]['Best Model']} ({snr_best_df[snr_best_df['SNR (dB)'] == 20].iloc[0]['Training Type']}).
The best model at 10 dB was {snr_best_df[snr_best_df['SNR (dB)'] == 10].iloc[0]['Best Model']} ({snr_best_df[snr_best_df['SNR (dB)'] == 10].iloc[0]['Training Type']}).

When all noisy conditions were evaluated together, the most noise-robust model was {best_noise_model['Model']} ({best_noise_model['Training Type']}).
Its mean noisy Macro F1 score was {best_noise_model['Mean Noisy Macro F1']:.4f}.
Its worst noisy Macro F1 score was {best_noise_model['Worst Noisy Macro F1']:.4f}.
Its overall Noise Robustness Score was {best_noise_model['Noise Robustness Score']:.4f}.

These findings show that model selection should not be based only on clean-test performance.
The most suitable model for real-world use should be selected according to its robustness under degraded input conditions.
"""

print_section("AUTOMATIC REPORT INTERPRETATION")
print(report_text)

with open(os.path.join(OUTPUT_DIR, "09_automatic_report_interpretation.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)


# ============================================================
# 24. OUTPUT FILE LIST
# ============================================================

print_section("ALL DONE")
print(f"All CSV files, figures, tables, and report text were saved in: {OUTPUT_DIR}")

print("\nSaved files:")
for file in sorted(os.listdir(OUTPUT_DIR)):
    print("-", file)
