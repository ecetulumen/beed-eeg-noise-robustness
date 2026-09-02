"""Predictions, metrics, repeated noisy evaluation, and summary tables."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from config import N_NOISE_REPEATS, RANDOM_STATE, SNR_LEVELS
from noise import add_awgn_noise


def predict_model(model, X_data, model_type):
    if model_type == "mlp":
        probabilities = model.predict(X_data, verbose=0)
        return np.argmax(probabilities, axis=1), probabilities

    predictions = model.predict(X_data)
    probabilities = model.predict_proba(X_data) if hasattr(model, "predict_proba") else None
    return predictions, probabilities


def get_specificity_per_class(y_true, y_pred, num_classes):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    specificities = []

    for class_index in range(num_classes):
        true_positive = matrix[class_index, class_index]
        false_negative = matrix[class_index, :].sum() - true_positive
        false_positive = matrix[:, class_index].sum() - true_positive
        true_negative = matrix.sum() - true_positive - false_negative - false_positive
        denominator = true_negative + false_positive
        specificities.append(true_negative / denominator if denominator else 0.0)

    return specificities


def safe_roc_auc(y_true, probabilities, num_classes):
    if probabilities is None:
        return np.nan
    try:
        if num_classes == 2:
            return roc_auc_score(y_true, probabilities[:, 1])
        y_binary = label_binarize(y_true, classes=np.arange(num_classes))
        return roc_auc_score(y_binary, probabilities, average="macro", multi_class="ovr")
    except ValueError:
        return np.nan


def safe_pr_auc(y_true, probabilities, num_classes):
    if probabilities is None:
        return np.nan
    try:
        if num_classes == 2:
            return average_precision_score(y_true, probabilities[:, 1])
        y_binary = label_binarize(y_true, classes=np.arange(num_classes))
        return average_precision_score(y_binary, probabilities, average="macro")
    except ValueError:
        return np.nan


def calculate_metrics(y_true, y_pred, probabilities, num_classes):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro Recall / Sensitivity": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Mean Specificity": np.mean(
            get_specificity_per_class(y_true, y_pred, num_classes)
        ),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC AUC": safe_roc_auc(y_true, probabilities, num_classes),
        "PR AUC": safe_pr_auc(y_true, probabilities, num_classes),
    }


def evaluate_model_group(
    model_group,
    X_eval,
    y_eval,
    num_classes,
    test_type,
    condition,
    snr_level=None,
    repeat=None,
):
    rows = []
    predictions = {}
    probabilities = {}

    for model_name, item in model_group.items():
        y_pred, y_prob = predict_model(item["model"], X_eval, item["type"])
        row = {
            "Model": model_name,
            "Training Type": item["training_type"],
            "Test Type": test_type,
            "Condition": condition,
            "SNR": snr_level,
            "Repeat": repeat,
        }
        row.update(calculate_metrics(y_eval, y_pred, y_prob, num_classes))
        rows.append(row)

        key = f"{model_name} | {item['training_type']} | {condition}"
        predictions[key] = y_pred
        probabilities[key] = y_prob

    return pd.DataFrame(rows), predictions, probabilities


def evaluate_clean_test(clean_models, robust_models, X_test, y_test, num_classes):
    clean_df, clean_predictions, clean_probabilities = evaluate_model_group(
        clean_models,
        X_test,
        y_test,
        num_classes,
        test_type="Clean Test",
        condition="Clean Train -> Clean Test",
    )
    robust_df, robust_predictions, robust_probabilities = evaluate_model_group(
        robust_models,
        X_test,
        y_test,
        num_classes,
        test_type="Clean Test",
        condition="Robust Train -> Clean Test",
    )

    predictions = {**clean_predictions, **robust_predictions}
    probabilities = {**clean_probabilities, **robust_probabilities}
    return pd.concat([clean_df, robust_df], ignore_index=True), predictions, probabilities


def evaluate_noisy_test(clean_models, robust_models, X_test, y_test, num_classes):
    rows = []

    for snr in SNR_LEVELS:
        for repeat in range(1, N_NOISE_REPEATS + 1):
            seed = RANDOM_STATE + snr * 100 + repeat
            noisy_test = add_awgn_noise(X_test, snr, seed)

            clean_df, _, _ = evaluate_model_group(
                clean_models,
                noisy_test,
                y_test,
                num_classes,
                test_type="Noisy Test",
                condition="Clean Train -> Noisy Test",
                snr_level=snr,
                repeat=repeat,
            )
            robust_df, _, _ = evaluate_model_group(
                robust_models,
                noisy_test,
                y_test,
                num_classes,
                test_type="Noisy Test",
                condition="Robust Train -> Noisy Test",
                snr_level=snr,
                repeat=repeat,
            )
            rows.extend([clean_df, robust_df])

    return pd.concat(rows, ignore_index=True)


def summarize_noisy_results(detailed_results):
    summary = detailed_results.groupby(
        ["Model", "Training Type", "Test Type", "Condition", "SNR"],
        as_index=False,
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
        PR_AUC_Mean=("PR AUC", "mean"),
    )
    scale = 1.96 / np.sqrt(N_NOISE_REPEATS)
    summary["Macro_F1_95CI"] = scale * summary["Macro_F1_Std"]
    summary["Accuracy_95CI"] = scale * summary["Accuracy_Std"]
    return summary


def select_best_models_by_snr(noisy_summary):
    rows = []
    for snr in SNR_LEVELS:
        best = noisy_summary[noisy_summary["SNR"] == snr].nlargest(
            1, "Macro_F1_Mean"
        ).iloc[0]
        rows.append(
            {
                "SNR (dB)": snr,
                "Best Model": best["Model"],
                "Training Type": best["Training Type"],
                "Condition": best["Condition"],
                "Accuracy Mean": best["Accuracy_Mean"],
                "Balanced Accuracy Mean": best["Balanced_Accuracy_Mean"],
                "Macro Precision Mean": best["Macro_Precision_Mean"],
                "Macro Recall Mean": best["Macro_Recall_Mean"],
                "Macro F1 Mean": best["Macro_F1_Mean"],
                "Macro F1 95% CI": best["Macro_F1_95CI"],
                "Weighted F1 Mean": best["Weighted_F1_Mean"],
                "Mean Specificity": best["Mean_Specificity_Mean"],
                "Cohen Kappa Mean": best["Cohen_Kappa_Mean"],
                "MCC Mean": best["MCC_Mean"],
                "ROC AUC Mean": best["ROC_AUC_Mean"],
                "PR AUC Mean": best["PR_AUC_Mean"],
            }
        )
    return pd.DataFrame(rows)


def calculate_robust_training_gain(noisy_summary):
    rows = []
    for snr in SNR_LEVELS:
        for model_name in noisy_summary["Model"].unique():
            model_rows = noisy_summary[
                (noisy_summary["SNR"] == snr)
                & (noisy_summary["Model"] == model_name)
            ]
            clean = model_rows[
                model_rows["Condition"] == "Clean Train -> Noisy Test"
            ]
            robust = model_rows[
                model_rows["Condition"] == "Robust Train -> Noisy Test"
            ]
            if clean.empty or robust.empty:
                continue
            clean = clean.iloc[0]
            robust = robust.iloc[0]
            rows.append(
                {
                    "SNR (dB)": snr,
                    "Model": model_name,
                    "Clean Train Macro F1": clean["Macro_F1_Mean"],
                    "Robust Train Macro F1": robust["Macro_F1_Mean"],
                    "Macro F1 Gain": robust["Macro_F1_Mean"] - clean["Macro_F1_Mean"],
                    "Clean Train Accuracy": clean["Accuracy_Mean"],
                    "Robust Train Accuracy": robust["Accuracy_Mean"],
                    "Accuracy Gain": robust["Accuracy_Mean"] - clean["Accuracy_Mean"],
                    "Clean Train MCC": clean["MCC_Mean"],
                    "Robust Train MCC": robust["MCC_Mean"],
                    "MCC Gain": robust["MCC_Mean"] - clean["MCC_Mean"],
                }
            )
    return pd.DataFrame(rows)


def build_noise_robustness_ranking(noisy_summary):
    rows = []
    grouped = noisy_summary.groupby(["Model", "Training Type"])

    for (model_name, training_type), group in grouped:
        mean_f1 = group["Macro_F1_Mean"].mean()
        worst_f1 = group["Macro_F1_Mean"].min()
        rows.append(
            {
                "Model": model_name,
                "Training Type": training_type,
                "Mean Noisy Macro F1": mean_f1,
                "Worst Noisy Macro F1": worst_f1,
                "Mean Noisy Accuracy": group["Accuracy_Mean"].mean(),
                "Worst Noisy Accuracy": group["Accuracy_Mean"].min(),
                "Noise Robustness Score": 0.60 * mean_f1 + 0.40 * worst_f1,
            }
        )
    return pd.DataFrame(rows)

