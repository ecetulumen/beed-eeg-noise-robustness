"""Plots and presentation-ready result tables."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from config import RANDOM_STATE
from evaluation import predict_model
from noise import add_awgn_noise
from utils import save_fig


MODEL_ORDER = ["SVM", "Random Forest", "XGBoost", "MLP"]


def _draw_confusion_matrix(axis, y_true, y_pred, title):
    classes = np.unique(y_true)
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    normalized = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)
    normalized = np.nan_to_num(normalized)
    image = axis.imshow(normalized, vmin=0, vmax=1)

    accuracy = accuracy_score(y_true, y_pred)
    axis.set_title(f"{title}\nAccuracy = {accuracy * 100:.2f}%", fontweight="bold")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(classes)
    axis.set_yticks(classes)

    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontweight="bold" if value > 0.50 else "normal",
            )
    return image


def get_clean_predictions(clean_models, X_test):
    predictions = {}
    for model_name, item in clean_models.items():
        y_pred, _ = predict_model(item["model"], X_test, item["type"])
        predictions[model_name] = y_pred
    return predictions


def get_awgn_predictions(clean_models, robust_models, X_test, snr):
    noisy_test = add_awgn_noise(X_test, snr, RANDOM_STATE)
    predictions = {"Clean Train": {}, "Robust Train": {}}

    for model_name, item in clean_models.items():
        predictions["Clean Train"][model_name] = predict_model(
            item["model"], noisy_test, item["type"]
        )[0]
    for model_name, item in robust_models.items():
        predictions["Robust Train"][model_name] = predict_model(
            item["model"], noisy_test, item["type"]
        )[0]
    return predictions


def plot_clean_confusion_grid(y_true, predictions):
    figure, axes = plt.subplots(1, 4, figsize=(22, 4.8), constrained_layout=True)

    for index, model_name in enumerate(MODEL_ORDER):
        if model_name not in predictions:
            axes[index].axis("off")
            continue
        image = _draw_confusion_matrix(
            axes[index], y_true, predictions[model_name], model_name
        )
        figure.colorbar(image, ax=axes[index], fraction=0.046, pad=0.04)

    figure.suptitle("Clean Train - Clean Test", fontsize=18, fontweight="bold")
    save_fig("clean_train_clean_test_confusion_grid.png")


def plot_awgn_confusion_grid(y_true, predictions, snr):
    figure, axes = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True)

    for row, training_type in enumerate(["Clean Train", "Robust Train"]):
        for column, model_name in enumerate(MODEL_ORDER):
            axis = axes[row, column]
            if model_name not in predictions[training_type]:
                axis.axis("off")
                continue
            image = _draw_confusion_matrix(
                axis,
                y_true,
                predictions[training_type][model_name],
                f"{model_name} - {training_type}",
            )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(
        f"AWGN Robustness Test - {snr} dB", fontsize=18, fontweight="bold"
    )
    save_fig(f"awgn_{snr}db_clean_vs_robust_confusion_grid.png")


def plot_mlp_history(history, title_prefix, filename_prefix):
    values = history.history

    plt.figure(figsize=(8, 4))
    plt.plot(values["accuracy"], label="Train accuracy")
    plt.plot(values["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy")
    plt.grid(alpha=0.4)
    plt.legend()
    save_fig(f"{filename_prefix}_accuracy.png")

    plt.figure(figsize=(8, 4))
    plt.plot(values["loss"], label="Train loss")
    plt.plot(values["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.grid(alpha=0.4)
    plt.legend()
    save_fig(f"{filename_prefix}_loss.png")


def plot_snr_comparison(noisy_summary, snr):
    subset = noisy_summary[noisy_summary["SNR"] == snr].copy()
    subset["Model Label"] = subset["Model"] + "\n" + subset["Training Type"]
    subset = subset.sort_values("Macro_F1_Mean", ascending=False)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        subset["Model Label"],
        subset["Macro_F1_Mean"],
        yerr=subset["Macro_F1_95CI"],
        capsize=4,
    )
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Macro F1 mean ± 95% CI")
    plt.ylim(0, 1.05)
    plt.title(f"AWGN {snr} dB - Model Robustness Comparison")
    plt.grid(axis="y", alpha=0.4)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save_fig(f"awgn_{snr}db_model_comparison_bar.png")


def plot_noise_trends(noisy_summary):
    plt.figure(figsize=(13, 7))

    for model_name in noisy_summary["Model"].unique():
        for condition in [
            "Clean Train -> Noisy Test",
            "Robust Train -> Noisy Test",
        ]:
            subset = noisy_summary[
                (noisy_summary["Model"] == model_name)
                & (noisy_summary["Condition"] == condition)
            ].sort_values("SNR", ascending=False)
            label = f"{model_name} - {'Robust' if condition.startswith('Robust') else 'Clean'}"
            plt.plot(
                subset["SNR"],
                subset["Macro_F1_Mean"],
                marker="o",
                linewidth=2,
                label=label,
            )

    plt.gca().invert_xaxis()
    plt.xlabel("SNR level (dB)")
    plt.ylabel("Macro F1 mean")
    plt.ylim(0, 1.05)
    plt.title("Noise Robustness Trend Across 30, 20, and 10 dB")
    plt.grid(alpha=0.4)
    plt.legend(fontsize=8, ncol=2)
    save_fig("all_models_30_20_10db_noise_trend.png")


def plot_robustness_ranking(ranking):
    subset = ranking.sort_values("Noise Robustness Score", ascending=False).copy()
    subset["Model Label"] = subset["Model"] + "\n" + subset["Training Type"]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(subset["Model Label"], subset["Noise Robustness Score"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Noise robustness score")
    plt.ylim(0, 1.05)
    plt.title("Overall Noise Robustness Ranking")
    plt.grid(axis="y", alpha=0.4)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save_fig("overall_noise_robustness_ranking.png")


def save_table_as_png(dataframe, title, filename, max_rows=20):
    table_data = dataframe.head(max_rows).copy()
    numeric_columns = table_data.select_dtypes(include=[np.number]).columns
    table_data[numeric_columns] = table_data[numeric_columns].round(4)

    width = min(24, 2.2 * len(table_data.columns))
    figure, axis = plt.subplots(figsize=(width, 0.7 * len(table_data) + 2.5))
    axis.axis("off")
    table = axis.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.35)
    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    save_fig(filename)


def plot_roc_and_pr(y_true, probabilities, title_prefix, filename_prefix):
    if probabilities is None:
        print("Probability output unavailable; ROC and PR curves were skipped.")
        return

    num_classes = len(np.unique(y_true))
    if num_classes == 2:
        y_score = probabilities[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        roc_label = f"AUC = {auc(fpr, tpr):.3f}"
        pr_label = f"PR-AUC = {auc(recall, precision):.3f}"
    else:
        y_binary = label_binarize(y_true, classes=np.arange(num_classes))
        fpr, tpr, _ = roc_curve(y_binary.ravel(), probabilities.ravel())
        precision, recall, _ = precision_recall_curve(
            y_binary.ravel(), probabilities.ravel()
        )
        roc_label = f"Micro AUC = {auc(fpr, tpr):.3f}"
        pr_label = f"Micro PR-AUC = {auc(recall, precision):.3f}"

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=roc_label)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{title_prefix} ROC Curve")
    plt.grid(alpha=0.4)
    plt.legend()
    save_fig(f"{filename_prefix}_roc_curve.png")

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=pr_label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} Precision-Recall Curve")
    plt.grid(alpha=0.4)
    plt.legend()
    save_fig(f"{filename_prefix}_pr_curve.png")

