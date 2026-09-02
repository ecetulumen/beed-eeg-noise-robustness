"""Classification reports, class-wise error analysis, and text summaries."""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from config import OUTPUT_DIR


def save_classification_report(y_true, y_pred, filename):
    report = classification_report(y_true, y_pred, zero_division=0)
    (OUTPUT_DIR / filename).write_text(report, encoding="utf-8")
    return report


def build_error_analysis(y_true, y_pred, num_classes):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    rows = []

    for class_index in range(num_classes):
        total = matrix[class_index, :].sum()
        correct = matrix[class_index, class_index]
        rows.append(
            {
                "Class": class_index,
                "Total Samples": total,
                "Correct": correct,
                "Wrong": total - correct,
                "Class Accuracy": correct / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_automatic_summary(best_clean, best_noise, best_by_snr):
    snr_lines = []
    for _, row in best_by_snr.sort_values("SNR (dB)", ascending=False).iterrows():
        snr_lines.append(
            f"The best model at {int(row['SNR (dB)'])} dB was "
            f"{row['Best Model']} ({row['Training Type']})."
        )

    return f"""AUTOMATIC RESULT INTERPRETATION

This project evaluated the clean-data performance and noise robustness of SVM,
Random Forest, XGBoost, and MLP classifiers.

The best clean baseline model was {best_clean['Model']}, with a clean-test
Macro F1 score of {best_clean['Macro F1']:.4f}.

{chr(10).join(snr_lines)}

Across all noisy conditions, the most robust model was {best_noise['Model']}
({best_noise['Training Type']}). Its mean noisy Macro F1 was
{best_noise['Mean Noisy Macro F1']:.4f}, its worst noisy Macro F1 was
{best_noise['Worst Noisy Macro F1']:.4f}, and its combined robustness score was
{best_noise['Noise Robustness Score']:.4f}.

Model selection should therefore consider performance under degraded inputs,
not only clean-test performance.
"""

