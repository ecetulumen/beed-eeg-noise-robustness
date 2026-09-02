"""Run the complete BEED clean-vs-robust AWGN experiment."""

import matplotlib.pyplot as plt

from config import (
    DATA_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    ROBUST_REPEAT_MAP,
    ROBUST_TRAIN_SNR_LEVELS,
    SNR_LEVELS,
)
from data_processing import load_dataset, split_and_scale
from evaluation import (
    build_noise_robustness_ranking,
    calculate_robust_training_gain,
    evaluate_clean_test,
    evaluate_noisy_test,
    predict_model,
    select_best_models_by_snr,
    summarize_noisy_results,
)
from models import train_ml_models, train_mlp
from noise import add_awgn_noise, create_robust_training_data
from reporting import (
    build_automatic_summary,
    build_error_analysis,
    save_classification_report,
)
from utils import display_and_save, print_section, save_df, set_seed
from visualization import (
    get_awgn_predictions,
    get_clean_predictions,
    plot_awgn_confusion_grid,
    plot_clean_confusion_grid,
    plot_mlp_history,
    plot_noise_trends,
    plot_robustness_ranking,
    plot_roc_and_pr,
    plot_snr_comparison,
    save_table_as_png,
)


def train_model_groups(X_train, y_train, input_dim, num_classes):
    clean_models = train_ml_models(
        X_train, y_train, training_type="Clean Train", num_classes=num_classes
    )
    clean_mlp, clean_history = train_mlp(
        X_train, y_train, input_dim, num_classes, training_type="Clean Train"
    )
    clean_models["MLP"] = clean_mlp

    X_robust, y_robust = create_robust_training_data(X_train, y_train)
    print_section("ROBUST TRAINING DATA")
    print("Clean train shape:", X_train.shape)
    print("Robust train shape:", X_robust.shape)
    for snr in ROBUST_TRAIN_SNR_LEVELS:
        print(f"AWGN {snr} dB copies: {ROBUST_REPEAT_MAP[snr]}")

    robust_models = train_ml_models(
        X_robust,
        y_robust,
        training_type="Robust Train",
        num_classes=num_classes,
    )
    robust_mlp, robust_history = train_mlp(
        X_robust,
        y_robust,
        input_dim,
        num_classes,
        training_type="Robust Train",
    )
    robust_models["MLP"] = robust_mlp

    histories = {"clean": clean_history, "robust": robust_history}
    return clean_models, robust_models, histories


def create_and_save_summary_tables(clean_results, noisy_details):
    noisy_summary = summarize_noisy_results(noisy_details)
    best_by_snr = select_best_models_by_snr(noisy_summary)
    robustness_gain = calculate_robust_training_gain(noisy_summary)
    robustness_ranking = build_noise_robustness_ranking(noisy_summary)

    display_and_save(
        clean_results.sort_values(["Condition", "Macro F1"], ascending=[True, False]),
        "CLEAN TEST RESULTS",
        "01_clean_test_results.csv",
    )
    display_and_save(
        noisy_details,
        "NOISY TEST DETAILED RESULTS",
        "02_noisy_test_detailed_results.csv",
    )
    display_and_save(
        noisy_summary.sort_values(["SNR", "Macro_F1_Mean"], ascending=[False, False]),
        "NOISY TEST SUMMARY",
        "03_noisy_test_summary.csv",
    )
    display_and_save(
        best_by_snr.sort_values("SNR (dB)", ascending=False),
        "BEST MODEL FOR EACH SNR LEVEL",
        "04_best_model_for_each_snr.csv",
    )
    display_and_save(
        robustness_gain.sort_values(
            ["SNR (dB)", "Macro F1 Gain"], ascending=[False, False]
        ),
        "ROBUST TRAINING GAIN BY SNR",
        "05_robust_training_gain_by_snr.csv",
    )
    display_and_save(
        robustness_ranking.sort_values(
            "Noise Robustness Score", ascending=False
        ),
        "OVERALL NOISE ROBUSTNESS RANKING",
        "06_overall_noise_robustness_ranking.csv",
    )
    return noisy_summary, best_by_snr, robustness_ranking


def create_figures(
    clean_models,
    robust_models,
    histories,
    X_test,
    y_test,
    clean_results,
    noisy_summary,
    best_by_snr,
    robustness_ranking,
):
    plot_mlp_history(histories["clean"], "Clean Train MLP", "clean_train_mlp")
    plot_mlp_history(histories["robust"], "Robust Train MLP", "robust_train_mlp")
    plot_clean_confusion_grid(y_test, get_clean_predictions(clean_models, X_test))

    for snr in SNR_LEVELS:
        predictions = get_awgn_predictions(clean_models, robust_models, X_test, snr)
        plot_awgn_confusion_grid(y_test, predictions, snr)
        plot_snr_comparison(noisy_summary, snr)

    plot_noise_trends(noisy_summary)
    plot_robustness_ranking(robustness_ranking)
    save_table_as_png(
        clean_results.sort_values(["Condition", "Macro F1"], ascending=[True, False]),
        "Clean Test Results",
        "clean_test_results_table.png",
    )
    save_table_as_png(
        noisy_summary.sort_values(["SNR", "Macro_F1_Mean"], ascending=[False, False]),
        "Noisy Test Summary",
        "noisy_test_summary_table.png",
    )
    save_table_as_png(
        best_by_snr.sort_values("SNR (dB)", ascending=False),
        "Best Model for Each SNR",
        "best_model_for_each_snr_table.png",
    )
    save_table_as_png(
        robustness_ranking.sort_values("Noise Robustness Score", ascending=False),
        "Overall Noise Robustness Ranking",
        "overall_noise_robustness_ranking_table.png",
    )


def report_best_model(
    clean_models,
    robust_models,
    X_test,
    y_test,
    num_classes,
    clean_results,
    best_by_snr,
    robustness_ranking,
):
    clean_baselines = clean_results[
        clean_results["Condition"] == "Clean Train -> Clean Test"
    ]
    best_clean = clean_baselines.nlargest(1, "Macro F1").iloc[0]
    best_noise = robustness_ranking.nlargest(1, "Noise Robustness Score").iloc[0]

    selected_group = (
        robust_models if best_noise["Training Type"] == "Robust Train" else clean_models
    )
    selected_item = selected_group[best_noise["Model"]]
    clean_pred, clean_prob = predict_model(
        selected_item["model"], X_test, selected_item["type"]
    )

    noisy_test = add_awgn_noise(X_test, snr_db=10, random_state=RANDOM_STATE)
    noisy_pred, noisy_prob = predict_model(
        selected_item["model"], noisy_test, selected_item["type"]
    )

    print_section("BEST MODEL - CLEAN TEST CLASSIFICATION REPORT")
    print(
        save_classification_report(
            y_test, clean_pred, "best_noise_model_clean_test_classification_report.txt"
        )
    )
    print_section("BEST MODEL - 10 dB CLASSIFICATION REPORT")
    print(
        save_classification_report(
            y_test, noisy_pred, "best_noise_model_10db_classification_report.txt"
        )
    )

    clean_errors = build_error_analysis(y_test, clean_pred, num_classes)
    noisy_errors = build_error_analysis(y_test, noisy_pred, num_classes)
    save_df(clean_errors, "07_error_analysis_best_noise_model_clean_test.csv")
    save_df(noisy_errors, "08_error_analysis_best_noise_model_10db.csv")
    save_table_as_png(
        clean_errors,
        "Error Analysis - Best Noise-Robust Model on Clean Test",
        "error_analysis_best_noise_model_clean_test.png",
    )
    save_table_as_png(
        noisy_errors,
        "Error Analysis - Best Noise-Robust Model on 10 dB",
        "error_analysis_best_noise_model_10db.png",
    )
    plot_roc_and_pr(
        y_test,
        clean_prob,
        f"Best Noise-Robust Model Clean Test - {best_noise['Model']}",
        "best_noise_model_clean_test",
    )
    plot_roc_and_pr(
        y_test,
        noisy_prob,
        f"Best Noise-Robust Model 10 dB - {best_noise['Model']}",
        "best_noise_model_10db",
    )

    summary = build_automatic_summary(best_clean, best_noise, best_by_snr)
    (OUTPUT_DIR / "09_automatic_report_interpretation.txt").write_text(
        summary, encoding="utf-8"
    )
    print_section("AUTOMATIC RESULT INTERPRETATION")
    print(summary)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["font.size"] = 10
    set_seed(RANDOM_STATE)

    print("DATA_PATH:", DATA_PATH)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    features, labels, encoder = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, _ = split_and_scale(features, labels)
    num_classes = len(encoder.classes_)

    clean_models, robust_models, histories = train_model_groups(
        X_train, y_train, features.shape[1], num_classes
    )
    clean_results, _, _ = evaluate_clean_test(
        clean_models, robust_models, X_test, y_test, num_classes
    )
    noisy_details = evaluate_noisy_test(
        clean_models, robust_models, X_test, y_test, num_classes
    )
    noisy_summary, best_by_snr, robustness_ranking = create_and_save_summary_tables(
        clean_results, noisy_details
    )
    create_figures(
        clean_models,
        robust_models,
        histories,
        X_test,
        y_test,
        clean_results,
        noisy_summary,
        best_by_snr,
        robustness_ranking,
    )
    report_best_model(
        clean_models,
        robust_models,
        X_test,
        y_test,
        num_classes,
        clean_results,
        best_by_snr,
        robustness_ranking,
    )

    print_section("ALL DONE")
    print(f"All outputs were saved in: {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print("-", path.name)


if __name__ == "__main__":
    main()

