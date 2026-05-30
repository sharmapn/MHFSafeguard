# misclassification_analysis.py
# Robust misclassification analysis utilities for MHF Safeguard models.

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in str(name))


def _as_array(values: Iterable) -> np.ndarray:
    return np.array(list(values))


def identify_incorrect_classifications(df_actual, y_test, y_pred, algorithm_name, output_dir: Path):
    y_test_arr = _as_array(y_test)
    y_pred_arr = _as_array(y_pred)

    results_df = pd.DataFrame({
        "Tweet": df_actual["Tweet"].astype(str).values,
        "Actual Label": y_test_arr,
        "Predicted Label": y_pred_arr,
    })

    results_df["Incorrectly Classified"] = results_df["Actual Label"] != results_df["Predicted Label"]
    results_df["Row Position"] = np.arange(len(results_df))

    incorrect_predictions = results_df[results_df["Incorrectly Classified"]].copy()

    filename = output_dir / f"incorrectly_classified_rows_{_safe_name(algorithm_name)}.csv"
    incorrect_predictions.to_csv(filename, index=False)
    print(f"Incorrectly classified rows saved to '{filename}'")

    return incorrect_predictions, results_df


def class_wise_misclassification(incorrect_predictions, algorithm_name, output_dir: Path):
    class_misclassifications = (
        incorrect_predictions
        .groupby(["Actual Label", "Predicted Label"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    filename = output_dir / f"class_wise_misclassifications_{_safe_name(algorithm_name)}.csv"
    class_misclassifications.to_csv(filename, index=False)
    print(f"Class-wise misclassification counts saved to '{filename}'")

    return class_misclassifications


def plot_confusion_matrix(y_test, y_pred, labels, algorithm_name, output_dir: Path):
    labels = list(labels)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"Confusion Matrix - {algorithm_name}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    filename = output_dir / f"confusion_matrix_{_safe_name(algorithm_name)}.png"
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Confusion matrix saved as '{filename}'")


def most_common_misclassified_texts(incorrect_predictions, algorithm_name, output_dir: Path):
    most_common = incorrect_predictions["Tweet"].value_counts().reset_index()
    most_common.columns = ["Tweet", "Misclassification Count"]

    filename = output_dir / f"most_common_misclassified_texts_{_safe_name(algorithm_name)}.csv"
    most_common.to_csv(filename, index=False)
    print(f"Most commonly misclassified texts saved to '{filename}'")

    return most_common


def misclassification_by_text_length(results_df, algorithm_name, output_dir: Path):
    results_df = results_df.copy()
    results_df["Text Length"] = results_df["Tweet"].astype(str).apply(len)

    length_analysis = (
        results_df
        .groupby(["Incorrectly Classified"])["Text Length"]
        .describe()
        .reset_index()
    )

    filename = output_dir / f"misclassification_by_text_length_{_safe_name(algorithm_name)}.csv"
    length_analysis.to_csv(filename, index=False)
    print(f"Misclassification by text length saved to '{filename}'")

    return length_analysis


def common_misclassified_words(incorrect_predictions, algorithm_name, output_dir: Path, top_n: int = 30):
    text = " ".join(incorrect_predictions["Tweet"].dropna().astype(str).tolist()).lower()
    words = [w.strip(".,!?;:\"'()[]{}") for w in text.split()]
    words = [w for w in words if len(w) > 2]

    most_common_words = pd.DataFrame(
        Counter(words).most_common(top_n),
        columns=["Word", "Frequency"]
    )

    filename = output_dir / f"common_misclassified_words_{_safe_name(algorithm_name)}.csv"
    most_common_words.to_csv(filename, index=False)
    print(f"Common misclassified words saved to '{filename}'")

    return most_common_words


def confidence_analysis_misclassifications(
    incorrect_predictions,
    y_pred_proba,
    algorithm_name,
    labels,
    output_dir: Path,
):
    if y_pred_proba is None:
        print("Confidence analysis skipped because y_pred_proba is not provided.")
        return None

    y_pred_proba = np.asarray(y_pred_proba)
    labels = list(labels)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    rows = []
    for _, row in incorrect_predictions.iterrows():
        pos = int(row["Row Position"])
        predicted_label = row["Predicted Label"]
        class_index = label_to_index.get(predicted_label)

        if class_index is None or pos >= len(y_pred_proba):
            confidence = np.nan
        else:
            confidence = float(y_pred_proba[pos, class_index])

        rows.append({
            "Tweet": row["Tweet"],
            "Actual Label": row["Actual Label"],
            "Predicted Label": predicted_label,
            "Confidence": confidence,
        })

    confidence_df = pd.DataFrame(rows)

    filename = output_dir / f"confidence_analysis_misclassifications_{_safe_name(algorithm_name)}.csv"
    confidence_df.to_csv(filename, index=False)
    print(f"Confidence analysis for misclassifications saved to '{filename}'")

    return confidence_df


def perform_misclassification_analysis(
    df_actual,
    y_test,
    y_pred,
    algorithm_name,
    y_pred_proba=None,
    labels=None,
    label_encoder=None,
    output_dir="outputs/misclassification",
):
    """Run misclassification analysis.

    Parameters are intentionally compatible with the older function.
    label_encoder is accepted for backward compatibility but is no longer needed.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting misclassification analysis for {algorithm_name}...")

    incorrect_predictions, results_df = identify_incorrect_classifications(
        df_actual=df_actual,
        y_test=y_test,
        y_pred=y_pred,
        algorithm_name=algorithm_name,
        output_dir=output_dir,
    )

    class_wise_misclassification(incorrect_predictions, algorithm_name, output_dir)

    if labels is not None and len(labels) > 0:
        plot_confusion_matrix(y_test, y_pred, labels, algorithm_name, output_dir)
    else:
        print("Skipping confusion matrix plot as labels are missing.")

    most_common_misclassified_texts(incorrect_predictions, algorithm_name, output_dir)
    misclassification_by_text_length(results_df, algorithm_name, output_dir)
    common_misclassified_words(incorrect_predictions, algorithm_name, output_dir)

    if labels is not None and len(labels) > 0:
        confidence_analysis_misclassifications(
            incorrect_predictions=incorrect_predictions,
            y_pred_proba=y_pred_proba,
            algorithm_name=algorithm_name,
            labels=labels,
            output_dir=output_dir,
        )
    else:
        print("Skipping confidence analysis because labels are missing.")

    print(f"Misclassification analysis completed for {algorithm_name}.")
