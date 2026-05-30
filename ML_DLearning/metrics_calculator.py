# metrics_calculator.py
# Stable metrics helper for the MHF Safeguard ML/DL training scripts.
#
# This version is designed to be safe for both:
# 1. string labels, e.g. "Not Suicide post"; and
# 2. already-encoded integer labels, e.g. 0, 1, 2.
#
# It also avoids calculating PR-AUC, ROC-AUC, and log-loss when probability
# scores are unavailable, which is common for LinearSVC.

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


def _as_array(values: Iterable) -> np.ndarray:
    """Convert iterable labels/predictions to a NumPy array."""

    return np.asarray(list(values))


def _normalise_proba(y_pred_proba: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return probability matrix only if it is 2-dimensional."""

    if y_pred_proba is None:
        return None

    proba = np.asarray(y_pred_proba)

    if proba.ndim != 2:
        return None

    return proba


def _looks_like_encoded_labels(values: np.ndarray, n_classes: int) -> bool:
    """Check whether labels are already encoded as integers 0..n_classes-1."""

    try:
        numeric_values = values.astype(int)
    except Exception:
        return False

    if len(numeric_values) == 0:
        return False

    if not np.all(values.astype(str) == numeric_values.astype(str)):
        return False

    min_value = int(np.min(numeric_values))
    max_value = int(np.max(numeric_values))

    return min_value >= 0 and max_value < n_classes


def _encode_labels(
    y_true: Iterable,
    y_pred: Iterable,
    classes: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Encode labels safely.

    If y_true/y_pred are already integer encoded and classes are provided, the
    encoded values are used directly. This prevents errors when the calling code
    passes y_test_encoded and y_pred_encoded along with label_encoder.classes_.
    """

    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)

    if classes is not None:
        class_names = [str(c) for c in classes]
        n_classes = len(class_names)

        if _looks_like_encoded_labels(y_true_arr, n_classes) and _looks_like_encoded_labels(y_pred_arr, n_classes):
            return y_true_arr.astype(int), y_pred_arr.astype(int), class_names

        encoder = LabelEncoder()
        encoder.fit(class_names)
        return encoder.transform(y_true_arr.astype(str)), encoder.transform(y_pred_arr.astype(str)), class_names

    encoder = LabelEncoder()
    encoder.fit(np.unique(np.concatenate([y_true_arr.astype(str), y_pred_arr.astype(str)])))
    class_names = [str(c) for c in encoder.classes_]

    return encoder.transform(y_true_arr.astype(str)), encoder.transform(y_pred_arr.astype(str)), class_names


def _safe_json_value(value):
    """Convert NumPy values to JSON-friendly Python values."""

    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def calculate_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    y_pred_proba: Optional[np.ndarray] = None,
    classes: Optional[Sequence[str]] = None,
    model_name: str = "model",
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Calculate standard classification metrics safely.

    This function handles:
    - string labels;
    - already-encoded integer labels;
    - classifiers with probability output;
    - classifiers without probability output, such as LinearSVC.

    When probabilities are not available, PR-AUC, ROC-AUC, and log-loss are set
    to None rather than calculated incorrectly from hard class predictions.
    """

    y_true_enc, y_pred_enc, class_names = _encode_labels(
        y_true=y_true,
        y_pred=y_pred,
        classes=classes,
    )

    label_ids = list(range(len(class_names)))

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true_enc, y_pred_enc)),
        "f1_weighted": float(f1_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true_enc, y_pred_enc)),
        "mcc": float(matthews_corrcoef(y_true_enc, y_pred_enc)),
        "f2_score": float(fbeta_score(y_true_enc, y_pred_enc, beta=2, average="weighted", zero_division=0)),
        "pr_auc": None,
        "roc_auc": None,
        "log_loss": None,
        "probability_metric_note": "Not calculated because probability scores are unavailable or incompatible.",
    }

    proba = _normalise_proba(y_pred_proba)

    if proba is not None and proba.shape[0] == len(y_true_enc) and proba.shape[1] == len(class_names):
        try:
            if len(class_names) == 2:
                metrics["pr_auc"] = float(average_precision_score(y_true_enc, proba[:, 1]))
                metrics["roc_auc"] = float(roc_auc_score(y_true_enc, proba[:, 1]))
            else:
                pr_scores = []
                for class_index in range(len(class_names)):
                    binary_true = (y_true_enc == class_index).astype(int)
                    pr_scores.append(average_precision_score(binary_true, proba[:, class_index]))

                metrics["pr_auc"] = float(np.mean(pr_scores))
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_true_enc,
                        proba,
                        multi_class="ovr",
                        average="weighted",
                        labels=label_ids,
                    )
                )

            metrics["log_loss"] = float(log_loss(y_true_enc, proba, labels=label_ids))
            metrics.pop("probability_metric_note", None)

        except Exception as exc:
            metrics["probability_metric_error"] = str(exc)

    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true_enc,
        y_pred_enc,
        labels=label_ids,
        zero_division=0,
    )

    classwise_df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": fscore,
        "Support": support,
    })

    report_text = classification_report(
        y_true_enc,
        y_pred_enc,
        labels=label_ids,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=label_ids)

    print("\n--- Model Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nClass-wise Precision, Recall, and F1-Score:")
    print(classwise_df.to_string(index=False))

    print("\nClassification Report:")
    print(report_text)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        safe_model_name = "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in str(model_name))

        classwise_df.to_csv(output_path / f"{safe_model_name}_classwise_metrics.csv", index=False)

        pd.DataFrame(
            cm,
            index=class_names,
            columns=class_names,
        ).to_csv(output_path / f"{safe_model_name}_confusion_matrix.csv")

        (output_path / f"{safe_model_name}_classification_report.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        json_safe_metrics = {key: _safe_json_value(value) for key, value in metrics.items()}
        (output_path / f"{safe_model_name}_metrics.json").write_text(
            json.dumps(json_safe_metrics, indent=2),
            encoding="utf-8",
        )

    metrics["classwise"] = classwise_df
    metrics["confusion_matrix"] = cm
    metrics["classification_report_text"] = report_text

    return metrics
