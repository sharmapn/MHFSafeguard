# metrics_calculator.py
# Stable metrics helper for the MHF Safeguard ML/DL training scripts.

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

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
    return np.asarray(list(values))


def _normalise_proba(y_pred_proba: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if y_pred_proba is None:
        return None
    proba = np.asarray(y_pred_proba)
    if proba.ndim != 2:
        return None
    return proba


def calculate_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    y_pred_proba: Optional[np.ndarray] = None,
    classes: Optional[Sequence[str]] = None,
    model_name: str = "model",
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Calculate standard classification metrics safely.

    This function accepts either string labels or numeric labels. If probability
    estimates are provided, PR-AUC, ROC-AUC, and log-loss are calculated only
    when the shape of the probability matrix is compatible with the encoded
    labels.
    """

    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)

    encoder = LabelEncoder()
    if classes is not None:
        encoder.fit(list(classes))
    else:
        encoder.fit(np.unique(np.concatenate([y_true_arr.astype(str), y_pred_arr.astype(str)])))

    y_true_enc = encoder.transform(y_true_arr.astype(str))
    y_pred_enc = encoder.transform(y_pred_arr.astype(str))
    class_names = list(encoder.classes_)

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
                    pr_scores.append(average_precision_score((y_true_enc == class_index).astype(int), proba[:, class_index]))
                metrics["pr_auc"] = float(np.mean(pr_scores))
                metrics["roc_auc"] = float(roc_auc_score(y_true_enc, proba, multi_class="ovr", average="weighted"))
            metrics["log_loss"] = float(log_loss(y_true_enc, proba, labels=list(range(len(class_names)))))
        except Exception as exc:
            metrics["probability_metric_error"] = str(exc)

    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true_enc,
        y_pred_enc,
        labels=list(range(len(class_names))),
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
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=list(range(len(class_names))))

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

        safe_model_name = "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in model_name)
        classwise_df.to_csv(output_path / f"{safe_model_name}_classwise_metrics.csv", index=False)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_path / f"{safe_model_name}_confusion_matrix.csv")
        (output_path / f"{safe_model_name}_classification_report.txt").write_text(report_text, encoding="utf-8")
        (output_path / f"{safe_model_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metrics["classwise"] = classwise_df
    metrics["confusion_matrix"] = cm
    metrics["classification_report_text"] = report_text

    return metrics
