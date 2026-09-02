# =============================================================================
# MHF Safeguard training12.py — Python 3.14 / PyTorch edition
# 12 August 2026
# =============================================================================
#
# This is the Python 3.14-compatible successor to training11.py.
#
# Key changes:
# - Requires CPython 3.14.x (tested syntactically against modern Python).
# - Removes TensorFlow 2.14, legacy Keras, ktrain, and the old TF Transformers API.
# - Ports LSTM, GRU, CNN+LSTM, hybrid, and attention models to PyTorch.
# - Ports BERT/RoBERTa fine-tuning to the PyTorch Transformers API.
# - Preserves all v11 ML/DL/Transformer functionality.
# - Adds MentalRoBERTa, DeBERTa-v3-base, and ModernBERT-base.
# - v12: DeBERTa-v3 uses its native SentencePiece tokenizer by default on Python 3.14.
# - v12: Adds clearer gated-model authentication diagnostics without changing HF auth behavior.
# - Adds standard, class-weighted cross-entropy, and focal-loss transformer training.
# - Adds optional best-transformer imbalance ablation and evaluation-only mode.
# - Keeps completed ML/DL blocks off by default; enable only what you need.
# - Retains the original SQLite dataset construction and scikit-learn workflow.
# - Makes all database paths and expensive experiment blocks configurable.
# - Saves PyTorch checkpoints as .pt and Transformers models with save_pretrained().
#
# Recommended first run in PowerShell:
#   py -3.14 -m venv .venv314
#   .\.venv314\Scripts\Activate.ps1
#   python -m pip install --upgrade pip
#   pip install -r requirements_py314.txt
#   $env:MHFS_DB_PATH="C:\full\path\all_datasets_relabelled.db"
#   $env:MHFS_SUICIDEFORUM_DB_PATH="C:\full\path\suicideforum_dot_com4_labelled.db"
#   $env:MHFS_SSF_DB_PATH="C:\full\path\sanctioned_suicide_forum.sqlite"
#   $env:MHFS_RUN_DEEP_LEARNING="0"       # first smoke test
#   $env:MHFS_RUN_TRANSFORMERS="0"        # first smoke test
#   python training12_py314.py
#
# For neural models later:
#   $env:MHFS_RUN_DEEP_LEARNING="1"
#   $env:MHFS_RUN_TRANSFORMERS="1"
#
# Recommended transformer-only v12 run (completed ML/DL models stay off):
#   pip install --upgrade transformers sentencepiece
#   $env:MHFS_RUN_MACHINE_LEARNING="0"
#   $env:MHFS_RUN_DEEP_LEARNING="0"
#   $env:MHFS_RUN_TRANSFORMERS="1"
#   $env:MHFS_TRANSFORMER_MODELS="bert,roberta,mental_roberta,deberta_v3,modernbert"
#   $env:MHFS_TRANSFORMER_LOSS_MODES="standard"
#   $env:MHFS_RUN_IMBALANCE_ABLATION="0"
#   python training12_py314.py
#
# After standard models, imbalance-aware ablation on a chosen model:
#   $env:MHFS_TRANSFORMER_MODELS="mental_roberta"
#   $env:MHFS_TRANSFORMER_LOSS_MODES="weighted_ce,focal"
#   python training12_py314.py
#
# Evaluate an already-saved transformer on a larger/full held-out test set:
#   $env:MHFS_TRANSFORMER_EVAL_ONLY="1"
#   $env:MHFS_TRANSFORMER_TEST_LIMIT="0"
# =============================================================================

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import psutil

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Optional project-local analysis modules
# -----------------------------------------------------------------------------
try:
    from metrics_calculator import calculate_metrics as _project_calculate_metrics
except ImportError:
    _project_calculate_metrics = None

try:
    from misclassification_analysis import (
        perform_misclassification_analysis as _project_misclassification_analysis,
    )
except ImportError:
    _project_misclassification_analysis = None


def calculate_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_pred_proba: np.ndarray | None = None,
    classes: Sequence[str] | None = None,
) -> dict[str, float]:
    """Use the project helper when available; otherwise calculate core metrics."""
    if _project_calculate_metrics is not None:
        return _project_calculate_metrics(
            y_true,
            y_pred,
            y_pred_proba=y_pred_proba,
            classes=classes,
        )

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }
    print("[METRICS]", json.dumps(result, indent=2))
    return result


class FixedLabelEncoder:
    """A fixed-order encoder whose probability-column order is predictable."""

    def __init__(self, classes: Sequence[str]):
        self.classes_ = np.asarray(list(classes), dtype=object)
        self._to_id = {label: i for i, label in enumerate(self.classes_)}

    def transform(self, values: Iterable[str]) -> np.ndarray:
        return np.asarray([self._to_id[str(value)] for value in values], dtype=np.int64)

    def inverse_transform(self, values: Iterable[int]) -> np.ndarray:
        return np.asarray([self.classes_[int(value)] for value in values], dtype=object)


def perform_misclassification_analysis(
    *,
    df_actual: pd.DataFrame,
    y_test: Sequence[Any],
    y_pred: Sequence[Any],
    algorithm_name: str,
    y_pred_proba: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    label_encoder: Any | None = None,
) -> None:
    """Call the project helper or save a safe fallback CSV."""
    if USE_PROJECT_MISCLASSIFICATION_HELPER and _project_misclassification_analysis is not None:
        try:
            _project_misclassification_analysis(
                df_actual=df_actual,
                y_test=y_test,
                y_pred=y_pred,
                algorithm_name=algorithm_name,
                y_pred_proba=y_pred_proba,
                labels=labels,
                label_encoder=label_encoder,
            )
            return
        except Exception as exc:
            print(f"[WARN] Project misclassification helper failed: {exc}")

    output = df_actual.reset_index(drop=True).copy()
    output["actual_label"] = list(y_test)
    output["predicted_label"] = list(y_pred)
    output["is_misclassified"] = output["actual_label"] != output["predicted_label"]
    if y_pred_proba is not None and len(y_pred_proba) == len(output):
        output["confidence"] = np.max(y_pred_proba, axis=1)
    path = MISCLASSIFICATION_DIR / f"{algorithm_name}_misclassifications.csv"
    output.loc[output["is_misclassified"]].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Misclassification rows saved to: {path}")


# -----------------------------------------------------------------------------
# Runtime and path configuration
# -----------------------------------------------------------------------------
MIN_PYTHON = (3, 14)
if sys.version_info < MIN_PYTHON:
    raise RuntimeError(
        f"training12_py314.py requires Python 3.14+. Current: {sys.version.split()[0]}"
    )

DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(os.environ.get("MHFS_BASE_DIR", DEFAULT_BASE_DIR)).resolve()
DATA_DIR = BASE_DIR / "databases"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs_py314"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MISCLASSIFICATION_DIR = OUTPUTS_DIR / "misclassification"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

for folder in (
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    MISCLASSIFICATION_DIR,
    LOGS_DIR,
    PLOTS_DIR,
):
    folder.mkdir(parents=True, exist_ok=True)

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DB_PATH = Path(
    os.environ.get("MHFS_DB_PATH", DATA_DIR / "dec-24/all_datasets_relabelled.db")
).resolve()

USE_ACTUAL_ONLY_TEST = os.environ.get("MHFS_USE_ACTUAL_ONLY_TEST", "1") == "1"
SKIP_EXPENSIVE_CV = os.environ.get("MHFS_SKIP_EXPENSIVE_CV", "1") == "1"
RUN_ENSEMBLE_CLASSIFIER = os.environ.get("MHFS_RUN_ENSEMBLE", "0") == "1"
RUN_ADABOOST_CLASSIFIER = os.environ.get("MHFS_RUN_ADABOOST", "0") == "1"
RUN_STACKING_CLASSIFIER = os.environ.get("MHFS_RUN_STACKING", "0") == "1"
RUN_DEEP_LEARNING = os.environ.get("MHFS_RUN_DEEP_LEARNING", "0") == "1"
#RUN_DEEP_LEARNING = 1
RUN_TRANSFORMERS = os.environ.get("MHFS_RUN_TRANSFORMERS", "0") == "1"
#RUN_TRANSFORMERS = 1

machine_learning = os.environ.get("MHFS_RUN_MACHINE_LEARNING", "0") == "1"
not_done = True
number_of_classes = 3
CV_N_JOBS = int(os.environ.get("MHFS_CV_N_JOBS", "1"))
USE_PROJECT_MISCLASSIFICATION_HELPER = os.environ.get("MHFS_USE_PROJECT_MISCLASSIFICATION_HELPER", "0") == "1"

RANDOM_SEED = int(os.environ.get("MHFS_RANDOM_SEED", "42"))
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tee:
    def __init__(self, filename: Path):
        self.terminal = sys.__stdout__
        self.log = filename.open("w", buffering=1, encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()

    @property
    def encoding(self):
        return self.terminal.encoding



sys.stdout = Tee(LOGS_DIR / f"output_{RUN_STAMP}.txt")
sys.stderr = sys.stdout

try:
    process = psutil.Process(os.getpid())
    if hasattr(psutil, "HIGH_PRIORITY_CLASS"):
        process.nice(psutil.HIGH_PRIORITY_CLASS)
except Exception as exc:
    print(f"[WARN] Could not set high process priority: {exc}")

print("\n[PYTHON 3.14 CONFIGURATION]")
print(f"Python     : {sys.version.split()[0]}")
print(f"PyTorch   : {torch.__version__}")
print(f"Device    : {DEVICE}")
print(f"BASE_DIR  : {BASE_DIR}")
print(f"DB_PATH   : {DB_PATH}")
print(f"DL enabled: {RUN_DEEP_LEARNING}")
print(f"Transformers enabled: {RUN_TRANSFORMERS}")

if not DB_PATH.exists():
    raise FileNotFoundError(
        f"Main database not found: {DB_PATH}. Set MHFS_DB_PATH to its full path."
    )


# **Load Dataset**




# Read sqlite query results into a pandas DataFrSuicide or Self Harm Methodame
con = sqlite3.connect(str(DB_PATH))
# multiple datasets
# SELECT sentence as Tweet, label as Suicide FROM sent_388_original_labelled ORDER BY ID ASC LIMIT 4000 
# UNION 
# SELECT sentence as Tweet, label as Suicide FROM generated_sentences


# Just out ut the number of rows for each label
import pandas as pd

# suicideforums dataset
SUICIDEFORUM_DB_PATH = Path(os.environ.get(
    "MHFS_SUICIDEFORUM_DB_PATH",
    str(DATA_DIR / "suicideforum_dot_com4_labelled.db"),
)).resolve()

if not SUICIDEFORUM_DB_PATH.exists():
    raise FileNotFoundError(
        f"SuicideForum database not found: {SUICIDEFORUM_DB_PATH}. "
        "Set MHFS_SUICIDEFORUM_DB_PATH to its full path."
    )
attached_dbs = pd.read_sql_query("PRAGMA database_list;", con)
if "sf3" not in attached_dbs["name"].tolist():
    con.execute("ATTACH DATABASE ? AS sf3", (str(SUICIDEFORUM_DB_PATH),))


# santioned suicide dataset
SSF_DB_PATH = Path(os.environ.get(
    "MHFS_SSF_DB_PATH",
    str(DATA_DIR / "sanctioned_suicide_forum.sqlite"),
)).resolve()

# Attach the sanctioned-suicide forum label database as "ssf"
if not SSF_DB_PATH.exists():
    raise FileNotFoundError(
        f"Sanctioned Suicide database not found: {SSF_DB_PATH}. "
        "Set MHFS_SSF_DB_PATH to its full path."
    )
attached_dbs = pd.read_sql_query("PRAGMA database_list;", con)
if "ssf" not in attached_dbs["name"].tolist():
    con.execute("ATTACH DATABASE ? AS ssf", (str(SSF_DB_PATH),))

   
# SQL query to get label counts across all sources
label_count_query = """
SELECT Suicide AS label, COUNT(*) AS count FROM (

    -- 1. MH_forum_388_sentences: existing labelled MHF sentences
    SELECT TRIM(label) AS Suicide
    FROM MH_forum_388_sentences
    WHERE TRIM(label) IN (
        'Not Suicide post',
        'Ideation of Suicide, Self-Harm or Harming Others',
        'Method or action of Suicide, Self-Harm or Harming others'
    )

    UNION ALL

    -- 2. MH_forum_387_sentences: labelled MHF 387 sentences
    SELECT TRIM(label) AS Suicide
    FROM MH_forum_387_sentences
    WHERE sentence IS NOT NULL
      AND TRIM(sentence) <> ''
      AND label IS NOT NULL
      AND TRIM(label) <> ''
      AND TRIM(label) IN (
          'Not Suicide post',
          'Ideation of Suicide, Self-Harm or Harming Others',
          'Method or action of Suicide, Self-Harm or Harming others'
      )

 
    UNION ALL

    -- 3. sent_labelled_389_classified_sentences: AI-classified sentences from 389
    -- Use first_label as the main three-class label.
    SELECT        
        TRIM(first_label) AS Suicide
    FROM sent_labelled_389_classified_sentences
    WHERE sentence IS NOT NULL
      AND TRIM(sentence) <> ''
      AND TRIM(first_label) IN (
          'Not Suicide post',
          'Ideation of Suicide, Self-Harm or Harming Others',
          'Method or action of Suicide, Self-Harm or Harming others'
      )

    UNION ALL

    -- 4. generated_sentences
    SELECT TRIM(label) AS Suicide
    FROM generated_sentences
    WHERE ID < 32000
      AND sentence IS NOT NULL
      AND TRIM(sentence) <> ''
      AND COALESCE(consider, 1) = 1
      AND TRIM(label) <> 'Ideation of Suicide, Self-Harm or Harming Others'
      AND TRIM(label) IN (
          'Not Suicide post',
          'Ideation of Suicide, Self-Harm or Harming Others',
          'Method or action of Suicide, Self-Harm or Harming others'
      )

    UNION ALL

    -- 5. paraphrases4
    SELECT TRIM(original_label) AS Suicide
    FROM paraphrases4
    WHERE to_consider = 1
      AND COALESCE(consider, 1) = 1
      AND paraphrases IS NOT NULL
      AND TRIM(paraphrases) <> ''
      AND TRIM(original_label) <> 'Ideation of Suicide, Self-Harm or Harming Others'
      AND TRIM(original_label) IN (
          'Not Suicide post',
          'Ideation of Suicide, Self-Harm or Harming Others',
          'Method or action of Suicide, Self-Harm or Harming others'
      )

    UNION ALL

    -- 6. Curated augmented Ideation reintroduction set
    SELECT TRIM(label) AS Suicide
    FROM augmented_ideation_selected_100k
    WHERE selected_for_training = 1
      AND sentence IS NOT NULL
      AND TRIM(sentence) <> ''
      AND TRIM(label) = 'Ideation of Suicide, Self-Harm or Harming Others'

    UNION ALL

    -- 7. Kaggle classified sentences
    SELECT TRIM(first_label) AS Suicide
    FROM SuicideAndDepressionDetectionKaggleDataset_classified_sentences
    WHERE sentence IS NOT NULL
      AND TRIM(sentence) <> ''
      AND TRIM(first_label) IN (
          'Not Suicide post',
          'Ideation of Suicide, Self-Harm or Harming Others',
          'Method or action of Suicide, Self-Harm or Harming others'
      )


    UNION ALL

    -- 7. SuicideForum dot com classified sentences
    SELECT
        TRIM(first_label) AS Suicide
    FROM sf3.SuicideForumPosts_classified_sentences
    WHERE sentence IS NOT NULL
    AND TRIM(sentence) <> ''
    AND TRIM(first_label) IN (
        'Not Suicide post',
        'Ideation of Suicide, Self-Harm or Harming Others',
        'Method or action of Suicide, Self-Harm or Harming others'
    )
      

    UNION ALL

    -- 8. Sanctioned Suicide Forum sentence evidence labels
    SELECT
        TRIM(COALESCE(json_extract(e.value, '$.label'), l.first_label)) AS Suicide
    FROM ssf.post_safety_labels l,
        json_each(l.labelled_sentences_json) e
    WHERE l.labelled_sentences_json IS NOT NULL
    AND l.labelled_sentences_json <> '[]'
    AND json_extract(e.value, '$.sentence') IS NOT NULL
    AND TRIM(json_extract(e.value, '$.sentence')) <> ''
    AND TRIM(COALESCE(json_extract(e.value, '$.label'), l.first_label)) IN (
        'Ideation of Suicide, Self-Harm or Harming Others',
        'Method or action of Suicide, Self-Harm or Harming others'
    )


)
GROUP BY Suicide
ORDER BY count DESC;
"""

# Execute and display the result
label_counts_df = pd.read_sql_query(label_count_query, con)
print("\nLabel Counts Across All Tables:")
print(label_counts_df.to_string(index=False))


df_full = pd.read_sql_query(
    """
    -- 1. Existing labelled MHF 388 sentences
    SELECT 
        sentence AS Tweet, 
        TRIM(label) AS Suicide
    FROM MH_forum_388_sentences
    WHERE TRIM(label) IN (
        'Not Suicide post',
        'Ideation of Suicide, Self-Harm or Harming Others',
        'Method or action of Suicide, Self-Harm or Harming others'
    )

    UNION ALL

    -- 2. MHF 387: labelled sentences
    SELECT 
        sentence AS Tweet,
        TRIM(label) AS Suicide
    FROM MH_forum_387_sentences
    WHERE sentence IS NOT NULL
      AND TRIM(sentence) <> ''
      AND label IS NOT NULL
      AND TRIM(label) <> ''
      AND TRIM(label) IN (
          'Not Suicide post',
          'Ideation of Suicide, Self-Harm or Harming Others',
          'Method or action of Suicide, Self-Harm or Harming others'
      )

    UNION ALL


    -- 3. sent_labelled_389_classified_sentences: AI-classified sentences from 389
    -- Use first_label as the main three-class label.
    SELECT
        sentence AS Tweet,
        TRIM(first_label) AS Suicide
    FROM sent_labelled_389_classified_sentences
    WHERE sentence IS NOT NULL
      AND TRIM(sentence) <> ''
      AND TRIM(first_label) IN (
          'Not Suicide post',
          'Ideation of Suicide, Self-Harm or Harming Others',
          'Method or action of Suicide, Self-Harm or Harming others'
      )

    UNION ALL


    -- 4. Synthetic generated sentences using MHF keywords
    SELECT 
        sentence AS Tweet, 
        TRIM(label) AS Suicide
    FROM generated_sentences
    WHERE ID < 32000
      AND sentence IS NOT NULL
      AND TRIM(sentence) <> ''
      AND COALESCE(consider, 1) = 1
      AND TRIM(label) <> 'Ideation of Suicide, Self-Harm or Harming Others'

    UNION ALL

    -- 5. Paraphrased sentences
    SELECT 
        paraphrases AS Tweet, 
        TRIM(original_label) AS Suicide
    FROM paraphrases4
    WHERE to_consider = 1
      AND COALESCE(consider, 1) = 1
      AND paraphrases IS NOT NULL
      AND TRIM(paraphrases) <> ''
      AND TRIM(original_label) <> 'Ideation of Suicide, Self-Harm or Harming Others'
      AND TRIM(original_label) IN (
          'Not Suicide post',
          'Ideation of Suicide, Self-Harm or Harming Others',
          'Method or action of Suicide, Self-Harm or Harming others'
      )

    UNION ALL

    -- 6. Curated augmented Ideation reintroduction set
    SELECT
        sentence AS Tweet,
        TRIM(label) AS Suicide
    FROM augmented_ideation_selected_100k
    WHERE selected_for_training = 1
      AND sentence IS NOT NULL
      AND TRIM(sentence) <> ''
      AND TRIM(label) = 'Ideation of Suicide, Self-Harm or Harming Others'

    UNION ALL

    -- 7. Kaggle classified sentences
    SELECT 
        sentence AS Tweet, 
        TRIM(first_label) AS Suicide
    FROM SuicideAndDepressionDetectionKaggleDataset_classified_sentences
    WHERE sentence IS NOT NULL
      AND TRIM(sentence) <> ''
      AND TRIM(first_label) IN (
          'Not Suicide post',
          'Ideation of Suicide, Self-Harm or Harming Others',
          'Method or action of Suicide, Self-Harm or Harming others'
      )


    UNION ALL

    -- 7. SuicideForum dot com classified sentences
    SELECT
        sentence AS Tweet,
        TRIM(first_label) AS Suicide
    FROM sf3.SuicideForumPosts_classified_sentences
    WHERE sentence IS NOT NULL
    AND TRIM(sentence) <> ''
    AND TRIM(first_label) IN (
        'Not Suicide post',
        'Ideation of Suicide, Self-Harm or Harming Others',
        'Method or action of Suicide, Self-Harm or Harming others'
    )
    
    
    
    UNION ALL

    -- 8. Sanctioned Suicide Forum sentence evidence labels
    SELECT
        json_extract(e.value, '$.sentence') AS Tweet,
        TRIM(COALESCE(json_extract(e.value, '$.label'), l.first_label)) AS Suicide
    FROM ssf.post_safety_labels l,
        json_each(l.labelled_sentences_json) e
    WHERE l.labelled_sentences_json IS NOT NULL
    AND l.labelled_sentences_json <> '[]'
    AND json_extract(e.value, '$.sentence') IS NOT NULL
    AND TRIM(json_extract(e.value, '$.sentence')) <> ''
    AND TRIM(COALESCE(json_extract(e.value, '$.label'), l.first_label)) IN (
        'Ideation of Suicide, Self-Harm or Harming Others',
        'Method or action of Suicide, Self-Harm or Harming others'
    )


    """,
    con
)

# Clean labels and remove empty text rows
df_full["Tweet"] = df_full["Tweet"].fillna("").astype(str).str.strip()
df_full["Suicide"] = df_full["Suicide"].fillna("").astype(str).str.strip()

df_full = df_full[
    (df_full["Tweet"] != "")
    & (df_full["Suicide"].isin([
        "Not Suicide post",
        "Ideation of Suicide, Self-Harm or Harming Others",
        "Method or action of Suicide, Self-Harm or Harming others"
    ]))
].copy()

# Recommended: remove exact duplicate sentence-label pairs
df_full = df_full.drop_duplicates(subset=["Tweet", "Suicide"]).reset_index(drop=True)

#print("\n[INFO] Updated df_full rows after adding 387 and 389:", len(df_full))
print("\n[INFO] Updated df_full rows after adding 387, 389, SuicideForum, and SanctionedSuicide:", len(df_full))
print(df_full["Suicide"].value_counts().to_string())

# C:\projectscode\mentalHealthForums\AI_Models\twitter-suicidal-ideation-detection\machine_learning>start /affinity FFFF /high python training5_additional_datasets_improved_code.py

# Load only actual sentences for testing
# we cant consider the paraphrased and generated ones here
# df_actual = pd.read_sql_query(
#     "SELECT sentence as Tweet, label as Suicide FROM MH_forum_388_sentences"
#     "UNION ALL SELECT sentence as Tweet, first_label as Suicide FROM SuicideAndDepressionDetectionKaggleDataset_classified_sentences"
#     "UNION ALL SELECT sentence as Tweet, first_label as Suicide FROM SuicideForumPosts_classified_sentences",
#     con
# )

#The model was trained using original and augmented data, but final evaluation was conducted only on non-paraphrased human/forum data.

if USE_ACTUAL_ONLY_TEST:
    print("\n[INFO] Building actual-only dataset for final held-out testing.")

    # ---------------------------------------------------------------------
    # 1. Load all actual labelled sentence sources
    # ---------------------------------------------------------------------
    actual_df = pd.read_sql_query(
        """
        -- 1. MHF Checked -- Edited
        SELECT 
            sentence AS Tweet,
            TRIM(label) AS Suicide
        FROM MH_forum_388_sentences
        WHERE sentence IS NOT NULL
          AND TRIM(sentence) <> ''
          AND TRIM(label) IN (
              'Not Suicide post',
              'Ideation of Suicide, Self-Harm or Harming Others',
              'Method or action of Suicide, Self-Harm or Harming others'
          )

        UNION ALL

        -- 2. MHF Checked -- 387 labelled sentences
        SELECT 
            sentence AS Tweet,
            TRIM(label) AS Suicide
        FROM MH_forum_387_sentences
        WHERE sentence IS NOT NULL
          AND TRIM(sentence) <> ''
          AND label IS NOT NULL
          AND TRIM(label) <> ''
          AND TRIM(label) IN (
              'Not Suicide post',
              'Ideation of Suicide, Self-Harm or Harming Others',
              'Method or action of Suicide, Self-Harm or Harming others'
          )

        UNION ALL

        -- 3. MHF Checked -- Already Edited / 389 classified
        SELECT 
            sentence AS Tweet,
            TRIM(first_label) AS Suicide
        FROM sent_labelled_389_classified_sentences
        WHERE sentence IS NOT NULL
          AND TRIM(sentence) <> ''
          AND TRIM(first_label) IN (
              'Not Suicide post',
              'Ideation of Suicide, Self-Harm or Harming Others',
              'Method or action of Suicide, Self-Harm or Harming others'
          )

        UNION ALL

        -- 4. Kaggle classified sentences
        SELECT 
            sentence AS Tweet,
            TRIM(first_label) AS Suicide
        FROM SuicideAndDepressionDetectionKaggleDataset_classified_sentences
        WHERE sentence IS NOT NULL
          AND TRIM(sentence) <> ''
          AND TRIM(first_label) IN (
              'Not Suicide post',
              'Ideation of Suicide, Self-Harm or Harming Others',
              'Method or action of Suicide, Self-Harm or Harming others'
          )

        UNION ALL

        -- 5. SuicideForum.com classified sentences
        SELECT 
            sentence AS Tweet,
            TRIM(first_label) AS Suicide
        FROM sf3.SuicideForumPosts_classified_sentences
        WHERE sentence IS NOT NULL
          AND TRIM(sentence) <> ''
          AND TRIM(first_label) IN (
              'Not Suicide post',
              'Ideation of Suicide, Self-Harm or Harming Others',
              'Method or action of Suicide, Self-Harm or Harming others'
          )

        UNION ALL

        -- 6. Sanctioned Suicide Forum labelled sentence evidence
        SELECT
            json_extract(e.value, '$.sentence') AS Tweet,
            TRIM(COALESCE(json_extract(e.value, '$.label'), l.first_label)) AS Suicide
        FROM ssf.post_safety_labels l,
             json_each(l.labelled_sentences_json) e
        WHERE l.labelled_sentences_json IS NOT NULL
          AND l.labelled_sentences_json <> '[]'
          AND json_extract(e.value, '$.sentence') IS NOT NULL
          AND TRIM(json_extract(e.value, '$.sentence')) <> ''
          AND TRIM(COALESCE(json_extract(e.value, '$.label'), l.first_label)) IN (
              'Ideation of Suicide, Self-Harm or Harming Others',
              'Method or action of Suicide, Self-Harm or Harming others'
          )
        """,
        con
    )

    # Clean actual dataset
    actual_df["Tweet"] = actual_df["Tweet"].fillna("").astype(str).str.strip()
    actual_df["Suicide"] = actual_df["Suicide"].fillna("").astype(str).str.strip()

    actual_df = actual_df[
        (actual_df["Tweet"] != "")
        & (actual_df["Suicide"].isin([
            "Not Suicide post",
            "Ideation of Suicide, Self-Harm or Harming Others",
            "Method or action of Suicide, Self-Harm or Harming others"
        ]))
    ].copy()

    actual_df = actual_df.drop_duplicates(subset=["Tweet", "Suicide"]).reset_index(drop=True)

    print("\n[INFO] Actual-only rows before train/test split:", len(actual_df))
    print(actual_df["Suicide"].value_counts().to_string())

    # ---------------------------------------------------------------------
    # 2. Split actual data into actual_train and actual_test
    # ---------------------------------------------------------------------
    actual_train_df, df_actual = train_test_split(
        actual_df,
        test_size=0.25,
        random_state=42,
        stratify=actual_df["Suicide"]
    )

    actual_train_df = actual_train_df.reset_index(drop=True)
    df_actual = df_actual.reset_index(drop=True)

    print("\n[INFO] Actual training rows:", len(actual_train_df))
    print(actual_train_df["Suicide"].value_counts().to_string())

    print("\n[INFO] Held-out actual-only df_actual test rows:", len(df_actual))
    print(df_actual["Suicide"].value_counts().to_string())

    # ---------------------------------------------------------------------
    # 3. Load augmented data separately
    # ---------------------------------------------------------------------
    augmented_df = pd.read_sql_query(
        """
        -- 1. Synthetic generated sentences
        SELECT 
            sentence AS Tweet,
            TRIM(label) AS Suicide
        FROM generated_sentences
        WHERE ID < 32000
          AND sentence IS NOT NULL
          AND TRIM(sentence) <> ''
          AND COALESCE(consider, 1) = 1
          AND TRIM(label) <> 'Ideation of Suicide, Self-Harm or Harming Others'
          AND TRIM(label) IN (
              'Not Suicide post',
              'Ideation of Suicide, Self-Harm or Harming Others',
              'Method or action of Suicide, Self-Harm or Harming others'
          )

        UNION ALL

        -- 2. Paraphrased sentences
        SELECT 
            paraphrases AS Tweet,
            TRIM(original_label) AS Suicide
        FROM paraphrases4
        WHERE to_consider = 1
          AND COALESCE(consider, 1) = 1
          AND paraphrases IS NOT NULL
          AND TRIM(paraphrases) <> ''
          AND TRIM(original_label) <> 'Ideation of Suicide, Self-Harm or Harming Others'
          AND TRIM(original_label) IN (
              'Not Suicide post',
              'Ideation of Suicide, Self-Harm or Harming Others',
              'Method or action of Suicide, Self-Harm or Harming others'
          )

        UNION ALL

        -- 3. Curated augmented Ideation reintroduction set
        SELECT
            sentence AS Tweet,
            TRIM(label) AS Suicide
        FROM augmented_ideation_selected_100k
        WHERE selected_for_training = 1
          AND sentence IS NOT NULL
          AND TRIM(sentence) <> ''
          AND TRIM(label) = 'Ideation of Suicide, Self-Harm or Harming Others'
        """,
        con
    )

    augmented_df["Tweet"] = augmented_df["Tweet"].fillna("").astype(str).str.strip()
    augmented_df["Suicide"] = augmented_df["Suicide"].fillna("").astype(str).str.strip()

    augmented_df = augmented_df[
        (augmented_df["Tweet"] != "")
        & (augmented_df["Suicide"].isin([
            "Not Suicide post",
            "Ideation of Suicide, Self-Harm or Harming Others",
            "Method or action of Suicide, Self-Harm or Harming others"
        ]))
    ].copy()

    augmented_df = augmented_df.drop_duplicates(subset=["Tweet", "Suicide"]).reset_index(drop=True)

    print("\n[INFO] Augmented rows for training only:", len(augmented_df))
    print(augmented_df["Suicide"].value_counts().to_string())

    # ---------------------------------------------------------------------
    # 4. Rebuild df_full for training
    # ---------------------------------------------------------------------
    # Important:
    # df_full now contains actual_train + augmented data.
    # df_actual remains held-out actual-only test data.
    df_full = pd.concat(
        [actual_train_df, augmented_df],
        ignore_index=True
    )

    df_full = df_full.drop_duplicates(subset=["Tweet", "Suicide"]).reset_index(drop=True)

    print("\n[INFO] Final df_full training rows after held-out split:", len(df_full))
    print(df_full["Suicide"].value_counts().to_string())

else:
    print(
        "\n[WARN] MHFS_USE_ACTUAL_ONLY_TEST=0, so df_actual = df_full will be used. "
        "This is not recommended for final paper results because it may include generated/paraphrased data "
        "and possible training-test overlap."
    )
    df_actual = df_full



# Check if DataFrame is empty



#x_train, x_test, y_train, y_test = train_test_split(df["tweet"],df["Suicide"], test_size = 0.25, random_state = 42)
# Split df_full for training and validation
# X_train_full, X_val, y_train_full, y_val = train_test_split(
#     df_full['Tweet'], df_full['Suicide'], test_size=0.25, random_state=42
# )

# # Use only actual sentences for testing
# X_test = df_actual['Tweet']
# y_test = df_actual['Suicide']

# Use all available training data for traditional ML models.
# df_actual remains the held-out actual-only test set.
NEED_TFIDF = machine_learning or RUN_STACKING_CLASSIFIER
if NEED_TFIDF:
    
    # **Applying N-gram**
    print('**Applying N-gram**')
    
    X_train_full = df_full["Tweet"]
    y_train_full = df_full["Suicide"]

    X_test = df_actual["Tweet"]
    y_test = df_actual["Suicide"]

    # Define the vectorizer and transformer for N-grams.
    # Use trigrams and preserve apostrophes so negation scopes such as
    # "don't think suicidal", "not want to die", and "never thought suicide"
    # are visible to the model as explicit features.
    NEGATION_AWARE_TOKEN_PATTERN = r"(?u)\b\w[\w']+\b"
    count_vect = CountVectorizer(
        ngram_range=(1, 3),
        token_pattern=NEGATION_AWARE_TOKEN_PATTERN,
    )
    transformer = TfidfTransformer(norm='l2', sublinear_tf=True)

    # Transform X_train_full and X_val using CountVectorizer and TfidfTransformer
    x_train_counts = count_vect.fit_transform(X_train_full)
    x_train_tfidf = transformer.fit_transform(x_train_counts)



    # ---------------------------------------------------------------------
    # Save fitted TF-IDF preprocessing objects for later prediction/auditing
    # ---------------------------------------------------------------------
    # These fitted objects are required to use saved ML models such as
    # Suicide_SVM.pkl on new raw sentences. Without these exact fitted
    # objects, new text cannot be converted into the same feature space used
    # during training.

    count_vectorizer_path = MODELS_DIR / 'count_vectorizer_ngram_1_3_negation_aware.joblib'
    tfidf_transformer_path = MODELS_DIR / 'tfidf_transformer_l2_sublinear.joblib'

    joblib.dump(count_vect, count_vectorizer_path)
    joblib.dump(transformer, tfidf_transformer_path)

    print(f"Fitted CountVectorizer saved as {count_vectorizer_path}")
    print(f"Fitted TfidfTransformer saved as {tfidf_transformer_path}")

    # x_val_counts = count_vect.transform(X_val)
    # x_val_tfidf = transformer.transform(x_val_counts)

    # Transform X_test, containing only actual sentences, for evaluation
    x_test_counts = count_vect.transform(X_test)
    x_test_tfidf = transformer.transform(x_test_counts)

    # Print shapes for consistency verification
    print(x_train_tfidf.shape, x_test_tfidf.shape, y_train_full.shape, y_test.shape)
    print("x_train_tfidf shape:", x_train_tfidf.shape)
    print("y_train_full shape:", y_train_full.shape)
    print("x_test_tfidf shape:", x_test_tfidf.shape)
    print("y_test shape:", y_test.shape)
    print('\n')

    # Convert y_train_full and y_test to strings if necessary
    y_train_full = y_train_full.astype(str)
    y_test = y_test.astype(str)


# =============================================================================
# PIPELINE-BASED CROSS-VALIDATION
# =============================================================================
#
# Important:
#   This function performs cross-validation correctly by fitting the
#   CountVectorizer and TfidfTransformer inside each fold.
#
#   This avoids vocabulary leakage, because the vectorizer is not fitted on the
#   full training set before cross-validation.
#
#   Use raw text and labels here, not x_train_tfidf.
# =============================================================================

def run_text_pipeline_cv(model, model_name, X_text, y_labels, cv_splits=5):
    """
    Run leakage-safe cross-validation using a full text-classification pipeline.

    Parameters
    ----------
    model:
        The sklearn classifier, for example LinearSVC() or LogisticRegression().

    model_name:
        Name printed in the output.

    X_text:
        Raw text sentences.

    y_labels:
        Class labels.

    cv_splits:
        Number of folds. Use 5 for current paper consistency, or 10 if needed.

    Returns
    -------
    scores:
        Cross-validation macro F1 scores.
    """

    print(f"\n[{model_name}] Running leakage-safe {cv_splits}-fold pipeline CV...")

    cv_pipeline = Pipeline([
        ("count_vectorizer", CountVectorizer(
            ngram_range=(1, 3),
            token_pattern=NEGATION_AWARE_TOKEN_PATTERN,
        )),
        ("tfidf_transformer", TfidfTransformer(norm="l2", sublinear_tf=True)),
        ("model", model)
    ])

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        cv_pipeline,
        X_text,
        y_labels,
        cv=cv,
        scoring="f1_macro",
        n_jobs=CV_N_JOBS
    )

    print(f"[{model_name}] Cross-validated macro F1 scores:", scores)
    print(f"[{model_name}] Mean macro F1:", scores.mean())
    print(f"[{model_name}] Std macro F1:", scores.std())

    return scores

if machine_learning:
    # for soem reason the entire code does not execute in one go. It gets stops just after this, so putting this into if/else so we can continue from here afterwards
    if not_done:

        #**Machine Learning Models**
        # **Logistic Regression**
        print('**Logistic Regression**')

        lr = LogisticRegression(C=2, max_iter=1000)
        lr.fit(x_train_tfidf, y_train_full)
        y_pred1 = lr.predict(x_test_tfidf)
        print("Accuracy: " + str(accuracy_score(y_test, y_pred1)))
        print(classification_report(y_test, y_pred1))

        if SKIP_EXPENSIVE_CV:
            scores = []
            print('[SKIPPED] pipeline cross-validation skipped because MHFS_SKIP_EXPENSIVE_CV=1')

        else:
            scores = run_text_pipeline_cv(
                model=LogisticRegression(C=2, max_iter=1000),
                model_name="Logistic Regression",
                X_text=X_train_full,
                y_labels=y_train_full,
                cv_splits=5
            )

            print("Held-out actual-only test accuracy:", accuracy_score(y_test, y_pred1))

        # Initialize the label encoder and fit on the actual classes
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)

        # Transform y_test and y_pred1 into encoded numeric labels
        y_test_encoded = label_encoder.transform(y_test)
        y_pred1_encoded = label_encoder.transform(y_pred1)

        # Calculate additional metrics and generate probability predictions for confidence analysis
        y_pred_proba = lr.predict_proba(x_test_tfidf)  # Probability predictions for each class
        #metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
        # Call calculate_metrics with encoded labels
        print('\n')
        metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)
        # Assuming you have y_test, y_pred1, and y_pred_proba available
        #metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=lr.predict_proba(x_test_tfidf), classes=lr.classes_)
        print('\n')

        # Save the model
        model_filename = str(MODELS_DIR / 'logistic_regression_model.joblib')
        joblib.dump(lr, model_filename)
        print(f"Model saved as {model_filename}")
        print('\n')
        #To load the model in the future, use:
        #loaded_model = joblib.load(str(MODELS_DIR / 'logistic_regression_model.joblib'))

        # Perform comprehensive misclassification analysis
        perform_misclassification_analysis(
            df_actual=df_actual,            # Actual test data
            y_test=y_test,                  # Actual labels
            y_pred=y_pred1,                 # Predicted labels
            algorithm_name="Logistic_Regression",
            y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
            labels=lr.classes_,             # Class labels for confusion matrix
            label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
        )
        print('\n')

        # **Support Vector Machine**
        print('**Support Vector Machine**')
        print('\n')
        svc = LinearSVC()
        svc.fit(x_train_tfidf, y_train_full)
        y_pred2 = svc.predict(x_test_tfidf)
        print("Accuracy: " + str(accuracy_score(y_test, y_pred2)))
        print(classification_report(y_test, y_pred2))

        if SKIP_EXPENSIVE_CV:
            scores = []
            print('[SKIPPED] cross_val_score skipped because MHFS_SKIP_EXPENSIVE_CV=1')

        else:
            scores = run_text_pipeline_cv(
                model=LinearSVC(),
                model_name="Support Vector Machine",
                X_text=X_train_full,
                y_labels=y_train_full,
                cv_splits=5,
            )
            print("Held-out actual-only test accuracy:", accuracy_score(y_test, y_pred2))

        # Assuming you have y_test, y_pred1, and y_pred_proba available
        #y_pred_proba=lr.predict_proba(x_test_tfidf)
        #metrics = calculate_metrics(y_test, y_pred1, y_pred_proba, classes=lr.classes_)

        # Initialize the label encoder and fit on the actual classes
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)

        # Transform y_test and y_pred1 into encoded numeric labels
        y_test_encoded = label_encoder.transform(y_test)
        y_pred1_encoded = label_encoder.transform(y_pred2)

        # Calculate additional metrics and generate probability predictions for confidence analysis
        y_pred_proba = None  # LinearSVC does not provide predict_proba by default
        #metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
        # Call calculate_metrics with encoded labels
        print('\n')
        metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)
        print('\n')

        # joblib.dump(svc, str(MODELS_DIR / 'Suicide_SVM.pkl'))
        # print(f"Model saved as " + " models/Suicide_SVM.pkl")
        # print('\n')

        joblib.dump(svc, str(MODELS_DIR / 'Suicide_SVM.pkl'))
        print(f"Model saved as " + " models/Suicide_SVM.pkl")
        print('\n')

        # -----------------------------------------------------------------
        # Save a complete SVM inference pipeline
        # -----------------------------------------------------------------
        # This pipeline allows future scripts to call:
        #     pipeline.predict(list_of_raw_sentences)
        # without manually loading and applying CountVectorizer and
        # TfidfTransformer separately.

        svm_pipeline = Pipeline([
            ('count_vectorizer', count_vect),
            ('tfidf_transformer', transformer),
            ('svm', svc)
        ])

        svm_pipeline_path = MODELS_DIR / 'Suicide_SVM_pipeline.joblib'
        joblib.dump(svm_pipeline, svm_pipeline_path)
        print(f"Complete SVM prediction pipeline saved as {svm_pipeline_path}")

        # Also save a small bundle with metadata for auditing and reuse.
        svm_bundle = {
            'model_name': 'LinearSVC',
            'model': svc,
            'count_vectorizer': count_vect,
            'tfidf_transformer': transformer,
            'classes': list(svc.classes_),
            'ngram_range': (1, 3),
            'token_pattern': NEGATION_AWARE_TOKEN_PATTERN,
            'tfidf_norm': 'l2',
            'tfidf_sublinear_tf': True,
            'trained_on_column': 'Tweet',
            'target_column': 'Suicide',
            'created_at': RUN_STAMP
        }

        svm_bundle_path = MODELS_DIR / 'Suicide_SVM_with_vectorizer_bundle.joblib'
        joblib.dump(svm_bundle, svm_bundle_path)
        print(f"SVM bundle with vectorizer and transformer saved as {svm_bundle_path}")

        # Quick sanity check using the saved pipeline.
        sample_texts = [
            'I need help and I feel unsafe',
            'This is a general discussion post'
        ]

        sample_predictions = svm_pipeline.predict(sample_texts)

        print("SVM pipeline sanity-check predictions:")
        for sample_text, sample_prediction in zip(sample_texts, sample_predictions):
            print(f"  TEXT: {sample_text}")
            print(f"  PRED: {sample_prediction}")
        print('\n')


        # to load the model in the future
        # loaded_model = joblib.load('logistic_regression_model.joblib')

        # Perform comprehensive misclassification analysis
        perform_misclassification_analysis(
            df_actual=df_actual,            # Actual test data
            y_test=y_test,                  # Actual labels
            y_pred=y_pred2,                 # Predicted labels
            algorithm_name="Support_Vector_Machine",
            y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
            labels=lr.classes_,              # Class labels for confusion matrix
            label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
        )

        ## **Naive Bayes (Multinomial)**
        print('**Naive Bayes (Multinomial)**')

        mnb = MultinomialNB()
        mnb.fit(x_train_tfidf, y_train_full)
        y_pred3 = mnb.predict(x_test_tfidf)
        print("Accuracy: " + str(accuracy_score(y_test, y_pred3)))
        print(classification_report(y_test, y_pred3))

        if SKIP_EXPENSIVE_CV:
            scores = []
            print('[SKIPPED] pipeline cross-validation skipped because MHFS_SKIP_EXPENSIVE_CV=1')

        else:
            scores = run_text_pipeline_cv(
                model=MultinomialNB(),
                model_name="Naive Bayes",
                X_text=X_train_full,
                y_labels=y_train_full,
                cv_splits=5
            )
            print("Held-out actual-only test accuracy:", accuracy_score(y_test, y_pred3))

        # Assuming you have y_test, y_pred1, and y_pred_proba available
        y_pred_proba=lr.predict_proba(x_test_tfidf)
        #metrics = calculate_metrics(y_test, y_pred3, y_pred_proba, classes=lr.classes_)

        # Initialize the label encoder and fit on the actual classes
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)

        # Transform y_test and y_pred1 into encoded numeric labels
        y_test_encoded = label_encoder.transform(y_test)
        y_pred1_encoded = label_encoder.transform(y_pred3)

        # Calculate additional metrics and generate probability predictions for confidence analysis
        y_pred_proba = mnb.predict_proba(x_test_tfidf)  # Probability predictions for Naive Bayes
        #metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
        print('\n')
        # Call calculate_metrics with encoded labels
        metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)
        print('\n')

        # Save the model
        model_filename = str(MODELS_DIR / 'naive_bayes_model.joblib')
        joblib.dump(mnb, model_filename)
        print(f"Model saved as {model_filename}")

        #To load this model later:
        #loaded_model = joblib.load(str(MODELS_DIR / 'naive_bayes_model.joblib'))
        print('\n')
        # Perform comprehensive misclassification analysis
        perform_misclassification_analysis(
            df_actual=df_actual,            # Actual test data
            y_test=y_test,                  # Actual labels
            y_pred=y_pred3,                 # Predicted labels
            algorithm_name="Naive_Bayes",
            y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
            labels=lr.classes_,             # Class labels for confusion matrix
            label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
        )
        print('\n')

        ## **Random Forest**
        print('**Random Forest**')

        rfc = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')
        rfc.fit(x_train_tfidf, y_train_full)
        y_pred4 = rfc.predict(x_test_tfidf)
        print("Accuracy: " + str(accuracy_score(y_test, y_pred4)))
        print(classification_report(y_test, y_pred4))

        if SKIP_EXPENSIVE_CV:


            scores = []


            print('[SKIPPED] cross_val_score skipped because MHFS_SKIP_EXPENSIVE_CV=1')


        else:
            scores = run_text_pipeline_cv(
                model=RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=1,
                ),
                model_name="Random Forest",
                X_text=X_train_full,
                y_labels=y_train_full,
                cv_splits=5,
            )
            print("Held-out actual-only test accuracy:", accuracy_score(y_test, y_pred4))

        # Assuming you have y_test, y_pred1, and y_pred_proba available
        #y_pred_proba=lr.predict_proba(x_test_tfidf),
        #metrics = calculate_metrics(y_test, y_pred4, y_pred_proba, classes=lr.classes_)

        # Initialize the label encoder and fit on the actual classes
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)

        # Transform y_test and y_pred1 into encoded numeric labels
        y_test_encoded = label_encoder.transform(y_test)
        y_pred1_encoded = label_encoder.transform(y_pred4)

        # Calculate additional metrics and generate probability predictions for confidence analysis
        y_pred_proba = rfc.predict_proba(x_test_tfidf)  # Probability predictions for Random Forest
        #metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
        # Call calculate_metrics with encoded labels
        print('\n')
        metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)
        print('\n')
        # Save the model
        model_filename = str(MODELS_DIR / 'random_forest_model.joblib')
        joblib.dump(rfc, model_filename)
        print(f"Model saved as {model_filename}")
        print('\n')
        #To load the saved Random Forest model later:
        #loaded_model = joblib.load('random_forest_model.joblib')

        # Perform comprehensive misclassification analysis
        perform_misclassification_analysis(
            df_actual=df_actual,            # Actual test data
            y_test=y_test,                  # Actual labels
            y_pred=y_pred4,                 # Predicted labels
            algorithm_name="Random_Forest",
            y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
            labels=lr.classes_,              # Class labels for confusion matrix
            label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
        )

        # **Gradient Boosting Classifier**
        print('**Gradient Boosting Classifier**')

        gbc = GradientBoostingClassifier(n_estimators=1000, max_features='sqrt', max_depth=4, random_state=1, verbose=1)
        gbc.fit(x_train_tfidf, y_train_full)
        y_pred5 = gbc.predict(x_test_tfidf)
        print("Accuracy: " + str(accuracy_score(y_test, y_pred5)))
        print(classification_report(y_test, y_pred5))

        if SKIP_EXPENSIVE_CV:


            scores = []


            print('[SKIPPED] cross_val_score skipped because MHFS_SKIP_EXPENSIVE_CV=1')


        else:
            scores = run_text_pipeline_cv(
                model=GradientBoostingClassifier(
                    n_estimators=1000,
                    max_features="sqrt",
                    max_depth=4,
                    random_state=1,
                    verbose=0,
                ),
                model_name="Gradient Boosting",
                X_text=X_train_full,
                y_labels=y_train_full,
                cv_splits=5,
            )
            print("Held-out actual-only test accuracy:", accuracy_score(y_test, y_pred5))

        # Assuming you have y_test, y_pred1, and y_pred_proba available
        #metrics = calculate_metrics(y_test, y_pred5, y_pred_proba=lr.predict_proba(x_test_tfidf), classes=lr.classes_)

        # Initialize the label encoder and fit on the actual classes
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)

        # Transform y_test and y_pred1 into encoded numeric labels
        y_test_encoded = label_encoder.transform(y_test)
        y_pred1_encoded = label_encoder.transform(y_pred5)

        # Calculate additional metrics and generate probability predictions for confidence analysis
        y_pred_proba = gbc.predict_proba(x_test_tfidf)  # Probability predictions for Gradient Boosting
        #metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
        # Call calculate_metrics with encoded labels
        metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)

        # Save the model
        model_filename = str(MODELS_DIR / 'gradient_boosting_model.joblib')
        joblib.dump(gbc, model_filename)
        print(f"Model saved as {model_filename}")

        #To load the saved model:
        #loaded_model = joblib.load('gradient_boosting_model.joblib')

        # Perform comprehensive misclassification analysis
        perform_misclassification_analysis(
            df_actual=df_actual,            # Actual test data
            y_test=y_test,                  # Actual labels
            y_pred=y_pred5,                 # Predicted labels
            algorithm_name="Gradient_Boosting_Classifier",
            y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
            labels=lr.classes_,              # Class labels for confusion matrix
            label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
        )



    ## **Ensemble Classifier**
    y_pred6 = None
    if RUN_ENSEMBLE_CLASSIFIER:
        print('**Ensemble Classifier**')
        print('[INFO] MHFS_RUN_ENSEMBLE=1, running the heavy ensemble block.')

        print("Training Multinomial Naive Bayes...")
        mnb = MultinomialNB()
        print("Training Random Forest Classifier...")
        rfc = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=1)
        print("Training Logistic Regression...")
        ensemble_lr = LogisticRegression(C=2, max_iter=1000)
        print("Training Support Vector Machine...")
        svc = SVC(probability=True)

        print("Fitting the ensemble classifier...")
        ec = VotingClassifier(
            estimators=[
                ('Multinomial NB', mnb),
                ('Random Forest', rfc),
                ('Logistic Regression', ensemble_lr),
                ('Support Vector Machine', svc),
            ],
            voting='soft',
            weights=[1, 2, 3, 4],
            n_jobs=1,
        )

        try:
            ec.fit(x_train_tfidf, y_train_full)
            y_pred6 = ec.predict(x_test_tfidf)
            print("Accuracy: " + str(accuracy_score(y_test, y_pred6)))
            print(classification_report(y_test, y_pred6))

            if SKIP_EXPENSIVE_CV:
                scores = []
                print('[SKIPPED] cross_val_score skipped because MHFS_SKIP_EXPENSIVE_CV=1')
            else:
                scores = cross_val_score(ec, x_train_tfidf, y_train_full, cv=5)
                print(accuracy_score(y_test, y_pred6))
                print("Cross-validated scores:", scores)

            label_encoder = LabelEncoder()
            label_encoder.fit(y_test)
            y_test_encoded = label_encoder.transform(y_test)
            y_pred6_encoded = label_encoder.transform(y_pred6)
            y_pred_proba = ec.predict_proba(x_test_tfidf)
            metrics = calculate_metrics(
                y_test_encoded,
                y_pred6_encoded,
                y_pred_proba=y_pred_proba,
                classes=label_encoder.classes_,
            )

            model_filename = str(MODELS_DIR / 'Suicide_Ensemble.pkl')
            joblib.dump(ec, model_filename)
            print(f"Model saved as {model_filename}")

            perform_misclassification_analysis(
                df_actual=df_actual,
                y_test=y_test,
                y_pred=y_pred6,
                algorithm_name="Ensemble_Classifier",
                y_pred_proba=y_pred_proba,
                labels=label_encoder.classes_,
                label_encoder=label_encoder,
            )
        except (MemoryError, OSError) as exc:
            y_pred6 = None
            print(f"[SKIPPED] Ensemble Classifier failed due to system resources: {exc}")
    else:
        print('[SKIPPED] Ensemble Classifier skipped. Set MHFS_RUN_ENSEMBLE=1 to run it.')

    ## **AdaBoost with Random Forest Classifier**
    y_pred7 = None
    if RUN_ADABOOST_CLASSIFIER:
        print('**AdaBoost with Random Forest Classifier**')
        print('[INFO] MHFS_RUN_ADABOOST=1, running the heavy AdaBoost block.')

        rfc = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=0, n_jobs=1)
        abc = AdaBoostClassifier(estimator=rfc, learning_rate=0.2, n_estimators=100)
        try:
            abc.fit(x_train_tfidf, y_train_full)
            y_pred7 = abc.predict(x_test_tfidf)
            print("Accuracy: " + str(accuracy_score(y_test, y_pred7)))
            print(classification_report(y_test, y_pred7))

            if SKIP_EXPENSIVE_CV:
                scores = []
                print('[SKIPPED] cross_val_score skipped because MHFS_SKIP_EXPENSIVE_CV=1')
            else:
                scores = cross_val_score(abc, x_train_tfidf, y_train_full, cv=5)
                print(accuracy_score(y_test, y_pred7))
                print("Cross-validated scores:", scores)

            label_encoder = LabelEncoder()
            label_encoder.fit(y_test)
            y_test_encoded = label_encoder.transform(y_test)
            y_pred7_encoded = label_encoder.transform(y_pred7)
            y_pred_proba = abc.predict_proba(x_test_tfidf)
            metrics = calculate_metrics(
                y_test_encoded,
                y_pred7_encoded,
                y_pred_proba=y_pred_proba,
                classes=label_encoder.classes_,
            )

            model_filename = str(MODELS_DIR / 'adaboost_random_forest_model.joblib')
            joblib.dump(abc, model_filename)
            print(f"Model saved as {model_filename}")

            perform_misclassification_analysis(
                df_actual=df_actual,
                y_test=y_test,
                y_pred=y_pred7,
                algorithm_name="AdaBoost_with_Random_Forest_Classifier",
                y_pred_proba=y_pred_proba,
                labels=label_encoder.classes_,
                label_encoder=label_encoder,
            )
        except (MemoryError, OSError) as exc:
            y_pred7 = None
            print(f"[SKIPPED] AdaBoost failed due to system resources: {exc}")
    else:
        print('[SKIPPED] AdaBoost with Random Forest skipped. Set MHFS_RUN_ADABOOST=1 to run it.')

    # **Comparison Between ML Models**
    print('**Comparison Between ML Models**')

    comparison_predictions = {
        'Logistic Regression': y_pred1,
        'SVM': y_pred2,
        'Naive Bayes': y_pred3,
        'Random Forest': y_pred4,
        'GradientBoosting': y_pred5,
    }
    if y_pred6 is not None:
        comparison_predictions['Ensembled'] = y_pred6
    if y_pred7 is not None:
        comparison_predictions['Adaboost'] = y_pred7

    Comparison_unibi = pd.DataFrame({
        name: [
            accuracy_score(y_test, pred) * 100,
            f1_score(y_test, pred, average='macro') * 100,
            recall_score(y_test, pred, average='macro', zero_division=0) * 100,
            precision_score(y_test, pred, average='macro', zero_division=0) * 100,
        ]
        for name, pred in comparison_predictions.items()
    })

    print('Comparison using uni-bi-trigram features (1,3)')
    Comparison_unibi.rename(index={0: 'Accuracy', 1: 'F1_score', 2: 'Recall', 3: 'Precision'}, inplace=True)
    print(Comparison_unibi.to_string())



# =============================================================================
# OPTIONAL STACKING CLASSIFIER — corrected to use TF-IDF rather than raw strings
# =============================================================================
if RUN_STACKING_CLASSIFIER:
    print("\n**Stacking Classifier**")
    stacking_model = StackingClassifier(
        estimators=[
            ("svc", SVC(probability=True, kernel="linear", C=1.0)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                ),
            ),
            ("lr", LogisticRegression(max_iter=1000)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=1,
    )
    stacking_model.fit(x_train_tfidf, y_train_full)
    stacking_pred = stacking_model.predict(x_test_tfidf)
    stacking_proba = stacking_model.predict_proba(x_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, stacking_pred))
    print(classification_report(y_test, stacking_pred, zero_division=0))
    joblib.dump(stacking_model, MODELS_DIR / "stacking_model.joblib")
    perform_misclassification_analysis(
        df_actual=df_actual,
        y_test=y_test,
        y_pred=stacking_pred,
        algorithm_name="Stacking_Classifier",
        y_pred_proba=stacking_proba,
        labels=list(stacking_model.classes_),
        label_encoder=FixedLabelEncoder(list(stacking_model.classes_)),
    )
else:
    print("[SKIPPED] Stacking classifier. Set MHFS_RUN_STACKING=1 to run it.")


# =============================================================================
# PYTORCH DEEP-LEARNING MODELS
# =============================================================================
CLASS_NAMES = [
    "Not Suicide post",
    "Method or action of Suicide, Self-Harm or Harming others",
    "Ideation of Suicide, Self-Harm or Harming Others",
]
LABEL_TO_ID = {label: idx for idx, label in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
FIXED_ENCODER = FixedLabelEncoder(CLASS_NAMES)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize_text(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def build_vocabulary(
    texts: Iterable[str],
    max_size: int,
    min_frequency: int = 2,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text_value in texts:
        counts.update(tokenize_text(text_value))
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counts.most_common(max(0, max_size - len(vocabulary))):
        if count < min_frequency:
            break
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int], max_length: int) -> list[int]:
    ids = [vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in tokenize_text(text)]
    ids = ids[:max_length]
    if len(ids) < max_length:
        ids.extend([vocabulary[PAD_TOKEN]] * (max_length - len(ids)))
    return ids


class TextSequenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        vocabulary: dict[str, int],
        max_length: int,
    ) -> None:
        self.texts = frame["Tweet"].astype(str).tolist()
        self.labels = [LABEL_TO_ID[label] for label in frame["Suicide"].astype(str)]
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = encode_text(self.texts[index], self.vocabulary, self.max_length)
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        cell: str = "lstm",
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        rnn_cls = nn.LSTM if cell.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        _, hidden = self.rnn(embedded)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        if self.rnn.bidirectional:
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            features = hidden[-1]
        return self.classifier(self.dropout(features))


class CNNLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, 96, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(96, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids).transpose(1, 2)
        x = self.pool(torch.relu(self.conv(x))).transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(features))


class HybridCNNLSTMGRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, 96, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(96, 128, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(256, 96, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids).transpose(1, 2)
        x = self.pool(torch.relu(self.conv(x))).transpose(1, 2)
        x, _ = self.lstm(x)
        _, hidden = self.gru(x)
        return self.classifier(self.dropout(hidden[-1]))


class AttentionLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            128,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(0)
        sequence, _ = self.lstm(self.embedding(input_ids))
        scores = self.attention(sequence).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(context))


@torch.no_grad()
def predict_torch_model(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_proba: list[np.ndarray] = []
    for input_ids, labels in loader:
        logits = model(input_ids.to(DEVICE))
        probabilities = torch.softmax(logits, dim=1)
        all_true.append(labels.numpy())
        all_pred.append(probabilities.argmax(dim=1).cpu().numpy())
        all_proba.append(probabilities.cpu().numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_proba),
    )


def evaluate_torch_model(
    model: nn.Module,
    loader: DataLoader,
    algorithm_name: str,
    test_frame: pd.DataFrame,
) -> dict[str, float]:
    y_true_ids, y_pred_ids, probabilities = predict_torch_model(model, loader)
    y_true_labels = [ID_TO_LABEL[int(value)] for value in y_true_ids]
    y_pred_labels = [ID_TO_LABEL[int(value)] for value in y_pred_ids]
    print(f"\n[{algorithm_name}] Held-out actual-only classification report:")
    print(
        classification_report(
            y_true_labels,
            y_pred_labels,
            labels=CLASS_NAMES,
            zero_division=0,
        )
    )
    metrics = calculate_metrics(
        y_true_ids,
        y_pred_ids,
        y_pred_proba=probabilities,
        classes=CLASS_NAMES,
    )
    report = pd.DataFrame(
        classification_report(
            y_true_labels,
            y_pred_labels,
            labels=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report.to_csv(
        REPORTS_DIR / f"classification_report_{algorithm_name}.csv",
        encoding="utf-8-sig",
    )
    perform_misclassification_analysis(
        df_actual=test_frame,
        y_test=y_true_labels,
        y_pred=y_pred_labels,
        algorithm_name=algorithm_name,
        y_pred_proba=probabilities,
        labels=CLASS_NAMES,
        label_encoder=FIXED_ENCODER,
    )
    return metrics


def train_torch_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    test_frame: pd.DataFrame,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> dict[str, float]:
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)

        val_true, val_pred, _ = predict_torch_model(model, val_loader)
        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        train_loss = running_loss / max(1, len(train_loader.dataset))
        print(
            f"[{model_name}] epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.5f} val_macro_f1={val_f1:.5f}"
        )

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[{model_name}] Early stopping.")
                break

    model.load_state_dict(best_state)
    checkpoint_path = MODELS_DIR / f"{model_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "class_names": CLASS_NAMES,
            "created_at": RUN_STAMP,
        },
        checkpoint_path,
    )
    print(f"[{model_name}] Best checkpoint saved to: {checkpoint_path}")
    return evaluate_torch_model(model, test_loader, model_name, test_frame)


if RUN_DEEP_LEARNING:
    print("\n" + "=" * 78)
    print("PYTORCH DEEP-LEARNING MODELS")
    print("=" * 78)

    DL_VOCAB_SIZE = int(os.environ.get("MHFS_DL_VOCAB_SIZE", "12000"))
    DL_MAX_LEN = int(os.environ.get("MHFS_DL_MAX_LEN", "60"))
    DL_EMBEDDING_DIM = int(os.environ.get("MHFS_DL_EMBEDDING_DIM", "128"))
    DL_BATCH_SIZE = int(os.environ.get("MHFS_DL_BATCH_SIZE", "64"))
    DL_EPOCHS = int(os.environ.get("MHFS_DL_EPOCHS", "10"))
    DL_PATIENCE = int(os.environ.get("MHFS_DL_PATIENCE", "3"))
    DL_LR = float(os.environ.get("MHFS_DL_LEARNING_RATE", "0.001"))
    DL_NUM_WORKERS = int(os.environ.get("MHFS_DL_NUM_WORKERS", "0"))

    dl_train_df, dl_val_df = train_test_split(
        df_full[["Tweet", "Suicide"]].copy(),
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=df_full["Suicide"],
    )
    dl_train_df = dl_train_df.reset_index(drop=True)
    dl_val_df = dl_val_df.reset_index(drop=True)
    dl_test_df = df_actual[["Tweet", "Suicide"]].reset_index(drop=True).copy()

    vocabulary = build_vocabulary(
        dl_train_df["Tweet"].astype(str),
        max_size=DL_VOCAB_SIZE,
        min_frequency=2,
    )
    with (MODELS_DIR / "pytorch_tokenizer.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "vocabulary": vocabulary,
                "max_length": DL_MAX_LEN,
                "token_pattern": TOKEN_RE.pattern,
                "class_names": CLASS_NAMES,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    train_dataset = TextSequenceDataset(dl_train_df, vocabulary, DL_MAX_LEN)
    val_dataset = TextSequenceDataset(dl_val_df, vocabulary, DL_MAX_LEN)
    test_dataset = TextSequenceDataset(dl_test_df, vocabulary, DL_MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=DL_BATCH_SIZE,
        shuffle=True,
        num_workers=DL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=DL_BATCH_SIZE,
        shuffle=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=DL_BATCH_SIZE,
        shuffle=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    vocab_size = len(vocabulary)
    model_factories: dict[str, Any] = {
        "lstm_1_layer": lambda: RNNClassifier(
            vocab_size,
            DL_EMBEDDING_DIM,
            128,
            number_of_classes,
            cell="lstm",
            num_layers=1,
        ),
        "lstm_2_layer": lambda: RNNClassifier(
            vocab_size,
            DL_EMBEDDING_DIM,
            128,
            number_of_classes,
            cell="lstm",
            num_layers=2,
        ),
        "gru": lambda: RNNClassifier(
            vocab_size,
            DL_EMBEDDING_DIM,
            128,
            number_of_classes,
            cell="gru",
            num_layers=1,
        ),
        "cnn_lstm": lambda: CNNLSTMClassifier(
            vocab_size,
            DL_EMBEDDING_DIM,
            number_of_classes,
        ),
        "hybrid_cnn_lstm_gru": lambda: HybridCNNLSTMGRUClassifier(
            vocab_size,
            DL_EMBEDDING_DIM,
            number_of_classes,
        ),
        "lstm_attention": lambda: AttentionLSTMClassifier(
            vocab_size,
            DL_EMBEDDING_DIM,
            number_of_classes,
        ),
    }

    requested_models = {
        item.strip()
        for item in os.environ.get(
            "MHFS_DL_MODELS",
            ",".join(model_factories),
        ).split(",")
        if item.strip()
    }

    dl_results: list[dict[str, Any]] = []
    for model_name, factory in model_factories.items():
        if model_name not in requested_models:
            print(f"[SKIPPED] {model_name} not selected in MHFS_DL_MODELS.")
            continue
        print("\n" + "-" * 78)
        print(f"Training {model_name}")
        print("-" * 78)
        metrics = train_torch_model(
            model=factory(),
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            test_frame=dl_test_df,
            epochs=DL_EPOCHS,
            learning_rate=DL_LR,
            patience=DL_PATIENCE,
        )
        dl_results.append({**metrics, "model": model_name})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if dl_results:
        dl_results_df = pd.DataFrame(dl_results).sort_values(
            by="macro_f1" if "macro_f1" in dl_results[0] else "accuracy",
            ascending=False,
        )
        dl_results_path = REPORTS_DIR / "pytorch_model_comparison.csv"
        dl_results_df.to_csv(dl_results_path, index=False, encoding="utf-8-sig")
        print("\n[PYTORCH MODEL COMPARISON]")
        print(dl_results_df.to_string(index=False))
        print(f"Saved to: {dl_results_path}")
else:
    print("[SKIPPED] PyTorch DL models. Set MHFS_RUN_DEEP_LEARNING=1 to run them.")


# =============================================================================
# PYTORCH TRANSFORMER MODELS — VERSION 12
# BERT, RoBERTa, MentalRoBERTa, DeBERTa-v3, ModernBERT
# Optional imbalance-aware losses: class-weighted CE and focal loss
# =============================================================================
def stratified_limit_df(
    input_df: pd.DataFrame,
    limit: int,
    label_col: str = "Suicide",
) -> pd.DataFrame:
    if limit <= 0 or len(input_df) <= limit:
        return input_df.reset_index(drop=True)
    sampled, _ = train_test_split(
        input_df,
        train_size=limit,
        random_state=RANDOM_SEED,
        stratify=input_df[label_col],
    )
    return sampled.reset_index(drop=True)


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional balanced class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", class_weights.detach().clone())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - target_probs).pow(self.gamma)) * target_log_probs
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return loss.mean()


def balanced_transformer_class_weights(frame: pd.DataFrame) -> torch.Tensor:
    counts = frame["Suicide"].value_counts()
    total = float(len(frame))
    n_classes = float(len(CLASS_NAMES))
    weights = []
    for label in CLASS_NAMES:
        count = float(counts.get(label, 0))
        if count <= 0:
            raise ValueError(f"Training subset contains no examples for class: {label}")
        weights.append(total / (n_classes * count))
    tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    print("[transformers] Balanced class weights:")
    for label, weight in zip(CLASS_NAMES, tensor.detach().cpu().tolist()):
        print(f"  {label}: {weight:.6f}")
    return tensor


def transformer_output_name(experiment_name: str, loss_mode: str) -> str:
    return experiment_name if loss_mode == "standard" else f"{experiment_name}_{loss_mode}"


if RUN_TRANSFORMERS:
    print("\n" + "=" * 78)
    print("PYTORCH TRANSFORMERS — VERSION 12")
    print("=" * 78)
    try:
        import transformers as hf_transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required. Install/upgrade transformers first."
        ) from exc

    print(f"Transformers: {hf_transformers.__version__}")

    TRANSFORMER_MAX_LEN = int(os.environ.get("MHFS_TRANSFORMER_MAX_LEN", "128"))
    TRANSFORMER_BATCH_SIZE = int(os.environ.get("MHFS_TRANSFORMER_BATCH_SIZE", "16"))
    TRANSFORMER_EPOCHS = int(os.environ.get("MHFS_TRANSFORMER_EPOCHS", "3"))
    TRANSFORMER_PATIENCE = int(os.environ.get("MHFS_TRANSFORMER_PATIENCE", "2"))
    TRANSFORMER_LR = float(os.environ.get("MHFS_TRANSFORMER_LR", "2e-5"))
    TRANSFORMER_TRAIN_LIMIT = int(
        os.environ.get("MHFS_TRANSFORMER_TRAIN_LIMIT", "200000")
    )
    TRANSFORMER_VAL_LIMIT = int(
        os.environ.get("MHFS_TRANSFORMER_VAL_LIMIT", "50000")
    )
    TRANSFORMER_TEST_LIMIT = int(
        os.environ.get("MHFS_TRANSFORMER_TEST_LIMIT", "0")
    )
    TRANSFORMER_VAL_SIZE = float(
        os.environ.get("MHFS_TRANSFORMER_VAL_SIZE", "0.10")
    )
    TRANSFORMER_NUM_WORKERS = int(
        os.environ.get("MHFS_TRANSFORMER_NUM_WORKERS", "0")
    )
    TRANSFORMER_FOCAL_GAMMA = float(
        os.environ.get("MHFS_TRANSFORMER_FOCAL_GAMMA", "2.0")
    )
    FOCAL_USE_CLASS_WEIGHTS = (
        os.environ.get("MHFS_FOCAL_USE_CLASS_WEIGHTS", "1") == "1"
    )
    TRANSFORMER_EVAL_ONLY = (
        os.environ.get("MHFS_TRANSFORMER_EVAL_ONLY", "0") == "1"
    )
    RUN_IMBALANCE_ABLATION = (
        os.environ.get("MHFS_RUN_IMBALANCE_ABLATION", "0") == "1"
    )
    ABLATION_MODEL = os.environ.get("MHFS_ABLATION_MODEL", "auto").strip()
    MODERNBERT_ATTN_IMPLEMENTATION = os.environ.get(
        "MHFS_MODERNBERT_ATTN_IMPLEMENTATION", "eager"
    ).strip()
    # DeBERTa-v3 ships a SentencePiece model. On Python 3.14, forcing the
    # native (slow) tokenizer avoids the fast-tokenizer conversion path that
    # may fall back to TikToken and fail. Override to 1 only if your local
    # Transformers/SentencePiece stack is known to support the fast path.
    DEBERTA_USE_FAST_TOKENIZER = (
        os.environ.get("MHFS_DEBERTA_USE_FAST_TOKENIZER", "0") == "1"
    )

    transformer_source = df_full[["Tweet", "Suicide"]].copy()
    transformer_train_df, transformer_val_df = train_test_split(
        transformer_source,
        test_size=TRANSFORMER_VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=transformer_source["Suicide"],
    )
    transformer_test_df = df_actual[["Tweet", "Suicide"]].copy()

    transformer_train_df = stratified_limit_df(
        transformer_train_df, TRANSFORMER_TRAIN_LIMIT
    )
    transformer_val_df = stratified_limit_df(
        transformer_val_df, TRANSFORMER_VAL_LIMIT
    )
    transformer_test_df = stratified_limit_df(
        transformer_test_df, TRANSFORMER_TEST_LIMIT
    )

    print("[transformers] Train rows:", len(transformer_train_df))
    print("[transformers] Validation rows:", len(transformer_val_df))
    print("[transformers] Held-out test rows:", len(transformer_test_df))
    print("[transformers] Training class distribution:")
    print(transformer_train_df["Suicide"].value_counts().to_string())

    class TransformerTextDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int):
            self.texts = frame["Tweet"].astype(str).tolist()
            self.labels = [LABEL_TO_ID[label] for label in frame["Suicide"].astype(str)]
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            encoded = self.tokenizer(
                self.texts[index],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
            return item

    @torch.no_grad()
    def predict_transformer(
        model: nn.Module,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        all_proba: list[np.ndarray] = []
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            logits = model(**batch).logits
            probabilities = torch.softmax(logits, dim=1)
            all_true.append(labels.numpy())
            all_pred.append(probabilities.argmax(dim=1).cpu().numpy())
            all_proba.append(probabilities.cpu().numpy())
        return (
            np.concatenate(all_true),
            np.concatenate(all_pred),
            np.concatenate(all_proba),
        )

    balanced_weights = balanced_transformer_class_weights(transformer_train_df)

    def build_transformer_criterion(loss_mode: str) -> nn.Module:
        mode = loss_mode.strip().lower()
        if mode == "standard":
            return nn.CrossEntropyLoss().to(DEVICE)
        if mode == "weighted_ce":
            return nn.CrossEntropyLoss(weight=balanced_weights).to(DEVICE)
        if mode == "focal":
            focal_weights = balanced_weights if FOCAL_USE_CLASS_WEIGHTS else None
            return FocalLoss(
                gamma=TRANSFORMER_FOCAL_GAMMA,
                class_weights=focal_weights,
            ).to(DEVICE)
        raise ValueError(
            f"Unknown transformer loss mode: {loss_mode}. "
            "Use standard, weighted_ce, or focal."
        )

    def load_transformer_model(checkpoint: str, experiment_name: str) -> nn.Module:
        model_kwargs: dict[str, Any] = {
            "num_labels": number_of_classes,
            "id2label": ID_TO_LABEL,
            "label2id": LABEL_TO_ID,
        }
        # Explicit eager attention is conservative on CPU and avoids accidental
        # Flash-Attention/Triton use for ModernBERT. Override with an env var.
        if experiment_name == "modernbert" and MODERNBERT_ATTN_IMPLEMENTATION:
            model_kwargs["attn_implementation"] = MODERNBERT_ATTN_IMPLEMENTATION
        return AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            **model_kwargs,
        ).to(DEVICE)


    def load_transformer_tokenizer(
        source: str | Path,
        experiment_name: str,
    ) -> Any:
        """Load the tokenizer without changing v11 behavior for other models.

        DeBERTa-v3 defaults to its native SentencePiece tokenizer on Python 3.14
        because the fast-tokenizer conversion path can invoke TikToken. All other
        transformer families retain the v11 fast-tokenizer default.
        """
        use_fast = True
        if experiment_name == "deberta_v3":
            use_fast = DEBERTA_USE_FAST_TOKENIZER
            if not use_fast:
                try:
                    import sentencepiece  # noqa: F401
                except ImportError as exc:
                    raise RuntimeError(
                        "DeBERTa-v3 native tokenization requires sentencepiece. "
                        "Install/upgrade it with: "
                        "python -m pip install --upgrade sentencepiece"
                    ) from exc

        print(
            f"[{experiment_name}] Tokenizer mode: "
            f"{'fast' if use_fast else 'native/slow SentencePiece'}"
        )
        return AutoTokenizer.from_pretrained(
            source,
            use_fast=use_fast,
        )

    def run_transformer_experiment(
        checkpoint: str,
        experiment_name: str,
        loss_mode: str = "standard",
        eval_only: bool = False,
    ) -> dict[str, Any]:
        run_name = transformer_output_name(experiment_name, loss_mode)
        print("\n" + "-" * 78)
        print(
            f"Transformer experiment: {run_name} "
            f"({checkpoint}; loss={loss_mode}; eval_only={eval_only})"
        )
        print("-" * 78)

        output_dir = MODELS_DIR / f"transformer_{run_name}"
        summary_path = output_dir / "training_summary.json"

        if eval_only:
            if not output_dir.exists():
                raise FileNotFoundError(
                    f"Saved transformer not found for evaluation-only run: {output_dir}"
                )
            tokenizer = load_transformer_tokenizer(output_dir, experiment_name)
            model = load_transformer_model(str(output_dir), experiment_name)
            best_f1 = float("nan")
            if summary_path.exists():
                try:
                    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    best_f1 = float(saved_summary.get("best_val_macro_f1", float("nan")))
                except Exception:
                    pass
        else:
            tokenizer = load_transformer_tokenizer(checkpoint, experiment_name)
            model = load_transformer_model(checkpoint, experiment_name)
            best_f1 = -1.0

        train_dataset = TransformerTextDataset(
            transformer_train_df, tokenizer, TRANSFORMER_MAX_LEN
        )
        val_dataset = TransformerTextDataset(
            transformer_val_df, tokenizer, TRANSFORMER_MAX_LEN
        )
        test_dataset = TransformerTextDataset(
            transformer_test_df, tokenizer, TRANSFORMER_MAX_LEN
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRANSFORMER_BATCH_SIZE,
            shuffle=True,
            num_workers=TRANSFORMER_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=TRANSFORMER_BATCH_SIZE,
            shuffle=False,
            num_workers=TRANSFORMER_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=TRANSFORMER_BATCH_SIZE,
            shuffle=False,
            num_workers=TRANSFORMER_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        if not eval_only:
            criterion = build_transformer_criterion(loss_mode)
            optimizer = AdamW(model.parameters(), lr=TRANSFORMER_LR, weight_decay=0.01)
            total_steps = max(1, len(train_loader) * TRANSFORMER_EPOCHS)
            warmup_steps = max(1, int(total_steps * 0.10))

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                remaining = total_steps - step
                return max(
                    0.0,
                    float(remaining) / float(max(1, total_steps - warmup_steps)),
                )

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0

            for epoch in range(1, TRANSFORMER_EPOCHS + 1):
                model.train()
                running_loss = 0.0
                seen = 0
                for batch in train_loader:
                    labels = batch.pop("labels").to(DEVICE)
                    inputs = {key: value.to(DEVICE) for key, value in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(**inputs).logits
                    loss = criterion(logits, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    batch_size_now = int(labels.shape[0])
                    running_loss += float(loss.item()) * batch_size_now
                    seen += batch_size_now

                val_true, val_pred, _ = predict_transformer(model, val_loader)
                val_f1 = f1_score(
                    val_true,
                    val_pred,
                    average="macro",
                    zero_division=0,
                )
                print(
                    f"[{run_name}] epoch={epoch}/{TRANSFORMER_EPOCHS} "
                    f"train_loss={running_loss / max(1, seen):.5f} "
                    f"val_macro_f1={val_f1:.5f}"
                )
                if val_f1 > best_f1 + 1e-6:
                    best_f1 = val_f1
                    best_state = copy.deepcopy(model.state_dict())
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement >= TRANSFORMER_PATIENCE:
                        print(f"[{run_name}] Early stopping.")
                        break

            model.load_state_dict(best_state)
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
            summary_path.write_text(
                json.dumps(
                    {
                        "experiment_name": experiment_name,
                        "checkpoint": checkpoint,
                        "loss_mode": loss_mode,
                        "best_val_macro_f1": best_f1,
                        "train_rows": len(transformer_train_df),
                        "validation_rows": len(transformer_val_df),
                        "max_length": TRANSFORMER_MAX_LEN,
                        "epochs_requested": TRANSFORMER_EPOCHS,
                        "created_at": RUN_STAMP,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[{run_name}] Model saved to: {output_dir}")

        y_true_ids, y_pred_ids, probabilities = predict_transformer(model, test_loader)
        y_true_labels = [ID_TO_LABEL[int(value)] for value in y_true_ids]
        y_pred_labels = [ID_TO_LABEL[int(value)] for value in y_pred_ids]

        print(
            classification_report(
                y_true_labels,
                y_pred_labels,
                labels=CLASS_NAMES,
                zero_division=0,
            )
        )
        calculate_metrics(
            y_true_ids,
            y_pred_ids,
            y_pred_proba=probabilities,
            classes=CLASS_NAMES,
        )

        report_dict = classification_report(
            y_true_labels,
            y_pred_labels,
            labels=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        report = pd.DataFrame(report_dict).transpose()
        report.to_csv(
            REPORTS_DIR / f"classification_report_{run_name}.csv",
            encoding="utf-8-sig",
        )

        perform_misclassification_analysis(
            df_actual=transformer_test_df.reset_index(drop=True),
            y_test=y_true_labels,
            y_pred=y_pred_labels,
            algorithm_name=run_name,
            y_pred_proba=probabilities,
            labels=CLASS_NAMES,
            label_encoder=FIXED_ENCODER,
        )

        ideation = report_dict["Ideation of Suicide, Self-Harm or Harming Others"]
        method = report_dict["Method or action of Suicide, Self-Harm or Harming others"]
        not_suicide = report_dict["Not Suicide post"]
        result = {
            "experiment": experiment_name,
            "checkpoint": checkpoint,
            "loss_mode": loss_mode,
            "eval_only": eval_only,
            "best_val_macro_f1": best_f1,
            "test_rows": len(transformer_test_df),
            "accuracy": accuracy_score(y_true_ids, y_pred_ids),
            "macro_f1": f1_score(y_true_ids, y_pred_ids, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true_ids, y_pred_ids, average="weighted", zero_division=0),
            "macro_precision": precision_score(y_true_ids, y_pred_ids, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true_ids, y_pred_ids, average="macro", zero_division=0),
            "ideation_precision": ideation["precision"],
            "ideation_recall": ideation["recall"],
            "ideation_f1": ideation["f1-score"],
            "method_precision": method["precision"],
            "method_recall": method["recall"],
            "method_f1": method["f1-score"],
            "not_suicide_precision": not_suicide["precision"],
            "not_suicide_recall": not_suicide["recall"],
            "not_suicide_f1": not_suicide["f1-score"],
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    transformer_experiments = {
        "bert": os.environ.get("MHFS_BERT_CHECKPOINT", "bert-base-uncased"),
        "roberta": os.environ.get("MHFS_ROBERTA_CHECKPOINT", "roberta-base"),
        "mental_roberta": os.environ.get(
            "MHFS_MENTAL_ROBERTA_CHECKPOINT", "mental/mental-roberta-base"
        ),
        "deberta_v3": os.environ.get(
            "MHFS_DEBERTA_V3_CHECKPOINT", "microsoft/deberta-v3-base"
        ),
        "modernbert": os.environ.get(
            "MHFS_MODERNBERT_CHECKPOINT", "answerdotai/ModernBERT-base"
        ),
        # Optional domain-specific BERT counterpart. Not part of the default
        # five-model comparison, but available without another code change.
        "mental_bert": os.environ.get(
            "MHFS_MENTAL_BERT_CHECKPOINT", "mental/mental-bert-base-uncased"
        ),
    }

    requested_transformers = {
        item.strip()
        for item in os.environ.get(
            "MHFS_TRANSFORMER_MODELS",
            "bert,roberta,mental_roberta,deberta_v3,modernbert",
        ).split(",")
        if item.strip()
    }
    requested_loss_modes = [
        item.strip().lower()
        for item in os.environ.get(
            "MHFS_TRANSFORMER_LOSS_MODES", "standard"
        ).split(",")
        if item.strip()
    ]
    valid_loss_modes = {"standard", "weighted_ce", "focal"}
    unknown_loss_modes = set(requested_loss_modes) - valid_loss_modes
    if unknown_loss_modes:
        raise ValueError(
            f"Unknown MHFS_TRANSFORMER_LOSS_MODES values: {sorted(unknown_loss_modes)}"
        )

    unknown_models = requested_transformers - set(transformer_experiments)
    if unknown_models:
        raise ValueError(
            f"Unknown MHFS_TRANSFORMER_MODELS values: {sorted(unknown_models)}. "
            f"Available: {sorted(transformer_experiments)}"
        )

    transformer_results: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, str]] = set()

    for experiment_name, checkpoint in transformer_experiments.items():
        if experiment_name not in requested_transformers:
            print(f"[SKIPPED] Transformer {experiment_name} not selected.")
            continue
        for loss_mode in requested_loss_modes:
            try:
                result = run_transformer_experiment(
                    checkpoint=checkpoint,
                    experiment_name=experiment_name,
                    loss_mode=loss_mode,
                    eval_only=TRANSFORMER_EVAL_ONLY,
                )
                transformer_results.append(result)
                completed_keys.add((experiment_name, loss_mode))
            except Exception as exc:
                print(
                    f"[ERROR] Transformer experiment {experiment_name} "
                    f"with loss={loss_mode} failed: {exc}"
                )
                error_text = str(exc).lower()
                if "gated repo" in error_text or "401 client error" in error_text:
                    print(
                        f"[AUTH] {experiment_name} requires Hugging Face model access. "
                        "Request/accept access in your browser, then authenticate this "
                        "machine with `hf auth login` (or set HF_TOKEN)."
                    )
                if experiment_name == "deberta_v3" and (
                    "sentencepiece" in error_text or "tiktoken" in error_text
                ):
                    print(
                        "[TOKENIZER] DeBERTa-v3 is configured to use native "
                        "SentencePiece in v12. Upgrade SentencePiece with: "
                        "python -m pip install --upgrade sentencepiece"
                    )

    # Optional two-stage ablation: after standard-loss models finish, select the
    # best one by validation macro-F1 and run weighted CE + focal loss only on it.
    if RUN_IMBALANCE_ABLATION and not TRANSFORMER_EVAL_ONLY:
        if ABLATION_MODEL == "auto":
            standard_results = [
                row
                for row in transformer_results
                if row["loss_mode"] == "standard"
                and not math.isnan(float(row["best_val_macro_f1"]))
            ]
            if not standard_results:
                print(
                    "[WARN] Imbalance ablation requested but no successful standard-loss "
                    "transformer result is available."
                )
                ablation_model_name = None
            else:
                ablation_model_name = max(
                    standard_results,
                    key=lambda row: float(row["best_val_macro_f1"]),
                )["experiment"]
        else:
            ablation_model_name = ABLATION_MODEL
            if ablation_model_name not in transformer_experiments:
                raise ValueError(
                    f"MHFS_ABLATION_MODEL={ablation_model_name!r} is not valid."
                )

        if ablation_model_name is not None:
            print(
                f"\n[ABLATION] Running imbalance-aware losses on: {ablation_model_name}"
            )
            checkpoint = transformer_experiments[ablation_model_name]
            for loss_mode in ("weighted_ce", "focal"):
                if (ablation_model_name, loss_mode) in completed_keys:
                    continue
                try:
                    result = run_transformer_experiment(
                        checkpoint=checkpoint,
                        experiment_name=ablation_model_name,
                        loss_mode=loss_mode,
                        eval_only=False,
                    )
                    transformer_results.append(result)
                    completed_keys.add((ablation_model_name, loss_mode))
                except Exception as exc:
                    print(
                        f"[ERROR] Ablation {ablation_model_name} "
                        f"loss={loss_mode} failed: {exc}"
                    )

    if transformer_results:
        transformer_results_df = pd.DataFrame(transformer_results).sort_values(
            by=["macro_f1", "ideation_recall"],
            ascending=False,
        )
        stamped_path = REPORTS_DIR / f"transformer_model_comparison_{RUN_STAMP}.csv"
        latest_path = REPORTS_DIR / "transformer_model_comparison_latest.csv"
        transformer_results_df.to_csv(stamped_path, index=False, encoding="utf-8-sig")
        transformer_results_df.to_csv(latest_path, index=False, encoding="utf-8-sig")
        print("\n[TRANSFORMER MODEL COMPARISON]")
        display_columns = [
            "experiment",
            "loss_mode",
            "best_val_macro_f1",
            "accuracy",
            "macro_f1",
            "ideation_precision",
            "ideation_recall",
            "ideation_f1",
            "method_recall",
            "method_f1",
        ]
        print(transformer_results_df[display_columns].to_string(index=False))
        print(f"Saved to: {stamped_path}")
        print(f"Latest copy: {latest_path}")
else:
    print(
        "[SKIPPED] Transformer models. Set MHFS_RUN_TRANSFORMERS=1 to run them."
    )


con.close()
print("\n[COMPLETE] training12_py314.py finished.")
