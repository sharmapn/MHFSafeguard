# training5_additional_datasets_improved_code_FULL_SAFE.py
#
# Full-safe runner for the original 2,000+ line training script.
#
# This file does NOT replace or condense the original experiment script.
# Instead, it reads the original script, applies a small set of safety patches
# at runtime, and then executes the patched version.
#
# Why this design?
# - It preserves the full original workflow and model list.
# - It avoids losing experimental sections during rewriting.
# - It fixes the path problem that caused long training runs to fail at the end.
# - It avoids editing the original file directly, so the original remains available.
#
# Run from inside ML_DLearning:
#   python training5_additional_datasets_improved_code_FULL_SAFE.py
#
# Optional:
#   set MHFS_DB_PATH=C:\path\to\all_datasets_labelled.db
#   set MHFS_USE_ACTUAL_ONLY_TEST=1
#   python training5_additional_datasets_improved_code_FULL_SAFE.py

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# Local folder setup
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_SCRIPT = BASE_DIR / "training5_additional_datasets_improved_code.py"

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MISCLASSIFICATION_DIR = OUTPUTS_DIR / "misclassification"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

for folder in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, MISCLASSIFICATION_DIR, LOGS_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

DEFAULT_DB_PATH = DATA_DIR / "all_datasets_labelled.db"
DB_PATH = Path(os.environ.get("MHFS_DB_PATH", str(DEFAULT_DB_PATH))).resolve()

# If this is 1, df_actual is loaded from actual labelled sources only.
# This prevents generated/paraphrased rows from being used as final test data.
USE_ACTUAL_ONLY_TEST = os.environ.get("MHFS_USE_ACTUAL_ONLY_TEST", "1") == "1"

# If this is 1, the script avoids the very long cross_val_score blocks.
# This is useful after a model has already trained and you mainly want final evaluation.
SKIP_EXPENSIVE_CV = os.environ.get("MHFS_SKIP_EXPENSIVE_CV", "0") == "1"


# ---------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------

def patch_quoted_paths(source: str) -> str:
    """Redirect common relative paths to local ML_DLearning folders."""

    # ../models/file.ext  -> str(MODELS_DIR / "file.ext")
    source = re.sub(
        r"(['\"])\.\./models/([^'\"]+)\1",
        lambda m: f"str(MODELS_DIR / {m.group(1)}{m.group(2)}{m.group(1)})",
        source,
    )

    # models/file.ext -> str(MODELS_DIR / "file.ext")
    source = re.sub(
        r"(['\"])models/([^'\"]+)\1",
        lambda m: f"str(MODELS_DIR / {m.group(1)}{m.group(2)}{m.group(1)})",
        source,
    )

    # ../output/file.ext and output/file.ext -> outputs/file.ext
    source = re.sub(
        r"(['\"])\.\./output/([^'\"]+)\1",
        lambda m: f"str(OUTPUTS_DIR / {m.group(1)}{m.group(2)}{m.group(1)})",
        source,
    )
    source = re.sub(
        r"(['\"])output/([^'\"]+)\1",
        lambda m: f"str(OUTPUTS_DIR / {m.group(1)}{m.group(2)}{m.group(1)})",
        source,
    )

    return source


def patch_logging(source: str) -> str:
    """Use a timestamped local log file and overwrite, not append."""

    source = source.replace(
        'self.log = open(filename, "a", buffering=1)',
        'self.log = open(filename, "w", buffering=1, encoding="utf-8")',
    )
    source = source.replace(
        "self.log = open(filename, \"a\", buffering=1)",
        "self.log = open(filename, \"w\", buffering=1, encoding=\"utf-8\")",
    )
    source = source.replace(
        'sys.stdout = Tee("output.txt")',
        'sys.stdout = Tee(str(LOGS_DIR / f"output_{RUN_STAMP}.txt"))',
    )
    source = source.replace(
        "sys.stdout = Tee('output.txt')",
        "sys.stdout = Tee(str(LOGS_DIR / f'output_{RUN_STAMP}.txt'))",
    )
    return source


def patch_database_path(source: str) -> str:
    """Use local data/all_datasets_labelled.db or MHFS_DB_PATH."""

    source = re.sub(
        r"con\s*=\s*sqlite3\.connect\(['\"][^'\"]*all_datasets_labelled\.db['\"]\)",
        "con = sqlite3.connect(str(DB_PATH))",
        source,
    )
    return source


def patch_actual_test_set(source: str) -> str:
    """Replace df_actual = df_full with an actual-only test dataset query."""

    actual_only_block = r'''
if USE_ACTUAL_ONLY_TEST:
    print("\n[INFO] Loading actual-only df_actual for final testing.")
    df_actual = pd.read_sql_query(
        """
        SELECT sentence AS Tweet, TRIM(label) AS Suicide
        FROM MH_forum_388_sentences
        WHERE TRIM(label) IN (
            'Not Suicide post',
            'Suicide or Self Harm Ideation',
            'Method or action of Suicide, Self-Harm or Harming others'
        )

        UNION ALL

        SELECT sentence AS Tweet, TRIM(first_label) AS Suicide
        FROM SuicideAndDepressionDetectionKaggleDataset_classified_sentences
        WHERE TRIM(first_label) IN (
            'Not Suicide post',
            'Suicide or Self Harm Ideation',
            'Method or action of Suicide, Self-Harm or Harming others'
        )
        """,
        con
    )
    df_actual['Tweet'] = df_actual['Tweet'].fillna("").astype(str)
    df_actual['Suicide'] = df_actual['Suicide'].fillna("").astype(str).str.strip()
    df_actual = df_actual[df_actual['Suicide'].isin([
        'Not Suicide post',
        'Suicide or Self Harm Ideation',
        'Method or action of Suicide, Self-Harm or Harming others'
    ])].copy()
    df_actual = df_actual.drop_duplicates(subset=['Tweet', 'Suicide']).reset_index(drop=True)
    print("[INFO] df_actual actual-only rows:", len(df_actual))
    print(df_actual['Suicide'].value_counts().to_string())
else:
    print("\n[WARN] MHFS_USE_ACTUAL_ONLY_TEST=0, so df_actual = df_full will be used.")
    df_actual = df_full
'''

    source = source.replace("df_actual = df_full", actual_only_block)
    return source


def patch_model_checkpoint_paths(source: str) -> str:
    """Fix bare checkpoint filenames and incorrect model reload paths."""

    replacements = {
        "'lstm-1-layer-best_model.h5'": "str(MODELS_DIR / 'lstm-1-layer-best_model.h5')",
        '"lstm-1-layer-best_model.h5"': 'str(MODELS_DIR / "lstm-1-layer-best_model.h5")',
        "'/pre/'": "str(MODELS_DIR / 'pre')",
        '"/pre/"': 'str(MODELS_DIR / "pre")',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)

    return source


def patch_three_class_dl_settings(source: str) -> str:
    """Correct obvious 3-class DL settings while preserving the full script."""

    # For the current three-class softmax setup, categorical_crossentropy is correct.
    source = source.replace("loss='binary_crossentropy'", "loss='categorical_crossentropy'")
    source = source.replace('loss="binary_crossentropy"', 'loss="categorical_crossentropy"')

    # Some later experimental models were still hard-coded for 8 classes.
    source = source.replace("Dense(8, activation='softmax')", "Dense(number_of_classes, activation='softmax')")
    source = source.replace('Dense(8, activation="softmax")', 'Dense(number_of_classes, activation="softmax")')
    source = source.replace("num_labels=8", "num_labels=number_of_classes")

    return source


def patch_evaluation_mistakes(source: str) -> str:
    """Fix known copy-paste mistakes in misclassification analysis calls."""

    exact_replacements = {
        # SVM block previously passed y_pred1 into misclassification analysis.
        'y_pred=y_pred1,                 # Predicted labels\n            algorithm_name="Support_Vector_Machine"':
        'y_pred=y_pred2,                 # Predicted labels\n            algorithm_name="Support_Vector_Machine"',

        # Naive Bayes block previously passed y_pred1.
        'y_pred=y_pred1,                 # Predicted labels\n            algorithm_name="Naive_Bayes"':
        'y_pred=y_pred3,                 # Predicted labels\n            algorithm_name="Naive_Bayes"',
    }

    for old, new in exact_replacements.items():
        source = source.replace(old, new)

    # LinearSVC has no native predict_proba. Avoid reusing LR probabilities for SVM.
    source = source.replace(
        "y_pred_proba = lr.predict_proba(x_test_tfidf)  # Probability predictions for each class\n        #metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)\n        # Call calculate_metrics with encoded labels\n        print('\\n')\n        metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)\n        print('\\n')\n\n        joblib.dump(svc, str(MODELS_DIR / 'Suicide_SVM.pkl'))",
        "y_pred_proba = None  # LinearSVC does not provide probabilities by default.\n        print('\\n')\n        metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)\n        print('\\n')\n\n        joblib.dump(svc, str(MODELS_DIR / 'Suicide_SVM.pkl'))",
    )

    return source


def patch_expensive_cv(source: str) -> str:
    """Optionally skip very expensive cross-validation calls."""

    if not SKIP_EXPENSIVE_CV:
        return source

    source = re.sub(
        r"scores\s*=\s*cross_val_score\(([^\n]+)\)\n\s*print\(accuracy_score\(([^\n]+)\)\)\n\s*print\(\"Cross-validated scores:\", scores\)",
        "scores = []\n        print('[SKIPPED] cross_val_score skipped because MHFS_SKIP_EXPENSIVE_CV=1')",
        source,
    )
    return source


def patch_source(source: str) -> str:
    source = patch_logging(source)
    source = patch_database_path(source)
    source = patch_actual_test_set(source)
    source = patch_quoted_paths(source)
    source = patch_model_checkpoint_paths(source)
    source = patch_three_class_dl_settings(source)
    source = patch_evaluation_mistakes(source)
    source = patch_expensive_cv(source)
    return source


# ---------------------------------------------------------------------
# Execute patched original script
# ---------------------------------------------------------------------

def main() -> None:
    if not ORIGINAL_SCRIPT.exists():
        raise FileNotFoundError(f"Original script not found: {ORIGINAL_SCRIPT}")

    print(f"[FULL_SAFE] Base directory: {BASE_DIR}")
    print(f"[FULL_SAFE] Original script: {ORIGINAL_SCRIPT}")
    print(f"[FULL_SAFE] Models directory: {MODELS_DIR}")
    print(f"[FULL_SAFE] Outputs directory: {OUTPUTS_DIR}")
    print(f"[FULL_SAFE] Database path: {DB_PATH}")
    print(f"[FULL_SAFE] Actual-only test set: {USE_ACTUAL_ONLY_TEST}")
    print(f"[FULL_SAFE] Skip expensive CV: {SKIP_EXPENSIVE_CV}")

    source = ORIGINAL_SCRIPT.read_text(encoding="utf-8")
    patched_source = patch_source(source)

    patched_copy_path = OUTPUTS_DIR / f"patched_training_script_{RUN_STAMP}.py"
    patched_copy_path.write_text(patched_source, encoding="utf-8")
    print(f"[FULL_SAFE] Patched copy saved for inspection: {patched_copy_path}")

    globals_dict = {
        "__name__": "__main__",
        "__file__": str(ORIGINAL_SCRIPT),
        "BASE_DIR": BASE_DIR,
        "DATA_DIR": DATA_DIR,
        "MODELS_DIR": MODELS_DIR,
        "OUTPUTS_DIR": OUTPUTS_DIR,
        "REPORTS_DIR": REPORTS_DIR,
        "MISCLASSIFICATION_DIR": MISCLASSIFICATION_DIR,
        "LOGS_DIR": LOGS_DIR,
        "PLOTS_DIR": PLOTS_DIR,
        "RUN_STAMP": RUN_STAMP,
        "DB_PATH": DB_PATH,
        "USE_ACTUAL_ONLY_TEST": USE_ACTUAL_ONLY_TEST,
        "SKIP_EXPENSIVE_CV": SKIP_EXPENSIVE_CV,
    }

    exec(compile(patched_source, str(patched_copy_path), "exec"), globals_dict)


if __name__ == "__main__":
    main()
