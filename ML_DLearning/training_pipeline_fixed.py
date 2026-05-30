# training_pipeline_fixed.py
# Robust ML/DL training script for MHF Safeguard.
#
# Main goals:
# 1. Keep every output path relative to this script directory.
# 2. Save models, tokenizers, reports, and logs before later steps run.
# 3. Avoid failures caused by inconsistent "../models" vs "models" paths.
# 4. Avoid the earlier evaluation leakage pattern where df_actual = df_full.
# 5. Allow interrupted runs to be resumed by skipping models that already exist.
#
# Typical use:
#   python training_pipeline_fixed.py --db data/all_datasets_labelled.db --run ml
#   python training_pipeline_fixed.py --db data/all_datasets_labelled.db --run dl --dl-models lstm1
#   python training_pipeline_fixed.py --db data/all_datasets_labelled.db --run all --force
#
# Put the SQLite database in ML_DLearning/data/ or pass its absolute path using --db.

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, Embedding, GRU, LSTM, MaxPooling1D, SpatialDropout1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from misclassification_analysis import perform_misclassification_analysis

LABEL_NOT = "Not Suicide post"
LABEL_METHOD = "Method or action of Suicide, Self-Harm or Harming others"
LABEL_IDEATION = "Suicide or Self Harm Ideation"

CLASS_NAMES = [LABEL_NOT, LABEL_METHOD, LABEL_IDEATION]
LABEL_TO_ID = {LABEL_NOT: 0, LABEL_METHOD: 1, LABEL_IDEATION: 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
MISCLASS_DIR = OUTPUT_DIR / "misclassification"
LOGS_DIR = OUTPUT_DIR / "logs"

for folder in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, REPORTS_DIR, MISCLASS_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, log_path: Path):
        self.terminal = sys.__stdout__
        self.log = open(log_path, "w", buffering=1, encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[INFO] No GPU detected by TensorFlow. DL will run on CPU unless configured otherwise.")
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Enabled memory growth for GPU: {gpu}")
        except RuntimeError as exc:
            print(f"[WARN] Could not set GPU memory growth: {exc}")


def resolve_db_path(db_arg: str) -> Path:
    db_path = Path(db_arg)
    if not db_path.is_absolute():
        db_path = SCRIPT_DIR / db_path
    return db_path.resolve()


def load_dataset(db_path: Path, generated_limit: int = 32000) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    placeholders = ",".join(["?"] * len(CLASS_NAMES))
    query = f"""
    SELECT sentence AS Tweet, TRIM(label) AS Suicide, 'mhf_real' AS source
    FROM MH_forum_388_sentences
    WHERE TRIM(label) IN ({placeholders})

    UNION ALL

    SELECT sentence AS Tweet, TRIM(label) AS Suicide, 'generated' AS source
    FROM generated_sentences
    WHERE ID < ?
      AND TRIM(label) IN ({placeholders})

    UNION ALL

    SELECT paraphrases AS Tweet, TRIM(original_label) AS Suicide, 'paraphrase' AS source
    FROM paraphrases4
    WHERE to_consider = 1
      AND TRIM(original_label) IN ({placeholders})

    UNION ALL

    SELECT sentence AS Tweet, TRIM(first_label) AS Suicide, 'kaggle_real' AS source
    FROM SuicideAndDepressionDetectionKaggleDataset_classified_sentences
    WHERE TRIM(first_label) IN ({placeholders})
    """

    params = CLASS_NAMES + [generated_limit] + CLASS_NAMES + CLASS_NAMES + CLASS_NAMES

    with sqlite3.connect(str(db_path)) as con:
        df = pd.read_sql_query(query, con, params=params)

    df["Tweet"] = df["Tweet"].fillna("").astype(str).str.strip()
    df["Suicide"] = df["Suicide"].fillna("").astype(str).str.strip()
    df = df[df["Tweet"] != ""].copy()
    df = df[df["Suicide"].isin(CLASS_NAMES)].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["Tweet", "Suicide"]).reset_index(drop=True)
    after = len(df)

    print(f"[INFO] Loaded rows after cleaning: {after:,}")
    print(f"[INFO] Removed duplicates: {before - after:,}")
    print("\n[INFO] Label counts:")
    print(df["Suicide"].value_counts().to_string())
    print("\n[INFO] Source counts:")
    print(df["source"].value_counts().to_string())

    return df


def make_splits(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    real_mask = df["source"].isin(["mhf_real", "kaggle_real"])
    real_df = df[real_mask].copy()

    if real_df.empty:
        print("[WARN] No real-source rows found. Falling back to stratified split over all rows.")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df["Suicide"]
        )
    else:
        _, test_df = train_test_split(
            real_df,
            test_size=test_size,
            random_state=seed,
            stratify=real_df["Suicide"]
        )
        train_df = df.drop(index=test_df.index).copy()

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    split_dir = OUTPUT_DIR / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_dir / "train_split.csv", index=False)
    test_df.to_csv(split_dir / "test_split.csv", index=False)

    print(f"\n[INFO] Train rows: {len(train_df):,}")
    print(f"[INFO] Test rows: {len(test_df):,}")
    print("\n[INFO] Test label counts:")
    print(test_df["Suicide"].value_counts().to_string())

    return train_df, test_df


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_predictions(model_name: str, y_true: Iterable, y_pred: Iterable, y_proba: Optional[np.ndarray], texts: Iterable[str]) -> Dict[str, object]:
    y_true = np.array(list(y_true))
    y_pred = np.array(list(y_pred))

    accuracy = float(accuracy_score(y_true, y_pred))
    report_text = classification_report(y_true, y_pred, labels=CLASS_NAMES, zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=CLASS_NAMES, output_dict=True, zero_division=0)

    print(f"\n\n========== {model_name} ==========")
    print(f"Accuracy: {accuracy}")
    print(report_text)

    report_path = REPORTS_DIR / f"{model_name}_classification_report.txt"
    report_path.write_text(f"Accuracy: {accuracy}\n\n{report_text}", encoding="utf-8")

    save_json({"model": model_name, "accuracy": accuracy, "classification_report": report_dict}, REPORTS_DIR / f"{model_name}_metrics.json")

    df_actual = pd.DataFrame({"Tweet": list(texts)})
    perform_misclassification_analysis(
        df_actual=df_actual,
        y_test=y_true,
        y_pred=y_pred,
        algorithm_name=model_name,
        y_pred_proba=y_proba,
        labels=CLASS_NAMES,
        label_encoder=None,
        output_dir=MISCLASS_DIR,
    )

    return {"accuracy": accuracy, "report": report_dict}


def build_tfidf_pipeline(classifier: BaseEstimator, max_features: Optional[int]) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    max_features=max_features,
                    dtype=np.float32,
                ),
            ),
            ("clf", classifier),
        ]
    )


def train_ml_models(train_df: pd.DataFrame, test_df: pd.DataFrame, args: argparse.Namespace) -> None:
    X_train = train_df["Tweet"].astype(str).tolist()
    y_train = train_df["Suicide"].astype(str).tolist()
    X_test = test_df["Tweet"].astype(str).tolist()
    y_test = test_df["Suicide"].astype(str).tolist()

    ml_models = {
        "Logistic_Regression": LogisticRegression(C=2, max_iter=1000, n_jobs=-1, class_weight="balanced"),
        "Linear_SVM": LinearSVC(class_weight="balanced"),
        "Naive_Bayes": MultinomialNB(),
    }

    summary = []
    for model_name, clf in ml_models.items():
        model_path = MODELS_DIR / f"{model_name}_pipeline.joblib"
        if model_path.exists() and not args.force:
            print(f"\n[SKIP] {model_name} already exists at {model_path}. Use --force to retrain.")
            pipeline = joblib.load(model_path)
        else:
            print(f"\n[TRAIN] {model_name}")
            pipeline = build_tfidf_pipeline(clf, max_features=args.max_features)
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, model_path)
            print(f"[SAVED] {model_path}")

        y_pred = pipeline.predict(X_test)
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_test)
            except Exception as exc:
                print(f"[WARN] Probability output unavailable for {model_name}: {exc}")

        metrics = evaluate_predictions(model_name, y_test, y_pred, y_proba, X_test)
        summary.append({"model": model_name, "accuracy": metrics["accuracy"], "model_path": str(model_path)})

    pd.DataFrame(summary).to_csv(REPORTS_DIR / "ml_model_summary.csv", index=False)


@dataclass
class DLData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test_onehot: np.ndarray
    y_test_labels: np.ndarray
    test_texts: List[str]


def prepare_dl_data(train_df: pd.DataFrame, test_df: pd.DataFrame, args: argparse.Namespace) -> DLData:
    X_train_text, X_val_text, y_train_label, y_val_label = train_test_split(
        train_df["Tweet"].astype(str),
        train_df["Suicide"].astype(str),
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_df["Suicide"].astype(str),
    )

    y_test_label = test_df["Suicide"].astype(str).values
    tokenizer_path = MODELS_DIR / "tokenizer.pickle"

    if tokenizer_path.exists() and not args.force:
        print(f"[LOAD] Existing tokenizer: {tokenizer_path}")
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
    else:
        print("[TRAIN] Tokenizer")
        tokenizer = Tokenizer(num_words=args.vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train_text.astype(str).values)
        with open(tokenizer_path, "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SAVED] {tokenizer_path}")

    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_text.astype(str).values), maxlen=args.max_text_len)
    X_val_pad = pad_sequences(tokenizer.texts_to_sequences(X_val_text.astype(str).values), maxlen=args.max_text_len)
    X_test_pad = pad_sequences(tokenizer.texts_to_sequences(test_df["Tweet"].astype(str).values), maxlen=args.max_text_len)

    y_train_ids = np.array([LABEL_TO_ID[label] for label in y_train_label])
    y_val_ids = np.array([LABEL_TO_ID[label] for label in y_val_label])
    y_test_ids = np.array([LABEL_TO_ID[label] for label in y_test_label])

    return DLData(
        X_train=X_train_pad,
        X_val=X_val_pad,
        X_test=X_test_pad,
        y_train=to_categorical(y_train_ids, num_classes=len(CLASS_NAMES)),
        y_val=to_categorical(y_val_ids, num_classes=len(CLASS_NAMES)),
        y_test_onehot=to_categorical(y_test_ids, num_classes=len(CLASS_NAMES)),
        y_test_labels=np.array([ID_TO_LABEL[int(i)] for i in y_test_ids]),
        test_texts=test_df["Tweet"].astype(str).tolist(),
    )


def build_lstm1(args: argparse.Namespace) -> Sequential:
    model = Sequential(name="lstm1")
    model.add(Embedding(args.vocab_size, args.embedding_dim, input_length=args.max_text_len))
    model.add(SpatialDropout1D(0.5))
    model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.0)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(len(CLASS_NAMES), activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return model


def build_gru(args: argparse.Namespace) -> Sequential:
    model = Sequential(name="gru")
    model.add(Embedding(args.vocab_size, args.embedding_dim, input_length=args.max_text_len))
    model.add(SpatialDropout1D(0.4))
    model.add(GRU(96, dropout=0.3, recurrent_dropout=0.0))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(len(CLASS_NAMES), activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return model


def build_cnn_lstm(args: argparse.Namespace) -> Sequential:
    model = Sequential(name="cnn_lstm")
    model.add(Embedding(args.vocab_size, args.embedding_dim, input_length=args.max_text_len))
    model.add(SpatialDropout1D(0.4))
    model.add(Conv1D(filters=64, kernel_size=5, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(96, dropout=0.3, recurrent_dropout=0.0)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(len(CLASS_NAMES), activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return model


def get_callbacks(model_name: str) -> Tuple[List, Path, Path]:
    best_path = MODELS_DIR / f"{model_name}_best.keras"
    final_path = MODELS_DIR / f"{model_name}_final.keras"
    csv_path = REPORTS_DIR / f"{model_name}_training_log.csv"

    callbacks = [
        ModelCheckpoint(filepath=str(best_path), save_best_only=True, monitor="val_acc", mode="max", verbose=1),
        EarlyStopping(monitor="val_acc", mode="max", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
        CSVLogger(str(csv_path), append=False),
    ]
    return callbacks, best_path, final_path


def train_single_dl_model(model_name: str, dl_data: DLData, args: argparse.Namespace) -> None:
    builders = {"lstm1": build_lstm1, "gru": build_gru, "cnn_lstm": build_cnn_lstm}
    if model_name not in builders:
        print(f"[WARN] Unknown DL model '{model_name}'. Skipping.")
        return

    callbacks, best_path, final_path = get_callbacks(model_name)

    if final_path.exists() and not args.force:
        print(f"\n[SKIP] {model_name} final model already exists at {final_path}. Use --force to retrain.")
        model = load_model(str(final_path))
    else:
        print(f"\n[TRAIN] {model_name}")
        model = builders[model_name](args)
        print(model.summary())

        history = model.fit(
            dl_data.X_train,
            dl_data.y_train,
            validation_data=(dl_data.X_val, dl_data.y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        model.save(str(final_path))
        print(f"[SAVED] Final model: {final_path}")

        history_path = REPORTS_DIR / f"{model_name}_history.json"
        save_json({k: [float(x) for x in v] for k, v in history.history.items()}, history_path)

        if best_path.exists():
            model = load_model(str(best_path))
            print(f"[LOAD] Best checkpoint: {best_path}")
        else:
            print(f"[WARN] Best checkpoint not found. Using final model: {final_path}")

    results = model.evaluate(dl_data.X_test, dl_data.y_test_onehot, verbose=0)
    print(f"[RESULT] {model_name} test loss={results[0]} accuracy={results[1]}")

    y_proba = model.predict(dl_data.X_test, batch_size=args.batch_size, verbose=1)
    y_pred_ids = y_proba.argmax(axis=1)
    y_pred_labels = np.array([ID_TO_LABEL[int(i)] for i in y_pred_ids])

    evaluate_predictions(model_name=model_name, y_true=dl_data.y_test_labels, y_pred=y_pred_labels, y_proba=y_proba, texts=dl_data.test_texts)


def train_dl_models(train_df: pd.DataFrame, test_df: pd.DataFrame, args: argparse.Namespace) -> None:
    configure_gpu_memory_growth()
    dl_data = prepare_dl_data(train_df, test_df, args)
    requested_models = [m.strip() for m in args.dl_models.split(",") if m.strip()]
    for model_name in requested_models:
        train_single_dl_model(model_name, dl_data, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MHF Safeguard ML/DL classifiers safely.")
    parser.add_argument("--db", default="data/all_datasets_labelled.db", help="SQLite database path. Relative paths are resolved from this script folder.")
    parser.add_argument("--run", choices=["ml", "dl", "all"], default="ml", help="Which model group to run.")
    parser.add_argument("--force", action="store_true", help="Retrain even if saved models already exist.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--generated-limit", type=int, default=32000)
    parser.add_argument("--max-features", type=int, default=500000, help="Max TF-IDF features. Use 0 for no limit.")
    parser.add_argument("--vocab-size", type=int, default=6000)
    parser.add_argument("--max-text-len", type=int, default=60)
    parser.add_argument("--embedding-dim", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dl-models", default="lstm1", help="Comma-separated: lstm1,gru,cnn_lstm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_features == 0:
        args.max_features = None

    log_file = LOGS_DIR / f"training_{now_stamp()}.txt"
    sys.stdout = Tee(log_file)
    sys.stderr = sys.stdout

    print(f"[INFO] Script directory: {SCRIPT_DIR}")
    print(f"[INFO] Models directory: {MODELS_DIR}")
    print(f"[INFO] Outputs directory: {OUTPUT_DIR}")
    print(f"[INFO] Log file: {log_file}")

    set_reproducibility(args.seed)
    db_path = resolve_db_path(args.db)
    print(f"[INFO] Database path: {db_path}")

    df = load_dataset(db_path, generated_limit=args.generated_limit)
    train_df, test_df = make_splits(df, test_size=args.test_size, seed=args.seed)

    if args.run in ["ml", "all"]:
        train_ml_models(train_df, test_df, args)

    if args.run in ["dl", "all"]:
        train_dl_models(train_df, test_df, args)

    print("\n[DONE] Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
