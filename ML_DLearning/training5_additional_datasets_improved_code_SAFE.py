# training5_additional_datasets_improved_code_SAFE.py
# Safer replacement runner for the long MHF Safeguard ML/DL experiment script.
#
# This script keeps every model, tokenizer, report, plot, and log inside the
# ML_DLearning folder so that long runs do not fail at the final save/load step.
#
# Recommended:
#   python training5_additional_datasets_improved_code_SAFE.py --run ml
#   python training5_additional_datasets_improved_code_SAFE.py --run dl --dl-model lstm1

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("TF_USE_LEGACY_KERAS", "True")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from metrics_calculator import calculate_metrics
from misclassification_analysis import perform_misclassification_analysis
from path_setup import BASE_DIR, DATA_DIR, LOGS_DIR, MISCLASSIFICATION_DIR, MODELS_DIR, REPORTS_DIR, ensure_directories

LABEL_NOT = "Not Suicide post"
LABEL_METHOD = "Method or action of Suicide, Self-Harm or Harming others"
LABEL_IDEATION = "Suicide or Self Harm Ideation"
CLASS_NAMES = [LABEL_NOT, LABEL_METHOD, LABEL_IDEATION]
LABEL_TO_ID = {LABEL_NOT: 0, LABEL_METHOD: 1, LABEL_IDEATION: 2}
ID_TO_LABEL = {0: LABEL_NOT, 1: LABEL_METHOD, 2: LABEL_IDEATION}


class Tee:
    def __init__(self, filename: Path):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w", buffering=1, encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[INFO] No TensorFlow GPU detected.")
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Enabled GPU memory growth: {gpu}")
        except RuntimeError as exc:
            print(f"[WARN] Could not set GPU memory growth: {exc}")


def load_dataset(db_path: Path, generated_limit: int) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    labels_sql = ",".join(["?"] * len(CLASS_NAMES))
    query = f"""
        SELECT sentence AS Tweet, TRIM(label) AS Suicide, 'mhf_real' AS source
        FROM MH_forum_388_sentences
        WHERE TRIM(label) IN ({labels_sql})

        UNION ALL

        SELECT sentence AS Tweet, TRIM(label) AS Suicide, 'generated' AS source
        FROM generated_sentences
        WHERE ID < ? AND TRIM(label) IN ({labels_sql})

        UNION ALL

        SELECT paraphrases AS Tweet, TRIM(original_label) AS Suicide, 'paraphrase' AS source
        FROM paraphrases4
        WHERE to_consider = 1 AND TRIM(original_label) IN ({labels_sql})

        UNION ALL

        SELECT sentence AS Tweet, TRIM(first_label) AS Suicide, 'kaggle_real' AS source
        FROM SuicideAndDepressionDetectionKaggleDataset_classified_sentences
        WHERE TRIM(first_label) IN ({labels_sql})
    """
    params = CLASS_NAMES + [generated_limit] + CLASS_NAMES + CLASS_NAMES + CLASS_NAMES

    with sqlite3.connect(str(db_path)) as con:
        df = pd.read_sql_query(query, con, params=params)

    df["Tweet"] = df["Tweet"].fillna("").astype(str).str.strip()
    df["Suicide"] = df["Suicide"].fillna("").astype(str).str.strip()
    df = df[(df["Tweet"] != "") & (df["Suicide"].isin(CLASS_NAMES))].copy()
    before = len(df)
    df = df.drop_duplicates(subset=["Tweet", "Suicide"]).reset_index(drop=True)

    print(f"[INFO] Rows loaded: {len(df):,}")
    print(f"[INFO] Duplicates removed: {before - len(df):,}")
    print("\n[INFO] Label counts:")
    print(df["Suicide"].value_counts().to_string())
    print("\n[INFO] Source counts:")
    print(df["source"].value_counts().to_string())
    return df


def make_train_test_split(df: pd.DataFrame, test_size: float, seed: int):
    # Prefer a real-data test set. Synthetic and paraphrased rows remain mainly in training.
    real_df = df[df["source"].isin(["mhf_real", "kaggle_real"])].copy()
    if real_df.empty:
        print("[WARN] No real-source rows found; using stratified split across all data.")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["Suicide"])
    else:
        _, test_df = train_test_split(real_df, test_size=test_size, random_state=seed, stratify=real_df["Suicide"])
        train_df = df.drop(index=test_df.index).copy()

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    split_dir = BASE_DIR / "outputs" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_dir / "train_split.csv", index=False)
    test_df.to_csv(split_dir / "test_split.csv", index=False)

    print(f"\n[INFO] Train rows: {len(train_df):,}")
    print(f"[INFO] Test rows: {len(test_df):,}")
    print("\n[INFO] Test label counts:")
    print(test_df["Suicide"].value_counts().to_string())
    return train_df, test_df


def safe_model_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in name)


def evaluate_and_save(model_name: str, y_true, y_pred, y_proba, test_texts) -> None:
    print(f"\n========== {model_name} ==========")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, labels=CLASS_NAMES, zero_division=0))

    calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_proba,
        classes=CLASS_NAMES,
        model_name=model_name,
        output_dir=REPORTS_DIR,
    )

    df_actual = pd.DataFrame({"Tweet": list(test_texts)})
    perform_misclassification_analysis(
        df_actual=df_actual,
        y_test=y_true,
        y_pred=y_pred,
        algorithm_name=model_name,
        y_pred_proba=y_proba,
        labels=CLASS_NAMES,
        output_dir=MISCLASSIFICATION_DIR,
    )


def train_ml(train_df: pd.DataFrame, test_df: pd.DataFrame, args) -> None:
    X_train = train_df["Tweet"].astype(str).tolist()
    y_train = train_df["Suicide"].astype(str).tolist()
    X_test = test_df["Tweet"].astype(str).tolist()
    y_test = test_df["Suicide"].astype(str).tolist()

    models = {
        "Logistic_Regression": LogisticRegression(C=2, max_iter=1000, n_jobs=-1, class_weight="balanced"),
        "Linear_SVM": LinearSVC(class_weight="balanced"),
        "Naive_Bayes": MultinomialNB(),
    }

    summary = []
    for name, clf in models.items():
        path = MODELS_DIR / f"{name}_pipeline.joblib"
        if path.exists() and not args.force:
            print(f"[LOAD] {name}: {path}")
            pipeline = joblib.load(path)
        else:
            print(f"[TRAIN] {name}")
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=args.max_features)),
                ("classifier", clf),
            ])
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, path)
            print(f"[SAVED] {path}")

        y_pred = pipeline.predict(X_test)
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_test)
            except Exception as exc:
                print(f"[WARN] No probability output for {name}: {exc}")

        evaluate_and_save(name, y_test, y_pred, y_proba, X_test)
        summary.append({"model": name, "accuracy": float(accuracy_score(y_test, y_pred)), "path": str(path)})

    pd.DataFrame(summary).to_csv(REPORTS_DIR / "ml_summary.csv", index=False)


def prepare_dl(train_df: pd.DataFrame, test_df: pd.DataFrame, args):
    X_train_text, X_val_text, y_train_text, y_val_text = train_test_split(
        train_df["Tweet"].astype(str),
        train_df["Suicide"].astype(str),
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_df["Suicide"],
    )

    tokenizer_path = MODELS_DIR / "tokenizer.pickle"
    if tokenizer_path.exists() and not args.force:
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
        print(f"[LOAD] Tokenizer: {tokenizer_path}")
    else:
        tokenizer = Tokenizer(num_words=args.vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train_text.tolist())
        with open(tokenizer_path, "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SAVED] Tokenizer: {tokenizer_path}")

    X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_text.tolist()), maxlen=args.max_len)
    X_val = pad_sequences(tokenizer.texts_to_sequences(X_val_text.tolist()), maxlen=args.max_len)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_df["Tweet"].astype(str).tolist()), maxlen=args.max_len)

    y_train = to_categorical([LABEL_TO_ID[x] for x in y_train_text], num_classes=len(CLASS_NAMES))
    y_val = to_categorical([LABEL_TO_ID[x] for x in y_val_text], num_classes=len(CLASS_NAMES))
    y_test_labels = test_df["Suicide"].astype(str).to_numpy()
    y_test = to_categorical([LABEL_TO_ID[x] for x in y_test_labels], num_classes=len(CLASS_NAMES))

    return X_train, X_val, X_test, y_train, y_val, y_test, y_test_labels, test_df["Tweet"].astype(str).tolist()


def build_lstm1(args) -> Sequential:
    model = Sequential(name="lstm1")
    model.add(Embedding(args.vocab_size, args.embedding_dim, input_length=args.max_len))
    model.add(SpatialDropout1D(0.5))
    model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.0)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(len(CLASS_NAMES), activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return model


def train_dl(train_df: pd.DataFrame, test_df: pd.DataFrame, args) -> None:
    configure_gpu()
    X_train, X_val, X_test, y_train, y_val, y_test, y_test_labels, test_texts = prepare_dl(train_df, test_df, args)

    model_name = args.dl_model
    if model_name != "lstm1":
        raise ValueError("This safe script currently supports --dl-model lstm1. Add others after this run is stable.")

    best_path = MODELS_DIR / "lstm1_best.keras"
    final_path = MODELS_DIR / "lstm1_final.keras"
    csv_log = REPORTS_DIR / "lstm1_training_log.csv"

    if final_path.exists() and not args.force:
        print(f"[LOAD] Existing DL model: {final_path}")
        model = load_model(final_path)
    else:
        model = build_lstm1(args)
        print(model.summary())
        callbacks = [
            ModelCheckpoint(str(best_path), save_best_only=True, monitor="val_acc", mode="max", verbose=1),
            EarlyStopping(monitor="val_acc", patience=3, mode="max", restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
            CSVLogger(str(csv_log), append=False),
        ]
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        model.save(final_path)
        print(f"[SAVED] Final model: {final_path}")
        (REPORTS_DIR / "lstm1_history.json").write_text(json.dumps({k: [float(vv) for vv in v] for k, v in history.history.items()}, indent=2), encoding="utf-8")
        if best_path.exists():
            model = load_model(best_path)
            print(f"[LOAD] Best checkpoint: {best_path}")
        else:
            print(f"[WARN] Best checkpoint missing; using final model: {final_path}")

    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] LSTM1 loss={results[0]} accuracy={results[1]}")
    y_proba = model.predict(X_test, batch_size=args.batch_size, verbose=1)
    y_pred = np.array([ID_TO_LABEL[int(i)] for i in y_proba.argmax(axis=1)])
    evaluate_and_save("LSTM1", y_test_labels, y_pred, y_proba, test_texts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/all_datasets_labelled.db")
    parser.add_argument("--run", choices=["ml", "dl", "all"], default="ml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generated-limit", type=int, default=32000)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--max-features", type=int, default=500000)
    parser.add_argument("--vocab-size", type=int, default=6000)
    parser.add_argument("--max-len", type=int, default=60)
    parser.add_argument("--embedding-dim", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dl-model", default="lstm1")
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()
    set_seed(args.seed)

    log_file = LOGS_DIR / f"training_safe_{stamp()}.txt"
    sys.stdout = Tee(log_file)
    sys.stderr = sys.stdout

    print(f"[INFO] BASE_DIR: {BASE_DIR}")
    print(f"[INFO] MODELS_DIR: {MODELS_DIR}")
    print(f"[INFO] OUTPUTS_DIR: {BASE_DIR / 'outputs'}")
    print(f"[INFO] LOG_FILE: {log_file}")

    db_path = resolve_path(args.db)
    print(f"[INFO] DB_PATH: {db_path}")

    df = load_dataset(db_path, generated_limit=args.generated_limit)
    train_df, test_df = make_train_test_split(df, test_size=args.test_size, seed=args.seed)

    if args.run in ["ml", "all"]:
        train_ml(train_df, test_df, args)
    if args.run in ["dl", "all"]:
        train_dl(train_df, test_df, args)

    print("\n[DONE] Safe training script completed.")


if __name__ == "__main__":
    main()
