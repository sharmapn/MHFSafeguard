# =============================================================================
# MHF Safeguard ML/DL Training Script - Full Safe Static Version
# 30 May 2026
# =============================================================================
#
# This script is the corrected full version of the original ML/DL experiment
# script for training the MHF Safeguard classifier. It preserves the original
# full workflow and model sections, including traditional machine-learning
# models, deep-learning models, transformer-based experiments, ensemble models,
# and misclassification analysis.
#
# The main purpose of this version is to make long-running experiments safer.
# The previous script could train for many hours or days and then fail near the
# end because model, tokenizer, checkpoint, output, or log paths were inconsistent.
# This version redirects all important files to folders inside the current
# ML_DLearning directory.
#
# Main safety improvements:
# - Saves models inside: ML_DLearning/models/
# - Saves logs inside: ML_DLearning/outputs/logs/
# - Saves reports inside: ML_DLearning/outputs/reports/
# - Saves misclassification outputs inside: ML_DLearning/outputs/misclassification/
# - Uses timestamped output logs instead of repeatedly appending to output.txt.
# - Fixes model checkpoint/load paths such as lstm-1-layer-best_model.h5.
# - Supports a local database path or a custom path through MHFS_DB_PATH.
# - Replaces df_actual = df_full with an actual-only testing dataset by default.
# - Excludes generated and paraphrased sentences from df_actual testing data.
# - Fixes several known ML/DL evaluation and misclassification-analysis issues.
#
# Important dataset note:
# By default, df_actual now contains actual labelled sentences only from:
#   1. MH_forum_388_sentences
#   2. SuicideAndDepressionDetectionKaggleDataset_classified_sentences
#
# It excludes:
#   - generated_sentences
#   - paraphrases4
#
# This improves the previous evaluation design, where df_actual = df_full caused
# the test set to include generated/paraphrased data and possible training overlap.
# However, the strongest final-paper design should still split actual data first:
#
#   actual_data -> actual_train + actual_test
#   training    -> actual_train + generated + paraphrased
#   testing     -> actual_test only
#
# Current target classes:
#   1. Not Suicide post
#   2. Suicide or Self Harm Ideation
#   3. Method or action of Suicide, Self-Harm or Harming others
#
# Recommended run:
#   cd ML_DLearning
#   set MHFS_USE_ACTUAL_ONLY_TEST=1
#   set MHFS_SKIP_EXPENSIVE_CV=1
#   python training6_additional_datasets_improved_code_FULL_SAFE_STATIC.py
#
# =============================================================================


# training5_additional_datasets_improved_code_FULL_SAFE_STATIC.py
# Static full corrected version of the original script.
# This preserves the full original script and applies safety fixes directly in the file.
# Main fixes: local paths, timestamped logs, actual-only test option, path-safe model saves/loads,
# 3-class DL loss/output settings, and known y_pred/probability copy-paste issues.

# **Import libraries**

import os
os.environ['TF_USE_LEGACY_KERAS'] = 'True'  # optional for legacy keras compat

import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from nltk.corpus import stopwords, reuters, brown, gutenberg
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import joblib
from collections import Counter
from textblob import Word
from wordcloud import WordCloud, ImageColorGenerator

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, f1_score, recall_score
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder

import sqlite3
from PIL import Image
import urllib
import requests

# TensorFlow Keras imports (for Python 3.11+, TF ≥ 2.16.2)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Activation, Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Flatten,
    GRU, Conv1D, MaxPooling1D, Bidirectional
)
from tensorflow.keras.regularizers import l2

# Custom module imports
from metrics_calculator import calculate_metrics
from misclassification_analysis import perform_misclassification_analysis

# Optional ML wrapper
import ktrain
from ktrain import text


import psutil
import os

import sys

from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# SAFE LOCAL PATH SETUP
# ---------------------------------------------------------------------
# All files are stored inside the folder where this script is located.
# This prevents long runs failing at the end because of mixed paths such as
# models/..., ../models/..., or bare checkpoint filenames.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MISCLASSIFICATION_DIR = OUTPUTS_DIR / "misclassification"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

for _folder in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, MISCLASSIFICATION_DIR, LOGS_DIR, PLOTS_DIR]:
    _folder.mkdir(parents=True, exist_ok=True)

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DB_PATH = Path(os.environ.get("MHFS_DB_PATH", str(DATA_DIR / "all_datasets_labelled.db"))).resolve()
USE_ACTUAL_ONLY_TEST = os.environ.get("MHFS_USE_ACTUAL_ONLY_TEST", "1") == "1"
SKIP_EXPENSIVE_CV = os.environ.get("MHFS_SKIP_EXPENSIVE_CV", "0") == "1"


# Define a custom Tee class to duplicate all output to both terminal and a file
class Tee:
    def __init__(self, filename):
        # Save the original terminal output stream
        self.terminal = sys.__stdout__

        # Open the log file in append mode with line buffering
        # buffering=1 ensures that each line is written immediately to disk
        self.log = open(filename, "w", buffering=1, encoding="utf-8")

    def write(self, message):
        # Write the message to both terminal and log file
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Flush both terminal and file to ensure all buffered output is written
        self.terminal.flush()
        self.log.flush()

# Redirect standard output and error to the Tee class
# This captures everything printed via `print()` and also error messages
sys.stdout = Tee(str(LOGS_DIR / f"output_{RUN_STAMP}.txt"))
sys.stderr = sys.stdout  # Optional: also capture error messages

# Set process to high priority on Windows when available.
# If unsupported, continue safely instead of crashing.
try:
    p = psutil.Process(os.getpid())
    if hasattr(psutil, 'HIGH_PRIORITY_CLASS'):
        p.nice(psutil.HIGH_PRIORITY_CLASS)
except Exception as _priority_error:
    print(f"[WARN] Could not set high priority: {_priority_error}")

# Plotting setup
sns.set()

# Download required NLTK corpora
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('gutenberg')
nltk.download('brown')
nltk.download("reuters")
nltk.download('words')

# Configs
number_of_classes = 3 #5  # or 8
machine_learning = True
not_done = True

#For TensorFlow: Allow Memory Growth
#Add this early in your script:

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# For TensorFlow 2.16.2 and later, use:
# tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])  # Set memory limit to 4GB for the first GPU


# **Load Dataset**




# Read sqlite query results into a pandas DataFrSuicide or Self Harm Methodame
con = sqlite3.connect(str(DB_PATH))


# Just out ut the number of rows for each label
import pandas as pd


# SQL query to get label counts across all sources
label_count_query = """
SELECT Suicide AS label, COUNT(*) AS count FROM (
    -- MH_forum_388_sentences
    SELECT TRIM(label) AS Suicide
    FROM MH_forum_388_sentences
    WHERE TRIM(label) IN (
        'Not Suicide post',
        'Suicide or Self Harm Ideation',
        'Method or action of Suicide, Self-Harm or Harming others'
    )

    UNION ALL

    -- generated_sentences
    SELECT TRIM(label) AS Suicide
    FROM generated_sentences
    
    UNION ALL

    -- paraphrases4 (using original_label and only those to be considered)
    SELECT TRIM(original_label) AS Suicide
    FROM paraphrases4
    WHERE to_consider = 1
      AND TRIM(original_label) IN (
          'Not Suicide post',
          'Suicide or Self Harm Ideation',
          'Method or action of Suicide, Self-Harm or Harming others'
      )

    UNION ALL

    -- Kaggle classified sentences
    SELECT TRIM(first_label) AS Suicide
    FROM SuicideAndDepressionDetectionKaggleDataset_classified_sentences
    WHERE TRIM(first_label) IN (
        'Not Suicide post',
        'Suicide or Self Harm Ideation',
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


# NOW THE SCRIPT STARTS
# Load the full dataset for training and validation
df_full = pd.read_sql_query(
    " SELECT sentence as Tweet, label as Suicide FROM MH_forum_388_sentences WHERE label IN ('Not Suicide post','Suicide or Self Harm Ideation','Method or action of Suicide, Self-Harm or Harming others')"  # actual sentences from MH_forum
    # synthetic generates sentences using the MHF 'keywords'
    " UNION ALL SELECT sentence as Tweet, label as Suicide FROM generated_sentences "  
    # paraphrases
    " UNION ALL SELECT paraphrases AS Tweet, original_label AS Suicide FROM paraphrases4 WHERE to_consider = 1 "
    # " UNION ALL SELECT paraphrases as Tweet, label as Suicide FROM paraphrases4 "  # paraphrased rows
    # Google Gemini labelled depression dataset from kaggle
    " UNION ALL SELECT sentence as Tweet, first_label as Suicide FROM SuicideAndDepressionDetectionKaggleDataset_classified_sentences WHERE first_label IN ('Not Suicide post','Suicide or Self Harm Ideation','Method or action of Suicide, Self-Harm or Harming others')", # 2024076 rows from Suicide And Depression Detection Kaggle Dataset
    # " UNION ALL SELECT sentence as Tweet, first_label as Suicide FROM SuicideForumPosts_classified_sentences ",
    con
)


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
    print("\n[WARN] MHFS_USE_ACTUAL_ONLY_TEST=0, so df_actual = df_full will be used. This is not recommended for final paper results.")
    df_actual = df_full



# Check if DataFrame is empty
if df_full.empty:
    print("The DataFrame is empty.")
else:
    # Data Cleaning
    df_full['Tweet'] = df_full['Tweet'].fillna("")
    df_full.isna().sum()

# **Data Cleaning**

df_full['Tweet']=df_full['Tweet'].fillna("")
df_full.isna().sum()

# **Preprocessing**

#Convert to lower case
df_full['lower_case']= df_full['Tweet'].apply(lambda x: x.lower())

#Tokenize
tokenizer = RegexpTokenizer(r'\w+')
df_full['Special_word'] = df_full.apply(lambda row: tokenizer.tokenize(row['lower_case']), axis=1)
#Stop words remove
stop = stopwords.words('english')
stop.remove("not")
stop.remove("here")
stop.remove("some")
df_full['stop_words'] = df_full['Special_word'].apply(lambda x: [item for item in x if item not in stop])
df_full['stop_words'] = df_full['stop_words'].astype('str')
#Filter words based on length
df_full['short_word'] = df_full['stop_words'].str.findall(r'\w{3,}')
df_full['string']=df_full['short_word'].str.join(' ')

#Removing non-english words(mention,emoji,link,special characters etc..)
words = set(nltk.corpus.words.words())
for w in reuters.words():
  words.add(w)
for w in brown.words():
  words.add(w)
for w in gutenberg.words():
  words.add(w)

df_full['NonEnglish'] = df_full['string'].apply(lambda x: " ".join(x for x in x.split() if x in words))
#Lemmatization
df_full['tweet'] = df_full['NonEnglish'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df_full.head(5)

# make them null
df_full['Tweet'] = df_full['Tweet'].fillna("")
df_full['Tweet'] = df_full['Tweet'].apply(lambda x: str(x))


# **Applying N-gram**
print('**Applying N-gram**')

#x_train, x_test, y_train, y_test = train_test_split(df["tweet"],df["Suicide"], test_size = 0.25, random_state = 42)
# Split df_full for training and validation
X_train_full, X_val, y_train_full, y_val = train_test_split(
    df_full['Tweet'], df_full['Suicide'], test_size=0.25, random_state=42
)

# Use only actual sentences for testing
X_test = df_actual['Tweet']
y_test = df_actual['Suicide']



# Define the vectorizer and transformer for N-grams
count_vect = CountVectorizer(ngram_range=(1, 2))
transformer = TfidfTransformer(norm='l2', sublinear_tf=True)

# Transform X_train_full and X_val using CountVectorizer and TfidfTransformer
x_train_counts = count_vect.fit_transform(X_train_full)
x_train_tfidf = transformer.fit_transform(x_train_counts)

x_val_counts = count_vect.transform(X_val)
x_val_tfidf = transformer.transform(x_val_counts)

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

if machine_learning:
    # for soem reason the entire code does not execute in one go. It gets stops just after this, so putting this into if/else so we can continue from here afterwards
    if not_done:

        #**Machine Learning Models**
        # **Logistic Regression**
        print('**Logistic Regression**')

        lr = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
        lr.fit(x_train_tfidf, y_train_full)
        y_pred1 = lr.predict(x_test_tfidf)
        print("Accuracy: " + str(accuracy_score(y_test, y_pred1)))
        print(classification_report(y_test, y_pred1))

        if SKIP_EXPENSIVE_CV:


            scores = []


            print('[SKIPPED] cross_val_score skipped because MHFS_SKIP_EXPENSIVE_CV=1')


        else:


            scores = cross_val_score(lr, x_train_tfidf, y_train_full, cv=5)


            print(accuracy_score(y_test, y_pred1))


            print("Cross-validated scores:", scores)

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


            scores = cross_val_score(svc, x_train_tfidf, y_train_full, cv=5)


            print(accuracy_score(y_test, y_pred2))


            print("Cross-validated scores:", scores)

        

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

        joblib.dump(svc, str(MODELS_DIR / 'Suicide_SVM.pkl'))
        print(f"Model saved as " + " models/Suicide_SVM.pkl")
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


            print('[SKIPPED] cross_val_score skipped because MHFS_SKIP_EXPENSIVE_CV=1')


        else:


            scores = cross_val_score(mnb, x_train_tfidf, y_train_full, cv=5)


            print(accuracy_score(y_test, y_pred3))


            print("Cross-validated scores:", scores)

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
            scores = cross_val_score(rfc, x_train_tfidf, y_train_full, cv=5)
            print(accuracy_score(y_test, y_pred4))
            print("Cross-validated scores:", scores)

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
            scores = cross_val_score(gbc, x_train_tfidf, y_train_full, cv=5)
            print(accuracy_score(y_test, y_pred5))
            print("Cross-validated scores:", scores)

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
    print('**Ensemble Classifier**')

    print("Training Multinomial Naive Bayes...")
    mnb = MultinomialNB()
    print("Training Random Forest Classifier...")
    rfc = RandomForestClassifier(n_estimators=1000, max_depth=12, random_state=42, n_jobs=-1)
    print("Training Logistic Regression...")
    lr = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
    print("Training Support Vector Machine...")
    svc = SVC(probability=True)
    
    print("Fitting the ensemble classifier...")
    ec = VotingClassifier(estimators=[('Multinomial NB', mnb), 
            ('Random Forest', rfc), 
            ('Logistic Regression', lr), 
            ('Support Vector Machine', svc)], 
            voting='soft', weights=[1, 2, 3, 4], n_jobs=-1)
    
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

  
    # Initialize the label encoder and fit on the actual classes
    label_encoder = LabelEncoder()
    label_encoder.fit(y_test)

    # Transform y_test and y_pred1 into encoded numeric labels
    y_test_encoded = label_encoder.transform(y_test)
    y_pred6_encoded = label_encoder.transform(y_pred6)

    # Calculate additional metrics and generate probability predictions for confidence analysis
    #y_pred_proba = abc.predict_proba(x_test_tfidf)  # Probability predictions for AdaBoost
    y_pred_proba = ec.predict_proba(x_test_tfidf)  # Probability predictions for each class from ensemble
    #metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
    # Call calculate_metrics with encoded labels
    metrics = calculate_metrics(y_test_encoded, y_pred6_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)

    #joblib.dump(ec, str(MODELS_DIR / 'Suicide_Ensemble.pkl'))
    # Save the model
    model_filename = str(MODELS_DIR / 'Suicide_Ensemble.pkl')
    joblib.dump(ec, model_filename)
    print(f"Model saved as {model_filename}")

    #To load the saved Ensemble model:
    #loaded_model = joblib.load(str(MODELS_DIR / 'Suicide_Ensemble.pkl'))

    # Perform comprehensive misclassification analysis
    perform_misclassification_analysis(
        df_actual=df_actual,            # Actual test data
        y_test=y_test,                  # Actual labels
        y_pred=y_pred6,                 # Predicted labels
        algorithm_name="Ensemble_Classifier",
        y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
        labels=lr.classes_,              # Class labels for confusion matrix
        label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
    )

    ## **AdaBoost with Random Forest Classifier**
    print('**AdaBoost with Random Forest Classifier**')

    rfc = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=0)
    abc = AdaBoostClassifier(estimator=rfc, learning_rate=0.2, n_estimators=100)
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

    # Assuming you have y_test, y_pred1, and y_pred_proba available
    #metrics = calculate_metrics(y_test, y_pred7, y_pred_proba=lr.predict_proba(x_test_tfidf), classes=lr.classes_)

    # Initialize the label encoder and fit on the actual classes
    label_encoder = LabelEncoder()
    label_encoder.fit(y_test)

    # Transform y_test and y_pred1 into encoded numeric labels
    y_test_encoded = label_encoder.transform(y_test)
    y_pred1_encoded = label_encoder.transform(y_pred7)

    # Calculate additional metrics and generate probability predictions for confidence analysis
    # y_pred_proba already computed from the current model above
    #metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
    # Call calculate_metrics with encoded labels
    metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)

    # Save the model
    model_filename = str(MODELS_DIR / 'adaboost_random_forest_model.joblib')
    joblib.dump(abc, model_filename)
    print(f"Model saved as {model_filename}")

    #To load this saved model:
    #loaded_model = joblib.load('adaboost_random_forest_model.joblib')

    # Perform comprehensive misclassification analysis
    perform_misclassification_analysis(
        df_actual=df_actual,            # Actual test data
        y_test=y_test,                  # Actual labels
        y_pred=y_pred7,                 # Predicted labels
        algorithm_name="AdaBoost_with_Random_Forest_Classifier",
        y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
        labels=lr.classes_,              # Class labels for confusion matrix
        label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
    )

    # **Comparison Between ML Models**
    print('**Comparison Between ML Models**')

    Comparison_unibi = pd.DataFrame({
        'Logistic Regression': [accuracy_score(y_test, y_pred1)*100, f1_score(y_test, y_pred1, average='macro')*100, recall_score(y_test, y_pred1, average='micro')*100, precision_score(y_test, y_pred1, average='micro')*100],
        'SVM': [accuracy_score(y_test, y_pred2)*100, f1_score(y_test, y_pred2, average='macro')*100, recall_score(y_test, y_pred2, average='micro')*100, precision_score(y_test, y_pred2, average='micro')*100],
        'Naive Bayes': [accuracy_score(y_test, y_pred3)*100, f1_score(y_test, y_pred3, average='macro')*100, recall_score(y_test, y_pred3, average='micro')*100, precision_score(y_test, y_pred3, average='micro')*100],
        'Random Forest': [accuracy_score(y_test, y_pred4)*100, f1_score(y_test, y_pred4, average='macro')*100, recall_score(y_test, y_pred4, average='micro')*100, precision_score(y_test, y_pred4, average='micro')*100],
        'GradientBoosting': [accuracy_score(y_test, y_pred5)*100, f1_score(y_test, y_pred5, average='macro')*100, recall_score(y_test, y_pred5, average='micro')*100, precision_score(y_test, y_pred5, average='micro')*100],
        'Ensembled': [accuracy_score(y_test, y_pred6)*100, f1_score(y_test, y_pred6, average='macro')*100, recall_score(y_test, y_pred6, average='micro')*100, precision_score(y_test, y_pred6, average='micro')*100],
        'Adaboost': [accuracy_score(y_test, y_pred7)*100, f1_score(y_test, y_pred7, average='macro')*100, recall_score(y_test, y_pred7, average='micro')*100, precision_score(y_test, y_pred7, average='micro')*100]
    })

    print('Comparison using uni-gram(1,1)')
    Comparison_unibi.rename(index={0: 'Accuracy', 1: 'F1_score', 2: 'Recall', 3: 'Precision'}, inplace=True)
    Comparison_unibi.head()

    print('Comparison using bi-gram(2,2)')
    Comparison_unibi.rename(index={0: 'Accuracy', 1: 'F1_score', 2: 'Recall', 3: 'Precision'}, inplace=True)
    Comparison_unibi.head()

    print('Comparison using uni-bi-gram(1,2)')
    Comparison_unibi.rename(index={0: 'Accuracy', 1: 'F1_score', 2: 'Recall', 3: 'Precision'}, inplace=True)
    Comparison_unibi.head()

# **Deep Learning Models**
print('**Deep Learning Models**')

# Define vocabulary size and max text length
vocabulary_size = 6000
max_text_len = 60

# Initialize and fit tokenizer on the full training data
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(df_full['Tweet'].values)  # Using df_full for training

# Check tokenizer vocabulary size
le = len(tokenizer.word_index) + 1
print(le)

# Convert training and testing texts to sequences and pad them
sequences_train = tokenizer.texts_to_sequences(df_full['Tweet'].values)
X_DeepLearning_train = pad_sequences(sequences_train, maxlen=max_text_len)

sequences_test = tokenizer.texts_to_sequences(df_actual['Tweet'].values)
X_DeepLearning_test = pad_sequences(sequences_test, maxlen=max_text_len)

# Check shape of padded sequences
print("Training Data Shape:", X_DeepLearning_train.shape)
print("Testing Data Shape:", X_DeepLearning_test.shape)

# Save the tokenizer object
with open(str(MODELS_DIR / 'tokenizer.pickle'), 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the tokenizer
#with open(str(MODELS_DIR / 'tokenizer.pickle'), 'rb') as handle:
#    tokenizer = pickle.load(handle)

# Update labels for categorical encoding
df_full.loc[df_full['Suicide'] == 'Not Suicide post', 'LABEL'] = 0       # Potential Suicide post
df_full.loc[df_full['Suicide'] == 'Method or action of Suicide, Self-Harm or Harming others', 'LABEL'] = 1
df_full.loc[df_full['Suicide'] == 'Suicide or Self Harm Ideation', 'LABEL'] = 2


# Assign labels to df_actual as well, using the same logic as df_full
df_actual.loc[df_actual['Suicide'] == 'Not Suicide post', 'LABEL'] = 0
df_actual.loc[df_actual['Suicide'] == 'Method or action of Suicide, Self-Harm or Harming others', 'LABEL'] = 1
df_actual.loc[df_actual['Suicide'] == 'Suicide or Self Harm Ideation', 'LABEL'] = 2


# Ensure the 'LABEL' column exists in both DataFrames
print("df_full LABEL column:", 'LABEL' in df_full.columns)
print("df_actual LABEL column:", 'LABEL' in df_actual.columns)

# Convert labels to categorical
labels_train = to_categorical(df_full['LABEL'], num_classes=number_of_classes)
labels_test = to_categorical(df_actual['LABEL'], num_classes=number_of_classes)

print("Training Labels Shape:", labels_train.shape)
print("Testing Labels Shape:", labels_test.shape)

# **Train-Test Split for Deep Learning Model**
XX_train, XX_val, y_train, y_val = train_test_split(X_DeepLearning_train, labels_train, test_size=0.25, random_state=42)
print("Train, Validation Shapes:", XX_train.shape, y_train.shape, XX_val.shape, y_val.shape)

# Use only actual sentences for final testing
print("Test Shapes:", X_DeepLearning_test.shape, labels_test.shape)



# **LSTM 1-Layer**
print('**LSTM 1-Layer**')

# Model parameters
epochs = 10
emb_dim = 120
batch_size = 50

# Define the LSTM model
model_lstm1 = Sequential()
model_lstm1.add(Embedding(vocabulary_size, emb_dim, input_length=X_DeepLearning_train.shape[1]))
model_lstm1.add(SpatialDropout1D(0.8))
model_lstm1.add(Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.5)))
model_lstm1.add(Dropout(0.5))
model_lstm1.add(Flatten())
model_lstm1.add(Dense(64, activation='relu'))
model_lstm1.add(Dropout(0.5))
model_lstm1.add(Dense(number_of_classes, activation='softmax'))  # Set to 8 output classes

model_lstm1.compile(optimizer=tf.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])
print(model_lstm1.summary())

# Define callbacks
checkpoint_callback = ModelCheckpoint(filepath=str(MODELS_DIR / "lstm-1-layer-best_model.h5"), save_best_only=True, monitor="val_acc", mode="max", verbose=1)
early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1, restore_best_weights=True)
reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_delta=0.0001, cooldown=0, min_lr=0)
callbacks = [checkpoint_callback, early_stopping_callback, reduce_lr_callback]

# Train the model
history_lstm1 = model_lstm1.fit(XX_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(XX_val, y_val), callbacks=callbacks)

# Evaluate on test data
results_1 = model_lstm1.evaluate(X_DeepLearning_test, labels_test, verbose=False)
print(f'Test results - Loss: {results_1[0]} - Accuracy: {100 * results_1[1]}%')

# Load the best model for later use
#model = load_model(str(MODELS_DIR / 'lstm-1-layer-best_model.h5'))

# Plot training and validation accuracy
acc = history_lstm1.history['acc']
val_acc = history_lstm1.history['val_acc']
loss = history_lstm1.history['loss']
val_loss = history_lstm1.history['val_loss']

plt.plot(acc, 'go', label='Train accuracy')
plt.plot(val_acc, 'g', label='Validate accuracy')
plt.title('Train and validate accuracy')
plt.legend()

plt.figure()
plt.plot(loss, 'go', label='Train loss')
plt.plot(val_loss, 'g', label='Validate loss')
plt.title('Train and validate loss')
plt.legend()
plt.show()

# **Load tokenizer object**
print('**Load tokenizer object**')
with open(str(MODELS_DIR / 'tokenizer.pickle'), 'rb') as handle:
    tokenizers = pickle.load(handle)

# Load and use the best model for predictions
model = load_model(str(MODELS_DIR / 'lstm-1-layer-best_model.h5'))

# Prediction on new data
twt = ['i will not kill myself ']
twt = tokenizers.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=60, dtype='int32')

# Make predictions
predicted = model.predict(twt, batch_size=1, verbose=True)
predicted_class = np.argmax(predicted)

# Interpret the prediction
if predicted_class == 0:
    print("Not Suicide post")
elif predicted_class == 1:
    print("Method or action of Suicide, Self-Harm or Harming others")
elif predicted_class == 2:
    print("Suicide or Self Harm Ideation")


#To call the calculate_metrics function after training and evaluating the LSTM model, follow these steps:
#Make predictions on the test set (X_DeepLearning_test).
#Convert the predicted probabilities to class labels.
#Convert labels_test from one-hot encoding to label format for compatibility with metrics calculation.
#Call calculate_metrics using the true and predicted labels, along with the predicted probabilities.

from tensorflow.keras.models import load_model
from metrics_calculator import calculate_metrics  # Import the custom metrics function

# After loading the tokenizer and model, proceed to make predictions on the test data
# Predictions on test data (X_DeepLearning_test)
y_pred_proba = model_lstm1.predict(X_DeepLearning_test)  # Probability predictions for each class
y_pred = y_pred_proba.argmax(axis=1)  # Convert probabilities to class labels

# Convert one-hot encoded `labels_test` back to single-label format
y_test_labels = labels_test.argmax(axis=1)

# Define class names (matching the order in your 'LABEL' column)
class_names = ['Not Suicide post', 'Method or action of Suicide, Self-Harm or Harming others',
               'Suicide or Self Harm Ideation'] #, 'Depression', 'Advice']

# Calculate metrics
#metrics = calculate_metrics(y_true=y_test_labels, y_pred=y_pred, y_pred_proba=y_pred_proba, classes=class_names)

# Initialize the label encoder and fit on the actual classes
label_encoder = LabelEncoder()
label_encoder.fit(y_test)

# Transform y_test and y_pred1 into encoded numeric labels
y_test_encoded = label_encoder.transform(y_test)
y_pred1_encoded = label_encoder.transform(y_pred)

# Calculate additional metrics and generate probability predictions for confidence analysis
# y_pred_proba already computed from the current model above
#metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
# Call calculate_metrics with encoded labels
metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)



# Perform comprehensive misclassification analysis
perform_misclassification_analysis(
    df_actual=df_actual,            # Actual test data
    y_test=y_test,                  # Actual labels
    y_pred=y_pred,                 # Predicted labels
    algorithm_name="LSTM_1_Layer",
    y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
    labels=lr.classes_,              # Class labels for confusion matrix
    label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
)


## **LSTM 2-Layers**

epochs = 10
emb_dim = 120
batch_size = 50
model_lstm2 = Sequential()
model_lstm2.add(Embedding(vocabulary_size,emb_dim ,input_length=X_DeepLearning_train.shape[1]))
model_lstm2.add(SpatialDropout1D(0.8))
model_lstm2.add(Bidirectional(LSTM(200, dropout=0.5, recurrent_dropout=0.5, return_sequences= True)))
model_lstm2.add(Dropout(0.5))
model_lstm2.add(Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout =0.5)))
model_lstm2.add(Dropout(0.5))
model_lstm2.add(Flatten())
model_lstm2.add(Dense(64, activation='relu'))
model_lstm2.add(Dropout(0.5))
model_lstm2.add(Dense(number_of_classes, activation='softmax'))  
##### update from 2 to 3
model_lstm2.compile(optimizer=tf.optimizers.Adam(),loss='categorical_crossentropy', metrics=['acc'])
print(model_lstm2.summary())

checkpoint_callback = ModelCheckpoint(filepath=str(MODELS_DIR / "lastm-2-layer-best_model.h5"), save_best_only=True, monitor="val_acc", mode="max", verbose=1)

early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1, restore_best_weights=True)

reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_delta=0.0001, cooldown=0, min_lr=0)

callbacks2=[checkpoint_callback, early_stopping_callback, reduce_lr_callback]

history_lstm2 = model_lstm2.fit(XX_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=callbacks2)

results_2 = model_lstm2.evaluate(X_DeepLearning_test, y_test, verbose=False)
print(f'Test results - Loss: {results_2[0]} - Accuracy: {100*results_2[1]}%')


# Load the best model
#model = load_model(str(MODELS_DIR / 'lastm-2-layer-best_model.h5'))

acc = history_lstm2.history['acc']
val_acc = history_lstm2.history['val_acc']
loss = history_lstm2.history['loss']
val_loss = history_lstm2.history['val_loss']

plt.plot( acc, 'go', label='Train accuracy')
plt.plot( val_acc, 'g', label='Validate accuracy')
plt.title('Train and validate accuracy')
plt.legend()

plt.figure()

plt.plot( loss, 'go', label='Train loss')
plt.plot( val_loss, 'g', label='Validate loss')
plt.title('Train and validate loss')
plt.legend()

plt.show()

# Load tokenizer object
with open(str(MODELS_DIR / 'tokenizer.pickle'), 'rb') as handle:
    tokenizers = pickle.load(handle)

model = load_model(str(MODELS_DIR / 'lastm-2-layer-best_model.h5'))
#model.save('/content/drive/MyDrive/Colab_Notebooks/DL Model/Twitter Suicide Ideation Detection/lstm 2-layer.h5')

twt = ["i will not kill myself. "]
twt = tokenizers.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=60, dtype='int32')




predicted = model.predict(twt,batch_size=1,verbose = True)
# if(np.argmax(predicted) == 0):
#     print("Not Suicide post")
# if (np.argmax(predicted) == 1):
#     print("Suicide method, Self Harm Method or Method to harm others")
# elif (np.argmax(predicted) == 2):
#     print("Suicide Ideation post")

if (np.argmax(predicted) == 0):
    print("Not Suicide post")
elif (np.argmax(predicted) == 1):
    print("Method or action of Suicide, Self-Harm or Harming others")
elif (np.argmax(predicted) == 2):
    print("Suicide or Self Harm Ideation")



from tensorflow.keras.models import load_model
from metrics_calculator import calculate_metrics  # Import the custom metrics function

# Predictions on test data (X_DeepLearning_test)
y_pred_proba = model_lstm2.predict(X_DeepLearning_test)  # Probability predictions for each class
y_pred = y_pred_proba.argmax(axis=1)  # Convert probabilities to class labels

# Convert one-hot encoded labels_test back to single labels for compatibility with metrics
y_test_labels = labels_test.argmax(axis=1)



class_names = ['Not Suicide post', 'Method or action of Suicide, Self-Harm or Harming others',
               'Suicide or Self Harm Ideation'] #, 'Depression', 'Advice']

# Calculate metrics
#metrics = calculate_metrics(y_true=y_test_labels, y_pred=y_pred, y_pred_proba=y_pred_proba, classes=class_names)

# Initialize the label encoder and fit on the actual classes
label_encoder = LabelEncoder()
label_encoder.fit(y_test)

# Transform y_test and y_pred1 into encoded numeric labels
y_test_encoded = label_encoder.transform(y_test)
y_pred1_encoded = label_encoder.transform(y_pred)

# Calculate additional metrics and generate probability predictions for confidence analysis
# y_pred_proba already computed from the current model above
#metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
# Call calculate_metrics with encoded labels
metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)


# Perform comprehensive misclassification analysis
perform_misclassification_analysis(
    df_actual=df_actual,            # Actual test data
    y_test=y_test,                  # Actual labels
    y_pred=y_pred,                 # Predicted labels
    algorithm_name="LSTM_2_Layers",
    y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
    labels=lr.classes_,              # Class labels for confusion matrix
    label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
)
    
## **GRU**

epochs = 10
emb_dim = 120
batch_size = 50
model_gru = Sequential()
model_gru.add(Embedding(vocabulary_size,emb_dim ,input_length=X_DeepLearning_train.shape[1]))
model_gru.add(SpatialDropout1D(0.5))
model_gru.add(GRU(units=16,  dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.01)))
model_gru.add(Dropout(0.5))
model_gru.add(Dense(228, activation='relu', kernel_regularizer=l2(0.01)))
model_gru.add(Dropout(0.5))
model_gru.add(Dense(number_of_classes, activation='softmax'))  
#### update from 2 to 3 to 8
model_gru.compile(optimizer=tf.optimizers.Adam(),loss='categorical_crossentropy', metrics=['acc'])
print(model_gru.summary())

checkpoint_callback = ModelCheckpoint(filepath=str(MODELS_DIR / "gru-best_model.h5"), save_best_only=True, monitor="val_acc", mode="max", verbose=1)

early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1, restore_best_weights=True)

reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_delta=0.0001, cooldown=0, min_lr=0)

callbacks3=[checkpoint_callback, early_stopping_callback, reduce_lr_callback]

history_gru = model_gru.fit(XX_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1, callbacks=callbacks3)

results_3 = model_gru.evaluate(X_DeepLearning_test, y_test, verbose=False)
print(f'Test results - Loss: {results_3[0]} - Accuracy: {100*results_3[1]}%')

acc = history_gru.history['acc']
val_acc = history_gru.history['val_acc']
loss = history_gru.history['loss']
val_loss = history_gru.history['val_loss']

# Load the GRU model
#model = load_model(str(MODELS_DIR / 'gru-best_model.h5'))

plt.plot( acc, 'go', label='Train accuracy')
plt.plot( val_acc, 'g', label='Validate accuracy')
plt.title('Train and validate accuracy')
plt.legend()

plt.figure()

plt.plot( loss, 'go', label='Train loss')
plt.plot( val_loss, 'g', label='Validate loss')
plt.title('Train and validate loss')
plt.legend()

plt.show()

# Load tokenizer object
with open(str(MODELS_DIR / 'tokenizer.pickle'), 'rb') as handle:
    tokenizers = pickle.load(handle)

model = load_model(str(MODELS_DIR / 'gru-best_model.h5'))
#model.save('/content/drive/MyDrive/Colab_Notebooks/DL Model/Twitter Suicide Ideation Detection/gru-best_model.h5')

twt = ["i will not kill myself."]
twt = tokenizers.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=60, dtype='int32')

predicted = model.predict(twt,batch_size=1,verbose = True)
if(np.argmax(predicted) == 0):
    print("Not Suicide post")
if (np.argmax(predicted) == 1):
    print("Method or action of Suicide, Self-Harm or Harming others")
elif (np.argmax(predicted) == 2):
    print("Suicide or Self Harm Ideation")
# elif (np.argmax(predicted) == 3):
#     print("Not Suicide post")

# After training, make predictions on the test data
y_pred_proba = model_gru.predict(X_DeepLearning_test)  # Probability predictions for each class
y_pred = y_pred_proba.argmax(axis=1)  # Convert probabilities to class labels

# Convert one-hot encoded labels_test back to single labels for compatibility with metrics
y_test_labels = labels_test.argmax(axis=1)


class_names = ['Not Suicide post', 'Method or action of Suicide, Self-Harm or Harming others',
               'Suicide or Self Harm Ideation'] 

# Calculate metrics
#metrics = calculate_metrics(y_true=y_test_labels, y_pred=y_pred, y_pred_proba=y_pred_proba, classes=class_names)

# Initialize the label encoder and fit on the actual classes
label_encoder = LabelEncoder()
label_encoder.fit(y_test)

# Transform y_test and y_pred1 into encoded numeric labels
y_test_encoded = label_encoder.transform(y_test)
y_pred1_encoded = label_encoder.transform(y_pred)

# Calculate additional metrics and generate probability predictions for confidence analysis
# y_pred_proba already computed from the current model above
#metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
# Call calculate_metrics with encoded labels
metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)

# Perform comprehensive misclassification analysis
perform_misclassification_analysis(
    df_actual=df_actual,            # Actual test data
    y_test=y_test,                  # Actual labels
    y_pred=y_pred,                 # Predicted labels
    algorithm_name="GRU",
    y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
    labels=lr.classes_,              # Class labels for confusion matrix
    label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
)




## **CNN+LSTM**

epochs = 10
emb_dim = 120
batch_size = 50
model_cl = Sequential()
model_cl.add(Embedding(vocabulary_size,emb_dim, input_length=X_DeepLearning_train.shape[1]))
model_cl.add(SpatialDropout1D(0.8))
model_cl.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu'))
model_cl.add(MaxPooling1D(pool_size=2))
model_cl.add(Conv1D(filters=32, kernel_size=6, activation='relu'))
model_cl.add(MaxPooling1D(pool_size=2))
model_cl.add(Bidirectional(LSTM(100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
model_cl.add(Dropout(0.5))
model_cl.add(Bidirectional(LSTM(400, dropout=0.5, recurrent_dropout=0.5)))
model_cl.add(Dropout(0.5))
model_cl.add(Flatten())
model_cl.add(Dense(64, activation='relu'))
model_cl.add(Dropout(0.5))
model_cl.add(Dense(number_of_classes, activation='softmax'))   
##### update from 2 to 3
# Reason: The loss function should be categorical_crossentropy instead of binary_crossentropy for multi-class classification, given that you have more than two classes.
#model_cl.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
model_cl.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
print(model_cl.summary())

checkpoint_callback = ModelCheckpoint(filepath=str(MODELS_DIR / "cnn+lastm-best_model.h5"), save_best_only=True, monitor="val_acc", mode="max", verbose=1)

early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1, restore_best_weights=True)

reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_delta=0.0001, cooldown=0, min_lr=0)

callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback]

history_cl = model_cl.fit(XX_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1, callbacks=callbacks)

# Reason: Since you are using multi-class labels, labels_test (the categorical form of y_test) should be used here.
#results_4 = model_cl.evaluate(X_DeepLearning_test, y_test, verbose=False)
results_4 = model_cl.evaluate(X_DeepLearning_test, labels_test, verbose=False)
print(f'Test results - Loss: {results_4[0]} - Accuracy: {100*results_4[1]}%')

acc = history_cl.history['acc']
val_acc = history_cl.history['val_acc']
loss = history_cl.history['loss']
val_loss = history_cl.history['val_loss']
plt.plot( acc, 'go', label='Train accuracy')
plt.plot( val_acc, 'g', label='Validate accuracy')
plt.title('Train and validate accuracy')
plt.legend()


# Load the best CNN+LSTM model
model = load_model(str(MODELS_DIR / 'cnn+lastm-best_model.h5'))

plt.figure()
plt.plot( loss, 'go', label='Train loss')
plt.plot( val_loss, 'g', label='Validate loss')
plt.title('Train and validate loss')
plt.legend()
plt.show()

# Load tokenizer object
with open(str(MODELS_DIR / 'tokenizer.pickle'), 'rb') as handle:
    #tokenizers = pickle.load(handle)
    tokenizer = pickle.load(handle)  # Renamed to tokenizer for consistency

# # Load the best model and save a copy
model = load_model(str(MODELS_DIR / 'cnn+lastm-best_model.h5'))
model_cl.save(str(MODELS_DIR / 'CNN+LSTM.h5'))

twt = ['I will not kill myself']
#twt = tokenizer.texts_to_sequences(twt)
twt = tokenizer.texts_to_sequences(twt)  # Using 'tokenizer' consistently
twt = pad_sequences(twt, maxlen=60, dtype='int32')

predicted = model.predict(twt,batch_size=1,verbose = True)


if (np.argmax(predicted) == 0):
    print("Not Suicide post")
elif (np.argmax(predicted) == 1):
    print("Suicide method, Self Harm Method or Method to harm others")
elif (np.argmax(predicted) == 2):
    print("Suicide or Self Harm Ideation")


# Make predictions on the test data
y_pred_proba = model_cl.predict(X_DeepLearning_test)  # Probability predictions for each class
y_pred = y_pred_proba.argmax(axis=1)  # Convert probabilities to class labels

# Convert one-hot encoded labels_test back to single labels for compatibility with metrics
y_test_labels = labels_test.argmax(axis=1)



class_names = ['Not Suicide post', 'Method or action of Suicide, Self-Harm or Harming others',
               'Suicide or Self Harm Ideation'] 

# Calculate metrics
#metrics = calculate_metrics(y_true=y_test_labels, y_pred=y_pred, y_pred_proba=y_pred_proba, classes=class_names)

# Initialize the label encoder and fit on the actual classes
label_encoder = LabelEncoder()
label_encoder.fit(y_test)

# Transform y_test and y_pred1 into encoded numeric labels
y_test_encoded = label_encoder.transform(y_test)
y_pred1_encoded = label_encoder.transform(y_pred)

# Calculate additional metrics and generate probability predictions for confidence analysis
# y_pred_proba already computed from the current model above
#metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
# Call calculate_metrics with encoded labels
metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)

# Perform comprehensive misclassification analysis
perform_misclassification_analysis(
    df_actual=df_actual,            # Actual test data
    y_test=y_test,                  # Actual labels
    y_pred=y_pred,                 # Predicted labels
    algorithm_name="CNN_LSTM",
    y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
    labels=lr.classes_,              # Class labels for confusion matrix
    label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
)


## **Model Comparision**
print('**Model Comparision**')

results=pd.DataFrame({'Model':['LSTM-1 Layer','LSTM-2 Layer','GRU','CNN+LSTM'],
                     'Accuracy Score':[results_1[1],results_2[1],results_3[1],results_4[1]]})
result_df=results.sort_values(by='Accuracy Score', ascending=False)
result_df=result_df.set_index('Model')
result_df


#Hybrid CNN + LSTM + GRU Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, GRU, Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score, classification_report

# Model definition
model = Sequential([
    Embedding(input_dim=vocabulary_size, output_dim=120, input_length=max_text_len),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    GRU(64),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(number_of_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # Train the model
model.fit(X_train_full, y_train_full, validation_data=(X_val, y_val), epochs=5, batch_size=32)
# Save the model
model_filename = str(MODELS_DIR / 'hybrid_cnn_lstm_gru_model.h5')
model.save(model_filename)
print(f"Model saved as {model_filename}")

# Load the model
#loaded_model = load_model(model_filename)


# Evaluate on test data
y_pred = model.predict(X_test).argmax(axis=1)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"CNN + LSTM + GRU Test Accuracy: {test_accuracy}")
print("CNN + LSTM + GRU Classification Report:\n", classification_report(y_test, y_pred))


# Calculate metrics
#metrics = calculate_metrics(y_true=y_test_labels, y_pred=y_pred, y_pred_proba=y_pred_proba, classes=class_names)

# Initialize the label encoder and fit on the actual classes
label_encoder = LabelEncoder()
label_encoder.fit(y_test)

# Transform y_test and y_pred1 into encoded numeric labels
y_test_encoded = label_encoder.transform(y_test)
y_pred1_encoded = label_encoder.transform(y_pred)

# Calculate additional metrics and generate probability predictions for confidence analysis
# y_pred_proba already computed from the current model above
#metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
# Call calculate_metrics with encoded labels
metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)

# Perform comprehensive misclassification analysis
perform_misclassification_analysis(
    df_actual=df_actual,            # Actual test data
    y_test=y_test,                  # Actual labels
    y_pred=y_pred,                 # Predicted labels
    algorithm_name="Hybrid_CNN_LSTM_GRU",
    y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
    labels=lr.classes_,              # Class labels for confusion matrix
    label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
)


#LSTM with Attention Mechanism
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, classification_report

# Model definition
input_seq = Input(shape=(max_text_len,))
embedding_layer = Embedding(vocabulary_size, emb_dim)(input_seq)
lstm_out = LSTM(128, return_sequences=True)(embedding_layer)
attention_layer = Attention()([lstm_out, lstm_out])
dropout_layer = Dropout(0.5)(attention_layer)
flatten_layer = Flatten()(dropout_layer)
output = Dense(number_of_classes, activation='softmax')(flatten_layer)

model = Model(inputs=input_seq, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_full, y_train_full, validation_data=(X_val, y_val), epochs=5, batch_size=32)

# Save the model
model_filename = str(MODELS_DIR / 'lstm_with_attention_model.h5')
model.save(model_filename)
print(f"Model saved as {model_filename}")

# Load the model
#loaded_model = load_model(model_filename)

# Evaluate on test data
y_pred = model.predict(X_test).argmax(axis=1)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"LSTM with Attention Test Accuracy: {test_accuracy}")
print("LSTM with Attention Classification Report:\n", classification_report(y_test, y_pred))


# Make probability predictions on the test data
y_pred_proba = model.predict(X_test)  # Probability predictions for each class
y_pred = y_pred_proba.argmax(axis=1)  # Convert probabilities to class labels


class_names = ['Not Suicide post', 'Method or action of Suicide, Self-Harm or Harming others',
               'Suicide or Self Harm Ideation'] 

# Calculate metrics
#metrics = calculate_metrics(y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba, classes=class_names)

# Initialize the label encoder and fit on the actual classes
label_encoder = LabelEncoder()
label_encoder.fit(y_test)

# Transform y_test and y_pred1 into encoded numeric labels
y_test_encoded = label_encoder.transform(y_test)
y_pred1_encoded = label_encoder.transform(y_pred)

# Calculate additional metrics and generate probability predictions for confidence analysis
# y_pred_proba already computed from the current model above
#metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
# Call calculate_metrics with encoded labels
metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)

# Perform comprehensive misclassification analysis
perform_misclassification_analysis(
    df_actual=df_actual,            # Actual test data
    y_test=y_test,                  # Actual labels
    y_pred=y_pred,                 # Predicted labels
    algorithm_name="LSTMwithAttentionMechanism",
    y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
    labels=lr.classes_,              # Class labels for confusion matrix
    label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
)



#Stacking Classifiers with Meta-Learner
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Base models
base_models = [
    ('svc', SVC(probability=True)),
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('lr', LogisticRegression())
]

# Meta-model
meta_model = XGBClassifier()
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
stacking_model.fit(X_train_full, y_train_full)

# Test accuracy and report
y_pred = stacking_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Model Test Accuracy: {test_accuracy}")
print("Stacking Model Classification Report:\n", classification_report(y_test, y_pred))


#To save and reload the stacking model, use joblib or pickle:
#import joblib
joblib.dump(stacking_model, str(MODELS_DIR / 'stacking_model.joblib'))
stacking_model = joblib.load(str(MODELS_DIR / 'stacking_model.joblib'))


from metrics_calculator import calculate_metrics  # Import the custom metrics function
import joblib

# Make probability predictions on the test data
y_pred_proba = stacking_model.predict_proba(X_test)  # Probability predictions for each class
y_pred = y_pred_proba.argmax(axis=1)  # Convert probabilities to class labels


class_names = ['Not Suicide post', 'Method or action of Suicide, Self-Harm or Harming others',
               'Suicide or Self Harm Ideation'] 

# Calculate metrics
#metrics = calculate_metrics(y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba, classes=class_names)

# Initialize the label encoder and fit on the actual classes
label_encoder = LabelEncoder()
label_encoder.fit(y_test)

# Transform y_test and y_pred1 into encoded numeric labels
y_test_encoded = label_encoder.transform(y_test)
y_pred1_encoded = label_encoder.transform(y_pred)

# Calculate additional metrics and generate probability predictions for confidence analysis

metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)

# Perform comprehensive misclassification analysis
perform_misclassification_analysis(
    df_actual=df_actual,            # Actual test data
    y_test=y_test,                  # Actual labels
    y_pred=y_pred,                 # Predicted labels
    algorithm_name="StackingClassifierswithMeta_Learner",
    y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
    labels=lr.classes_,              # Class labels for confusion matrix
    label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
)




## **Bert Model**
print('\n############Bert Model')

#X_train, X_test, y_train, y_test = train_test_split(df['Tweet'], df['Suicide'], test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(df_full['Tweet'], df_full['LABEL'], test_size=0.33, random_state=42)

X_train = X_train.tolist()
X_test = X_test.tolist()
y_train = y_train.tolist()
y_test = y_test.tolist()


class_names = ['Not Suicide post', 'Method or action of Suicide, Self-Harm or Harming others',
               'Suicide or Self Harm Ideation'] 

# added code

import numpy as np

# Flatten y_train and y_test if they are arrays or one-hot encoded
y_train_flat = [np.argmax(y) if isinstance(y, np.ndarray) else y for y in y_train]
y_test_flat = [np.argmax(y) if isinstance(y, np.ndarray) else y for y in y_test]

# Display unique labels in y_train and y_test
print("Unique labels in y_train 1:", set(y_train_flat))
print("Unique labels in y_test 1:", set(y_test_flat))

# Count the occurrences of each label to inspect any inconsistencies
from collections import Counter
print("Label counts in y_train 1:", Counter(y_train_flat))
print("Label counts in y_test 1:", Counter(y_test_flat))

(x_train,y_train), (x_val,y_val), preproc = text.texts_from_array(x_train=X_train, y_train=y_train,
                                                                       x_test=X_test, y_test=y_test,
                                                                       class_names=class_names,
                                                                       preprocess_mode='bert',
                                                                       maxlen=140,
                                                                       max_features=5000)

import numpy as np

# Flatten y_train and y_test if they are arrays or one-hot encoded
y_train_flat = [np.argmax(y) if isinstance(y, np.ndarray) else y for y in y_train]
y_test_flat = [np.argmax(y) if isinstance(y, np.ndarray) else y for y in y_test]

# Display unique labels in y_train and y_test
print("Unique labels in y_train 2:", set(y_train_flat))
print("Unique labels in y_test 2:", set(y_test_flat))

# Count the occurrences of each label to inspect any inconsistencies
from collections import Counter
print("Label counts in y_train 2:", Counter(y_train_flat))
print("Label counts in y_test 2:", Counter(y_test_flat))
# added code

model = text.text_classifier('bert', train_data=(x_train,y_train), preproc=preproc)

learner = ktrain.get_learner(model, train_data=(x_train,y_train),
                             val_data=(x_val,y_val),
                             batch_size=16)

learner.fit_onecycle(2e-5, number_of_classes) #### changed from 3 to 8

learner.plot()

# @title
learner.validate(val_data=(x_val,y_val), class_names=class_names)

predictor = ktrain.get_predictor(learner.model, preproc)
predictor.get_classes()

message = 'i will not kill myself'
prediction = predictor.predict(message)
print('predicted: {}'.format(prediction))


# more metrics
from metrics_calculator import calculate_metrics  # Import the custom metrics function
import numpy as np

# Predict on the test set using the predictor
y_pred_proba = np.array([predictor.predict_proba(text) for text in X_test])  # Probability predictions for each class
y_pred = y_pred_proba.argmax(axis=1)  # Convert probabilities to class labels

# Calculate metrics using y_test and y_pred
#metrics = calculate_metrics(y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba, classes=class_names)

# Initialize the label encoder and fit on the actual classes
label_encoder = LabelEncoder()
label_encoder.fit(y_test)

# Transform y_test and y_pred1 into encoded numeric labels
y_test_encoded = label_encoder.transform(y_test)
y_pred1_encoded = label_encoder.transform(y_pred)

# Calculate additional metrics and generate probability predictions for confidence analysis
# y_pred_proba already computed from the current model above
#metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
# Call calculate_metrics with encoded labels
metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)

# Perform comprehensive misclassification analysis
perform_misclassification_analysis(
    df_actual=df_actual,            # Actual test data
    y_test=y_test,                  # Actual labels
    y_pred=y_pred,                 # Predicted labels
    algorithm_name="Bert_Model",
    y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
    labels=lr.classes_,              # Class labels for confusion matrix
    label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
)

#**Save Bert Model**

#from google.colab import drive
#drive.mount('/content/drive/colab/')

predictor.save(str(MODELS_DIR / "pre/"))

#**Load Saved Model and Predict**
predictor1 = ktrain.load_predictor(str(MODELS_DIR / 'pre'))

data = "I'm so tired of pretending that everything is okay. I just want to cut myself"
print('Now going to try predict for: ' + str(data))
prediction = predictor1.predict(data)
print('predicted: {}'.format(prediction))






# RoBERTa for Text Classification
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# Initialize tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=number_of_classes)

# Tokenize training, validation, and test data
train_encodings = tokenizer(X_train_full.tolist(), padding=True, truncation=True, return_tensors="tf")
val_encodings = tokenizer(X_val.tolist(), padding=True, truncation=True, return_tensors="tf")
test_encodings = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors="tf")

# Compile and train the model
optimizer = Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
model.fit(train_encodings['input_ids'], y_train_full, validation_data=(val_encodings['input_ids'], y_val), epochs=3, batch_size=16)

# Save the model and tokenizer
model.save_pretrained(str(MODELS_DIR / 'roberta_text_classifier'))
tokenizer.save_pretrained(str(MODELS_DIR / 'roberta_text_classifier'))
print("Model and tokenizer saved successfully.")

# Load the model and tokenizer
#loaded_tokenizer = RobertaTokenizer.from_pretrained(str(MODELS_DIR / 'roberta_text_classifier'))
#loaded_model = TFRobertaForSequenceClassification.from_pretrained(str(MODELS_DIR / 'roberta_text_classifier'))



# Evaluate on test data
y_pred = model.predict(test_encodings['input_ids']).logits.argmax(axis=1)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"RoBERTa Test Accuracy: {test_accuracy}")
print("RoBERTa Classification Report:\n", classification_report(y_test, y_pred))

from metrics_calculator import calculate_metrics  # Import the custom metrics function
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

# Evaluate on test data
y_pred_proba = model.predict(test_encodings['input_ids']).logits  # Get logits (pre-softmax scores)
y_pred = y_pred_proba.argmax(axis=1)  # Convert logits to class labels

# Convert logits to probability predictions using softmax
import numpy as np
y_pred_proba = np.exp(y_pred_proba) / np.sum(np.exp(y_pred_proba), axis=1, keepdims=True)


class_names = ['Not Suicide post', 'Method or action of Suicide, Self-Harm or Harming others',
               'Suicide or Self Harm Ideation'] 

# Calculate metrics
#metrics = calculate_metrics(y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba, classes=class_names)

# Initialize the label encoder and fit on the actual classes
label_encoder = LabelEncoder()
label_encoder.fit(y_test)

# Transform y_test and y_pred1 into encoded numeric labels
y_test_encoded = label_encoder.transform(y_test)
y_pred1_encoded = label_encoder.transform(y_pred)

# Calculate additional metrics and generate probability predictions for confidence analysis
# y_pred_proba already computed from the current model above
#metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
# Call calculate_metrics with encoded labels
metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)

# Perform comprehensive misclassification analysis
perform_misclassification_analysis(
    df_actual=df_actual,            # Actual test data
    y_test=y_test,                  # Actual labels
    y_pred=y_pred,                 # Predicted labels
    algorithm_name="RoBERTa",
    y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
    labels=lr.classes_,              # Class labels for confusion matrix
    label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
)


#DistilBERT for Lightweight Model Performance
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=number_of_classes)

# Tokenize data
train_encodings = tokenizer(X_train_full.tolist(), truncation=True, padding=True, return_tensors="tf")
val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True, return_tensors="tf")
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, return_tensors="tf")

# Compile and train the model
model.compile(optimizer=Adam(learning_rate=2e-5), loss=model.compute_loss, metrics=['accuracy'])
model.fit(train_encodings['input_ids'], y_train_full, validation_data=(val_encodings['input_ids'], y_val), epochs=3, batch_size=16)

# Save the model and tokenizer
model.save_pretrained(str(MODELS_DIR / 'distilbert_text_classifier'))
tokenizer.save_pretrained(str(MODELS_DIR / 'distilbert_text_classifier'))
print("Model and tokenizer saved successfully.")

# Load the model and tokenizer
#loaded_tokenizer = DistilBertTokenizer.from_pretrained(str(MODELS_DIR / 'distilbert_text_classifier'))
#loaded_model = TFDistilBertForSequenceClassification.from_pretrained(str(MODELS_DIR / 'distilbert_text_classifier'))


# Test accuracy and report
y_pred = model.predict(test_encodings['input_ids']).logits.argmax(axis=1)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"DistilBERT Test Accuracy: {test_accuracy}")
print("DistilBERT Classification Report:\n", classification_report(y_test, y_pred))


from metrics_calculator import calculate_metrics  # Import the custom metrics function
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np

# Predict on the test data
y_pred_proba = model.predict(test_encodings['input_ids']).logits  # Get logits (pre-softmax scores)
y_pred = y_pred_proba.argmax(axis=1)  # Convert logits to class labels

# Convert logits to probability predictions using softmax
y_pred_proba = np.exp(y_pred_proba) / np.sum(np.exp(y_pred_proba), axis=1, keepdims=True)


class_names = ['Not Suicide post', 'Method or action of Suicide, Self-Harm or Harming others',
               'Suicide or Self Harm Ideation'] 

# Calculate metrics
#metrics = calculate_metrics(y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba, classes=class_names)

# Initialize the label encoder and fit on the actual classes
label_encoder = LabelEncoder()
label_encoder.fit(y_test)

# Transform y_test and y_pred1 into encoded numeric labels
y_test_encoded = label_encoder.transform(y_test)
y_pred1_encoded = label_encoder.transform(y_pred)

# Calculate additional metrics and generate probability predictions for confidence analysis
# y_pred_proba already computed from the current model above
#metrics = calculate_metrics(y_test, y_pred1, y_pred_proba=y_pred_proba, classes=lr.classes_)
# Call calculate_metrics with encoded labels
metrics = calculate_metrics(y_test_encoded, y_pred1_encoded, y_pred_proba=y_pred_proba, classes=label_encoder.classes_)

# Perform comprehensive misclassification analysis
perform_misclassification_analysis(
    df_actual=df_actual,            # Actual test data
    y_test=y_test,                  # Actual labels
    y_pred=y_pred,                 # Predicted labels
    algorithm_name="DistilBERT",
    y_pred_proba=y_pred_proba,      # Probability predictions for confidence analysis
    labels=lr.classes_,              # Class labels for confusion matrix
    label_encoder=label_encoder     # Label encoder for probability-based confidence analysis
)

con.close()





###### Anomaly Detection
print('\n############Anomaly Detection')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

# We'll use the BERT model to get embeddings for all the data
all_data = X_train + X_test

# Get embeddings using the BERT model
# This might take some time depending on the size of your dataset
all_embeddings = []
batch_size = 32  # Adjust this based on your available memory
for i in range(0, len(all_data), batch_size):
    batch = all_data[i:i+batch_size]
    batch_embeddings = predictor.predict(batch, return_proba=True)
    all_embeddings.extend(batch_embeddings)

all_embeddings = np.array(all_embeddings)

# Standardize the embeddings
scaler = StandardScaler()
all_embeddings_scaled = scaler.fit_transform(all_embeddings)

# Initialize and fit the Isolation Forest
contamination = 0.1  # Adjust this based on your expectations of anomalies
clf = IsolationForest(contamination=contamination, random_state=42)
clf.fit(all_embeddings_scaled)

# Predict anomalies
anomaly_labels = clf.predict(all_embeddings_scaled)

# Create a DataFrame with the original text and anomaly labels
anomaly_df = pd.DataFrame({
    'text': all_data,
    'is_anomaly': anomaly_labels
})

# Print summary
print(f"Total samples: {len(anomaly_df)}")
print(f"Number of anomalies detected: {sum(anomaly_df['is_anomaly'] == -1)}")

# Print some example anomalies
print("\nExample anomalies:")
print(anomaly_df[anomaly_df['is_anomaly'] == -1].head())

# Optional: Save results to a CSV file
anomaly_df.to_csv(str(OUTPUTS_DIR / 'anomaly_detection_results.csv'), index=False)
print("\nResults saved to 'anomaly_detection_results.csv'")

# Visualize the distribution of normal vs anomalous samples
plt.figure(figsize=(10, 6))
sns.countplot(x='is_anomaly', data=anomaly_df)
plt.title('Distribution of Normal vs Anomalous Samples')
plt.xlabel('Is Anomaly (-1: Yes, 1: No)')
plt.ylabel('Count')
plt.savefig('anomaly_distribution.png')
plt.close()

print("\nAnomaly distribution plot saved as 'anomaly_distribution.png'")




## Error Analysis and CSV Output
## Capture incorrectly classified rows

# Make predictions on the test set
predictions = predictor.predict(X_test)  # Predict on test set

# Convert predicted class labels to match the original label format
predicted_labels = [class_names[np.argmax(pred)] for pred in predictions]

# Compare predicted labels with true labels
incorrectly_classified = []
for i in range(len(X_test)):
    if predicted_labels[i] != y_test[i]:
        incorrectly_classified.append({
            'Tweet': X_test[i],
            'True Label': y_test[i],
            'Predicted Label': predicted_labels[i]
        })

# Convert to DataFrame for easier visualization
incorrect_df = pd.DataFrame(incorrectly_classified)

# Output the incorrectly classified rows
print("\nIncorrectly classified rows:")
print(incorrect_df)

# Optionally, save the incorrectly classified rows to a CSV for further analysis
incorrect_df.to_csv(str(OUTPUTS_DIR / 'incorrect_classifications.csv'), index=False)
