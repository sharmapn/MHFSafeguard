# ML_DLearning

This folder contains the machine-learning and deep-learning scripts used to train and evaluate the MHF Safeguard classifier.

The classifier is intended to support the XenForo MHF Safeguard plugin by classifying forum posts, replies, or edited messages into three safety-related categories:

1. `Not Suicide post`
2. `Suicide or Self Harm Ideation`
3. `Method or action of Suicide, Self-Harm or Harming others`

The main purpose of this folder is to train models that can identify high-risk method/action content and return a label, score, and recommended moderation action to the MHF Safeguard server/plugin workflow.

## Current files

The main files in this folder are:

```text
ML_DLearning/
├── training5_additional_datasets_improved_code.py
├── training5_additional_datasets_improved_code_FULL_SAFE.py
├── training5_additional_datasets_improved_code_SAFE.py
├── metrics_calculator.py
├── misclassification_analysis.py
├── path_setup.py
├── requirements.txt
├── data/
├── models/
└── outputs/
```

### Important note about the scripts

`training5_additional_datasets_improved_code.py` is the original long experiment script. It contains the full model workflow, including machine-learning, deep-learning, BERT-style models, error analysis, and anomaly-detection experiments.

`training5_additional_datasets_improved_code_SAFE.py` is a shorter simplified runner. It was created only as a safe test version and does **not** preserve all models from the original script.

`training5_additional_datasets_improved_code_FULL_SAFE.py` is the recommended safer version. It does **not** condense the original script. Instead, it reads the original script, applies path and safety patches at runtime, saves a patched copy for inspection, and then runs the patched version. This preserves the original workflow while reducing the risk of the script running for days and then failing because of path errors.

## Why the FULL_SAFE wrapper was created

The original script had several risks:

- Model paths were inconsistent, for example `models/...`, `../models/...`, and bare filenames such as `lstm-1-layer-best_model.h5`.
- The LSTM model completed training, but then failed because the script attempted to load a checkpoint from the wrong folder.
- Output was written to `output.txt` in append mode, making logs from multiple runs difficult to separate.
- `df_actual = df_full` meant the test set became the full dataset, including generated and paraphrased rows.
- Some misclassification-analysis calls used the wrong prediction variable.
- Some metrics for SVM, Naive Bayes, Random Forest, Gradient Boosting, and deep-learning models reused Logistic Regression probabilities.
- Some deep-learning sections used `binary_crossentropy` even though the current task is three-class softmax classification.
- Some later neural models were still hard-coded for 8 output classes, even though the current task has 3 classes.

The FULL_SAFE wrapper fixes the most dangerous issues without removing the original experimental sections.

## Recommended script to run

Use:

```bash
python training5_additional_datasets_improved_code_FULL_SAFE.py
```

This script keeps the original full workflow but redirects paths and fixes known safety issues.

Before running, place the SQLite database here:

```text
ML_DLearning/data/all_datasets_labelled.db
```

or set the database path manually:

### Windows PowerShell

```powershell
$env:MHFS_DB_PATH="C:\path\to\all_datasets_labelled.db"
python training5_additional_datasets_improved_code_FULL_SAFE.py
```

### Windows CMD

```cmd
set MHFS_DB_PATH=C:\path\to\all_datasets_labelled.db
python training5_additional_datasets_improved_code_FULL_SAFE.py
```

### Linux/macOS

```bash
export MHFS_DB_PATH="/path/to/all_datasets_labelled.db"
python training5_additional_datasets_improved_code_FULL_SAFE.py
```

## Local path design

All output paths are now kept inside the `ML_DLearning` folder.

```text
ML_DLearning/data/                 database location
ML_DLearning/models/               saved ML/DL models and tokenizers
ML_DLearning/outputs/              general outputs
ML_DLearning/outputs/logs/         timestamped run logs
ML_DLearning/outputs/reports/      reports and metrics
ML_DLearning/outputs/misclassification/  misclassification output
ML_DLearning/outputs/plots/        plots and figures
```

This design avoids failures caused by mixed relative paths.

For example, instead of saving or loading from:

```text
../models/lstm-1-layer-best_model.h5
models/lstm-1-layer-best_model.h5
lstm-1-layer-best_model.h5
```

the safe wrapper redirects these to:

```text
ML_DLearning/models/lstm-1-layer-best_model.h5
```

## Current testing dataset behaviour

By default, the FULL_SAFE wrapper sets:

```text
MHFS_USE_ACTUAL_ONLY_TEST=1
```

This means `df_actual` is loaded as an actual-only testing dataset.

The current actual-only test dataset contains rows from:

```text
MH_forum_388_sentences
SuicideAndDepressionDetectionKaggleDataset_classified_sentences
```

It includes only these labels:

```text
Not Suicide post
Suicide or Self Harm Ideation
Method or action of Suicide, Self-Harm or Harming others
```

It excludes:

```text
generated_sentences
paraphrases4
```

The wrapper also removes duplicate `(Tweet, Suicide)` rows from `df_actual`.

This is better than testing on the full augmented dataset, but it is not yet the strongest final-paper design because the original `df_full` training pool still contains real MHF and Kaggle rows. Therefore, there may still be overlap between the training pool and actual-only test pool.

## Why `df_actual = df_full` was problematic

In the original script, the intended design appeared to be:

```text
df_full   = full training pool containing actual + generated + paraphrased + Kaggle-labelled data
df_actual = actual-only testing pool
```

This intention is visible from the comment in the original script:

```python
# Load only actual sentences for testing
# we cant consider the paraphrased and generated ones here
```

However, the script then used:

```python
df_actual = df_full
```

This made the final test set equal to the full dataset.

The consequence was:

```text
Training pool = part of df_full
Validation pool = part of df_full
Testing pool = all of df_full
```

This can inflate evaluation results because many test rows may overlap with training or validation rows. It also means generated and paraphrased rows may be included in testing, which is not ideal for final reporting.

## Why augmentation was used

The reason for using generated and paraphrased data is still valid. The original labelled forum data likely had too few examples in some categories, especially the method/action class. Therefore, `generated_sentences` and `paraphrases4` were created to increase the training signal for underrepresented categories.

The correct research design is:

```text
Generated/paraphrased data = training support
Actual-only data = final testing evidence
```

In other words, synthetic and paraphrased data should help the model learn, but final evaluation should be based on actual unseen sentences.

## Best final-paper dataset design

The strongest design for final paper results would be:

```text
1. Load actual-only labelled data.
2. Split actual-only data into actual_train and actual_test.
3. Load generated and paraphrased data separately.
4. Train on actual_train + generated + paraphrased data.
5. Test only on actual_test.
```

This prevents leakage and gives a cleaner estimate of real-world performance.

The current FULL_SAFE wrapper improves the original behaviour by excluding generated/paraphrased rows from `df_actual`, but it does not yet fully remove all possible actual-row overlap between `df_full` and `df_actual`. This should be the next improvement before final paper results are reported.

## Suggested SQL to inspect source-wise label counts

Use this query to understand how many examples come from each source and label:

```sql
SELECT 'MH_forum_388_sentences' AS source, label, COUNT(*) AS count
FROM MH_forum_388_sentences
GROUP BY label

UNION ALL

SELECT 'generated_sentences' AS source, label, COUNT(*) AS count
FROM generated_sentences
WHERE ID < 32000
GROUP BY label

UNION ALL

SELECT 'paraphrases4' AS source, original_label AS label, COUNT(*) AS count
FROM paraphrases4
WHERE to_consider = 1
GROUP BY original_label

UNION ALL

SELECT 'Kaggle_classified' AS source, first_label AS label, COUNT(*) AS count
FROM SuicideAndDepressionDetectionKaggleDataset_classified_sentences
GROUP BY first_label;
```

This will help confirm whether the original real labelled data was too small or imbalanced and how much augmentation contributed to each class.

## Output previously observed

A previous run showed the combined label counts across all tables as:

```text
Suicide or Self Harm Ideation                                  2,715,151
Not Suicide post                                               1,506,043
Method or action of Suicide, Self-Harm or Harming others         839,939
```

The same run showed:

```text
x_train_tfidf shape: (3,795,849, 3,289,623)
y_train_full shape: (3,795,849,)
x_test_tfidf shape: (5,061,133, 3,289,623)
y_test shape: (5,061,133,)
```

This indicated that the test set was larger than the training set because `df_actual = df_full` was being used.

The deep-learning section showed:

```text
Training Data Shape: (5,061,133, 60)
Testing Data Shape: (5,061,133, 60)
Training Labels Shape: (5,061,133, 3)
Testing Labels Shape: (5,061,133, 3)
Train, Validation Shapes: (3,795,849, 60), (3,795,849, 3), (1,265,284, 60), (1,265,284, 3)
```

Again, this confirmed that testing was being performed on the full dataset.

## Previously observed ML results

A previous run produced the following approximate results:

| Model | Accuracy | Method/action F1 | Comment |
|---|---:|---:|---|
| Logistic Regression | 93.34% | 0.903 | Strong baseline |
| Linear SVM | 95.50% | 0.940 | Best observed ML model |
| Naive Bayes | 87.15% | 0.820 | Weaker recall for method/action |
| Random Forest | 76.84% | 0.733 | Weaker than LR/SVM |
| Gradient Boosting | 83.16% | 0.720 | Too slow for the gain |

These results should be treated as preliminary because the previous test design used `df_actual = df_full`.

## Previously observed LSTM result

The LSTM-1 training completed, but the script failed afterwards when loading the checkpoint from the wrong path.

The run showed:

```text
Epoch 10 completed
Test results - Loss: 0.34394702315330505 - Accuracy: 87.29081153869629%
```

Then it failed with:

```text
OSError: No file or directory found at lstm-1-layer-best_model.h5
```

This was a path error, not a training failure. The FULL_SAFE wrapper fixes this by redirecting model paths to `ML_DLearning/models/`.

## Using actual-only test mode

The default is:

```text
MHFS_USE_ACTUAL_ONLY_TEST=1
```

This means generated and paraphrased rows are excluded from `df_actual`.

To explicitly set it:

### Windows CMD

```cmd
set MHFS_USE_ACTUAL_ONLY_TEST=1
python training5_additional_datasets_improved_code_FULL_SAFE.py
```

### Windows PowerShell

```powershell
$env:MHFS_USE_ACTUAL_ONLY_TEST="1"
python training5_additional_datasets_improved_code_FULL_SAFE.py
```

### Linux/macOS

```bash
export MHFS_USE_ACTUAL_ONLY_TEST=1
python training5_additional_datasets_improved_code_FULL_SAFE.py
```

To reproduce the old behaviour, use:

```text
MHFS_USE_ACTUAL_ONLY_TEST=0
```

This will make `df_actual = df_full`, but this is not recommended for final paper results.

## Skipping expensive cross-validation

The original script performs expensive cross-validation, including for slow models such as Gradient Boosting.

To skip expensive cross-validation blocks:

### Windows CMD

```cmd
set MHFS_SKIP_EXPENSIVE_CV=1
python training5_additional_datasets_improved_code_FULL_SAFE.py
```

### Windows PowerShell

```powershell
$env:MHFS_SKIP_EXPENSIVE_CV="1"
python training5_additional_datasets_improved_code_FULL_SAFE.py
```

### Linux/macOS

```bash
export MHFS_SKIP_EXPENSIVE_CV=1
python training5_additional_datasets_improved_code_FULL_SAFE.py
```

This is useful because previous output showed Gradient Boosting cross-validation could take many hours.

## What the FULL_SAFE wrapper patches

The wrapper applies the following patches at runtime:

1. Redirects `../models/...`, `models/...`, and bare checkpoint paths to `ML_DLearning/models/`.
2. Redirects output paths to `ML_DLearning/outputs/`.
3. Replaces `output.txt` with a timestamped log in `ML_DLearning/outputs/logs/`.
4. Opens logs in write mode instead of append mode.
5. Replaces the database path with `ML_DLearning/data/all_datasets_labelled.db` or `MHFS_DB_PATH`.
6. Replaces `df_actual = df_full` with an actual-only test query when `MHFS_USE_ACTUAL_ONLY_TEST=1`.
7. Changes `binary_crossentropy` to `categorical_crossentropy` for the three-class softmax task.
8. Replaces hard-coded `Dense(8)` and `num_labels=8` with `number_of_classes`.
9. Fixes known misclassification-analysis mistakes for SVM and Naive Bayes.
10. Allows expensive cross-validation to be skipped with `MHFS_SKIP_EXPENSIVE_CV=1`.

## What the FULL_SAFE wrapper does not yet fully fix

It does not fully redesign the dataset split.

The best final design should explicitly split actual data first, then remove actual-test rows from the training pool. The current wrapper improves the original script by making `df_actual` actual-only, but the training pool may still contain some actual rows that also appear in the test pool.

It also does not fully rewrite all later experimental sections such as BERT, RoBERTa, DistilBERT, stacking, and anomaly detection. It only patches obvious 3-class and path issues while preserving the original workflow.

## Suggested next improvement

Create a new explicit dataset preparation block:

```python
# 1. Load actual-only rows
actual_df = load_actual_only_rows()

# 2. Split actual-only rows
actual_train, actual_test = train_test_split(
    actual_df,
    test_size=0.25,
    stratify=actual_df['Suicide'],
    random_state=42
)

# 3. Load augmentation rows
aug_df = load_generated_and_paraphrased_rows()

# 4. Train on actual_train + augmentation
train_df = pd.concat([actual_train, aug_df], ignore_index=True)

# 5. Test only on actual_test
test_df = actual_test
```

This should be the final version used for publication.

## Recommended workflow now

For a safe first rerun:

```cmd
cd ML_DLearning
set MHFS_USE_ACTUAL_ONLY_TEST=1
set MHFS_SKIP_EXPENSIVE_CV=1
python training5_additional_datasets_improved_code_FULL_SAFE.py
```

For PowerShell:

```powershell
cd ML_DLearning
$env:MHFS_USE_ACTUAL_ONLY_TEST="1"
$env:MHFS_SKIP_EXPENSIVE_CV="1"
python training5_additional_datasets_improved_code_FULL_SAFE.py
```

After it completes, inspect:

```text
ML_DLearning/outputs/logs/
ML_DLearning/outputs/reports/
ML_DLearning/outputs/misclassification/
ML_DLearning/models/
```

The wrapper also saves the patched script it actually ran into:

```text
ML_DLearning/outputs/patched_training_script_<timestamp>.py
```

Open this file before a long run if you want to verify exactly what will execute.

## Recommendation for deployment

Based on previous preliminary results, Linear SVM was the strongest and lightest traditional ML model. It may be a good first backend for the MHF Safeguard server while deep-learning models are refined.

However, final deployment should use results from the corrected actual-only or actual-held-out test design, not the older `df_actual = df_full` evaluation.
