# ML_DLearning

This folder contains the safer, path-consistent training code for the MHF Safeguard machine-learning and deep-learning classifiers.

The earlier long script had a few risks:

- paths were mixed, such as `models/...`, `../models/...`, and bare filenames;
- training could finish successfully but fail later when loading a model from the wrong path;
- output was appended to the same `output.txt`, making multiple runs hard to separate;
- the test set was effectively the full dataset in one place (`df_actual = df_full`), which can inflate results;
- some misclassification calls used the wrong prediction variable;
- some advanced metrics used Logistic Regression probabilities even when evaluating other models;
- some deep-learning models used `binary_crossentropy` even though the task is three-class classification.

The new files are:

```text
ML_DLearning/
├── training_pipeline_fixed.py
├── misclassification_analysis.py
├── data/
├── models/
└── outputs/
```

## Path design

All paths are relative to the folder where `training_pipeline_fixed.py` is located.

So if the script is here:

```text
ML_DLearning/training_pipeline_fixed.py
```

then it automatically uses:

```text
ML_DLearning/data/
ML_DLearning/models/
ML_DLearning/outputs/
ML_DLearning/outputs/reports/
ML_DLearning/outputs/misclassification/
ML_DLearning/outputs/logs/
```

This avoids the previous problem where training completed but the script failed at the final model-loading step because it looked in the wrong folder.

## Database location

Put the SQLite database here:

```text
ML_DLearning/data/all_datasets_labelled.db
```

or pass another path:

```bash
python training_pipeline_fixed.py --db "C:\path\to\all_datasets_labelled.db" --run ml
```

## Run machine-learning models

```bash
cd ML_DLearning
python training_pipeline_fixed.py --run ml
```

This trains:

- Logistic Regression
- Linear SVM
- Naive Bayes

Each model is saved as a complete scikit-learn pipeline, including TF-IDF vectorisation:

```text
ML_DLearning/models/Logistic_Regression_pipeline.joblib
ML_DLearning/models/Linear_SVM_pipeline.joblib
ML_DLearning/models/Naive_Bayes_pipeline.joblib
```

This is important because deployment needs both the vectoriser and the classifier.

## Run deep-learning models

For LSTM-1 only:

```bash
python training_pipeline_fixed.py --run dl --dl-models lstm1
```

For multiple DL models:

```bash
python training_pipeline_fixed.py --run dl --dl-models lstm1,gru,cnn_lstm
```

Saved files:

```text
ML_DLearning/models/tokenizer.pickle
ML_DLearning/models/lstm1_best.keras
ML_DLearning/models/lstm1_final.keras
ML_DLearning/models/gru_best.keras
ML_DLearning/models/gru_final.keras
ML_DLearning/models/cnn_lstm_best.keras
ML_DLearning/models/cnn_lstm_final.keras
```

The script always saves a final model immediately after training. If the best checkpoint is missing for any reason, it falls back to the final model instead of crashing.

## Run everything

```bash
python training_pipeline_fixed.py --run all
```

For a very large dataset, it is safer to run ML and DL separately.

## Force retraining

By default, if a model already exists, the script skips retraining and evaluates the saved model.

To retrain:

```bash
python training_pipeline_fixed.py --run ml --force
```

## Important options

```text
--max-features 500000
```

Controls TF-IDF vocabulary size. Use `--max-features 0` for no limit, but this can create millions of features.

```text
--batch-size 256
```

Batch size for DL models.

```text
--epochs 10
```

Number of DL epochs.

```text
--dl-models lstm1,gru,cnn_lstm
```

Which DL models to train.

## Recommended first run

```bash
cd ML_DLearning
python training_pipeline_fixed.py --db data/all_datasets_labelled.db --run ml
```

Then run:

```bash
python training_pipeline_fixed.py --db data/all_datasets_labelled.db --run dl --dl-models lstm1 --batch-size 256 --epochs 10
```

## Outputs

Reports are saved to:

```text
ML_DLearning/outputs/reports/
```

Misclassification analysis is saved to:

```text
ML_DLearning/outputs/misclassification/
```

Logs are saved to:

```text
ML_DLearning/outputs/logs/
```

Train/test splits are saved to:

```text
ML_DLearning/outputs/splits/
```

## Main fixes made

1. All output paths are now based on `Path(__file__).resolve().parent`.
2. The script creates all folders automatically.
3. Logs are timestamped and written in `write` mode, not append mode.
4. Saved model paths are consistent.
5. The best checkpoint and final model paths are clearly separated.
6. Deep-learning models use `categorical_crossentropy` for the three-class softmax task.
7. DL evaluation uses one-hot test labels, not string labels.
8. Predictions are converted back to the original class names before reports are generated.
9. Misclassification analysis now receives the correct predictions for each model.
10. Probability-based confidence analysis no longer assumes the original DataFrame index aligns with the probability matrix.
11. The test split is no longer simply the full dataset.
12. Complete scikit-learn pipelines are saved for easier deployment.

## Notes for paper reporting

Your previous output showed strong ML results, especially Linear SVM, but those results should be rerun with this corrected script because the previous evaluation used `df_actual = df_full`, which can inflate test performance.

The earlier LSTM training completed, but failed after training because the script tried to load `lstm-1-layer-best_model.h5` from a path where it did not exist. This fixed script prevents that issue by consistently saving and loading from the local `models/` folder.
