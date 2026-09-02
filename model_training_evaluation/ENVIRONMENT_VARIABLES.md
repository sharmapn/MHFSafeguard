# Environment Variables and Run Profiles

`training12_py314.py` is controlled primarily through environment variables. This makes it possible to use the same script for traditional machine learning, neural deep learning, transformer fine-tuning, saved-model evaluation, and imbalance-ablation experiments without editing the Python source.

The examples below assume **Windows PowerShell**, including the integrated terminal in Visual Studio or Visual Studio Code.

> **Important:** PowerShell variables set with `$env:NAME="value"` apply to the current terminal session and any processes started from it. This is useful for experiments because settings do not permanently alter the machine. Open a new terminal, or remove/change the variables, when switching experiment profiles.

## Python 3.14 environment

From the repository root:

```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\model_training_evaluation\requirements.txt
```

When using Visual Studio Code, select `.venv314` as the Python interpreter. If the script is started from the integrated PowerShell terminal, the `$env:` variables below are inherited automatically. If it is started through an IDE debugger instead, configure the same variables in that debugger profile.

## Database paths

The safest approach is to use full local paths rather than copying research databases into the public repository.

```powershell
$env:MHFS_DB_PATH="C:\full\path\all_datasets_relabelled.db"
$env:MHFS_SUICIDEFORUM_DB_PATH="C:\full\path\suicideforum_dot_com4_labelled.db"
$env:MHFS_SSF_DB_PATH="C:\full\path\sanctioned_suicide_forum.sqlite"
```

Optional base directory override:

```powershell
$env:MHFS_BASE_DIR="C:\projectscode\mentalHealthForums\AI_Models\twitter-suicidal-ideation-detection"
```

Do not commit database paths, private datasets, Hugging Face tokens, or other credentials to this repository.

## Run profiles used in the project

### 1. Traditional machine learning only

Runs Logistic Regression, Linear SVM, Naive Bayes, Random Forest, and Gradient Boosting while leaving neural and transformer blocks off.

```powershell
$env:MHFS_RUN_MACHINE_LEARNING="1"
$env:MHFS_RUN_DEEP_LEARNING="0"
$env:MHFS_RUN_TRANSFORMERS="0"
$env:MHFS_SKIP_EXPENSIVE_CV="1"
python -u .\model_training_evaluation\training12_py314.py
```

To enable the expensive cross-validation blocks:

```powershell
$env:MHFS_SKIP_EXPENSIVE_CV="0"
$env:MHFS_CV_N_JOBS="1"
```

### 2. Neural deep-learning models only

```powershell
$env:MHFS_RUN_MACHINE_LEARNING="0"
$env:MHFS_RUN_DEEP_LEARNING="1"
$env:MHFS_RUN_TRANSFORMERS="0"
$env:MHFS_DL_MODELS="lstm_1_layer,lstm_2_layer,gru,cnn_lstm,hybrid_cnn_lstm_gru,lstm_attention"
python -u .\model_training_evaluation\training12_py314.py
```

To run only one neural model, for example the attention LSTM:

```powershell
$env:MHFS_DL_MODELS="lstm_attention"
```

### 3. Transformer training/fine-tuning only

```powershell
$env:MHFS_RUN_MACHINE_LEARNING="0"
$env:MHFS_RUN_DEEP_LEARNING="0"
$env:MHFS_RUN_TRANSFORMERS="1"
$env:MHFS_TRANSFORMER_EVAL_ONLY="0"
$env:MHFS_TRANSFORMER_MODELS="bert,roberta,mental_roberta,modernbert"
$env:MHFS_TRANSFORMER_LOSS_MODES="standard"
$env:MHFS_TRANSFORMER_TRAIN_LIMIT="200000"
$env:MHFS_TRANSFORMER_VAL_LIMIT="50000"
python -u .\model_training_evaluation\training12_py314.py
```

`deberta_v3` and `mental_bert` are also supported by the script but were not part of the final completed four-transformer comparison reported in the paper.

### 4. Train or evaluate one transformer only

For MentalRoBERTa only:

```powershell
$env:MHFS_RUN_MACHINE_LEARNING="0"
$env:MHFS_RUN_DEEP_LEARNING="0"
$env:MHFS_RUN_TRANSFORMERS="1"
$env:MHFS_TRANSFORMER_MODELS="mental_roberta"
$env:MHFS_TRANSFORMER_LOSS_MODES="standard"
$env:MHFS_TRANSFORMER_EVAL_ONLY="0"
python -u .\model_training_evaluation\training12_py314.py
```

### 5. Evaluate saved transformers on the complete held-out test set

This is the profile used for the final full-test transformer evaluation. `MHFS_TRANSFORMER_TEST_LIMIT=0` means **no test-set limit**, so the complete 957,154-sentence held-out set is used.

```powershell
$env:MHFS_RUN_MACHINE_LEARNING="0"
$env:MHFS_RUN_DEEP_LEARNING="0"
$env:MHFS_RUN_TRANSFORMERS="1"
$env:MHFS_TRANSFORMER_MODELS="bert,roberta,mental_roberta,modernbert"
$env:MHFS_TRANSFORMER_LOSS_MODES="standard"
$env:MHFS_TRANSFORMER_EVAL_ONLY="1"
$env:MHFS_TRANSFORMER_TEST_LIMIT="0"
$env:MHFS_RUN_IMBALANCE_ABLATION="0"
python -u .\model_training_evaluation\training12_py314.py 2>&1 | Tee-Object -FilePath transformer_full_eval.txt
```

### 6. Imbalance-aware transformer ablation

The script can compare weighted cross-entropy and focal loss after standard transformer training.

Automatic selection of the best standard-loss transformer by validation macro F1:

```powershell
$env:MHFS_RUN_MACHINE_LEARNING="0"
$env:MHFS_RUN_DEEP_LEARNING="0"
$env:MHFS_RUN_TRANSFORMERS="1"
$env:MHFS_TRANSFORMER_EVAL_ONLY="0"
$env:MHFS_TRANSFORMER_LOSS_MODES="standard"
$env:MHFS_RUN_IMBALANCE_ABLATION="1"
$env:MHFS_ABLATION_MODEL="auto"
python -u .\model_training_evaluation\training12_py314.py
```

To force the ablation to MentalRoBERTa:

```powershell
$env:MHFS_ABLATION_MODEL="mental_roberta"
```

Alternatively, weighted/focal loss can be requested directly:

```powershell
$env:MHFS_TRANSFORMER_MODELS="mental_roberta"
$env:MHFS_TRANSFORMER_LOSS_MODES="weighted_ce,focal"
```

## Core switches

| Variable | Default | Purpose |
|---|---:|---|
| `MHFS_RUN_MACHINE_LEARNING` | `0` | Enable traditional ML block |
| `MHFS_RUN_DEEP_LEARNING` | `0` | Enable PyTorch neural DL block |
| `MHFS_RUN_TRANSFORMERS` | `0` | Enable transformer block |
| `MHFS_USE_ACTUAL_ONLY_TEST` | `1` | Keep final testing actual-only |
| `MHFS_SKIP_EXPENSIVE_CV` | `1` | Skip expensive cross-validation |
| `MHFS_CV_N_JOBS` | `1` | Number of parallel CV jobs |
| `MHFS_RUN_ENSEMBLE` | `0` | Enable ensemble classifier experiment |
| `MHFS_RUN_ADABOOST` | `0` | Enable AdaBoost experiment |
| `MHFS_RUN_STACKING` | `0` | Enable stacking experiment |
| `MHFS_RANDOM_SEED` | `42` | Random seed |
| `MHFS_USE_PROJECT_MISCLASSIFICATION_HELPER` | `0` | Use project-specific misclassification helper |

## Neural deep-learning controls

| Variable | Default | Purpose |
|---|---:|---|
| `MHFS_DL_MODELS` | all six | Comma-separated neural models to run |
| `MHFS_DL_VOCAB_SIZE` | `12000` | Vocabulary size |
| `MHFS_DL_MAX_LEN` | `60` | Maximum sequence length |
| `MHFS_DL_EMBEDDING_DIM` | `128` | Embedding dimension |
| `MHFS_DL_BATCH_SIZE` | `64` | Batch size |
| `MHFS_DL_EPOCHS` | `10` | Maximum epochs |
| `MHFS_DL_PATIENCE` | `3` | Early-stopping patience |
| `MHFS_DL_LEARNING_RATE` | `0.001` | Learning rate |
| `MHFS_DL_NUM_WORKERS` | `0` | PyTorch DataLoader workers |

Valid `MHFS_DL_MODELS` names are:

```text
lstm_1_layer
lstm_2_layer
gru
cnn_lstm
hybrid_cnn_lstm_gru
lstm_attention
```

## Transformer controls

| Variable | Default | Purpose |
|---|---:|---|
| `MHFS_TRANSFORMER_MODELS` | `bert,roberta,mental_roberta,deberta_v3,modernbert` | Models to run |
| `MHFS_TRANSFORMER_LOSS_MODES` | `standard` | `standard`, `weighted_ce`, and/or `focal` |
| `MHFS_TRANSFORMER_EVAL_ONLY` | `0` | Load saved checkpoint and evaluate instead of fine-tuning |
| `MHFS_TRANSFORMER_TRAIN_LIMIT` | `200000` | Stratified transformer training limit |
| `MHFS_TRANSFORMER_VAL_LIMIT` | `50000` | Stratified validation limit |
| `MHFS_TRANSFORMER_TEST_LIMIT` | `0` | Test limit; `0` means full test set |
| `MHFS_TRANSFORMER_VAL_SIZE` | `0.10` | Validation fraction before limiting |
| `MHFS_TRANSFORMER_MAX_LEN` | `128` | Maximum token length |
| `MHFS_TRANSFORMER_BATCH_SIZE` | `16` | Batch size |
| `MHFS_TRANSFORMER_EPOCHS` | `3` | Maximum epochs |
| `MHFS_TRANSFORMER_PATIENCE` | `2` | Early-stopping patience |
| `MHFS_TRANSFORMER_LR` | `2e-5` | Learning rate |
| `MHFS_TRANSFORMER_NUM_WORKERS` | `0` | DataLoader workers |
| `MHFS_TRANSFORMER_FOCAL_GAMMA` | `2.0` | Focal-loss gamma |
| `MHFS_FOCAL_USE_CLASS_WEIGHTS` | `1` | Apply class weights in focal loss |
| `MHFS_RUN_IMBALANCE_ABLATION` | `0` | Run weighted-CE/focal ablation after standard models |
| `MHFS_ABLATION_MODEL` | `auto` | Model used for the imbalance ablation |

Supported transformer identifiers are:

```text
bert
roberta
mental_roberta
deberta_v3
modernbert
mental_bert
```

## Transformer checkpoint overrides

These are optional. They allow a different Hugging Face checkpoint to be substituted without changing the script.

| Variable | Default checkpoint |
|---|---|
| `MHFS_BERT_CHECKPOINT` | `bert-base-uncased` |
| `MHFS_ROBERTA_CHECKPOINT` | `roberta-base` |
| `MHFS_MENTAL_ROBERTA_CHECKPOINT` | `mental/mental-roberta-base` |
| `MHFS_DEBERTA_V3_CHECKPOINT` | `microsoft/deberta-v3-base` |
| `MHFS_MODERNBERT_CHECKPOINT` | `answerdotai/ModernBERT-base` |
| `MHFS_MENTAL_BERT_CHECKPOINT` | `mental/mental-bert-base-uncased` |
| `MHFS_MODERNBERT_ATTN_IMPLEMENTATION` | `eager` |
| `MHFS_DEBERTA_USE_FAST_TOKENIZER` | `0` |

## Hugging Face authentication

Some checkpoints may require Hugging Face access. Authenticate the workstation before the run:

```powershell
hf auth login
```

or set a token in the current terminal session:

```powershell
$env:HF_TOKEN="your-token"
```

Never place the token in the Python script, README, committed `.env` file, or console log uploaded to GitHub.

## Clearing settings before another experiment

Because old terminal variables can accidentally affect a later run, either open a fresh PowerShell terminal or clear the variables explicitly. For example:

```powershell
Remove-Item Env:MHFS_RUN_MACHINE_LEARNING -ErrorAction SilentlyContinue
Remove-Item Env:MHFS_RUN_DEEP_LEARNING -ErrorAction SilentlyContinue
Remove-Item Env:MHFS_RUN_TRANSFORMERS -ErrorAction SilentlyContinue
Remove-Item Env:MHFS_TRANSFORMER_MODELS -ErrorAction SilentlyContinue
Remove-Item Env:MHFS_TRANSFORMER_EVAL_ONLY -ErrorAction SilentlyContinue
Remove-Item Env:MHFS_TRANSFORMER_TEST_LIMIT -ErrorAction SilentlyContinue
```

For reproducibility, record the environment-variable profile used for each reported experiment together with the corresponding output log.