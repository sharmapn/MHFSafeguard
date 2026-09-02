# Model Training and Evaluation Research Companion

This folder contains the research code and aggregate evaluation outputs used for the **traditional machine-learning, neural deep-learning, and transformer experiments** associated with MHFSafeguard and the accompanying paper on detecting suicide/self-harm ideation and method/action content in online mental-health communities.

The XenForo implementation is maintained separately in **[SuicideSelfHarmDetector-XenForoPlugin](https://github.com/sharmapn/SuicideSelfHarmDetector-XenForoPlugin)**. This folder is the main research/reproducibility area for classifier development and evaluation.

## Final classification task

The models classify sentences into three categories:

1. `Not Suicide post`
2. `Ideation of Suicide, Self-Harm or Harming Others`
3. `Method or action of Suicide, Self-Harm or Harming others`

In the paper these are described more readably as:

- **Not Ideation or method or action**
- **Suicide or Self Harm Ideation**
- **Method or action of Suicide, Self-Harm or Harming others**

## Publication pipeline

The current publication script is:

```text
training12_py314.py
```

This supersedes the older experimental `training.py` workflow in this folder. The older file and supporting utilities are retained for provenance but should not be treated as the final publication pipeline.

The final design separates actual forum/public-source data **before augmentation**:

```text
Actual-only labelled data
        |
        +--> 75% actual training split
        |         + generated/paraphrased/curated augmentation
        |         = final training pool
        |
        +--> 25% actual-only held-out test set
                  = 957,154 unseen sentences
```

Generated, paraphrased, and curated augmentation data are used only for training and are not included in the final held-out test set.

## Final dataset sizes

The publication run produced:

| Dataset component | Rows |
|---|---:|
| Actual-only rows before split | 3,828,613 |
| Actual training rows | 2,871,459 |
| Augmented training-only rows | 1,069,489 |
| Final training rows | 3,940,894 |
| Final actual-only held-out test set | **957,154** |

Held-out test distribution:

| Class | Rows |
|---|---:|
| Not Suicide post | 755,408 |
| Ideation | 103,510 |
| Method/action | 98,236 |

## Evaluated model families

### Traditional machine learning

- Logistic Regression
- Linear SVM
- Naive Bayes
- Random Forest
- Gradient Boosting

### Neural deep learning

- LSTM (1 layer)
- LSTM (2 layers)
- GRU
- CNN-LSTM
- Hybrid CNN-LSTM-GRU
- LSTM with Attention

### Transformer models

- BERT
- RoBERTa
- MentalRoBERTa
- ModernBERT

All final reported models were evaluated on the **same 957,154-sentence actual-only held-out test set**.

## Selected final results

| Model | Accuracy | Macro F1 | Method recall | Method F1 | Ideation recall | Ideation F1 |
|---|---:|---:|---:|---:|---:|---:|
| Linear SVM | **93.64%** | 0.880 | **0.937** | **0.897** | 0.738 | 0.781 |
| MentalRoBERTa | 93.52% | **0.880** | 0.920 | 0.887 | **0.792** | **0.793** |
| RoBERTa | 93.42% | 0.878 | 0.917 | 0.884 | 0.782 | 0.789 |
| LSTM with Attention | 93.44% | 0.878 | 0.922 | 0.889 | 0.762 | 0.783 |
| ModernBERT | 93.44% | 0.874 | 0.918 | 0.883 | 0.725 | 0.779 |
| BERT | 93.15% | 0.873 | 0.921 | 0.880 | 0.773 | 0.781 |

The main finding is not a large overall separation between SVM and MentalRoBERTa. Instead, they show complementary strengths: **Linear SVM is particularly strong for explicit method/action content, whereas MentalRoBERTa provides substantially stronger ideation recall.**

## Folder structure

```text
model_training_evaluation/
├── README.md
├── ENVIRONMENT_VARIABLES.md
├── training12_py314.py
├── results/
│   ├── README.md
│   ├── transformer_full_eval_957154.txt
│   └── transformer_model_comparison_full_957154.csv
├── metrics_calculator.py
├── misclassification_analysis.py
├── path_setup.py
├── requirements.txt
└── training.py                      # legacy experimental script
```

Raw datasets, private forum content, and row-level misclassification exports should not be committed to this public repository.

## Python environment and run configuration

The publication pipeline uses Python 3.14 and PyTorch. `training12_py314.py` is deliberately controlled through environment variables so that traditional ML, neural DL, transformer training, saved-model evaluation, and ablation experiments can be run from the same script without editing its source.

See **[ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md)** for:

- Python 3.14 virtual-environment setup;
- Visual Studio / VS Code PowerShell usage;
- database path variables;
- ML-only, DL-only and transformer-only run profiles;
- running an individual model;
- full 957,154-row transformer evaluation;
- weighted cross-entropy and focal-loss ablation;
- all major `MHFS_*` variables and defaults;
- Hugging Face authentication; and
- clearing old environment settings before a different experiment.

For example, the completed full transformer evaluation used saved fine-tuned checkpoints and evaluated the entire held-out test set:

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

`MHFS_TRANSFORMER_TEST_LIMIT=0` means that the complete **957,154-sentence** held-out test set is evaluated.

## Reproducibility and data availability

The repository provides code and aggregate evaluation outputs. It intentionally does **not** publish raw MentalHealthForum.net content, private forum data, local SQLite databases, model checkpoints containing project-specific artefacts, or row-level outputs that reproduce sensitive user-generated text.

Researchers wishing to reproduce the workflow with their own appropriately authorised data can use `training12_py314.py` as the reference implementation.

## Relationship to the XenForo plugin

The models developed here provide the classifier research underlying the moderation workflow. The separate XenForo add-on sends content to a self-managed classifier API and can use the returned label/risk result to support logging, moderation, or secondary human review.

Plugin repository: **https://github.com/sharmapn/SuicideSelfHarmDetector-XenForoPlugin**

The research findings support **human-supervised moderation assistance**, not autonomous replacement of moderators.
