# Aggregate Evaluation Results

This directory contains **publication-safe aggregate outputs** from the final MHFSafeguard model-evaluation workflow.

## Current files

```text
FULL_TEST_RESULTS.md
all_models_full_test_957154.csv
transformer_full_eval_957154.txt
transformer_model_comparison_full_957154.csv
```

### `FULL_TEST_RESULTS.md`

Human-readable summary of the final traditional ML, neural DL and transformer results on the complete held-out test set.

### `all_models_full_test_957154.csv`

Machine-readable comparison of **all 15 final models** on the same 957,154-sentence test set, including accuracy, macro F1 and class-wise precision/recall/F1.

### `transformer_full_eval_957154.txt`

Completed evaluation-only console run of BERT, RoBERTa, MentalRoBERTa and ModernBERT on the complete held-out test set.

### `transformer_model_comparison_full_957154.csv`

Transformer-only machine-readable comparison generated from the full held-out evaluation.

## Final held-out test set

All final machine-learning, neural deep-learning and transformer model comparisons use the same **957,154-sentence actual-only held-out test set**:

| Reference class | Support |
|---|---:|
| Not Suicide post | 755,408 |
| Ideation of Suicide, Self-Harm or Harming Others | 103,510 |
| Method or action of Suicide, Self-Harm or Harming others | 98,236 |
| **Total** | **957,154** |

Generated and paraphrased examples are used only for training support and are excluded from this held-out evaluation.

## Important distinction between final and intermediate runs

Several development logs were produced while configuring the transformer pipeline. These include smoke tests, DeBERTa/tokenizer setup attempts and the earlier **100,000-sentence transformer test run**. They are useful development provenance but are **not the final paper evaluation**.

The publication-facing results are the complete **957,154-sentence held-out evaluations** recorded in the files above. In particular, the transformer results in `transformer_full_eval_957154.txt` supersede the earlier 100,000-row transformer metrics.

## Overall summary

| Model | Family | Accuracy | Macro F1 | Method recall | Ideation recall |
|---|---|---:|---:|---:|---:|
| Linear SVM | Traditional ML | **0.936447** | 0.879840 | **0.937141** | 0.738344 |
| MentalRoBERTa | Transformer | 0.935248 | **0.880341** | 0.920141 | **0.792184** |
| LSTM (2 layers) | Neural DL | 0.934792 | 0.876996 | 0.917668 | 0.749174 |
| LSTM with Attention | Neural DL | 0.934367 | 0.877795 | 0.922126 | 0.762496 |

Linear SVM achieved the highest overall accuracy and strongest Method/action recall, while MentalRoBERTa achieved a marginally higher macro F1 and substantially stronger Ideation recall.

## Public-data policy

Do **not** commit raw MentalHealthForum.net text, SQLite databases, row-level misclassification exports, or other outputs that reproduce sensitive forum content. Aggregate metrics and execution logs that contain no raw user text are appropriate for this public directory.
