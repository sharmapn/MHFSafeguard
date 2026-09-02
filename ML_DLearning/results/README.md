# Aggregate Evaluation Results

This directory contains **publication-safe aggregate outputs** from the final MHFSafeguard model-evaluation workflow.

## Current files

```text
transformer_full_eval_957154.txt
transformer_model_comparison_full_957154.csv
```

`transformer_full_eval_957154.txt` records the completed evaluation-only run of BERT, RoBERTa, MentalRoBERTa and ModernBERT on the complete held-out test set.

`transformer_model_comparison_full_957154.csv` provides the same final transformer results in machine-readable form, including overall and class-wise metrics.

## Final held-out test set

All final machine-learning, neural deep-learning and transformer model comparisons use the same **957,154-sentence actual-only held-out test set**:

| Reference class | Support |
|---|---:|
| Not Suicide post | 755,408 |
| Ideation of Suicide, Self-Harm or Harming Others | 103,510 |
| Method or action of Suicide, Self-Harm or Harming others | 98,236 |
| **Total** | **957,154** |

Generated and paraphrased examples are used only for training support and are excluded from this held-out evaluation.

## Transformer summary

| Model | Accuracy | Macro F1 | Method recall | Method F1 | Ideation recall | Ideation F1 |
|---|---:|---:|---:|---:|---:|---:|
| BERT | 0.9315 | 0.8733 | 0.9212 | 0.8798 | 0.7734 | 0.7810 |
| RoBERTa | 0.9342 | 0.8779 | 0.9174 | 0.8840 | 0.7819 | 0.7889 |
| MentalRoBERTa | **0.9352** | **0.8803** | 0.9201 | **0.8869** | **0.7922** | **0.7927** |
| ModernBERT | 0.9344 | 0.8744 | 0.9178 | 0.8828 | 0.7246 | 0.7793 |

The full paper also compares these models with traditional machine-learning and neural deep-learning models. Linear SVM remains particularly strong for Method/action detection, while MentalRoBERTa provides stronger Ideation recall.

## Public-data policy

Do **not** commit raw MentalHealthForum.net text, SQLite databases, row-level misclassification exports, or other outputs that reproduce sensitive forum content. Aggregate metrics and execution logs that contain no raw user text are appropriate for this public directory.
