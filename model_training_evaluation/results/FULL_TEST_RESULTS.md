# Full held-out evaluation results

These are the final publication-facing results on the same **957,154-sentence actual-only held-out test set** for all three model families.

Test-set composition:

| Class | Support |
|---|---:|
| Not Suicide post | 755,408 |
| Ideation of Suicide, Self-Harm or Harming Others | 103,510 |
| Method or action of Suicide, Self-Harm or Harming others | 98,236 |
| **Total** | **957,154** |

Generated/paraphrased examples were used for training support only and were excluded from final testing.

## Traditional machine learning

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.930614 | 0.868702 |
| **Linear SVM** | **0.936447** | **0.879840** |
| Multinomial Naive Bayes | 0.844342 | 0.557935 |
| Random Forest | 0.819402 | 0.653586 |
| Gradient Boosting | 0.822716 | 0.485376 |

## Neural deep learning

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| LSTM (1 layer) | 0.934623 | 0.876353 |
| LSTM (2 layers) | **0.934792** | 0.876996 |
| GRU | 0.933054 | 0.873718 |
| CNN-LSTM | 0.933437 | 0.873940 |
| Hybrid CNN-LSTM-GRU | 0.932595 | 0.875923 |
| LSTM with Attention | 0.934367 | **0.877795** |

## Transformers

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| BERT | 0.931489 | 0.873296 |
| RoBERTa | 0.934192 | 0.877907 |
| **MentalRoBERTa** | **0.935248** | **0.880341** |
| ModernBERT | 0.934446 | 0.874402 |

## Overall interpretation

Linear SVM achieved the highest overall accuracy (**93.64%**) and strongest Method/action recall (**0.937**). MentalRoBERTa achieved the highest overall macro F1 (**0.880341**, marginally above SVM) and substantially higher Ideation recall (**0.792** versus **0.738** for SVM).

The final comparison therefore indicates complementary strengths rather than one model dominating every criterion.

For full class-wise precision, recall and F1 values for every model, see:

```text
all_models_full_test_957154.csv
```

For the full transformer evaluation console output, see:

```text
transformer_full_eval_957154.txt
```

The traditional ML and neural-DL values originate from the complete actual-only held-out evaluation run used for the paper. Row-level misclassification outputs are intentionally not published because they can reproduce sensitive user-generated text.
