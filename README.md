# MHFSafeguard Research Repository

This repository contains the **research, data-preparation, model-training, evaluation and reproducibility material** for the study on detecting suicide and self-harm content in online mental-health communities.

The XenForo implementation has been separated into its own repository:

**[SuicideSelfHarmDetector-XenForoPlugin](https://github.com/sharmapn/SuicideSelfHarmDetector-XenForoPlugin)**

## Research task

The study classifies sentences into three categories:

1. **Not Ideation or method or action**
2. **Suicide or Self Harm Ideation**
3. **Method or action of Suicide, Self-Harm or Harming others**

The modelling pipeline evaluates traditional machine learning, neural deep learning and transformer models. The final model comparison uses the same **957,154-sentence actual-only held-out test set** for all model families.

## Main repository areas

```text
MHFSafeguard/
├── model_training_evaluation/       model training, evaluation and aggregate results
├── categories_and_terms/            category/term development material
├── labelling_using_Gemini_API/      data-labelling scripts and supporting material
├── paraphrasing_using_Gemini_AI/    augmentation/paraphrasing material
├── docs/                             research documentation and labelling prompts
└── keywords.txt                      keyword list used in the research workflow
```

## Model training and evaluation

The `model_training_evaluation/` directory is the main reproducibility area for the classifier experiments. It contains the current training pipeline, traditional ML, neural DL and transformer evaluation utilities, requirements, and aggregate result files.

The final experimental design separates actual data before augmentation:

```text
Actual labelled data
        |
        +--> 75% actual training data
        |
        +--> 25% actual-only held-out test data (n = 957,154)

Actual training data
        + generated/paraphrased training-only augmentation
        |
        v
Model training
        |
        v
Evaluation on the untouched actual-only held-out test set
```

Generated and paraphrased examples are used for **training support only** and are excluded from final held-out testing.

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
- CNN--LSTM
- Hybrid CNN--LSTM--GRU
- LSTM with Attention

### Transformers

- BERT
- RoBERTa
- MentalRoBERTa
- ModernBERT

All final model results are evaluated on the same 957,154 held-out sentences. Linear SVM and MentalRoBERTa provide the strongest overall results, with complementary strengths across Method/action and Ideation detection.

## Reproducibility and public-data policy

This public repository is intended to provide code, configuration, prompts and aggregate experimental outputs that support reproducibility. It does **not** publish private MentalHealthForum.net message data, raw moderator-reviewed sentences, confidential databases, API credentials or other sensitive user content.

Some third-party/public datasets used in the research remain subject to their original licences and distribution conditions and therefore may need to be obtained from their original sources.

## Related XenForo plugin

The moderation integration is maintained separately at:

**https://github.com/sharmapn/SuicideSelfHarmDetector-XenForoPlugin**

That repository contains the XenForo add-on code and API integration. Keeping the implementation separate prevents the software plugin from being mixed with the research training and evaluation artefacts.

## Research status

This repository accompanies ongoing academic research into machine-learning and deep-learning support for human-supervised moderation of suicide and self-harm content in online mental-health communities.
