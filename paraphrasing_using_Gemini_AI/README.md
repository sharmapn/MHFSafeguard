# Paraphrasing and Distraction Sentence Generation Using Gemini AI

This folder contains the Python script used to generate paraphrased and synthetic training sentences using the Google Gemini API. The generated sentences are used for data augmentation in the mental-health forum moderation study.

## Purpose

The purpose of the script is to expand the labelled training dataset by generating additional sentence variants for underrepresented classes, especially:

1. **Suicide or self-harm ideation**
2. **Methods or actions of suicide, self-harm, or harming others**
3. **Non-harmful distraction sentences**, where safe keywords are used in neutral everyday contexts

This helps improve class balance and exposes the ML/DL models to more varied linguistic expressions.

## What the script does

For each labelled sentence, the script sends the sentence, label, and associated keyword(s) to the Gemini API and asks it to generate three kinds of output:

| Output type | Description | Intended label/use |
|---|---|---|
| `similar_context` | Paraphrases that preserve the original meaning, keyword use, and context. | Same label as the original sentence. |
| `diverse_situations_same_context` | New sentences where the same keyword(s) remain associated with suicide, self-harm, or harmful action, but appear in different scenarios. | Same label as the original sentence. |
| `distraction_sentences` | Neutral, non-harmful sentences using safe keywords in everyday contexts. | Non-harmful / not-suicide examples. |

The generated outputs are stored in the `paraphrases4` table in the local SQLite database.

## Important privacy and ethics note

Mental-health forum data can contain highly sensitive personal disclosures. This script should only be used with data for which appropriate ethics, consent, and data-governance safeguards are in place.

Do **not** upload private forum posts, identifiable user content, or moderator-protected data to an external AI API unless this is explicitly permitted by the relevant ethics approval and data-use agreements.

Only scripts and aggregated or derived outputs should be shared publicly. Raw private MHF forum posts should not be released in this repository.

## Security note: API keys

Do **not** hard-code Gemini API keys in the script.

Use a local `.env` file or an environment variable instead.

Example `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

Recommended `.gitignore` entries:

```text
.env
*.db
*.sqlite
*.sqlite3
__pycache__/
```

If an API key is accidentally committed, revoke/rotate it immediately and remove it from Git history.

## Input database

The script expects a local SQLite database. The default path may need to be edited in the script before running.

Typical database path:

```python
db_path = "../databases/dec-24/all_datasets_labelled.db"
```

The script reads labelled sentences from tables such as:

```text
MH_forum_388_sentences
SuicideAndDepressionDetectionKaggleDataset_classified_sentences
```

Typical input fields include:

| Field | Description |
|---|---|
| `id` | Sentence or row identifier. |
| `sentence` | Original labelled sentence. |
| `label` or `first_label` | Assigned class label. |
| `keyword` or `keywords_phrases_method_action` | Keyword(s) used to guide generation. |

## Output table

The script writes generated sentences into a local table, typically:

```text
paraphrases4
```

Typical output fields include:

| Field | Description |
|---|---|
| `id` | ID of the source sentence. |
| `sentence` | Original source sentence. |
| `paraphrases` | Generated sentence. |
| `label` | Label assigned to the generated sentence. |
| `keywords` | Keyword(s) used in the prompt. |
| `context_type` | Generation type, such as `similar_context`, `diverse_situations_same_context`, or `distraction_sentences`. |
| `gemini_raw_response` | Optional raw Gemini response retained for auditability. |

## Workflow relationship

This script belongs to the data augmentation stage of the paper workflow:

```text
Labelled sentence dataset
        ↓
Keyword-guided Gemini paraphrasing
        ↓
Similar-context paraphrases
        ↓
Diverse same-context synthetic sentences
        ↓
Neutral distraction sentences where safe
        ↓
Augmented training dataset
        ↓
ML/DL model training and evaluation
```

## How to run

1. Install dependencies:

```bash
pip install google-generativeai python-dotenv nltk
```

2. Create a local `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

3. Confirm that the SQLite database path and table names in the script are correct.

4. Run the script:

```bash
python <script_name>.py
```

Replace `<script_name>.py` with the actual filename in this folder.

## Important implementation note

When inserting paraphrases, the generated label should normally preserve the original sentence label for `similar_context` and `diverse_situations_same_context`. For example, if the source sentence is labelled as `Suicide or Self Harm Ideation`, the generated paraphrases should keep that label rather than being automatically changed to a method/action label.

Neutral `distraction_sentences`, when generated, should be inserted as non-harmful examples.

## Reproducibility note

The original local SQLite database is not included in this repository because it may contain sensitive or restricted text. Therefore, readers can inspect the augmentation logic and prompt design, but cannot fully reproduce the original generation run without access to the same local dataset.

## Limitations

Generated paraphrases are not automatically guaranteed to be clinically or ethically correct. They should be reviewed, filtered, and evaluated before being used for final model training. Particular care is needed to avoid generating text that unintentionally intensifies harmful content or introduces misleading labels.

## Suggested citation/footnote in the paper

```latex
\footnote{The AI-assisted labelling, paraphrasing, and category-term analysis scripts are available in the project repository: \url{https://github.com/sharmapn/MHFSafeguard}.}
```
