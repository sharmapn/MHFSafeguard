# AI-Assisted Sentence Labelling Using Gemini API

This folder contains the Python script used to perform **AI-assisted sentence-level labelling** of mental-health-related text using the Google Gemini API. The labelled output is used as an upstream input for later category-wise term-frequency analysis, lexicon construction, synthetic sentence generation, and ML/DL model training.

## Purpose

The script processes posts from the `SuicideAndDepressionDetectionKaggleDataset` table and classifies each post sentence-by-sentence into moderation-relevant labels. The aim is to support construction of a labelled research corpus for detecting:

1. suicide or self-harm ideation;
2. methods or actions of suicide, self-harm, or harming others;
3. depression-related expressions;
4. advice/supportive content; and
5. non-suicide-related text.

The script also asks Gemini to extract relevant keywords/phrases and assign mid-level method/action and ideation categories. These outputs are stored in a classified SQLite table and later used by the scripts in `categories_and_terms/`.

## Important privacy and safety note

This script should be used only with data for which appropriate ethical, privacy, and governance approvals are in place. Mental-health forum posts can contain highly sensitive personal disclosures. Do not use this script to upload private forum posts, identifiable user content, or moderator-protected data to an external AI API unless consent, ethics approval, and data-governance safeguards are in place.

For this project, raw private MHF posts should not be released in the repository. Only aggregated outputs, scripts, and non-sensitive derived files should be shared publicly.

## Security note: API keys

Do **not** hard-code Gemini API keys in the script.

Use an environment variable or a local `.env` file that is not committed to GitHub.

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

The script expects a local SQLite database. The default database path and table names may need to be edited in the script before running.

Typical input table:

```text
SuicideAndDepressionDetectionKaggleDataset
```

Expected source columns include:

| Column | Description |
|---|---|
| `id` | Unique post/message identifier. |
| `text` | Raw post/message text to be classified. |

## Output table

The script creates or updates a classified sentence table, typically named:

```text
SuicideAndDepressionDetectionKaggleDataset_classified_sentences
```

Typical output columns:

| Column | Description |
|---|---|
| `post_id` | ID of the original post/message. |
| `sentence` | Sentence extracted from the post. |
| `first_label` | Primary sentence-level label. |
| `keywords_phrases_method_action` | Extracted keywords/phrases related to methods/actions. |
| `keywords_phrases_ideation` | Extracted keywords/phrases related to ideation. |
| `mid_label` | Method/action category or subcategory where applicable. |
| `next_label` | Ideation category where applicable. |
| `second_label` | Secondary risk-style label. |
| `full_response` | Full JSON response returned by Gemini for the sentence. |

## Relationship to the paper workflow

This script is part of the data-labelling stage of the paper workflow:

```text
Raw dataset posts
        ↓
AI-assisted sentence-level classification using Gemini
        ↓
Classified sentence table in SQLite
        ↓
Category and keyword extraction scripts
        ↓
Aggregated term-frequency and distinctive-term summaries
        ↓
ML/DL model training and evaluation
```

The later category/term analysis is stored in:

```text
categories_and_terms/
```

## How to run

1. Install dependencies:

```bash
pip install google-generativeai python-dotenv beautifulsoup4 nltk
```

2. Create a local `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

3. Update the database path and table names in the script if needed:

```python
db_path = "../databases/dec-24/all_datasets_labelled.db"
source_db_table = "SuicideAndDepressionDetectionKaggleDataset"
classified_table = "SuicideAndDepressionDetectionKaggleDataset_classified_sentences"
```

4. Run the script:

```bash
python <script_name>.py
```

Replace `<script_name>.py` with the actual filename in this folder.

## Reproducibility notes

The original SQLite database is not included in this public repository because it may contain sensitive or restricted text. Therefore, readers can inspect the labelling logic and prompt design, but cannot fully reproduce the original labelling run without access to the same local dataset.

For reproducible downstream analysis, use the aggregated CSV outputs provided in:

```text
categories_and_terms/data/
categories_and_terms/outputs/
```

## Limitations

AI-assisted labels should not be treated as final clinical or moderation decisions. They are provisional annotations used to support research corpus construction. Human review, error analysis, and careful interpretation remain necessary, especially because false positives and false negatives in mental-health moderation can have serious consequences.

## Suggested citation/footnote in the paper

```latex
\footnote{The AI-assisted labelling script and aggregated category-term outputs are available in the project repository: \url{https://github.com/sharmapn/MHFSafeguard}.}
```
