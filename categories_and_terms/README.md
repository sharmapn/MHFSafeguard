# Categories and Terms Analysis

This folder contains the scripts and aggregated CSV outputs used to produce the category-wise term-frequency and distinctive-term table reported in the paper.

## Purpose

The purpose of this analysis is to summarise terms associated with two moderation-relevant aspects:

1. **Suicide or self-harm ideation**
2. **Methods or actions of suicide, self-harm, or harming others**

The outputs are descriptive summaries generated after AI-assisted sentence-level classification of the *Suicide and Depression Detection* dataset. They are used to support lexicon construction, synthetic sentence generation, model training, and error analysis.

## Privacy and ethics note

This folder should not contain raw Mental Health Forum posts, private forum text, or any other sensitive user-generated content. The CSV files included here are aggregated term/category outputs only. MHF-specific frequency tables are not released because those posts come from moderator-reviewed private forum areas and contain sensitive user disclosures.

## Recommended folder structure

```text
categories_and_terms/
├── README.md
├── scripts/
│   ├── 3_aspect_category_terms3.py
│   └── make_top10_category_reports_distinctive.py
├── data/
│   ├── aspect_category_terms.csv
│   └── top10_terms_by_aspect_category.csv
└── outputs/
    ├── top10_categories_with_frequent_and_distinctive_terms.csv
    └── top10_categories_with_frequent_and_distinctive_terms.tex
```

## Files to include

### Core scripts

| File | Purpose |
|---|---|
| `scripts/3_aspect_category_terms3.py` | Reads the classified SQLite table, uses the existing `category` column, extracts keywords from `full_response`, assigns each row to an aspect using `first_label`, and produces the base term-frequency CSV files. |
| `scripts/make_top10_category_reports_distinctive.py` | Reads `aspect_category_terms.csv` and produces the final publication-ready CSV and LaTeX table containing both frequent and distinctive terms. |

### Core data/output files

| File | Purpose |
|---|---|
| `data/aspect_category_terms.csv` | Main aggregated input file. Columns: `aspect`, `category`, `term`, `count`. This is the base file from which later summaries are generated. |
| `data/top10_terms_by_aspect_category.csv` | Intermediate top-10 term list per aspect/category. Useful for audit and comparison with earlier tables. |
| `outputs/top10_categories_with_frequent_and_distinctive_terms.csv` | Final CSV used for the improved results table. It includes frequent and distinctive terms. |
| `outputs/top10_categories_with_frequent_and_distinctive_terms.tex` | LaTeX version of the final table used in the manuscript. |

## How to reproduce the outputs

### Option 1: Rebuild from the SQLite database

This requires access to the classified SQLite database, which is not included in the repository because it may contain sensitive text.

1. Edit `DB_FILE` and `TABLE_NAME` inside `scripts/3_aspect_category_terms3.py`.
2. Run:

```bash
python scripts/3_aspect_category_terms3.py
```

This produces:

```text
aspect_category_terms.csv
top10_terms_by_aspect_category.csv
```

Move these files into the `data/` folder if needed.

### Option 2: Rebuild the final paper table from the aggregated CSV

This does not require the original SQLite database.

1. Make sure `aspect_category_terms.csv` is available in the same working directory as `make_top10_category_reports_distinctive.py`, or update the `IN_CSV` path in the script.
2. Run:

```bash
python scripts/make_top10_category_reports_distinctive.py
```

This produces:

```text
top10_categories_with_frequent_and_distinctive_terms.csv
top10_categories_with_frequent_and_distinctive_terms.tex
```

## What the final table represents

The final table is not topic modelling. It is a category-wise term-frequency and distinctive-term summary.

- **Term count** means the total number of extracted keyword-token occurrences within a category.
- **Frequent terms** are the most common cleaned tokens within that category after removing common stop words.
- **Distinctive terms** are terms that are more characteristic of that category compared with other categories within the same aspect.
- Residual or catch-all categories such as `None`, `Other`, `Unspecified`, and `Unknown` are excluded from the final table to avoid confusing readers.

## Files not recommended for the main repository folder

The following files were intermediate, older, duplicate, or exploratory outputs and are not needed for reproducing the final table:

```text
aspect_category_terms.py
aspect_category_terms2.py
category_term_frequencies.py
category_term_frequency_combined.py
4_latex_summary_table3.py
make_top10_category_reports.py
top10_categories_with_top20_terms.py
categories_extracted.csv
top10_categories_with_top20_terms.csv
top10_categories_with_top20_terms_general.csv
top10_categories_with_top20_terms_Methods_or_actions_of_Suicide,_Self_Harm_or_harming_others.csv
top10_categories_with_top20_terms_Suicide_or_Self_Harm_Ideation.csv
all_categories_ranked_by_aspect.csv
top10_categories_by_aspect.csv
summary_report_from_top10.md
top10_categories_with_top20_terms.md
ideation_top_categories.png
```

These can be kept in a separate `archive/` folder if you want to preserve the development history, but they should not be placed beside the final files because they may confuse reviewers.

## Suggested Git commands

From the repository root:

```bash
mkdir -p categories_and_terms/scripts categories_and_terms/data categories_and_terms/outputs

# copy the required files into the folders above

git add categories_and_terms
git commit -m "Add category-term analysis scripts and outputs"
git push
```

## Citation in the paper

A suitable footnote in the paper is:

```latex
\footnote{The category-wise term-frequency scripts and aggregated outputs are available in the project repository under \texttt{categories\_and\_terms}: \url{https://github.com/sharmapn/MHFSafeguard/tree/main/categories_and_terms}}
```
