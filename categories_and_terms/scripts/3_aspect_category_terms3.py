#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aspect_category_terms_from_category_col.py

Purpose
-------
Rebuild the term frequency tables using the existing `category` column
instead of extracting categories from the JSON. We still read `full_response`
to grab the right keywords field (ideation vs. methods), and use `first_label`
to decide the aspect.

Outputs
-------
1) aspect_category_terms.csv
   columns: aspect, category, term, count
   (global sort by count DESC)

2) top10_terms_by_aspect_category.csv
   columns: aspect, category, term, count, rank
   (top 10 terms per (aspect, category))

Design
------
- Pure stdlib (sqlite3, json, csv, re, os, collections)
- Defensive parsing of JSON (list/dict) for keyword fields
- Robust parsing of the `category` TEXT (semicolon or comma separated)
- Minimal token filtering + tiny lemmatizer
"""

import sqlite3
import json
import csv
import re
import os
from collections import defaultdict

# =========================
# CONFIG (EDIT THESE)
# =========================

DB_FILE = "../databases/dec-24/all_datasets_labelled.db"  # <-- set your path
TABLE_NAME = "SuicideAndDepressionDetectionKaggleDataset_classified_sentences"

# Column names
ID_COL = "id"
FIRST_LABEL_COL = "first_label"
FULL_RESPONSE_COL = "full_response"
CATEGORY_COL = "category"   # <-- we use this instead of mid_label JSON

# Optional filter (leave empty to scan all rows)
OPTIONAL_WHERE = ""   # e.g., "WHERE id BETWEEN 1 AND 500000"

# Output files
OUT_ALL = "aspect_category_terms.csv"
OUT_TOP10 = "top10_terms_by_aspect_category.csv"

# Token rules
MIN_TOKEN_LEN = 2
DROP_NUMERIC_TOKENS = True

# Aspect labels
ASPECT_IDEATION = "Suicide or Self Harm Ideation"
ASPECT_METHODS  = "Methods or actions of Suicide, Self Harm or harming others"

# Category parsing: split on semicolon (preferred) and comma fallbacks
CATEGORY_SPLIT_PATTERN = r"\s*[;|,]\s*"


# =========================
# Tiny heuristic lemmatizer
# =========================
def simple_lemma(token: str) -> str:
    """
    Conservative normalization to reduce inflections without over-stemming.
    """
    t = token
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    if len(t) > 4 and t.endswith("ed"):
        t = t[:-2]
    if len(t) > 4 and t.endswith("ing"):
        t = t[:-3]
    return t


# =========================
# Basic tokenization helper
# =========================
def tokenize(text: str) -> list:
    """
    Lowercase, remove punctuation, collapse spaces, split, filter, lemma.
    """
    if text is None:
        return []
    s = str(text).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    out = []
    for tok in s.split():
        if DROP_NUMERIC_TOKENS and tok.isdigit():
            continue
        if len(tok) < MIN_TOKEN_LEN:
            continue
        out.append(simple_lemma(tok))
    return out


# =========================
# JSON safety helpers
# =========================
def is_empty_like(x) -> bool:
    """
    Treat None, empty string, and literal "[]"
    (after trimming) as empty values.
    """
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or s == "[]"


def ensure_list_of_objects(parsed_json):
    """
    Normalize to a list of dict sentence-objects.
    """
    if isinstance(parsed_json, list):
        return parsed_json
    if isinstance(parsed_json, dict):
        return [parsed_json]
    return []


def coerce_to_list_of_phrases(value) -> list:
    """
    The keyword fields can be:
      - string,
      - JSON-stringified list of strings,
      - list of strings,
      - or None.
    Return a list of strings (possibly empty).
    """
    if value is None:
        return []
    if isinstance(value, str):
        txt = value.strip()
        if txt.startswith("[") and txt.endswith("]"):
            try:
                arr = json.loads(txt)
                if isinstance(arr, list):
                    return [str(x) for x in arr if x is not None]
            except json.JSONDecodeError:
                pass
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    return [str(value)]


# =========================
# Aspect + keywords dispatch
# =========================
def decide_aspect(first_label: str) -> str:
    """
    If first_label is exactly the IDEATION label, choose IDEATION;
    otherwise bucket into METHODS.
    """
    if (first_label or "").strip() == ASPECT_IDEATION:
        return ASPECT_IDEATION
    return ASPECT_METHODS


def choose_keywords_for_aspect(obj: dict, aspect: str) -> list:
    """
    For Ideation: use 'keywords_phrases_ideation'
    For Methods:  use 'keywords_phrases_method_action'
    """
    if aspect == ASPECT_IDEATION:
        return coerce_to_list_of_phrases(obj.get("keywords_phrases_ideation"))
    return coerce_to_list_of_phrases(obj.get("keywords_phrases_method_action"))


# =========================
# Category parsing (from column)
# =========================
def parse_categories_from_column(cat_value: str) -> list:
    """
    Parse the TEXT in the `category` column.
    Supports multi-value cells joined by ';' (primary) or ',' / '|' as fallbacks.
    Trims whitespace and drops empties. If nothing usable, returns ["None"].
    """
    cats = []
    if cat_value is None:
        cats = []
    else:
        txt = str(cat_value).strip()
        if txt:
            # split on ; , or | with optional spaces
            parts = re.split(CATEGORY_SPLIT_PATTERN, txt)
            cats = [p.strip() for p in parts if p and p.strip()]

    if not cats:
        return ["None"]
    return cats


# =========================
# MAIN LOGIC
# =========================
def main():
    # Connect to the database
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Build SELECT
    base_sql = f"SELECT {ID_COL}, {FIRST_LABEL_COL}, {FULL_RESPONSE_COL}, {CATEGORY_COL} FROM {TABLE_NAME} WHERE first_label <> 'Not Suicide post'"
    sql = base_sql if not OPTIONAL_WHERE.strip() else f"{base_sql} {OPTIONAL_WHERE}"
    cur.execute(sql)

    # counts[(aspect, category, term)] = int
    counts = defaultdict(int)

    total_rows = 0
    sentence_objs = 0

    for db_row in cur:
        total_rows += 1
        first_label = db_row[FIRST_LABEL_COL]
        full_response_raw = db_row[FULL_RESPONSE_COL]
        category_text = db_row[CATEGORY_COL]

        # 1) Aspect from first_label
        aspect = decide_aspect(first_label)

        # 2) Categories from CATEGORY column (may be multi-value)
        categories = parse_categories_from_column(category_text)

        # 3) Parse full_response JSON to get keyword phrases for the chosen aspect
        if is_empty_like(full_response_raw):
            # no keywords to count for this row
            continue
        try:
            parsed = json.loads(full_response_raw)
        except json.JSONDecodeError:
            continue

        objects = ensure_list_of_objects(parsed)
        if not objects:
            continue

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            sentence_objs += 1

            phrases = choose_keywords_for_aspect(obj, aspect)
            if not phrases:
                continue

            # 4) Tokenize and accumulate per (aspect, category, token)
            for phrase in phrases:
                tokens = tokenize(phrase)
                if not tokens:
                    continue
                for cat in categories:
                    for tok in tokens:
                        counts[(aspect, cat, tok)] += 1

    conn.close()

    # ---- Write the full table (sorted by count DESC) ----
    rows = [(a, c, t, n) for (a, c, t), n in counts.items()]
    rows.sort(key=lambda r: (-r[3], r[0], r[1], r[2]))  # primary: count desc; then stable tie-breakers

    with open(OUT_ALL, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["aspect", "category", "term", "count"])
        for a, c, t, n in rows:
            w.writerow([a, c, t, n])

    # ---- Build Top-10 per (aspect, category) ----
    grouped = defaultdict(list)
    for a, c, t, n in rows:
        grouped[(a, c)].append((t, n))

    top10_rows = []
    for (a, c), items in grouped.items():
        items_sorted = sorted(items, key=lambda x: (-x[1], x[0]))
        top_k = items_sorted[:10]
        for rank, (term, cnt) in enumerate(top_k, start=1):
            top10_rows.append((a, c, term, cnt, rank))

    top10_rows.sort(key=lambda r: (r[0], r[1], r[4]))

    with open(OUT_TOP10, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["aspect", "category", "term", "count", "rank"])
        for a, c, t, n, rank in top10_rows:
            w.writerow([a, c, t, n, rank])

    # ---- Console summary ----
    print("SCAN SUMMARY")
    print("------------")
    print(f"DB rows scanned:          {total_rows}")
    print(f"Sentence objects parsed:  {sentence_objs}")
    print(f"Unique (aspect, category, term) triples: {len(counts)}")
    print(f"Saved: {OUT_ALL}")
    print(f"Saved: {OUT_TOP10}")

    # Preview: show a few blocks
    print("\nSAMPLE TOP-10 PER (aspect, category):")
    seen_blocks = 0
    for (a, c), items in grouped.items():
        if seen_blocks >= 5:
            break
        items_sorted = sorted(items, key=lambda x: (-x[1], x[0]))[:10]
        print(f"\n[{a}  →  {c}]")
        for rank, (term, cnt) in enumerate(items_sorted, start=1):
            print(f"  {rank:>2}. {term:20} {cnt}")
        seen_blocks += 1


# Entry point
if __name__ == "__main__":
    main()
