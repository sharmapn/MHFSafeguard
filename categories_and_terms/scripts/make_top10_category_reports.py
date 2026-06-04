#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_top10_category_reports.py

What it does
------------
1) Reads 'aspect_category_terms.csv' (columns: aspect, category, term, count).
2) Ranks categories per aspect using:
     - max single-term count (desc)
     - distinct terms (desc)
     - total term count (desc)
     - category name (asc)
3) For each aspect, emits exactly 10 categories (one line per category) with
   the top 20 terms formatted as "term (count); term (count); ...".
4) Writes multiple outputs:
   a) top10_categories_with_top20_terms.csv             (one file, both aspects)
   b) top10_categories_with_top20_terms_<aspect>.csv    (split by aspect)
   c) top10_categories_with_top20_terms.md              (Markdown report)
   d) top10_categories_with_top20_terms_<aspect>.tex    (LaTeX longtable per aspect)

Tweakables
----------
- EXCLUDE_CATEGORIES: drop 'Other/Unspecified/Unknown' (case-insensitive exact match).
- TOP_K_CATEGORIES, TOP_K_TERMS: change to 10/20 as you like.
- MAX_TERMS_PER_LINE_MD/LATEX: for visual wrapping in Markdown/LaTeX.

Author: You
"""

import csv
import os
from collections import defaultdict

# =============== CONFIG ===============

IN_CSV = "aspect_category_terms.csv"

OUT_CSV_ALL = "top10_categories_with_top20_terms.csv"
OUT_MD = "top10_categories_with_top20_terms.md"
OUT_TEX_TEMPLATE = "top10_categories_with_top20_terms_{aspect}.tex"
OUT_CSV_PER_ASPECT_TEMPLATE = "top10_categories_with_top20_terms_{aspect}.csv"

TOP_K_CATEGORIES = 10
TOP_K_TERMS = 20

# Exclude these categories entirely (case-insensitive exact match)
EXCLUDE_CATEGORIES = {"other", "unspecified", "unknown"}

# For Markdown/LaTeX wrapping:
MAX_TERMS_PER_LINE_MD = 6      # wrap after N items in Markdown "Top terms"
MAX_TERMS_PER_LINE_LATEX = 5   # wrap after N items in LaTeX "Top terms"

# =============== HELPERS ===============

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def should_exclude_category(cat: str) -> bool:
    if not cat:
        return False
    return cat.strip().lower() in EXCLUDE_CATEGORIES

def join_terms_list(pairs, max_terms=None, fmt="{t} ({c})", sep="; "):
    items = []
    for i, (t, c) in enumerate(pairs):
        if max_terms is not None and i >= max_terms:
            break
        items.append(fmt.format(t=t, c=c))
    return sep.join(items)

def wrap_list_for_md(pairs, per_line, fmt="`{t}` ({c})"):
    """Return multi-line Markdown string for top terms with soft wrapping."""
    lines, current = [], []
    for i, (t, c) in enumerate(pairs[:]):
        current.append(fmt.format(t=t, c=c))
        if (i + 1) % per_line == 0:
            lines.append("; ".join(current))
            current = []
    if current:
        lines.append("; ".join(current))
    return "<br/>".join(lines) if lines else ""

def wrap_list_for_latex(pairs, per_line, fmt=r"\texttt{{{t}}} ({c})"):
    """Return LaTeX-safe string with line breaks (\\) for long term lists."""
    # Escape LaTeX special chars in terms
    def latex_escape(s: str) -> str:
        return (s.replace("\\", r"\textbackslash{}")
                 .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
                 .replace("#", r"\#").replace("_", r"\_").replace("{", r"\{")
                 .replace("}", r"\}").replace("~", r"\textasciitilde{}")
                 .replace("^", r"\textasciicircum{}"))

    lines, current = [], []
    for i, (t, c) in enumerate(pairs[:]):
        current.append(fmt.format(t=latex_escape(t), c=c))
        if (i + 1) % per_line == 0:
            lines.append("; ".join(current))
            current = []
    if current:
        lines.append("; ".join(current))
    return r" \\ ".join(lines)

# =============== LOAD & AGGREGATE ===============

def load_stats(path: str):
    """
    Build:
      stats[(aspect, category)] = {
          "term_counts": {term: count, ...},
          "total_count": int
      }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    stats = defaultdict(lambda: {"term_counts": defaultdict(int), "total_count": 0})

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"aspect", "category", "term", "count"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        for row in reader:
            aspect = (row.get("aspect") or "").strip()
            category = (row.get("category") or "").strip()
            term = (row.get("term") or "").strip()
            count = safe_int(row.get("count"), 0)
            if not aspect or not category or not term or count <= 0:
                continue
            stats[(aspect, category)]["term_counts"][term] += count
            stats[(aspect, category)]["total_count"] += count
    return stats

# =============== METRICS & RANKING ===============

def compute_metrics(stats):
    """
    Per (aspect, category):
      - total_count, distinct_terms
      - max_term_count
      - top_terms_sorted: [(term, count), ...] by count desc, then term asc
    Group by aspect.
    """
    by_aspect = defaultdict(list)
    for (aspect, category), data in stats.items():
        if should_exclude_category(category):
            continue
        tc = data["term_counts"]
        total = data["total_count"]
        top_terms_sorted = sorted(tc.items(), key=lambda x: (-x[1], x[0]))
        distinct_terms = len(tc)
        max_term_count = top_terms_sorted[0][1] if top_terms_sorted else 0
        by_aspect[aspect].append({
            "aspect": aspect,
            "category": category,
            "total_count": total,
            "distinct_terms": distinct_terms,
            "max_term_count": max_term_count,
            "top_terms_sorted": top_terms_sorted,
        })
    return by_aspect

def rank_categories(rows_for_aspect):
    return sorted(
        rows_for_aspect,
        key=lambda r: (-r["max_term_count"], -r["distinct_terms"], -r["total_count"], r["category"])
    )

# =============== WRITERS ===============

def write_master_csv(by_aspect, out_path, k_cats=10, k_terms=20):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aspect", "rank", "category", "total_count", "distinct_terms", "top_terms"])
        for aspect, rows in by_aspect.items():
            ranked = rank_categories(rows)[:k_cats]
            for i, rec in enumerate(ranked, start=1):
                top_terms_str = join_terms_list(rec["top_terms_sorted"], max_terms=k_terms)
                w.writerow([
                    rec["aspect"], i, rec["category"],
                    rec["total_count"], rec["distinct_terms"], top_terms_str
                ])

def write_per_aspect_csvs(by_aspect, template, k_cats=10, k_terms=20):
    for aspect, rows in by_aspect.items():
        ranked = rank_categories(rows)[:k_cats]
        path = template.format(aspect=aspect.replace(" ", "_"))
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "category", "total_count", "distinct_terms", "top_terms"])
            for i, rec in enumerate(ranked, start=1):
                top_terms_str = join_terms_list(rec["top_terms_sorted"], max_terms=k_terms)
                w.writerow([i, rec["category"], rec["total_count"], rec["distinct_terms"], top_terms_str])

def write_markdown(by_aspect, out_path, k_cats=10, k_terms=20, wrap_after=6):
    lines = []
    lines.append("# Top-10 Categories per Aspect\n")
    for aspect, rows in by_aspect.items():
        ranked = rank_categories(rows)[:k_cats]
        lines.append(f"## {aspect}\n")
        lines.append("| Rank | Category | Total | Distinct Terms | Top terms |")
        lines.append("|:----:|:---------|------:|:--------------:|:---------|")
        for i, rec in enumerate(ranked, start=1):
            wrapped = wrap_list_for_md(rec["top_terms_sorted"][:k_terms], per_line=wrap_after)
            lines.append(f"| {i} | {rec['category']} | {rec['total_count']} | {rec['distinct_terms']} | {wrapped} |")
        lines.append("")  # blank line
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def write_latex_tables(by_aspect, template, k_cats=10, k_terms=20, wrap_after=5):
    for aspect, rows in by_aspect.items():
        ranked = rank_categories(rows)[:k_cats]
        aspect_safe = aspect.replace("&", "and")  # tiny safety
        body_lines = []
        for i, rec in enumerate(ranked, start=1):
            wrapped = wrap_list_for_latex(rec["top_terms_sorted"][:k_terms], per_line=wrap_after)
            # Columns: Rank & Category & Total & Distinct & Top terms
            body_lines.append(f"{i} & {rec['category']} & {rec['total_count']} & {rec['distinct_terms']} & {wrapped} \\\\ \\hline")

        tex = r"""\begin{longtable}{r l r r p{9cm}}
\caption{Top-10 Categories for %s}\\
\hline
\textbf{Rank} & \textbf{Category} & \textbf{Total} & \textbf{Distinct} & \textbf{Top terms} \\
\hline
\endfirsthead
\hline
\textbf{Rank} & \textbf{Category} & \textbf{Total} & \textbf{Distinct} & \textbf{Top terms} \\
\hline
\endhead
%s
\end{longtable}
""" % (aspect_safe, "\n".join(body_lines))

        path = template.format(aspect=aspect.replace(" ", "_"))
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)

# =============== MAIN ===============

def main():
    stats = load_stats(IN_CSV)
    by_aspect = compute_metrics(stats)

    # Outputs
    write_master_csv(by_aspect, OUT_CSV_ALL, TOP_K_CATEGORIES, TOP_K_TERMS)
    write_per_aspect_csvs(by_aspect, OUT_CSV_PER_ASPECT_TEMPLATE, TOP_K_CATEGORIES, TOP_K_TERMS)
    write_markdown(by_aspect, OUT_MD, TOP_K_CATEGORIES, TOP_K_TERMS, MAX_TERMS_PER_LINE_MD)
    write_latex_tables(by_aspect, OUT_TEX_TEMPLATE, TOP_K_CATEGORIES, TOP_K_TERMS, MAX_TERMS_PER_LINE_LATEX)

    # Console summary
    aspects = list(by_aspect.keys())
    print("Done ✅")
    print("------")
    print(f"Aspects: {aspects}")
    print(f"Saved CSV (all): {OUT_CSV_ALL}")
    print(f"Saved Markdown:  {OUT_MD}")
    print(f"Saved per-aspect CSVs and LaTeX tables.")

if __name__ == "__main__":
    main()
