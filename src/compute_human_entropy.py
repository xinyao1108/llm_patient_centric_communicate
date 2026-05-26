"""
Per-subgroup human response entropy from the pre-aggregated summary file.

This script reads `results/human_response_summary.txt` (produced by
`summarize_combined_response.py`) and computes, for each gender and education
subgroup, the response distribution over A--E and the corresponding Shannon
entropy.  It re-emits two LaTeX tables (tab:human-gender-dist,
tab:human-edu-dist) and a console summary.

Two normalizations are reported per row:

  * H            = - sum_i p_i log2 p_i           (bits)
  * H_n (A-E)    = H / log2(5)                    (5-letter support, our default)
  * H_n (A-D)    = H_AD / log2(4)                 (legacy A-D pooling used by
                                                   `old_results_comparison.tex`,
                                                   for direct comparability)

The summary file's per-cell `n` is the count of *mapped responses*, not
respondents (each respondent answered ~37 questions); the column is labeled
accordingly.  F and G are reported as a single "Other" bucket and dropped
from the entropy computation.

Caveat: the input is pooled across heterogeneous questions (different k_q),
so this matches the old paper's recipe exactly but does NOT correct for the
varying option count per question -- see Q1 (3 options) vs. Q9 (up to 7
options).  For the per-question-normalized variant, use the xlsx directly.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_PATH = BASE_DIR / "results" / "human_response_summary.txt"
OUTPUT_TEX = BASE_DIR / "results" / "human_entropy_tables.tex"

LETTERS_FULL = ["A", "B", "C", "D", "E", "F", "G"]
LETTERS_AE = ["A", "B", "C", "D", "E"]
LETTERS_AD = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_section(text: str, section_header: str) -> Dict[str, Dict[str, int]]:
    """Extract `{subgroup_value: {letter: count}}` for a named section.

    The summary blocks look like

        ============================================================
          RESPONSE DISTRIBUTION BY <FIELD>
        ============================================================

          <field> = <value>  (n=NNNN)
                     A:    439  (23.3%)
                     B:    451  (23.9%)
                     ...
    """
    pattern = rf"  {re.escape(section_header)}\s*\n=+"
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"Section not found: {section_header}")
    body = text[m.end():]
    # Stop at the next "===" header.
    end = re.search(r"\n=+\n", body)
    if end:
        body = body[: end.start()]

    groups: Dict[str, Dict[str, int]] = {}
    cur_label: str | None = None
    label_re = re.compile(r"^\s+\S+\s*=\s*(.+?)\s+\(n=\d+\)\s*$")
    line_re = re.compile(r"^\s+([A-G]):\s*(\d+)\s*\(")
    for line in body.splitlines():
        lm = label_re.match(line)
        if lm:
            cur_label = lm.group(1).strip()
            groups[cur_label] = {l: 0 for l in LETTERS_FULL}
            continue
        cm = line_re.match(line)
        if cm and cur_label is not None:
            groups[cur_label][cm.group(1)] = int(cm.group(2))
    return groups


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def _entropy(counts: List[int]) -> float:
    arr = np.asarray(counts, dtype=float)
    s = arr.sum()
    if s == 0:
        return float("nan")
    p = arr / s
    p_pos = p[p > 0]
    return float(-(p_pos * np.log2(p_pos)).sum())


def _row_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    n_full = sum(counts[l] for l in LETTERS_FULL)
    n_ae = sum(counts[l] for l in LETTERS_AE)
    n_ad = sum(counts[l] for l in LETTERS_AD)

    H_ae = _entropy([counts[l] for l in LETTERS_AE])
    H_ad = _entropy([counts[l] for l in LETTERS_AD])

    pct_ae = {l: counts[l] / n_ae if n_ae else float("nan") for l in LETTERS_AE}
    other = sum(counts[l] for l in ("F", "G"))

    return {
        "n_full": n_full,
        "n_ae": n_ae,
        "n_ad": n_ad,
        "pct_A": pct_ae["A"],
        "pct_B": pct_ae["B"],
        "pct_C": pct_ae["C"],
        "pct_D": pct_ae["D"],
        "pct_E": pct_ae["E"],
        "pct_other": other / n_full if n_full else float("nan"),
        "H_ae": H_ae,
        "Hn_ae": H_ae / np.log2(5),
        "H_ad": H_ad,
        "Hn_ad": H_ad / np.log2(4),
    }


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def _fmt_pct(p: float) -> str:
    return f"{100 * p:.1f}\\%" if p == p else "--"


def _fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}" if v == v else "--"


def _build_table(
    rows: List[Tuple[str, Dict[str, float]]],
    *,
    caption: str,
    label: str,
    group_col_header: str,
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lrrrrrrrrrrr}",
        r"\toprule",
        (rf"\textbf{{{group_col_header}}} & \textbf{{n (responses)}} "
         r"& \textbf{A} & \textbf{B} & \textbf{C} & \textbf{D} & \textbf{E} "
         r"& \textbf{F/G} "
         r"& \textbf{$H$ (A--E)} & \textbf{$H_n$ (A--E)} "
         r"& \textbf{$H$ (A--D)} & \textbf{$H_n$ (A--D)} \\"),
        r"\midrule",
    ]
    for label_str, m in rows:
        lines.append(
            f"{label_str} & {int(m['n_full'])} & "
            f"{_fmt_pct(m['pct_A'])} & {_fmt_pct(m['pct_B'])} & "
            f"{_fmt_pct(m['pct_C'])} & {_fmt_pct(m['pct_D'])} & "
            f"{_fmt_pct(m['pct_E'])} & {_fmt_pct(m['pct_other'])} & "
            f"{_fmt(m['H_ae'])} & {_fmt(m['Hn_ae'])} & "
            f"{_fmt(m['H_ad'])} & {_fmt(m['Hn_ad'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.3em}",
        r"\begin{minipage}{0.95\linewidth}\scriptsize",
        (r"$H$ = Shannon entropy in bits; $H_n = H / \log_2 k$ with $k=5$ for "
         r"A--E and $k=4$ for the legacy A--D variant (matching "
         r"\texttt{old\_results\_comparison.tex}). F/G (don't know / not "
         r"applicable) reported as a single percentage and excluded from $H$. "
         r"Percentages are over the A--E support. $n$ counts mapped responses, "
         r"not respondents (each respondent answered $\sim$37 questions). "
         r"Counts pooled across heterogeneous questions, so no per-question "
         r"$k_q$ normalization is applied."),
        r"\end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not SUMMARY_PATH.exists():
        raise SystemExit(
            f"Missing {SUMMARY_PATH}. Run `make analysis` first to generate it."
        )

    text = SUMMARY_PATH.read_text()

    gender_groups = _parse_section(text, "RESPONSE DISTRIBUTION BY GENDER")
    edu_groups = _parse_section(text, "RESPONSE DISTRIBUTION BY EDUCATION_LEVEL")

    # Gender: order Female, Male, Non-binary
    gender_order = [
        ("Female", "Female"),
        ("Male", "Male"),
        ("Non-binary / third gender", "Non-binary"),
    ]
    gender_rows = [
        (display, _row_metrics(gender_groups[key]))
        for key, display in gender_order
        if key in gender_groups
    ]

    edu_order = [("low", "Low"), ("high", "High")]
    edu_rows = [
        (display, _row_metrics(edu_groups[key]))
        for key, display in edu_order
        if key in edu_groups
    ]

    # ---- Console summary ----
    print("=" * 72)
    print("  HUMAN RESPONSE ENTROPY (pooled across all questions)")
    print("=" * 72)
    for title, rows in [("Gender", gender_rows), ("Education", edu_rows)]:
        print(f"\n{title}")
        print(f"{'':<14}{'n':>6}  {'A%':>5} {'B%':>5} {'C%':>5} {'D%':>5} "
              f"{'E%':>5} {'F/G%':>5}  "
              f"{'H(AE)':>6} {'Hn(AE)':>7}  {'H(AD)':>6} {'Hn(AD)':>7}")
        for lbl, m in rows:
            print(
                f"{lbl:<14}{int(m['n_full']):>6}  "
                f"{100*m['pct_A']:>4.1f} {100*m['pct_B']:>4.1f} "
                f"{100*m['pct_C']:>4.1f} {100*m['pct_D']:>4.1f} "
                f"{100*m['pct_E']:>4.1f} {100*m['pct_other']:>5.1f}  "
                f"{m['H_ae']:>6.3f} {m['Hn_ae']:>7.3f}  "
                f"{m['H_ad']:>6.3f} {m['Hn_ad']:>7.3f}"
            )

    # ---- LaTeX output ----
    OUTPUT_TEX.write_text(
        "% Auto-generated by src/compute_human_entropy.py\n"
        "% Reproducibility table: human response distribution + Shannon entropy.\n\n"
        + _build_table(
            gender_rows,
            caption=(
                "Human overall response distribution by gender (pooled across "
                "all questions, mapped responses)."
            ),
            label="tab:human-gender-dist",
            group_col_header="Gender",
        )
        + "\n\n"
        + _build_table(
            edu_rows,
            caption=(
                "Human overall response distribution by education level "
                "(pooled across all questions, mapped responses)."
            ),
            label="tab:human-edu-dist",
            group_col_header="Education",
        )
        + "\n"
    )
    print(f"\nWrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
