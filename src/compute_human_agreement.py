"""
Within-group pairwise agreement + cross-group plurality counts (humans only).

Reproduces tab:human-gender-agree and tab:human-edu-agree from the published
human_response.xlsx.  The published version of these tables in
old_results_comparison.tex was computed against a slightly earlier xlsx
snapshot; this script re-derives them from the data currently shipped with
the repository, using a single documented methodology.

Methodology
-----------
* Multi-select aware: a comma-separated raw answer contributes each of its
  mapped letters to that respondent's response set for the question (same
  rule used by ``summarize_combined_response.py``).
* Letters restricted to A-D (matches the ``random baseline = 25%'' footnote
  in the original tables and the A-D pooling used by
  ``compute_human_entropy.py``).
* Pairwise agreement per question is Simpson's repeat-pair probability,
  i.e. P(two randomly chosen respondents in the subgroup picked the same
  letter) = sum_l n_l (n_l - 1) / (n (n - 1)).  The reported subgroup value
  is the unweighted mean across questions for which the subgroup has at
  least two responses.
* Plurality per question is the most common A-D letter, with alphabetical
  tie-breaking.  A question contributes to the cross-group plurality
  denominator only when both compared subgroups have a defined plurality.

Outputs
-------
* ``results/human_agreement.txt`` -- console summary (this script's stdout).
* ``results/human_agreement_tables.tex`` -- LaTeX for
  ``tab:human-gender-agree`` and ``tab:human-edu-agree``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openpyxl

from compute_distance_alignment import parse_mapping_from_rtf

BASE_DIR = Path(__file__).resolve().parent.parent
XLSX_PATH = BASE_DIR / "data" / "human_response.xlsx"
MAP_PATH = BASE_DIR / "data" / "mapping.txt"
OUTPUT_TEX = BASE_DIR / "results" / "human_agreement_tables.tex"

DS_BLOCKS = {
    "DS1": ["Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"],
    "DS2": ["Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26"],
    "DS3": ["Q28", "Q29", "Q30", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37"],
    "DS4": ["Q39", "Q40", "Q41", "Q42", "Q43", "Q44", "Q45", "Q46", "Q47", "Q48"],
}
Q_TYPE: Dict[str, str] = {}
Q_COLS: List[str] = []
for _ds, _qs in DS_BLOCKS.items():
    for _i, _q in enumerate(_qs):
        Q_COLS.append(_q)
        Q_TYPE[_q] = "perception" if _i in (0, 9) else "information"

LETTERS_AD = {"A", "B", "C", "D"}


def _map_multi(raw, qmap: Dict[str, str]) -> List[str]:
    """Map a raw cell value to a list of letters (multi-select aware)."""
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    if s in qmap:
        return [qmap[s]]
    lower = {k.lower(): v for k, v in qmap.items()}
    if s.lower() in lower:
        return [lower[s.lower()]]
    parts = [p.strip() for p in s.split(",")]
    out: List[str] = []
    for p in parts:
        if p in qmap:
            out.append(qmap[p])
        elif p.lower() in lower:
            out.append(lower[p.lower()])
    return out


def _subgroup_labels(row: dict) -> List[str]:
    labels: List[str] = []
    gender = str(row.get("Q49") or "").strip()
    if gender == "Male":
        labels.append("Male")
    elif gender == "Female":
        labels.append("Female")
    elif "Non-binary" in gender:
        labels.append("Non-binary")
    edu = str(row.get("Q58") or "").strip().lower()
    if edu == "low":
        labels.append("Edu-Low")
    elif edu == "high":
        labels.append("Edu-High")
    return labels


def _load_per_question_counts(
    xlsx_path: Path, mappings: Dict[str, Dict[str, str]]
) -> Tuple[Dict[str, Dict[str, Counter]], Dict[str, int]]:
    """Return per-(subgroup, question) Counter and per-subgroup respondent N."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) if h else f"col_{i}" for i, h in enumerate(rows[0])]
    wb.close()

    counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    n_resp: Counter = Counter()
    for row in rows[1:]:
        rd = dict(zip(headers, row))
        gv = rd.get("Q49")
        if not gv or str(gv).strip() in ("", "Gender"):
            continue
        labels = _subgroup_labels(rd)
        for lab in labels:
            n_resp[lab] += 1
        for q in Q_COLS:
            for letter in _map_multi(rd.get(q), mappings.get(q, {})):
                if letter in LETTERS_AD:
                    for lab in labels:
                        counts[lab][q][letter] += 1
    return counts, n_resp


def _pairwise(c: Counter) -> Optional[float]:
    n = sum(c.values())
    if n < 2:
        return None
    return sum(x * (x - 1) for x in c.values()) / (n * (n - 1))


def _agg_pairwise(counts: Dict[str, Counter], qtype: Optional[str] = None) -> float:
    vals = [
        v
        for q in Q_COLS
        if (qtype is None or Q_TYPE[q] == qtype)
        for v in [_pairwise(counts[q])]
        if v is not None
    ]
    return sum(vals) / len(vals) if vals else float("nan")


def _plurality(c: Counter) -> Optional[str]:
    """Most common A-D letter; alphabetical tie-break.  None if no data."""
    if not c:
        return None
    items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[0][0]


def _cross_plurality(
    counts_a: Dict[str, Counter], counts_b: Dict[str, Counter]
) -> Tuple[int, int, int, List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Return (n_agree, n_total_compared, n_excluded, agreements, disagreements)."""
    agree = 0
    disagreements: List[Tuple[str, str, str]] = []
    agreements: List[Tuple[str, str, str]] = []
    excluded = 0
    for q in Q_COLS:
        a = _plurality(counts_a[q])
        b = _plurality(counts_b[q])
        if a is None or b is None:
            excluded += 1
            continue
        if a == b:
            agree += 1
            agreements.append((q, a, b))
        else:
            disagreements.append((q, a, b))
    return agree, agree + len(disagreements), excluded, agreements, disagreements


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{100 * v:.1f}\\%"


def _fmt_dis_list(disagreements: List[Tuple[str, str, str]], label_a: str, label_b: str) -> str:
    if not disagreements:
        return "no disagreements"
    parts = []
    for q, a, b in sorted(disagreements, key=lambda x: int(x[0][1:])):
        qt = Q_TYPE[q]
        parts.append(f"{q} ({qt}: {label_a}\\,=\\,{a}, {label_b}\\,=\\,{b})")
    return ", ".join(parts)


def _build_gender_table(
    counts: Dict[str, Dict[str, Counter]], n_resp: Dict[str, int], dis_mf: List[Tuple[str, str, str]],
    agree_mf: int, denom_mf: int
) -> str:
    rows = []
    for gname, display in (("Male", "Male"), ("Female", "Female"), ("Non-binary", "Non-binary$^*$")):
        o = _agg_pairwise(counts[gname])
        i = _agg_pairwise(counts[gname], "information")
        p = _agg_pairwise(counts[gname], "perception")
        rows.append(f"{display} & {n_resp[gname]} & {_pct(o)} & {_pct(i)} & {_pct(p)} \\\\")
    dis_str = _fmt_dis_list(dis_mf, "M", "F")
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{Within-group pairwise agreement by gender (human respondents). "
         r"Pairwise agreement is restricted to options A--D; see footnote~1 for the "
         r"inclusion rule.}"),
        r"\label{tab:human-gender-agree}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Gender} & \textbf{N} & \textbf{Overall} & \textbf{Info Qs} & \textbf{Perception Qs} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.3em}",
        r"\begin{minipage}{0.95\linewidth}\scriptsize",
        (r"Pairwise agreement = probability that two randomly chosen respondents in the "
         r"subgroup selected the same response (A--D), averaged across questions; random "
         rf"baseline = 25\%. Male and Female agree on the plurality response for {agree_mf} "
         rf"of {denom_mf} questions ({100*agree_mf/denom_mf:.1f}\\%); disagreements occur on {dis_str}.\\"),
        r"$^*$ N=6; estimates unreliable for this group.",
        r"\end{minipage}",
        r"\end{table}",
    ])


def _build_edu_table(
    counts: Dict[str, Dict[str, Counter]], n_resp: Dict[str, int],
    dis_e: List[Tuple[str, str, str]], agree_e: int, denom_e: int, excluded_e: int
) -> str:
    rows = []
    for ename, display in (("Edu-Low", "Low"), ("Edu-High", "High")):
        o = _agg_pairwise(counts[ename])
        i = _agg_pairwise(counts[ename], "information")
        p = _agg_pairwise(counts[ename], "perception")
        rows.append(f"{display} & {n_resp[ename]} & {_pct(o)} & {_pct(i)} & {_pct(p)} \\\\")
    dis_str = _fmt_dis_list(dis_e, "Low", "High")
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{Within-group pairwise agreement by education level (human "
         r"respondents). Pairwise agreement is restricted to options A--D; see "
         r"footnote~1 for the inclusion rule.}"),
        r"\label{tab:human-edu-agree}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Education} & \textbf{N} & \textbf{Overall} & \textbf{Info Qs} & \textbf{Perception Qs} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.3em}",
        r"\begin{minipage}{0.95\linewidth}\scriptsize",
        (r"Pairwise agreement = probability that two randomly chosen respondents in the "
         r"subgroup selected the same response (A--D), averaged across questions; random "
         rf"baseline = 25\%. Low- and high-education groups agree on the plurality response "
         rf"for {agree_e} of {denom_e} questions ({100*agree_e/denom_e:.1f}\\%); "
         rf"disagreements occur on {dis_str}. "
         rf"{excluded_e} questions excluded because neither education stratum "
         r"produced a uniquely-defined plurality on those questions."),
        r"\end{minipage}",
        r"\end{table}",
    ])


def main() -> None:
    mappings = parse_mapping_from_rtf(MAP_PATH)
    counts, n_resp = _load_per_question_counts(XLSX_PATH, mappings)

    # Cross-group plurality
    a_mf, d_mf, x_mf, _, dis_mf = _cross_plurality(counts["Male"], counts["Female"])
    a_e, d_e, x_e, _, dis_e = _cross_plurality(counts["Edu-Low"], counts["Edu-High"])

    # Console summary
    print("=" * 72)
    print("  WITHIN-GROUP PAIRWISE AGREEMENT (A-D restriction, multi-select aware)")
    print("=" * 72)
    print(f"\n{'Group':<14} {'N':>3}  {'Overall':>8} {'Info':>8} {'Percep':>8}")
    for g in ("Male", "Female", "Non-binary", "Edu-Low", "Edu-High"):
        o = _agg_pairwise(counts[g])
        i = _agg_pairwise(counts[g], "information")
        p = _agg_pairwise(counts[g], "perception")
        print(f"{g:<14} {n_resp[g]:>3}  "
              f"{100*o:>7.1f}% {100*i:>7.1f}% {100*p:>7.1f}%")

    print("\n--- Cross-group plurality (A-D, alphabetical tie-break) ---")
    print(f"Male vs Female: {a_mf}/{d_mf} agree ({100*a_mf/d_mf:.1f}%); "
          f"{x_mf} excluded.")
    print(f"  Disagreements: {[(q,a,b) for q,a,b in sorted(dis_mf, key=lambda x:int(x[0][1:]))]}")
    print(f"Edu-Low vs Edu-High: {a_e}/{d_e} agree ({100*a_e/d_e:.1f}%); "
          f"{x_e} excluded.")
    print(f"  Disagreements: {[(q,a,b) for q,a,b in sorted(dis_e, key=lambda x:int(x[0][1:]))]}")

    # LaTeX
    tex = (
        "% Auto-generated by src/compute_human_agreement.py\n"
        "% Reproducible within-group pairwise agreement tables (A-D restricted).\n\n"
        + _build_gender_table(counts, n_resp, dis_mf, a_mf, d_mf)
        + "\n\n"
        + _build_edu_table(counts, n_resp, dis_e, a_e, d_e, x_e)
        + "\n"
    )
    OUTPUT_TEX.write_text(tex)
    print(f"\nWrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
