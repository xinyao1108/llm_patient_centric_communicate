"""Robustness check: "High School or Less" vs merged "Low" education reference.

The binary education subgroup ``Edu-Low`` (Q58 == "low") aggregates
three Q52 sub-levels: "No post-high school degree", "Some High School",
"High School Graduate", and "Some College". To test whether the
"Some College" respondents dilute the low-education signal, we recompute
the per-question human majority response using only the "HS or Less"
sub-population and compare LLM alignment rates against both reference
definitions.

Subgroups
---------
Low (merged)   Q58 == "low"                                            -> n approx 49
HS or Less     Q52 in {No post-high school degree, Some High School,
                       High School Graduate}                           -> n approx 36

Outputs
-------
* Console summary with per-subgroup question counts, mode flips, and
  within-group pairwise agreement rates.
* Per-(model, question type) alignment rate of each subgroup's
  ``Edu-Low`` LLM persona against the human majority computed under
  each reference definition.
* LaTeX rows ready to paste into Tables ``tab:edu-robust`` and
  ``tab:edu-robust-align``.
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openpyxl

from compute_distance_alignment import (
    DS_BLOCKS,
    INFO_KEYS,
    JSON_MODELS,
    MAP_PATH,
    ORDINAL,
    PERC_KEYS,
    XLSX_PATH,
    _first_letter,
    parse_mapping_from_rtf,
)

# HS-or-Less subset definition (exact Q52 spellings observed in the xlsx)
HS_OR_LESS_LABELS = {
    "No post-high school degree",
    "Some High School",
    "High School Graduate",
}

MODEL_ORDER = ["Sonnet 4.5", "Opus 4.6", "GPT-5.2", "GPT-4.1"]


def _norm(text: object) -> str:
    return str(text).strip() if text is not None else ""


# ---------------------------------------------------------------------------
# Build per-respondent letter map (questions x respondent)
# ---------------------------------------------------------------------------

def load_human_responses(
    xlsx_path: Path,
    mappings: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[Tuple[int, int], str]], List[str], List[str]]:
    """Return per-respondent (dsq -> letter) maps along with their Q52/Q58 labels."""
    col_to_dsq: Dict[str, Tuple[int, int]] = {}
    for _, (qs, ds_num) in DS_BLOCKS.items():
        for offset, col in enumerate(qs):
            col_to_dsq[col] = (ds_num, offset + 1)

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) if h else f"col_{i}" for i, h in enumerate(rows[0])]
    wb.close()

    answers_per_resp: List[Dict[Tuple[int, int], str]] = []
    q52_labels: List[str] = []
    q58_labels: List[str] = []
    for row in rows[1:]:
        row_dict = dict(zip(headers, row))
        if not _norm(row_dict.get("Q49")) or _norm(row_dict.get("Q49")) == "Gender":
            continue
        per_q: Dict[Tuple[int, int], str] = {}
        for col, dsq in col_to_dsq.items():
            raw = row_dict.get(col)
            if raw is None or _norm(raw) == "":
                continue
            qmap = mappings.get(col, {})
            letter = qmap.get(_norm(raw)) or {k.lower(): v for k, v in qmap.items()}.get(
                _norm(raw).lower()
            )
            if letter in ORDINAL:
                per_q[dsq] = letter
        answers_per_resp.append(per_q)
        q52_labels.append(_norm(row_dict.get("Q52")))
        q58_labels.append(_norm(row_dict.get("Q58")).lower())
    return answers_per_resp, q52_labels, q58_labels


def majority_per_question(
    answers_per_resp: List[Dict[Tuple[int, int], str]],
    member_mask: List[bool],
) -> Dict[Tuple[int, int], Optional[str]]:
    accum: Dict[Tuple[int, int], List[str]] = {}
    for resp, keep in zip(answers_per_resp, member_mask):
        if not keep:
            continue
        for dsq, letter in resp.items():
            accum.setdefault(dsq, []).append(letter)
    return {
        dsq: Counter(letters).most_common(1)[0][0] if letters else None
        for dsq, letters in accum.items()
    }


def within_group_agreement(
    answers_per_resp: List[Dict[Tuple[int, int], str]],
    member_mask: List[bool],
    keys: List[Tuple[int, int]],
) -> float:
    """Fraction of same-question respondent pairs that selected the same letter."""
    keyset = set(keys)
    n_same = 0
    n_total = 0
    members = [a for a, keep in zip(answers_per_resp, member_mask) if keep]
    for a, b in combinations(members, 2):
        shared = (set(a) & set(b)) & keyset
        for k in shared:
            n_total += 1
            if a[k] == b[k]:
                n_same += 1
    return n_same / n_total if n_total else float("nan")


# ---------------------------------------------------------------------------
# LLM Edu-Low subgroup majority per question
# ---------------------------------------------------------------------------

def llm_low_edu_majority(json_paths: List[Path]) -> Dict[Tuple[int, int], Optional[str]]:
    accum: Dict[Tuple[int, int], List[str]] = {}
    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        for r in data:
            if _norm(r["persona"].get("education")).lower() != "low":
                continue
            ds = int(r["discharge_summary_id"][2:])
            q = int(r["question_id"][1:])
            letter = _first_letter(r["response"])
            if letter in ORDINAL:
                accum.setdefault((ds, q), []).append(letter)
    return {
        dsq: Counter(letters).most_common(1)[0][0] if letters else None
        for dsq, letters in accum.items()
    }


def alignment_rate(
    llm_maj: Dict[Tuple[int, int], Optional[str]],
    human_maj: Dict[Tuple[int, int], Optional[str]],
    keys: List[Tuple[int, int]],
) -> float:
    hits = 0
    for k in keys:
        l = llm_maj.get(k)
        h = human_maj.get(k)
        if l is None or h is None:
            continue
        if l == h:
            hits += 1
    return hits / len(keys)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Parsing mapping...")
    mappings = parse_mapping_from_rtf(MAP_PATH)

    print("Loading human responses...")
    answers, q52, q58 = load_human_responses(XLSX_PATH, mappings)

    mask_low = [v == "low" for v in q58]
    mask_hs = [label in HS_OR_LESS_LABELS for label in q52]
    n_low = sum(mask_low)
    n_hs = sum(mask_hs)
    print(f"  Low (merged, Q58=='low'):    n={n_low}")
    print(f"  HS or Less (Q52 in HS set): n={n_hs}")

    maj_low = majority_per_question(answers, mask_low)
    maj_hs = majority_per_question(answers, mask_hs)

    keys_all = PERC_KEYS + INFO_KEYS
    keys_used = [k for k in keys_all if maj_low.get(k) and maj_hs.get(k)]
    diffs = [(k, maj_low[k], maj_hs[k]) for k in keys_used if maj_low[k] != maj_hs[k]]
    print(f"\n  Questions with majority data in both: {len(keys_used)}")
    print(f"  Majority differences (HS-only vs Low merged): {len(diffs)}")
    for k, low, hs in diffs:
        print(f"    DS{k[0]}-Q{k[1]:>2}: Low={low}  HS-only={hs}")

    agree_low = {
        "overall":    within_group_agreement(answers, mask_low, keys_all),
        "info":       within_group_agreement(answers, mask_low, INFO_KEYS),
        "perception": within_group_agreement(answers, mask_low, PERC_KEYS),
    }
    agree_hs = {
        "overall":    within_group_agreement(answers, mask_hs, keys_all),
        "info":       within_group_agreement(answers, mask_hs, INFO_KEYS),
        "perception": within_group_agreement(answers, mask_hs, PERC_KEYS),
    }
    print("\nWithin-group pairwise agreement:")
    print(f"  Low (merged)    overall={agree_low['overall']*100:.1f}%  "
          f"info={agree_low['info']*100:.1f}%  perception={agree_low['perception']*100:.1f}%")
    print(f"  HS or Less only overall={agree_hs['overall']*100:.1f}%  "
          f"info={agree_hs['info']*100:.1f}%  perception={agree_hs['perception']*100:.1f}%")

    # LLM alignment under each reference
    print("\nLLM alignment rate of Edu-Low personas vs human majority:")
    print(f"  {'Model':<12} {'Info(merged)':>13} {'Info(HS-only)':>14} {'Delta':>8} "
          f"{'Perc(merged)':>13} {'Perc(HS-only)':>14} {'Delta':>8}")
    rows = []
    for model in MODEL_ORDER:
        llm_maj = llm_low_edu_majority(JSON_MODELS[model])
        info_low = alignment_rate(llm_maj, maj_low, INFO_KEYS)
        info_hs = alignment_rate(llm_maj, maj_hs, INFO_KEYS)
        perc_low = alignment_rate(llm_maj, maj_low, PERC_KEYS)
        perc_hs = alignment_rate(llm_maj, maj_hs, PERC_KEYS)
        rows.append((model, info_low, info_hs, perc_low, perc_hs))
        print(
            f"  {model:<12} {info_low:13.3f} {info_hs:14.3f} {info_hs - info_low:+8.3f} "
            f"{perc_low:13.3f} {perc_hs:14.3f} {perc_hs - perc_low:+8.3f}"
        )

    # LaTeX rows
    print("\n% Paste into tab:edu-robust:")
    print(
        f"Low (merged)    & {n_low} & {agree_low['overall']*100:.1f}\\% & "
        f"{agree_low['info']*100:.1f}\\% & {agree_low['perception']*100:.1f}\\% & --- \\\\"
    )
    print(
        f"HS or Less only & {n_hs} & {agree_hs['overall']*100:.1f}\\% & "
        f"{agree_hs['info']*100:.1f}\\% & {agree_hs['perception']*100:.1f}\\% & "
        f"{len(diffs)} / {len(keys_used)} \\\\"
    )

    print("\n% Paste into tab:edu-robust-align:")
    for model, info_low, info_hs, perc_low, perc_hs in rows:
        di = info_hs - info_low
        dp = perc_hs - perc_low
        print(
            f"{model} & {info_low:.3f} & {info_hs:.3f} & ${di:+.3f}$ & "
            f"{perc_low:.3f} & {perc_hs:.3f} & ${dp:+.3f}$ \\\\"
        )


if __name__ == "__main__":
    main()
