"""Export per-question response distributions by education level and discharge summary.

Produces ``ds_edu_distributions.csv`` with one row per (respondent or model
iteration) × question, restricted to respondents/personas labelled as
``Edu-Low`` or ``Edu-High``. The CSV feeds ``R/ds_edu_distribution_plot.R``
which renders Figures 1-3:

  Figure 1: ``ds_edu_dumbbell.png``        - per-DS education-gap dumbbell
  Figure 2: ``ds_edu_perception_dist.png`` - stacked-bar response distributions
  Figure 3: ``ds_edu_information_dist.png`` - stacked-bar response distributions

Columns
-------
source         : 'Human' or 'Sonnet 4.5'
education      : 'Edu-Low' or 'Edu-High'
ds             : discharge summary number (1..4)
question_num   : question within DS (1..10)
question_type  : 'Perception' (Q1, Q10) or 'Information' (Q2-Q9)
response       : single letter A..E
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openpyxl

from compute_distance_alignment import (
    DS_BLOCKS,
    JSON_MODELS,
    MAP_PATH,
    ORDINAL,
    XLSX_PATH,
    _first_letter,
    parse_mapping_from_rtf,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_CSV = DATA_DIR / "ds_edu_distributions.csv"

PERC_Q_NUMS = {1, 10}


def question_type(q: int) -> str:
    return "Perception" if q in PERC_Q_NUMS else "Information"


def edu_label(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s == "low":
        return "Edu-Low"
    if s == "high":
        return "Edu-High"
    return None


def collect_sonnet_rows() -> List[Tuple[str, str, int, int, str, str]]:
    rows: List[Tuple[str, str, int, int, str, str]] = []
    for path in JSON_MODELS["Sonnet 4.5"]:
        with open(path) as f:
            data = json.load(f)
        for r in data:
            edu = edu_label(r["persona"].get("education"))
            if edu is None:
                continue
            ds = int(r["discharge_summary_id"][2:])
            q = int(r["question_id"][1:])
            if q not in range(1, 11):
                continue
            letter = _first_letter(r["response"])
            if letter not in ORDINAL:
                continue
            rows.append(("Sonnet 4.5", edu, ds, q, question_type(q), letter))
    return rows


def collect_human_rows(mappings: Dict[str, Dict[str, str]]) -> List[Tuple[str, str, int, int, str, str]]:
    col_to_dsq: Dict[str, Tuple[int, int]] = {}
    for _, (qs, ds_num) in DS_BLOCKS.items():
        for offset, col in enumerate(qs):
            col_to_dsq[col] = (ds_num, offset + 1)

    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True)
    ws = wb.active
    raw_rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) if h else f"col_{i}" for i, h in enumerate(raw_rows[0])]
    wb.close()

    out: List[Tuple[str, str, int, int, str, str]] = []
    for row in raw_rows[1:]:
        row_dict = dict(zip(headers, row))
        gender = row_dict.get("Q49")
        if not gender or str(gender).strip() in ("", "Gender"):
            continue
        edu = edu_label(row_dict.get("Q58"))
        if edu is None:
            continue
        for col, (ds, q) in col_to_dsq.items():
            raw = row_dict.get(col)
            if raw is None or str(raw).strip() == "":
                continue
            s = str(raw).strip()
            qmap = mappings.get(col, {})
            letter = qmap.get(s) or {k.lower(): v for k, v in qmap.items()}.get(s.lower())
            if letter not in ORDINAL:
                continue
            out.append(("Human", edu, ds, q, question_type(q), letter))
    return out


def main() -> None:
    print("Parsing mapping...")
    mappings = parse_mapping_from_rtf(MAP_PATH)

    print("Collecting Sonnet 4.5 persona responses...")
    sonnet_rows = collect_sonnet_rows()
    print(f"  Sonnet 4.5 rows: {len(sonnet_rows)}")

    print("Collecting human responses...")
    human_rows = collect_human_rows(mappings)
    print(f"  Human rows: {len(human_rows)}")

    out_rows = sonnet_rows + human_rows
    print(f"Writing {OUTPUT_CSV} ({len(out_rows)} total rows)...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "education", "ds", "question_num", "question_type", "response"])
        w.writerows(out_rows)
    print("Done.")


if __name__ == "__main__":
    main()
