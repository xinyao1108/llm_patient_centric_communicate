"""Export pooled human + model responses in long format for R violin plot.

Output: responses_long.csv with columns
    source         - 'Human' or model name
    question_type  - 'Perception' or 'Information'
    subgroup       - one of the 8 subgroup labels
    ds             - discharge summary number (1..4)
    question_num   - question within DS (1..10)
    response       - integer ordinal 0..4 (A..E)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from compute_distance_alignment import (
    INFO_KEYS,
    JSON_MODELS,
    MAP_PATH,
    ORDINAL,
    PERC_KEYS,
    XLSX_PATH,
    _first_letter,
    classify_freq,
    parse_mapping_from_rtf,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_CSV = DATA_DIR / "responses_long.csv"

PERC_SET = set(PERC_KEYS)
INFO_SET = set(INFO_KEYS)


def question_type(ds: int, q: int) -> str:
    if (ds, q) in PERC_SET:
        return "Perception"
    if (ds, q) in INFO_SET:
        return "Information"
    return "Other"


def model_subgroups(persona: dict) -> List[str]:
    labels: List[str] = []
    edu = str(persona.get("education", "")).strip().lower()
    if edu == "low":
        labels.append("Edu-Low")
    elif edu == "high":
        labels.append("Edu-High")
    g = str(persona.get("gender", "")).strip().lower()
    if g == "male":
        labels.append("Male")
    elif g == "female":
        labels.append("Female")
    doc = classify_freq(persona.get("doctor_visit"))
    if doc == "low":
        labels.append("OutLow")
    elif doc == "high":
        labels.append("OutHigh")
    er = classify_freq(persona.get("er_visit_frequency"))
    if er == "low":
        labels.append("ERLow")
    return labels


def human_subgroups(row: dict) -> List[str]:
    labels: List[str] = []
    edu = row.get("Q58")
    if edu is not None:
        e = str(edu).strip().lower()
        if e == "low":
            labels.append("Edu-Low")
        elif e == "high":
            labels.append("Edu-High")
    gender = row.get("Q49")
    if gender is not None:
        g = str(gender).strip().lower()
        if g == "male":
            labels.append("Male")
        elif g == "female":
            labels.append("Female")
    doc = classify_freq(row.get("Q56"))
    if doc == "low":
        labels.append("OutLow")
    elif doc == "high":
        labels.append("OutHigh")
    er = classify_freq(row.get("Q57"))
    if er == "low":
        labels.append("ERLow")
    return labels


def main() -> None:
    print("Parsing mapping...")
    mappings = parse_mapping_from_rtf(MAP_PATH)

    print("Streaming model responses to CSV...")
    f = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["source", "question_type", "subgroup", "ds", "question_num", "response"])

    n_model_rows = 0
    for model_name, paths in JSON_MODELS.items():
        for path in paths:
            with open(path) as fp:
                data = json.load(fp)
            for r in data:
                ds = int(r["discharge_summary_id"][2:])
                q = int(r["question_id"][1:])
                qtype = question_type(ds, q)
                if qtype == "Other":
                    continue
                letter = _first_letter(r["response"])
                if letter not in ORDINAL:
                    continue
                ord_val = ORDINAL[letter]
                for sub in model_subgroups(r["persona"]):
                    w.writerow([model_name, qtype, sub, ds, q, ord_val])
                    n_model_rows += 1
    print(f"  wrote {n_model_rows} model rows")

    print("Streaming human responses to CSV...")
    import openpyxl
    ds_blocks = {
        "DS1": (["Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"], 1),
        "DS2": (["Q17", "Q18", "Q19", "Q20", "Q21", "Q23", "Q24", "Q25", "Q22", "Q26"], 2),
        "DS3": (["Q28", "Q29", "Q30", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37"], 3),
        "DS4": (["Q39", "Q40", "Q41", "Q42", "Q43", "Q44", "Q45", "Q46", "Q47", "Q48"], 4),
    }
    col_to_dsq: Dict[str, Tuple[int, int]] = {}
    for _, (qs, ds_num) in ds_blocks.items():
        for offset, col in enumerate(qs):
            col_to_dsq[col] = (ds_num, offset + 1)

    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) if h else f"col_{i}" for i, h in enumerate(rows[0])]
    wb.close()

    n_human_rows = 0
    for row in rows[1:]:
        row_dict = dict(zip(headers, row))
        if not row_dict.get("Q49") or str(row_dict.get("Q49")).strip() in ("", "Gender"):
            continue
        subs = human_subgroups(row_dict)
        if not subs:
            continue
        for col, (ds, q) in col_to_dsq.items():
            qtype = question_type(ds, q)
            if qtype == "Other":
                continue
            raw = row_dict.get(col)
            if raw is None or str(raw).strip() == "":
                continue
            s = str(raw).strip()
            qmap = mappings.get(col, {})
            letter = qmap.get(s) or {k.lower(): v for k, v in qmap.items()}.get(s.lower())
            if letter not in ORDINAL:
                continue
            ord_val = ORDINAL[letter]
            for sub in subs:
                w.writerow(["Human", qtype, sub, ds, q, ord_val])
                n_human_rows += 1
    print(f"  wrote {n_human_rows} human rows")
    f.close()
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
