"""
Summarize combined_response.xlsx (human survey data)

Uses mapping.txt to convert text responses to letter codes (A, B, C, D, ...),
then produces the same style of summary as summarize_claude_45sonnet.py:
  - Overall statistics
  - Response distribution per question
  - Breakdown by each persona/demographic dimension
  - Cross-tabulations for key dimension pairs

Note: The xlsx contains 100 human respondents. Each respondent answered ~3 of 4
discharge summary blocks, yielding ~73-77 responses per question column.
Questions are grouped by discharge summary:
  DS1: Q6-Q15  |  DS2: Q17-Q26  |  DS3: Q28-Q37  |  DS4: Q39-Q48

Within each DS, the 10 questions are split by type:
  - Perception-based:   Q1 and Q10 (first and last)
  - Information-based:  Q2 through Q9

Frequency classification applied to doctor_visit (Q56) and
er_visit_frequency (Q57):
  weekly, monthly -> "high"
  yearly, never   -> "low"
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import openpyxl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "combined_response.xlsx"
if not INPUT_FILE.exists():
    fallback_file = DATA_DIR / "human_response.xlsx"
    if fallback_file.exists():
        INPUT_FILE = fallback_file

MAPPING_FILE = DATA_DIR / "mapping.txt"

# ── Demographic column mapping ──────────────────────────────────
DEMO_COLS = {
    "Q49": "gender",
    "Q50": "age",
    "Q51": "ethnicity",
    "Q52": "highest_education",
    "Q56": "doctor_visit",
    "Q57": "er_visit_frequency",
    "Q58": "education_level",
}

# Discharge summary block assignments
DS_BLOCKS = {
    "DS1": ["Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"],
    "DS2": ["Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26"],
    "DS3": ["Q28", "Q29", "Q30", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37"],
    "DS4": ["Q39", "Q40", "Q41", "Q42", "Q43", "Q44", "Q45", "Q46", "Q47", "Q48"],
}

# Reverse lookup: question -> discharge summary and question-type
Q_TO_DS: dict[str, str] = {}
Q_TO_QNUM: dict[str, int] = {}   # 1..10 within DS
Q_TO_TYPE: dict[str, str] = {}   # 'perception' or 'information'
for ds, qs in DS_BLOCKS.items():
    for offset, q in enumerate(qs):
        qnum = offset + 1
        Q_TO_DS[q] = ds
        Q_TO_QNUM[q] = qnum
        Q_TO_TYPE[q] = "perception" if qnum in (1, 10) else "information"

DEMO_COL_SET = set(DEMO_COLS.keys())


def parse_mapping_from_rtf(path: Path) -> dict[str, dict[str, str]]:
    """Parse the RTF mapping file and return {question_id: {text_value: letter}}."""
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Strip RTF control words and formatting, remove braces
    clean = re.sub(r"\\[a-zA-Z]+[\d]*\s?", " ", raw)
    clean = re.sub(r"\\['\-]", "", clean)
    clean = re.sub(r"[{}]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    mappings: dict[str, dict[str, str]] = {}

    # Find positions of all 'Qnn' tokens
    q_positions = list(re.finditer(r"'(Q\d+)'\s*:", clean))

    for i, q_match in enumerate(q_positions):
        qid = q_match.group(1)
        start = q_match.end()
        end = q_positions[i + 1].start() if i + 1 < len(q_positions) else len(clean)
        block = clean[start:end]

        # Extract 'text' : 'Letter' pairs where Letter is a single A-G
        kv_pattern = re.compile(r"""['"]([^'"]+)['"]\s*:\s*['"]([A-G])['"]""")
        text_to_letter = {}
        for kv_match in kv_pattern.finditer(block):
            text_val = kv_match.group(1).strip()
            letter = kv_match.group(2)
            text_to_letter[text_val] = letter
        if text_to_letter:
            mappings[qid] = text_to_letter

    return mappings


def classify_frequency(freq: str) -> str:
    freq = freq.lower().strip()
    if freq in ["weekly", "monthly"]:
        return "high"
    elif freq in ["yearly", "never"]:
        return "low"
    else:
        return "unknown"


def map_response(raw_value: str | None, question_mapping: dict[str, str]) -> list[str]:
    """Map a raw text response to letter code(s). Returns list for multi-select."""
    if raw_value is None or str(raw_value).strip() == "":
        return []

    raw_str = str(raw_value).strip()

    # Try exact match first
    if raw_str in question_mapping:
        return [question_mapping[raw_str]]

    # Try case-insensitive match
    lower_map = {k.lower(): v for k, v in question_mapping.items()}
    if raw_str.lower() in lower_map:
        return [lower_map[raw_str.lower()]]

    # Handle comma-separated multi-select responses
    parts = [p.strip() for p in raw_str.split(",")]
    if len(parts) > 1:
        letters = []
        for part in parts:
            if part in question_mapping:
                letters.append(question_mapping[part])
            elif part.lower() in lower_map:
                letters.append(lower_map[part.lower()])
        if letters:
            return letters

    return ["UNMAPPED"]


def load_data(xlsx_path: Path, mappings: dict[str, dict[str, str]]) -> tuple[list[dict], int, dict[str, int]]:
    """Load xlsx and return (records, respondent_count, per_column_response_counts)."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) if h else f"col_{i}" for i, h in enumerate(rows[0])]

    records = []
    respondent_count = 0
    col_response_counts: dict[str, int] = Counter()

    for row in rows[1:]:
        row_dict = dict(zip(headers, row))

        # Skip rows with no demographic data (empty padding rows)
        gender_val = row_dict.get("Q49")
        if gender_val is None or str(gender_val).strip() == "":
            continue

        respondent_count += 1

        # Extract demographics
        demo = {}
        for col, field_name in DEMO_COLS.items():
            val = row_dict.get(col)
            demo[field_name] = str(val).strip() if val is not None else "Unknown"

        # Derived binary classifications for visit frequencies
        demo["doctor_visit_class"] = classify_frequency(demo.get("doctor_visit", ""))
        demo["er_visit_class"] = classify_frequency(demo.get("er_visit_frequency", ""))

        # Extract question responses
        for col_name in headers:
            if "_TEXT" in col_name or col_name in DEMO_COL_SET or not col_name.startswith("Q"):
                continue
            if col_name.startswith("col_"):
                continue

            raw_val = row_dict.get(col_name)
            if raw_val is None:
                continue

            q_mapping = mappings.get(col_name, {})
            if not q_mapping:
                continue

            col_response_counts[col_name] += 1
            letters = map_response(raw_val, q_mapping)
            ds_id = Q_TO_DS.get(col_name, "Unknown")
            q_type = Q_TO_TYPE.get(col_name, "unknown")
            q_num = Q_TO_QNUM.get(col_name, -1)
            for letter in letters:
                records.append({
                    "response": letter,
                    "question_id": col_name,
                    "discharge_summary_id": ds_id,
                    "question_num": q_num,
                    "question_type": q_type,
                    "persona": demo,
                })

    wb.close()
    return records, respondent_count, dict(col_response_counts)


# ── Printing helpers ────────────────────────────────────────────

def pct(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%" if total else "0.0%"


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_counter(counter: Counter, total: int, indent: int = 2) -> None:
    pad = " " * indent
    for key, count in sorted(counter.items()):
        print(f"{pad}{key:>10s}: {count:>6d}  ({pct(count, total)})")


def response_distribution_by(data: list[dict], field: str, *, persona_field: bool = True) -> None:
    groups: dict[str, list[str]] = {}
    for record in data:
        key = record["persona"][field] if persona_field else record[field]
        groups.setdefault(key, []).append(record["response"])

    for group_key in sorted(groups):
        responses = groups[group_key]
        counter = Counter(responses)
        total = len(responses)
        print(f"\n  {field} = {group_key}  (n={total})")
        print_counter(counter, total, indent=4)


def cross_tab(data: list[dict], field1: str, field2: str) -> None:
    combos: dict[tuple[str, str], list[str]] = {}
    for record in data:
        k1 = record["persona"][field1]
        k2 = record["persona"][field2]
        combos.setdefault((k1, k2), []).append(record["response"])

    for (v1, v2) in sorted(combos):
        responses = combos[(v1, v2)]
        counter = Counter(responses)
        total = len(responses)
        print(f"\n  {field1}={v1}, {field2}={v2}  (n={total})")
        print_counter(counter, total, indent=4)


def resolve_input_file() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return INPUT_FILE


def main() -> None:
    input_file = resolve_input_file()
    print("Parsing mapping file...")
    mappings = parse_mapping_from_rtf(MAPPING_FILE)
    print(f"  Found mappings for {len(mappings)} questions: {sorted(mappings.keys())}")

    print(f"Loading and mapping xlsx data from {input_file.name}...")
    data, respondent_count, col_counts = load_data(input_file, mappings)

    # Get unique respondents
    unique_personas = set(tuple(sorted(r["persona"].items())) for r in data)
    respondent_count = len(unique_personas)

    # ── Overall statistics ──────────────────────────────────────
    print_section("OVERALL STATISTICS")
    print(f"  Total respondents:             {respondent_count}")
    print(f"  Total mapped response records: {len(data)}")
    print(f"  Unique questions:              {len({r['question_id'] for r in data})}")
    print(f"  Discharge summaries:           {sorted({r['discharge_summary_id'] for r in data})}")

    print(f"\n  Responses per question column:")
    for q in sorted(col_counts, key=lambda x: int(x[1:])):
        print(f"    {q:>4s}: {col_counts[q]:>3d} respondents")

    # Respondents per discharge summary block
    ds_respondents: dict[str, set] = {}
    for record in data:
        ds_id = record["discharge_summary_id"]
        # Use a tuple of demo values as respondent key
        rkey = tuple(sorted(record["persona"].items()))
        ds_respondents.setdefault(ds_id, set()).add(rkey)
    print(f"\n  Respondents per discharge summary block:")
    for ds_id in sorted(ds_respondents):
        print(f"    {ds_id}: ~{len(ds_respondents[ds_id])} respondents")

    persona_fields = [
        "age",
        "gender",
        "education_level",
        "highest_education",
        "ethnicity",
        "doctor_visit",
        "er_visit_frequency",
    ]
    print()
    for field in persona_fields:
        unique_vals = sorted({r["persona"][field] for r in data})
        print(f"  Unique {field + ':':23s} {unique_vals}")

    # ── Frequency classifications ──────────────────────────────────────
    print_section("FREQUENCY CLASSIFICATIONS")
    doctor_class = Counter(classify_frequency(dict(p)["doctor_visit"]) for p in unique_personas)
    er_class = Counter(classify_frequency(dict(p)["er_visit_frequency"]) for p in unique_personas)

    print(f"\n  Doctor visit frequency classifications:")
    for cls, count in sorted(doctor_class.items()):
        print(f"    {cls}: {count} ({pct(count, respondent_count)})")

    print(f"\n  ER visit frequency classifications:")
    for cls, count in sorted(er_class.items()):
        print(f"    {cls}: {count} ({pct(count, respondent_count)})")

    # ── Unmapped responses ──────────────────────────────────────
    unmapped = [r for r in data if r["response"] == "UNMAPPED"]
    if unmapped:
        print(f"\n  WARNING: {len(unmapped)} unmapped responses ({pct(len(unmapped), len(data))})")
        unmapped_qs = Counter(r["question_id"] for r in unmapped)
        for q, cnt in sorted(unmapped_qs.items()):
            print(f"    {q}: {cnt}")

    # Filter out unmapped for the rest of the analysis
    data = [r for r in data if r["response"] != "UNMAPPED"]
    total = len(data)
    print(f"\n  Records used for analysis (after removing unmapped): {total}")

    # ── Overall response distribution ───────────────────────────
    print_section("OVERALL RESPONSE DISTRIBUTION")
    all_responses = [r["response"] for r in data]
    print_counter(Counter(all_responses), total)

    # ── By question ─────────────────────────────────────────────
    print_section("RESPONSE DISTRIBUTION BY QUESTION")
    response_distribution_by(data, "question_id", persona_field=False)

    # ── By discharge summary ────────────────────────────────────
    print_section("RESPONSE DISTRIBUTION BY DISCHARGE SUMMARY")
    response_distribution_by(data, "discharge_summary_id", persona_field=False)

    # ── By question type (Perception vs Information) ───────────
    print_section("RESPONSE DISTRIBUTION BY QUESTION TYPE")
    print("  Perception-based = Q1 & Q10 per DS")
    print("  Information-based = Q2 through Q9 per DS")
    response_distribution_by(data, "question_type", persona_field=False)

    # ── By each persona dimension ───────────────────────────────
    for field in persona_fields:
        print_section(f"RESPONSE DISTRIBUTION BY {field.upper()}")
        response_distribution_by(data, field)

    # ── By classified frequency (high vs low) ──────────────────
    for field in ["doctor_visit_class", "er_visit_class"]:
        print_section(f"RESPONSE DISTRIBUTION BY {field.upper()}")
        print("  Classification: weekly/monthly -> high, yearly/never -> low")
        response_distribution_by(data, field)

    # ── Question type x classified frequency x subgroup ────────
    def type_split_by_subgroup(subgroup_field: str, label: str) -> None:
        print_section(f"QUESTION-TYPE SPLIT BY {label.upper()}")
        types = ["perception", "information"]
        groups: dict[tuple[str, str], list[str]] = {}
        for record in data:
            key = (record["persona"][subgroup_field], record["question_type"])
            groups.setdefault(key, []).append(record["response"])
        for t in types:
            print(f"\n  --- question_type = {t} ---")
            subkeys = sorted(k for k in groups if k[1] == t)
            for (sub, _) in subkeys:
                responses = groups[(sub, t)]
                counter = Counter(responses)
                total = len(responses)
                print(f"\n    {subgroup_field}={sub}  (n={total})")
                print_counter(counter, total, indent=6)

    type_split_by_subgroup("education_level", "Education Level")
    type_split_by_subgroup("gender", "Gender")
    type_split_by_subgroup("doctor_visit_class", "Doctor Visit (classified)")
    type_split_by_subgroup("er_visit_class", "ER Visit (classified)")

    # ── Cross-tabulations ───────────────────────────────────────
    cross_pairs = [
        ("gender", "education_level"),
        ("age", "education_level"),
        ("ethnicity", "education_level"),
        ("age", "gender"),
    ]
    for f1, f2 in cross_pairs:
        print_section(f"CROSS-TAB: {f1.upper()} x {f2.upper()}")
        cross_tab(data, f1, f2)

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
