"""Per-discharge-summary sex gap on perception modal MAE (Male - Female).

For each of the four LLMs and each discharge summary DS1-DS4, this script
computes the modal mean absolute error (MAE) over the two perception
questions (Q1, Q10) for the Male and Female persona subgroups, using the
matched human Male/Female majority response as the reference. The per-DS
sex gap is then::

    gap_{model, DS} = MAE_Male(model, DS) - MAE_Female(model, DS)

Positive values indicate worse alignment for the male subgroup. The
"Overall" row reproduces the perception modal MAE Male - Female values
already reported in Table 6 (summary_results.tex tab:sex-gap).

Outputs
-------
* Console summary table
* LaTeX rows ready to paste into Table 8 (``tab:ds-sex-gap-perc``)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openpyxl

from compute_distance_alignment import (
    DS_BLOCKS,
    JSON_MODELS,
    MAP_PATH,
    ORDINAL,
    PERC_KEYS,
    XLSX_PATH,
    _first_letter,
    parse_mapping_from_rtf,
)

MODEL_ORDER = ["Sonnet 4.5", "Opus 4.6", "GPT-5.2", "GPT-4.1"]
PERC_Q_NUMS = {1, 10}


def _norm(text: object) -> str:
    return str(text).strip() if text is not None else ""


# ---------------------------------------------------------------------------
# Human Male/Female majority per (DS, Q) for perception questions only
# ---------------------------------------------------------------------------

def human_sex_majority(
    xlsx_path: Path,
    mappings: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[Tuple[int, int], Optional[str]]]:
    col_to_dsq: Dict[str, Tuple[int, int]] = {}
    for _, (qs, ds_num) in DS_BLOCKS.items():
        for offset, col in enumerate(qs):
            col_to_dsq[col] = (ds_num, offset + 1)

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) if h else f"col_{i}" for i, h in enumerate(rows[0])]
    wb.close()

    accum: Dict[str, Dict[Tuple[int, int], List[str]]] = {"male": {}, "female": {}}
    perc_keyset = set(PERC_KEYS)
    for row in rows[1:]:
        row_dict = dict(zip(headers, row))
        gender = _norm(row_dict.get("Q49")).lower()
        if gender not in ("male", "female"):
            continue
        for col, dsq in col_to_dsq.items():
            if dsq not in perc_keyset:
                continue
            raw = row_dict.get(col)
            if raw is None or _norm(raw) == "":
                continue
            qmap = mappings.get(col, {})
            letter = qmap.get(_norm(raw)) or {k.lower(): v for k, v in qmap.items()}.get(
                _norm(raw).lower()
            )
            if letter in ORDINAL:
                accum[gender].setdefault(dsq, []).append(letter)
    return {
        sex: {
            dsq: Counter(letters).most_common(1)[0][0] if letters else None
            for dsq, letters in per_q.items()
        }
        for sex, per_q in accum.items()
    }


# ---------------------------------------------------------------------------
# LLM Male/Female persona majority per (DS, Q) for perception questions only
# ---------------------------------------------------------------------------

def llm_sex_majority(
    json_paths: List[Path],
) -> Dict[str, Dict[Tuple[int, int], Optional[str]]]:
    accum: Dict[str, Dict[Tuple[int, int], List[str]]] = {"male": {}, "female": {}}
    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        for r in data:
            sex = _norm(r["persona"].get("gender")).lower()
            if sex not in ("male", "female"):
                continue
            ds = int(r["discharge_summary_id"][2:])
            q = int(r["question_id"][1:])
            if q not in PERC_Q_NUMS:
                continue
            letter = _first_letter(r["response"])
            if letter in ORDINAL:
                accum[sex].setdefault((ds, q), []).append(letter)
    return {
        sex: {
            dsq: Counter(letters).most_common(1)[0][0] if letters else None
            for dsq, letters in per_q.items()
        }
        for sex, per_q in accum.items()
    }


# ---------------------------------------------------------------------------
# Modal MAE on a key list
# ---------------------------------------------------------------------------

def modal_mae(
    llm_maj: Dict[Tuple[int, int], Optional[str]],
    human_maj: Dict[Tuple[int, int], Optional[str]],
    keys: List[Tuple[int, int]],
) -> float:
    dists: List[float] = []
    for k in keys:
        l = llm_maj.get(k)
        h = human_maj.get(k)
        if l in ORDINAL and h in ORDINAL:
            dists.append(abs(ORDINAL[l] - ORDINAL[h]))
    return sum(dists) / len(dists) if dists else float("nan")


def main() -> None:
    print("Parsing mapping...")
    mappings = parse_mapping_from_rtf(MAP_PATH)

    print("Loading human sex-stratified perception majorities...")
    human = human_sex_majority(XLSX_PATH, mappings)

    ds_labels = {
        1: "DS1: Short & Difficult",
        2: "DS2: Long & Simple",
        3: "DS3: Long & Difficult",
        4: "DS4: Short & Simple",
    }
    ds_keys = {ds: [(ds, q) for q in sorted(PERC_Q_NUMS)] for ds in range(1, 5)}

    print(f"\n  {'DS':<24} " + " ".join(f"{m:>12}" for m in MODEL_ORDER))
    row_cells: Dict[int, List[float]] = {}
    overall_cells: List[float] = []

    for model in MODEL_ORDER:
        # Pre-load once per model
        pass

    # Compute per-(model, DS) gaps
    table: Dict[int, Dict[str, float]] = {ds: {} for ds in range(1, 5)}
    overall: Dict[str, Tuple[float, float, float]] = {}
    for model in MODEL_ORDER:
        llm = llm_sex_majority(JSON_MODELS[model])
        for ds in range(1, 5):
            mae_m = modal_mae(llm["male"], human["male"], ds_keys[ds])
            mae_f = modal_mae(llm["female"], human["female"], ds_keys[ds])
            table[ds][model] = mae_m - mae_f
        # Overall across all 8 perception keys (= column "Sex Male - Female" in Table 1/6)
        mae_m_all = modal_mae(llm["male"], human["male"], PERC_KEYS)
        mae_f_all = modal_mae(llm["female"], human["female"], PERC_KEYS)
        overall[model] = (mae_m_all, mae_f_all, mae_m_all - mae_f_all)

    print()
    print(f"{'DS Type':<24} " + " ".join(f"{m:>12}" for m in MODEL_ORDER))
    print("-" * (24 + 13 * len(MODEL_ORDER)))
    for ds in range(1, 5):
        cells = " ".join(f"{table[ds][m]:>+12.2f}" for m in MODEL_ORDER)
        print(f"{ds_labels[ds]:<24} {cells}")
    print("-" * (24 + 13 * len(MODEL_ORDER)))
    cells = " ".join(f"{overall[m][2]:>+12.3f}" for m in MODEL_ORDER)
    print(f"{'Overall (Male-Female)':<24} {cells}")

    print("\n% Paste into tab:ds-sex-gap-perc:")
    for ds in range(1, 5):
        cells = " & ".join(_fmt_signed(table[ds][m]) for m in MODEL_ORDER)
        label = ds_labels[ds].replace("&", "\\&")
        print(f"{label} & {cells} \\\\")
    overall_cells_str = " & ".join(_fmt_signed(overall[m][2], 2) for m in MODEL_ORDER)
    print(f"\\midrule")
    print(f"Overall                 & {overall_cells_str} \\\\")


def _fmt_signed(v: float, nd: int = 2) -> str:
    if v != v:  # NaN
        return "--"
    if v == 0:
        return f" 0.00"
    if v > 0:
        return f"+{v:.{nd}f}"
    return f"$-${abs(v):.{nd}f}"


if __name__ == "__main__":
    main()
