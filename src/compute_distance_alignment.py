"""
Compute distance-based alignment metrics for model vs. human responses.

Question split:
  - Perception-based: Q1 and Q10 (per DS)         -> 8 keys total
  - Information-based: Q2 through Q9 (per DS)     -> 32 keys total

Metrics (computed per subgroup, per question-type):
  1. Modal MAE  - mean |ord(model_mode) - ord(human_mode)|
  2. Exp. dist  - mean E[|ord(LLM) - ord(Human)|] over marginal distributions

Ordinal coding: A=0, B=1, C=2, D=3, E=4 (max distance = 4).

Subgroups are aligned between model and human:
  Edu-Low/High, Male/Female, OutLow/High, ERLow/High.

Human frequency classification:
  weekly, monthly -> "high"
  yearly, never   -> "low"

Models:
  - Sonnet 4.5, Opus 4.6, GPT-5.2, GPT-4.1 - from add_exp_results/*_iter*.json
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openpyxl

# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

ORDINAL: Dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

PERC_Q_NUMS = {1, 10}                     # Perception-based
INFO_Q_NUMS = {2, 3, 4, 5, 6, 7, 8, 9}    # Information-based
PERC_KEYS = [(ds, q) for ds in range(1, 5) for q in PERC_Q_NUMS]   # 8
INFO_KEYS = [(ds, q) for ds in range(1, 5) for q in INFO_Q_NUMS]   # 32

JSON_MODELS: Dict[str, List[Path]] = {
    "Sonnet 4.5": sorted(DATA_DIR.glob("add_exp_results/claude-sonnet-4-5-20250929_iter*.json")),
    "Opus 4.6":   sorted(DATA_DIR.glob("add_exp_results/claude-opus-4-6_iter*.json")),
    "GPT-5.2":    sorted(DATA_DIR.glob("add_exp_results/gpt-5-2-2025-12-11_iter*.json")),
    "GPT-4.1":    sorted(DATA_DIR.glob("add_exp_results/gpt-4-1-2025-04-14_iter*.json")),
}

XLSX_PATH = DATA_DIR / "human_response.xlsx"
MAP_PATH = DATA_DIR / "mapping.txt"

# Human xlsx column -> (DS number, Q-within-DS 1..10)
DS_BLOCKS = {
    "DS1": (["Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"], 1),
    "DS2": (["Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26"], 2),
    "DS3": (["Q28", "Q29", "Q30", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37"], 3),
    "DS4": (["Q39", "Q40", "Q41", "Q42", "Q43", "Q44", "Q45", "Q46", "Q47", "Q48"], 4),
}

SUBGROUP_LABELS = ["Edu-Low", "Edu-High", "Male", "Female", "OutLow", "OutHigh", "ERLow"]

# Type aliases
PerqDist = Dict[Tuple[int, int], Dict[str, float]]
PerqMaj = Dict[Tuple[int, int], Optional[str]]


# ---------------------------------------------------------------------------
# Frequency classification (shared rule)
# ---------------------------------------------------------------------------

def classify_freq(raw: Optional[str]) -> Optional[str]:
    """Map raw visit-frequency label to 'high' or 'low'.

    weekly/monthly -> 'high'
    yearly/never   -> 'low'
    Already-classified values ('high'/'low'/'High'/'Low') are normalized.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("weekly", "monthly", "high"):
        return "high"
    if s in ("yearly", "never", "low"):
        return "low"
    return None


# ---------------------------------------------------------------------------
# RTF mapping parser
# ---------------------------------------------------------------------------

def parse_mapping_from_rtf(path: Path) -> Dict[str, Dict[str, str]]:
    """Parse the RTF mapping file into ``{Qid: {option_text: letter}}``.

    The mapping mixes single-quoted keys (``'Avoid fruit'``) with
    double-quoted keys (``"I don't know"``) so that option texts containing
    apostrophes can be encoded.  The key/value regex must therefore accept
    matching outer-quote pairs and tolerate the opposite quote inside.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    clean = re.sub(r"\\[a-zA-Z]+[\d]*\s?", " ", raw)
    clean = re.sub(r"\\['\-]", "", clean)
    clean = re.sub(r"[{}]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    mappings: Dict[str, Dict[str, str]] = {}
    q_positions = list(re.finditer(r"'(Q\d+)'\s*:", clean))
    # Accept keys wrapped in '...' (no internal singles) OR "..." (no internal
    # doubles); values are always a single letter wrapped in either quote.
    kv = re.compile(
        r"""(?:'([^']+)'|"([^"]+)")\s*:\s*['"]([A-G])['"]"""
    )
    for i, m in enumerate(q_positions):
        qid = m.group(1)
        start = m.end()
        end = q_positions[i + 1].start() if i + 1 < len(q_positions) else len(clean)
        block = clean[start:end]
        t2l: Dict[str, str] = {}
        for kv_m in kv.finditer(block):
            key = (kv_m.group(1) or kv_m.group(2)).strip()
            t2l[key] = kv_m.group(3)
        if t2l:
            mappings[qid] = t2l
    return mappings


def _map_response(raw: Optional[str], qmap: Dict[str, str]) -> Optional[str]:
    if not raw or str(raw).strip() == "":
        return None
    s = str(raw).strip()
    if s in qmap:
        return qmap[s]
    lower = {k.lower(): v for k, v in qmap.items()}
    return lower.get(s.lower())


# ---------------------------------------------------------------------------
# Human data loading (split into 8 subgroups)
# ---------------------------------------------------------------------------

def load_human_data(
    xlsx_path: Path,
    mappings: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, PerqMaj], Dict[str, PerqDist]]:
    """Return per-subgroup human majority + distribution dictionaries.

    Keys are the 8 SUBGROUP_LABELS. Each respondent contributes to all
    subgroups for which their demographics match (e.g. Edu-Low AND Male).
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) if h else f"col_{i}" for i, h in enumerate(rows[0])]
    wb.close()

    col_to_dsq: Dict[str, Tuple[int, int]] = {}
    for _, (qs, ds_num) in DS_BLOCKS.items():
        for offset, col in enumerate(qs):
            col_to_dsq[col] = (ds_num, offset + 1)

    def persona_labels(row: dict) -> List[str]:
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
        doc_cls = classify_freq(row.get("Q56"))
        if doc_cls == "low":
            labels.append("OutLow")
        elif doc_cls == "high":
            labels.append("OutHigh")
        er_cls = classify_freq(row.get("Q57"))
        if er_cls == "low":
            labels.append("ERLow")
        return labels

    # accumulate letters per (subgroup_label, (ds, q))
    accum: Dict[str, Dict[Tuple[int, int], List[str]]] = {lbl: {} for lbl in SUBGROUP_LABELS}

    for row in rows[1:]:
        row_dict = dict(zip(headers, row))
        if not row_dict.get("Q49") or str(row_dict.get("Q49")).strip() in ("", "Gender"):
            continue
        labels = persona_labels(row_dict)
        if not labels:
            continue
        for col, dsq in col_to_dsq.items():
            raw = row_dict.get(col)
            letter = _map_response(raw, mappings.get(col, {}))
            if not letter:
                continue
            for lbl in labels:
                accum[lbl].setdefault(dsq, []).append(letter)

    subgroup_maj: Dict[str, PerqMaj] = {}
    subgroup_dist: Dict[str, PerqDist] = {}
    for lbl in SUBGROUP_LABELS:
        maj: PerqMaj = {}
        dist: PerqDist = {}
        for dsq, letters in accum[lbl].items():
            c = Counter(letters)
            top = c.most_common(1)
            maj[dsq] = top[0][0] if top else None
            total = sum(c.values())
            dist[dsq] = {ltr: cnt / total for ltr, cnt in c.items()}
        subgroup_maj[lbl] = maj
        subgroup_dist[lbl] = dist

    return subgroup_maj, subgroup_dist


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def _modal_mae(model_maj: PerqMaj, human_maj: PerqMaj, keys: List[Tuple[int, int]]) -> float:
    dists: List[float] = []
    for k in keys:
        lm = model_maj.get(k)
        hm = human_maj.get(k)
        if lm in ORDINAL and hm in ORDINAL:
            dists.append(abs(ORDINAL[lm] - ORDINAL[hm]))
    return sum(dists) / len(dists) if dists else float("nan")


def _expected_dist(
    model_dist: PerqDist,
    human_dist: PerqDist,
    keys: List[Tuple[int, int]],
) -> float:
    vals: List[float] = []
    for k in keys:
        llm_d = model_dist.get(k, {})
        hum_d = human_dist.get(k, {})
        e_dist = sum(
            abs(ORDINAL[li] - ORDINAL[hi]) * lp * hp
            for li, lp in llm_d.items() if li in ORDINAL
            for hi, hp in hum_d.items() if hi in ORDINAL
        )
        vals.append(e_dist)
    return sum(vals) / len(vals) if vals else float("nan")


def compute_modal_mae_perc(m: PerqMaj, h: PerqMaj) -> float:
    return _modal_mae(m, h, PERC_KEYS)


def compute_modal_mae_info(m: PerqMaj, h: PerqMaj) -> float:
    return _modal_mae(m, h, INFO_KEYS)


def compute_expected_dist_perc(m: PerqDist, h: PerqDist) -> float:
    return _expected_dist(m, h, PERC_KEYS)


def compute_expected_dist_info(m: PerqDist, h: PerqDist) -> float:
    return _expected_dist(m, h, INFO_KEYS)


# ---------------------------------------------------------------------------
# Load JSON model data
# ---------------------------------------------------------------------------

def _first_letter(raw) -> Optional[str]:
    s = str(raw).strip() if raw is not None else ""
    c = s[0].upper() if s else ""
    return c if c in ORDINAL else None


def load_json_model(
    json_paths: List[Path],
) -> Tuple[Dict[str, PerqMaj], Dict[str, PerqDist]]:
    """Return (subgroup_maj, subgroup_dist) for a model.

    Robust to model persona where doctor_visit / er_visit_frequency may be
    either raw strings ('weekly', 'yearly', ...) or already classified
    ('High'/'Low').
    """
    data: List[dict] = []
    for json_path in json_paths:
        with open(json_path) as f:
            data.extend(json.load(f))

    def dsq(r: dict) -> Tuple[int, int]:
        return (int(r["discharge_summary_id"][2:]), int(r["question_id"][1:]))

    def edu(r: dict) -> str:
        return str(r["persona"].get("education", "")).strip().lower()

    def gender(r: dict) -> str:
        return str(r["persona"].get("gender", "")).strip().lower()

    subgroup_defs = {
        "Edu-Low":  lambda r: edu(r) == "low",
        "Edu-High": lambda r: edu(r) == "high",
        "Male":     lambda r: gender(r) == "male",
        "Female":   lambda r: gender(r) == "female",
        "OutLow":   lambda r: classify_freq(r["persona"].get("doctor_visit")) == "low",
        "OutHigh":  lambda r: classify_freq(r["persona"].get("doctor_visit")) == "high",
        "ERLow":    lambda r: classify_freq(r["persona"].get("er_visit_frequency")) == "low",
    }

    subgroup_maj: Dict[str, PerqMaj] = {}
    subgroup_dist: Dict[str, PerqDist] = {}

    for lbl, pred in subgroup_defs.items():
        accum: Dict[Tuple[int, int], List[str]] = {}
        for r in data:
            if not pred(r):
                continue
            letter = _first_letter(r["response"])
            if letter:
                accum.setdefault(dsq(r), []).append(letter)

        maj: PerqMaj = {}
        dist: PerqDist = {}
        for k, letters in accum.items():
            c = Counter(letters)
            top = c.most_common(1)
            maj[k] = top[0][0] if top else None
            total = sum(c.values())
            dist[k] = {ltr: cnt / total for ltr, cnt in c.items()}

        subgroup_maj[lbl] = maj
        subgroup_dist[lbl] = dist

    return subgroup_maj, subgroup_dist


# ---------------------------------------------------------------------------
# Per-question diagnostics
# ---------------------------------------------------------------------------

def per_question_distances(
    model_maj: PerqMaj,
    human_maj: PerqMaj,
    keys: List[Tuple[int, int]],
    limit: Optional[int] = None,
) -> None:
    ks = sorted(keys)
    if limit is not None:
        ks = ks[:limit]
    for k in ks:
        lm = model_maj.get(k, "?")
        hm = human_maj.get(k, "?")
        d = abs(ORDINAL[lm] - ORDINAL[hm]) if lm in ORDINAL and hm in ORDINAL else "-"
        print(f"    DS{k[0]}-Q{k[1]:>2d}: LLM={lm}  Human={hm}  |dist|={d}")


# ---------------------------------------------------------------------------
# LaTeX table printers
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.2f}" if v == v else "--"


def _print_latex_table(
    all_metric: Dict[str, Dict[str, float]],
    *,
    caption: str,
    label: str,
) -> None:
    print(r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\caption{%s}
\label{%s}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Model}
& \multicolumn{2}{c}{\textbf{Education}}
& \multicolumn{2}{c}{\textbf{Sex}}
& \multicolumn{2}{c}{\textbf{Outpatient}}
& \textbf{ED Visits}
& \textbf{Max} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
& Low & High & Male & Female & Low & High & Low & \\
\midrule""" % (caption, label))

    order = ["Sonnet 4.5", "Opus 4.6", "GPT-5.2", "GPT-4.1"]
    for model in order:
        if model not in all_metric:
            continue
        row = all_metric[model]
        v = [_fmt(row[l]) for l in SUBGROUP_LABELS]
        cells = " & ".join(v)
        print(f"{model} & {cells} & 4 \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Parsing mapping file...")
    mappings = parse_mapping_from_rtf(MAP_PATH)

    print("Loading human data (per-subgroup split)...")
    human_maj, human_dist = load_human_data(XLSX_PATH, mappings)

    print("\nHuman subgroup sample sizes (perception keys coverage):")
    for lbl in SUBGROUP_LABELS:
        covered = sum(1 for k in PERC_KEYS if k in human_maj[lbl])
        total_records = sum(
            sum(int(round(p * 1e6)) for p in human_dist[lbl].get(k, {}).values()) > 0
            for k in PERC_KEYS
        )
        print(f"  {lbl:<9}: {covered}/8 perception keys populated")

    # Collect results per question type
    all_mae_perc: Dict[str, Dict[str, float]] = {}
    all_ead_perc: Dict[str, Dict[str, float]] = {}
    all_mae_info: Dict[str, Dict[str, float]] = {}
    all_ead_info: Dict[str, Dict[str, float]] = {}

    for model_name, json_paths in JSON_MODELS.items():
        if not json_paths:
            print(f"\n[WARN] No files for {model_name}; skipping.")
            continue
        print(f"\nLoading {model_name} ({len(json_paths)} files)...")
        smaj, sdist = load_json_model(json_paths)
        all_mae_perc[model_name] = {}
        all_ead_perc[model_name] = {}
        all_mae_info[model_name] = {}
        all_ead_info[model_name] = {}
        for label in SUBGROUP_LABELS:
            all_mae_perc[model_name][label] = compute_modal_mae_perc(smaj[label], human_maj[label])
            all_ead_perc[model_name][label] = compute_expected_dist_perc(sdist[label], human_dist[label])
            all_mae_info[model_name][label] = compute_modal_mae_info(smaj[label], human_maj[label])
            all_ead_info[model_name][label] = compute_expected_dist_info(sdist[label], human_dist[label])

    col_w = 8
    hdr = f"{'Model':<14}" + "".join(f"{l:>{col_w}}" for l in SUBGROUP_LABELS)

    def print_table(title: str, metric_map: Dict[str, Dict[str, float]]) -> None:
        print("\n" + "=" * 78)
        print(f"  {title}")
        print("=" * 78)
        print(hdr)
        for model, row in metric_map.items():
            vals = "".join(f"{row[l]:>{col_w}.3f}" for l in SUBGROUP_LABELS)
            print(f"  {model:<12}{vals}")

    print_table("PERCEPTION (Q1, Q10) - MODAL MAE", all_mae_perc)
    print_table("PERCEPTION (Q1, Q10) - EXPECTED ABSOLUTE DISTANCE", all_ead_perc)
    print_table("INFORMATION (Q2-Q9) - MODAL MAE", all_mae_info)
    print_table("INFORMATION (Q2-Q9) - EXPECTED ABSOLUTE DISTANCE", all_ead_info)

    # Per-question diagnostics for Edu-Low (perception + first 10 info)
    print("\n" + "=" * 78)
    print("  PER-QUESTION MODAL DISTANCES  (Perception, Edu-Low subgroup)")
    print("=" * 78)
    for model_name, json_paths in JSON_MODELS.items():
        if not json_paths:
            continue
        smaj, _ = load_json_model(json_paths)
        print(f"\n  {model_name}:")
        per_question_distances(smaj["Edu-Low"], human_maj["Edu-Low"], PERC_KEYS)

    print("\n" + "=" * 78)
    print("  PER-QUESTION MODAL DISTANCES  (Information, Edu-Low subgroup, first 10)")
    print("=" * 78)
    for model_name, json_paths in JSON_MODELS.items():
        if not json_paths:
            continue
        smaj, _ = load_json_model(json_paths)
        print(f"\n  {model_name}:")
        per_question_distances(smaj["Edu-Low"], human_maj["Edu-Low"], INFO_KEYS, limit=10)

    # LaTeX tables
    print("\n" + "=" * 78)
    print("  LATEX TABLE - Perception Modal MAE")
    print("=" * 78)
    _print_latex_table(
        all_mae_perc,
        caption=(
            "Distance-based alignment (Perception-based questions Q1, Q10; $n=8$). "
            "Modal MAE between LLM subgroup and corresponding human subgroup. "
            "Ordinal coding A\\,=\\,0\\ldots E\\,=\\,4; lower is better."
        ),
        label="tab:perc-distance-alignment",
    )

    print("\n" + "=" * 78)
    print("  LATEX TABLE - Perception Expected Absolute Distance")
    print("=" * 78)
    _print_latex_table(
        all_ead_perc,
        caption=(
            "Distribution-level alignment (Perception-based questions Q1, Q10; $n=8$). "
            "Expected absolute ordinal distance between LLM subgroup and human subgroup marginals."
        ),
        label="tab:perc-expdist-alignment",
    )

    print("\n" + "=" * 78)
    print("  LATEX TABLE - Information Modal MAE")
    print("=" * 78)
    _print_latex_table(
        all_mae_info,
        caption=(
            "Distance-based alignment (Information-based questions Q2--Q9; $n=32$). "
            "Modal MAE between LLM subgroup and corresponding human subgroup."
        ),
        label="tab:info-distance-alignment",
    )

    print("\n" + "=" * 78)
    print("  LATEX TABLE - Information Expected Absolute Distance")
    print("=" * 78)
    _print_latex_table(
        all_ead_info,
        caption=(
            "Distribution-level alignment (Information-based questions Q2--Q9; $n=32$). "
            "Expected absolute ordinal distance between LLM subgroup and human subgroup marginals."
        ),
        label="tab:info-expdist-alignment",
    )


if __name__ == "__main__":
    main()
