# LLM Persona Alignment with Human Responses on Discharge Summaries

Reproducibility package for evaluating how four large language models (Sonnet 4.5,
Opus 4.6, GPT-5.2, GPT-4.1) simulate patient personas when answering discharge
summary comprehension questions, against human survey responses.

## Repository layout

```
.
├── README.md
├── data/
│   ├── human_response.xlsx                Raw human survey responses (88 valid respondents)
│   ├── mapping.txt                        Response-text -> letter (A..G) mapping (RTF)
│   ├── responses_long.csv                 Long-format pooled responses (generated)
│   ├── ds_edu_distributions.csv           Per-question response distributions (generated)
│   └── add_exp_results/                   LLM iteration outputs (4 models x 10 iters = 40 JSONs)
├── src/                                   Python: data prep + statistical analysis
│   ├── summarize_combined_response.py     Result 1 - human distributions by gender/education/highest_education
│   ├── compute_distance_alignment.py      Per-subgroup Modal MAE and EAD
│   ├── statistical_analysis.py            Stability / KS / ANOVA / Tukey + LaTeX tables
│   ├── compute_robustness_hs_only.py      Result 2 - HS-or-Less vs merged Low robustness check
│   ├── compute_per_ds_sex_gap.py          Result 8 - per-DS sex gap on perception modal MAE
│   ├── export_responses_long.py           Builds data/responses_long.csv (for violin plots)
│   └── export_ds_edu_distributions.py     Builds data/ds_edu_distributions.csv (for Figs 1-3)
├── R/                                     R: figure rendering
│   ├── violin_plot.R                      Figs: violin_plot_overall.png, violin_plot_by_type.png
│   ├── ds_edu_distribution_plot.R         Figs 1-3: dumbbell + two stacked-bar charts
│   └── misalignment_plot.R                Fig 4: misalignment_worst_questions.png (data hardcoded)
└── results/                               Final tables and figures
    ├── summary_results.tex                Curated main results (Tables 1-10 + narrative)
    ├── auto_alignment_tables.tex          Auto-regenerated alignment + stat tables (output of statistical_analysis.py)
    ├── other_results.tex                  Stability / KS / ANOVA / Tukey supplementary tables
    ├── violin_plot_overall.png
    ├── violin_plot_by_type.png
    ├── ds_edu_dumbbell.png
    ├── ds_edu_perception_dist.png
    ├── ds_edu_information_dist.png
    └── misalignment_worst_questions.png
```

## Question structure

The human survey covers four discharge summaries (DS1-DS4), each with 10 questions:

- **Perception** questions: Q1, Q10 within each DS (8 keys total)
- **Information** questions: Q2-Q9 within each DS (32 keys total)

Responses use a 5-point ordinal scale `A=0, B=1, C=2, D=3, E=4`
(some questions allow `F` or `G` for "other / don't know" - excluded from ordinal metrics).

## Subgroups

Seven matched subgroups are evaluated on both human and LLM data:

`Edu-Low`, `Edu-High`, `Male`, `Female`, `OutLow`, `OutHigh`, `ERLow`

Human visit-frequency classification: `weekly|monthly -> high`, `yearly|never -> low`.

## Quick reproduction with `make`

```bash
make                # data -> analysis -> figures (full pipeline)
make data           # regenerate the two derived CSVs in data/
make analysis       # run Python analyses, write results/*.txt and auto_alignment_tables.tex
make figures        # rebuild every PNG in results/ from the CSVs
make clean          # remove everything that `make` produces
```

`make` is idempotent; re-running it only redoes work whose inputs changed.

## Reproducing every reported result

### Result 1 - Human response distribution by gender and education

```bash
cd src && python summarize_combined_response.py
```

Reports overall response counts per gender and education subgroup; output written
to stdout (redirect to a file if you want a transcript).

### Result 2 - Robustness check: High-School-Only vs merged Low

```bash
cd src && python compute_robustness_hs_only.py
```

Compares the merged `Edu-Low` subgroup (Q58 == "low", n=49) against the
stricter "High School or Less" subset (Q52 in {No post-high school degree,
Some High School, High School Graduate}, n=36). Reports:

- Per-question majority differences (DS3-Q5 and DS4-Q7 flip; 2/37 questions)
- Within-group pairwise agreement rates for both definitions
- LLM alignment rates for each model under both human reference definitions

LaTeX rows for `tab:edu-robust` and `tab:edu-robust-align` are printed at
the bottom of the output, ready to paste into the manuscript.

### Result 3 - Alignment between LLM and human responses

3a/3b - violin plots:

```bash
cd src && python export_responses_long.py        # builds data/responses_long.csv
cd ../R && Rscript violin_plot.R                 # renders both violin PNGs
```

3c/3d - Modal MAE and EAD tables (Tables 1-4 in summary_results.tex):

```bash
cd src && python compute_distance_alignment.py   # prints subgroup tables
cd src && python statistical_analysis.py         # writes results/auto_alignment_tables.tex
```

`auto_alignment_tables.tex` contains the freshly-computed Tables 1-4 (Modal MAE
and EAD, perception/information) plus the supplementary stability, Shapiro,
KS, ANOVA, and Tukey tables. The curated `summary_results.tex` also includes
these plus hand-typed Tables 5-8 (equity gaps, per-DS gaps) and the narrative
discussion; running `statistical_analysis.py` does NOT overwrite it.

### Result 4 - Education and gender equity gaps

Tables 5 and 6 of `summary_results.tex` are currently typeset by hand from
the per-subgroup outputs of `compute_distance_alignment.py`. To verify:
the table cell at (model, metric) is `subgroup_low_value - subgroup_high_value`
or `male_value - female_value`.

### Result 5 - DS-level education-gap dumbbell (Sonnet 4.5)

```bash
cd src && python export_ds_edu_distributions.py  # builds data/ds_edu_distributions.csv
cd ../R && Rscript ds_edu_distribution_plot.R    # writes all three Edu figures
```

This single R run produces Figures 1, 2, and 3:

- `ds_edu_dumbbell.png`
- `ds_edu_perception_dist.png`
- `ds_edu_information_dist.png`

### Results 6, 7 - Education-stratified stacked bars (Sonnet 4.5 vs Human)

Same command as Result 5; the R script produces all three figures in one call.

### Result 8 - Per-DS sex gap on perception modal MAE

```bash
cd src && python compute_per_ds_sex_gap.py
```

For each of the four LLMs and each DS, computes the modal MAE on the two
perception questions (Q1, Q10) separately for Male and Female personas
against the matched human Male/Female majority, then reports
`gap = MAE_Male - MAE_Female` per DS. LaTeX rows for `tab:ds-sex-gap-perc`
are printed at the bottom of the output. The "Overall" row reproduces the
Sex column of Table 6 (perception modal MAE gap) by construction.

### Result 9 - Worst-aligned information questions (Sonnet 4.5)

```bash
cd R && Rscript misalignment_plot.R
```

The 7 worst-aligned distributions are hardcoded in `misalignment_plot.R` and
mirror Table 10 (`tab:misalignment-detail`) of `summary_results.tex`.

## Dependencies

**Python** (>=3.9): `numpy`, `pandas`, `scipy`, `statsmodels`, `openpyxl`,
  `matplotlib`, `seaborn`.

**R**: `ggplot2`, `dplyr`, `readr`, `scales`.

## Notes on data integrity

- The `add_exp_results/` directory contains only the four models used in the
  analysis (Sonnet 4.5, Opus 4.6, GPT-5.2, GPT-4.1), ten iterations each.
- Frequency classification (`weekly/monthly -> high`, `yearly/never -> low`)
  is applied identically to human and model personas to ensure matched subgroups.
- `Edu-Low`/`Edu-High` for humans uses the binarized `Q58` column in the xlsx;
  for models, the `persona.education` field already takes those two values.
