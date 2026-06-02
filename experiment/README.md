# LLM Experiment Generation

This directory contains the upstream code that produces the per-model JSON
iteration files consumed by the analysis pipeline (`data/add_exp_results/*.json`).
It is included here for full reproducibility; the analysis pipeline itself does
not depend on running this code.

## Files

| File | Purpose |
|---|---|
| `experiment_config.json` | Persona grid, discharge-summary IDs, question IDs, model list, iteration counts. |
| `persona_discharge_query.py` | Query engine: builds persona prompts, dispatches to OpenAI / Anthropic, parses responses. |
| `run_experiment.py` | Driver that loads the config, enumerates persona × DS × question × iteration, and writes one JSON per (model, iteration). |

## Persona grid

The configured grid is 3 ages × 2 genders × 2 education levels × 4 ethnicities
× 2 doctor-visit frequencies × 2 ED-visit frequencies = **192 personas** per
iteration, each answering 10 questions on each of 4 discharge summaries =
7,680 queries per model-iteration. The analysis pipeline expects 10 iterations
per model.

## Running

```bash
pip install -r requirements.txt          # see upstream repo
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
python run_experiment.py experiment_config.json
```

Output files are written as `results/<model>_iter<N>.json`. Copy or symlink
them into `../data/add_exp_results/` so the analysis pipeline can find them.

## Per-DS question variants

Some Q-numbers reuse the same number across discharge summaries but the
underlying question content varies by clinical case. The question dictionary
in `persona_discharge_query.py` is therefore structured as
`{Q-id: {DS-id: text, "default": text}}` with a per-DS override mechanism
(`get_question_text(q_id, ds_id)`). The variants currently encoded are:

- **Q6** uses the diagnosis-identification stem on DS1, DS3, DS4 and the
  "when to call the doctor" stem on DS2 (six options including E = "All of
  the above").
- **Q7** uses the 4-option treatment stem on DS1 and DS4, a 6-option
  "activities prohibited" multi-select on DS2, and a 5-option orthopedic
  variant on DS3.

## Provenance

Source repository:
[xinyao1108/llm-persona-discharge-study](https://github.com/xinyao1108/llm-persona-discharge-study).
The vendored copy of `persona_discharge_query.py` differs from upstream in
that the question dictionary has been restructured to accept per-DS overrides
(see "Per-DS question variants" above). The other two files are unchanged
copies; refer to the upstream repository for the full development history
and the discharge-summary text content.
