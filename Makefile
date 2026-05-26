# Reproduces every result in this repository.
#
#   make             - run the full pipeline (data -> analysis -> figures)
#   make data        - regenerate the two derived CSVs in data/
#   make analysis    - run all Python analyses and capture text outputs
#   make figures     - rebuild every PNG in results/ from the CSVs
#   make clean       - remove generated CSVs, text logs, and figures

PYTHON  ?= python
RSCRIPT ?= Rscript

DATA_CSVS   = data/responses_long.csv data/ds_edu_distributions.csv
TXT_LOGS    = results/human_response_summary.txt \
              results/alignment_subgroup_summary.txt \
              results/robustness_hs_only.txt \
              results/per_ds_sex_gap.txt \
              results/human_entropy.txt \
              results/human_agreement.txt
AUTO_TEX    = results/auto_alignment_tables.tex \
              results/human_entropy_tables.tex \
              results/human_agreement_tables.tex
PLOTS       = results/violin_plot_overall.png \
              results/violin_plot_by_type.png \
              results/violin_plot_by_subgroup_perception.png \
              results/violin_plot_by_subgroup_information.png \
              results/ds_edu_dumbbell.png \
              results/ds_edu_perception_dist.png \
              results/ds_edu_information_dist.png \
              results/misalignment_worst_questions.png

.PHONY: all data analysis figures clean

all: data analysis figures

# ---------- Stage 1: Python exporters (write to data/) ----------
data: $(DATA_CSVS)

data/responses_long.csv: src/export_responses_long.py src/compute_distance_alignment.py
	cd src && $(PYTHON) export_responses_long.py

data/ds_edu_distributions.csv: src/export_ds_edu_distributions.py src/compute_distance_alignment.py
	cd src && $(PYTHON) export_ds_edu_distributions.py

# ---------- Stage 2: Python analyses ----------
analysis: $(TXT_LOGS) $(AUTO_TEX)

results/human_response_summary.txt: src/summarize_combined_response.py
	cd src && $(PYTHON) summarize_combined_response.py > ../$@

results/alignment_subgroup_summary.txt: src/compute_distance_alignment.py
	cd src && $(PYTHON) compute_distance_alignment.py > ../$@

results/robustness_hs_only.txt: src/compute_robustness_hs_only.py src/compute_distance_alignment.py
	cd src && $(PYTHON) compute_robustness_hs_only.py > ../$@

results/per_ds_sex_gap.txt: src/compute_per_ds_sex_gap.py src/compute_distance_alignment.py
	cd src && $(PYTHON) compute_per_ds_sex_gap.py > ../$@

results/auto_alignment_tables.tex: src/statistical_analysis.py src/compute_distance_alignment.py
	cd src && $(PYTHON) statistical_analysis.py

results/human_entropy.txt results/human_entropy_tables.tex: src/compute_human_entropy.py results/human_response_summary.txt
	cd src && $(PYTHON) compute_human_entropy.py > ../results/human_entropy.txt

results/human_agreement.txt results/human_agreement_tables.tex: src/compute_human_agreement.py src/compute_distance_alignment.py
	cd src && $(PYTHON) compute_human_agreement.py > ../results/human_agreement.txt

# ---------- Stage 3: R figures (depend on CSVs) ----------
figures: data $(PLOTS)

results/violin_plot_overall.png results/violin_plot_by_type.png \
results/violin_plot_by_subgroup_perception.png \
results/violin_plot_by_subgroup_information.png: R/violin_plot.R data/responses_long.csv
	cd R && $(RSCRIPT) violin_plot.R

results/ds_edu_dumbbell.png results/ds_edu_perception_dist.png \
results/ds_edu_information_dist.png: R/ds_edu_distribution_plot.R data/ds_edu_distributions.csv
	cd R && $(RSCRIPT) ds_edu_distribution_plot.R

results/misalignment_worst_questions.png: R/misalignment_plot.R
	cd R && $(RSCRIPT) misalignment_plot.R

# ---------- Cleanup ----------
clean:
	rm -f $(DATA_CSVS) $(TXT_LOGS) $(AUTO_TEX) $(PLOTS)
