# violin_plot.R
# Visualize density and variance of human + model responses.
#
# Input : responses_long.csv (produced by export_responses_long.py)
# Output: violin_plot_overall.png
#         violin_plot_by_type.png
#         violin_plot_by_subgroup_perception.png
#         violin_plot_by_subgroup_information.png
#
# Required packages: ggplot2, dplyr, readr, scales
#   install.packages(c("ggplot2", "dplyr", "readr", "scales"))

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(scales)
})

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df <- read_csv(
  "../data/responses_long.csv",
  col_types = cols(
    source        = col_character(),
    question_type = col_character(),
    subgroup      = col_character(),
    ds            = col_integer(),
    question_num  = col_integer(),
    response      = col_double()
  )
)

# Order sources so Human is first, then models
source_levels <- c("Human", "Sonnet 4.5", "Opus 4.6", "GPT-5.2", "GPT-4.1")
df$source <- factor(df$source, levels = source_levels)
df$question_type <- factor(df$question_type, levels = c("Perception", "Information"))

subgroup_levels <- c("Edu-Low", "Edu-High", "Male", "Female",
                     "OutLow", "OutHigh", "ERLow")
df$subgroup <- factor(df$subgroup, levels = subgroup_levels)

# Color palette: Human in dark grey, four models in distinct hues
source_colors <- c(
  "Human"      = "#444444",
  "Sonnet 4.5" = "#1f77b4",
  "Opus 4.6"   = "#2ca02c",
  "GPT-5.2"    = "#d62728",
  "GPT-4.1"    = "#9467bd"
)

ordinal_breaks <- 0:4
ordinal_labels <- c("A", "B", "C", "D", "E")

base_theme <- theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    legend.position    = "none",
    strip.background   = element_rect(fill = "grey92", color = NA),
    strip.text         = element_text(face = "bold"),
    plot.title         = element_text(face = "bold", size = 14),
    plot.subtitle      = element_text(color = "grey40")
  )

# ---------------------------------------------------------------------------
# 2. Overall violin (all questions, all subgroups pooled)
# ---------------------------------------------------------------------------
p_overall <- ggplot(df, aes(x = source, y = response, fill = source)) +
  geom_violin(trim = FALSE, scale = "area", alpha = 0.75,
              color = "grey20", size = 0.4) +
  geom_boxplot(width = 0.12, outlier.shape = NA, fill = "white",
               color = "grey20", size = 0.4) +
  stat_summary(fun = mean, geom = "point", shape = 23,
               size = 2.5, fill = "white", color = "black") +
  scale_y_continuous(breaks = ordinal_breaks, labels = ordinal_labels,
                     limits = c(-0.3, 4.3)) +
  scale_fill_manual(values = source_colors) +
  labs(
    title    = "Response distribution: Human vs. LLM persona simulators",
    subtitle = "Pooled across all subgroups, discharge summaries, and questions",
    x = NULL, y = "Response (ordinal)"
  ) +
  base_theme

ggsave("../results/violin_plot_overall.png", p_overall,
       width = 7, height = 4.5, dpi = 300)

# ---------------------------------------------------------------------------
# 3. Faceted by question type (Perception vs Information)
# ---------------------------------------------------------------------------
p_by_type <- ggplot(df, aes(x = source, y = response, fill = source)) +
  geom_violin(trim = FALSE, scale = "area", alpha = 0.75,
              color = "grey20", size = 0.4) +
  geom_boxplot(width = 0.12, outlier.shape = NA, fill = "white",
               color = "grey20", size = 0.4) +
  stat_summary(fun = mean, geom = "point", shape = 23,
               size = 2.2, fill = "white", color = "black") +
  facet_wrap(~ question_type, ncol = 2) +
  scale_y_continuous(breaks = ordinal_breaks, labels = ordinal_labels,
                     limits = c(-0.3, 4.3)) +
  scale_fill_manual(values = source_colors) +
  labs(
    title    = "Response distribution by question type",
    subtitle = "Perception (Q1, Q10)  vs.  Information (Q2-Q9)",
    x = NULL, y = "Response (ordinal)"
  ) +
  base_theme

ggsave("../results/violin_plot_by_type.png", p_by_type,
       width = 9, height = 4.5, dpi = 300)

# ---------------------------------------------------------------------------
# 4. Subgroup breakdown per question type
# ---------------------------------------------------------------------------
make_subgroup_plot <- function(qt) {
  d <- df[df$question_type == qt, ]
  ggplot(d, aes(x = source, y = response, fill = source)) +
    geom_violin(trim = FALSE, scale = "area", alpha = 0.75,
                color = "grey20", size = 0.3) +
    geom_boxplot(width = 0.13, outlier.shape = NA, fill = "white",
                 color = "grey20", size = 0.3) +
    stat_summary(fun = mean, geom = "point", shape = 23,
                 size = 1.8, fill = "white", color = "black") +
    facet_wrap(~ subgroup, ncol = 4) +
    scale_y_continuous(breaks = ordinal_breaks, labels = ordinal_labels,
                       limits = c(-0.3, 4.3)) +
    scale_fill_manual(values = source_colors) +
    labs(
      title    = sprintf("Response distribution by subgroup - %s questions", qt),
      subtitle = "Each panel: matched human subgroup vs. corresponding LLM subgroup",
      x = NULL, y = "Response (ordinal)"
    ) +
    base_theme +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 9))
}

p_perc <- make_subgroup_plot("Perception")
ggsave("../results/violin_plot_by_subgroup_perception.png", p_perc,
       width = 12, height = 6.5, dpi = 300)

p_info <- make_subgroup_plot("Information")
ggsave("../results/violin_plot_by_subgroup_information.png", p_info,
       width = 12, height = 6.5, dpi = 300)

# ---------------------------------------------------------------------------
# 5. Console summary: per-source descriptive stats
# ---------------------------------------------------------------------------
summary_tbl <- aggregate(
  response ~ source + question_type,
  data = df,
  FUN = function(x) c(
    n      = length(x),
    mean   = mean(x, na.rm = TRUE),
    median = median(x, na.rm = TRUE),
    sd     = sd(x, na.rm = TRUE),
    iqr    = IQR(x, na.rm = TRUE)
  )
)
summary_tbl <- cbind(
  summary_tbl[, c("source", "question_type")],
  as.data.frame(summary_tbl$response)
)

cat("\nDescriptive statistics (Mean / SD / IQR) per source x question type:\n")
print(as.data.frame(summary_tbl), row.names = FALSE, digits = 3)

cat("\nPlots saved:\n")
cat("  violin_plot_overall.png\n")
cat("  violin_plot_by_type.png\n")
cat("  violin_plot_by_subgroup_perception.png\n")
cat("  violin_plot_by_subgroup_information.png\n")
