# ds_edu_distribution_plot.R
# Paired stacked-bar + dumbbell chart: Sonnet 4.5 vs Human
# response distributions by education level and DS type.
#
# Uses base R aggregation (compatible with older dplyr).

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
})

df <- read.csv("../data/ds_edu_distributions.csv", stringsAsFactors = FALSE)

ds_labels <- c(
  "1" = "DS1: Short & Difficult",
  "2" = "DS2: Long & Simple",
  "3" = "DS3: Long & Difficult",
  "4" = "DS4: Short & Simple"
)
df$ds_label   <- factor(ds_labels[as.character(df$ds)], levels = ds_labels)
df$source     <- factor(df$source, levels = c("Human", "Sonnet 4.5"))
df$education  <- factor(df$education, levels = c("Edu-Low", "Edu-High"))
df$response   <- factor(df$response, levels = c("A", "B", "C", "D", "E"))

resp_colors <- c(
  "A" = "#2166AC", "B" = "#67A9CF", "C" = "#D1D1D1",
  "D" = "#EF8A62", "E" = "#B2182B"
)

base_theme <- theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    strip.background   = element_rect(fill = "grey92", color = NA),
    strip.text         = element_text(face = "bold", size = 9),
    legend.position    = "bottom",
    legend.title       = element_text(face = "bold"),
    plot.title         = element_text(face = "bold", size = 13),
    plot.subtitle      = element_text(color = "grey40", size = 10),
    axis.text.y        = element_text(size = 9)
  )

# ---------------------------------------------------------------------------
# Aggregate using base R
# ---------------------------------------------------------------------------
agg_counts <- function(data) {
  counts <- aggregate(
    list(n = data$response),
    by = list(
      ds_label      = data$ds_label,
      education     = data$education,
      source        = data$source,
      question_type = data$question_type,
      response      = data$response
    ),
    FUN = length
  )
  # compute group totals
  totals <- aggregate(
    n ~ ds_label + education + source + question_type,
    data = counts, FUN = sum
  )
  names(totals)[names(totals) == "n"] <- "total"
  merged <- merge(counts, totals)
  merged$pct <- merged$n / merged$total
  merged
}

agg <- agg_counts(df)

# ---------------------------------------------------------------------------
# Stacked bar chart
# ---------------------------------------------------------------------------
make_stacked_bar <- function(qt, title_suffix) {
  d <- agg[agg$question_type == qt, ]

  ggplot(d, aes(x = source, y = pct, fill = response)) +
    geom_bar(stat = "identity", position = "stack", width = 0.75,
             color = "grey30", size = 0.2) +
    facet_grid(ds_label ~ education, switch = "y") +
    scale_y_continuous(labels = function(x) paste0(round(x*100), "%"),
                       expand = c(0, 0, 0.02, 0)) +
    scale_fill_manual(values = resp_colors, name = "Response",
                      drop = FALSE) +
    coord_flip() +
    labs(
      title    = paste0("Sonnet 4.5 vs Human \u2014 ", title_suffix),
      subtitle = "Response distribution by discharge summary type and education level",
      x = NULL, y = "Proportion of responses"
    ) +
    base_theme +
    theme(
      strip.placement  = "outside",
      panel.spacing.x  = unit(0.8, "lines"),
      panel.spacing.y  = unit(0.4, "lines")
    )
}

p_perc <- make_stacked_bar("Perception", "Perception Questions (Q1, Q10)")
ggsave("../results/ds_edu_perception_dist.png", p_perc,
       width = 8, height = 7, dpi = 300)

p_info <- make_stacked_bar("Information", "Information Questions (Q2\u2013Q9)")
ggsave("../results/ds_edu_information_dist.png", p_info,
       width = 8, height = 7, dpi = 300)

# ---------------------------------------------------------------------------
# Dumbbell chart
# ---------------------------------------------------------------------------
ord_map <- c("A" = 0, "B" = 1, "C" = 2, "D" = 3, "E" = 4)

get_mode <- function(x) {
  tt <- table(x)
  names(tt)[which.max(tt)]
}

# Mode per (source, education, ds, question_num, question_type)
mode_df <- aggregate(
  response ~ source + education + ds + ds_label + question_type + question_num,
  data = df, FUN = get_mode
)
names(mode_df)[names(mode_df) == "response"] <- "mode_resp"
mode_df$mode_ord <- ord_map[mode_df$mode_resp]

# Separate Human and LLM
human_modes <- mode_df[mode_df$source == "Human",
                       c("education","ds","ds_label","question_type","question_num","mode_ord")]
names(human_modes)[names(human_modes) == "mode_ord"] <- "human_ord"

llm_modes <- mode_df[mode_df$source == "Sonnet 4.5",
                     c("education","ds","ds_label","question_type","question_num","mode_ord")]
names(llm_modes)[names(llm_modes) == "mode_ord"] <- "llm_ord"

merged_modes <- merge(human_modes, llm_modes)
merged_modes$abs_diff <- abs(merged_modes$llm_ord - merged_modes$human_ord)

# MAE per (education, ds, question_type)
mae_df <- aggregate(
  abs_diff ~ education + ds + ds_label + question_type,
  data = merged_modes, FUN = mean
)
names(mae_df)[names(mae_df) == "abs_diff"] <- "mae"

# Pivot education
mae_lo <- mae_df[mae_df$education == "Edu-Low", c("ds","ds_label","question_type","mae")]
names(mae_lo)[names(mae_lo) == "mae"] <- "Edu_Low"
mae_hi <- mae_df[mae_df$education == "Edu-High", c("ds","ds_label","question_type","mae")]
names(mae_hi)[names(mae_hi) == "mae"] <- "Edu_High"

mae_wide <- merge(mae_lo, mae_hi)
mae_wide$gap <- mae_wide$Edu_Low - mae_wide$Edu_High
mae_wide$question_type <- factor(mae_wide$question_type,
                                  levels = c("Perception", "Information"))

p_dumbbell <- ggplot(mae_wide, aes(y = reorder(ds_label, -ds))) +
  geom_segment(aes(x = Edu_High, xend = Edu_Low,
                    yend = reorder(ds_label, -ds)),
               color = "grey50", size = 1.5) +
  geom_point(aes(x = Edu_High), color = "#2166AC", size = 3.5) +
  geom_point(aes(x = Edu_Low),  color = "#B2182B", size = 3.5) +
  geom_text(aes(x = (Edu_Low + Edu_High) / 2,
                label = sprintf("%+.2f", sign(gap) * round(abs(gap) + 1e-9, 2))),
            vjust = -0.8, size = 3.2, fontface = "bold") +
  facet_wrap(~ question_type, ncol = 2, scales = "free_x") +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.15))) +
  labs(
    title    = "Education gap by discharge summary type (Sonnet 4.5)",
    subtitle = paste0(
      "\u25CF Edu-High (blue)    \u25CF Edu-Low (red)    ",
      "Gap = Low \u2212 High (positive = worse for low-edu)"),
    x = "Modal MAE (lower = better alignment with humans)",
    y = NULL
  ) +
  base_theme +
  theme(
    panel.grid.major.y = element_blank(),
    strip.text = element_text(size = 11)
  )

ggsave("../results/ds_edu_dumbbell.png", p_dumbbell,
       width = 9, height = 4.5, dpi = 300)

cat("\nPlots saved:\n")
cat("  ds_edu_perception_dist.png   - stacked bar (perception)\n")
cat("  ds_edu_information_dist.png  - stacked bar (information)\n")
cat("  ds_edu_dumbbell.png          - dumbbell chart (education gap)\n")
