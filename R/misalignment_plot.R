# misalignment_plot.R
# Diverging stacked bar chart for the 7 worst-aligned information questions
# (Sonnet 4.5 vs Human), showing paired response distributions.

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
})

# ---------------------------------------------------------------------------
# Data from Table 10 (hardcoded to match exact values; multi-select aware)
# ---------------------------------------------------------------------------
rows <- list(
  list(q="DS2-Q6\nWhen to call doctor", dist=4,
       hA=0.34, hB=0.33, hC=0.00, hD=0.32, hE=0.00,
       lA=0.00, lB=0.00, lC=0.00, lD=0.00, lE=1.00),
  list(q="DS1-Q9\nDischarge plan coverage", dist=2,
       hA=0.19, hB=0.12, hC=0.22, hD=0.22, hE=0.20,
       lA=0.00, lB=0.71, lC=0.00, lD=0.29, lE=0.00),
  list(q="DS2-Q7\nActivities prohibited", dist=2,
       hA=0.00, hB=0.41, hC=0.21, hD=0.37, hE=0.01,
       lA=0.00, lB=0.40, lC=0.11, lD=0.48, lE=0.00),
  list(q="DS2-Q9\nDischarge plan coverage", dist=2,
       hA=0.30, hB=0.22, hC=0.09, hD=0.16, hE=0.11,
       lA=0.00, lB=0.06, lC=0.40, lD=0.37, lE=0.17),
  list(q="DS3-Q5\nOther prescriptions identified", dist=2,
       hA=0.40, hB=0.10, hC=0.50, hD=0.00, hE=0.00,
       lA=1.00, lB=0.00, lC=0.00, lD=0.00, lE=0.00),
  list(q="DS3-Q9\nDischarge plan coverage", dist=2,
       hA=0.20, hB=0.14, hC=0.24, hD=0.10, hE=0.21,
       lA=0.00, lB=0.00, lC=0.39, lD=0.00, lE=0.61),
  list(q="DS4-Q5\nOther prescriptions identified", dist=2,
       hA=0.11, hB=0.09, hC=0.80, hD=0.00, hE=0.00,
       lA=1.00, lB=0.00, lC=0.00, lD=0.00, lE=0.00)
)

# Build long-format data frame
build_df <- function() {
  out <- data.frame(
    question = character(), source = character(),
    response = character(), pct = numeric(),
    stringsAsFactors = FALSE
  )
  for (r in rows) {
    for (resp in c("A","B","C","D","E")) {
      h_val <- r[[paste0("h", resp)]]
      l_val <- r[[paste0("l", resp)]]
      if (h_val > 0) {
        out <- rbind(out, data.frame(
          question = r$q, source = "Human",
          response = resp, pct = h_val,
          stringsAsFactors = FALSE))
      }
      if (l_val > 0) {
        out <- rbind(out, data.frame(
          question = r$q, source = "Sonnet 4.5",
          response = resp, pct = l_val,
          stringsAsFactors = FALSE))
      }
    }
  }
  out
}

df <- build_df()

# Preserve question order (worst first)
q_order <- sapply(rows, function(r) r$q)
df$question <- factor(df$question, levels = rev(q_order))
df$source   <- factor(df$source, levels = c("Human", "Sonnet 4.5"))
df$response <- factor(df$response, levels = c("A","B","C","D","E"))

# Dist labels for annotation
dist_labels <- data.frame(
  question = factor(sapply(rows, function(r) r$q), levels = rev(q_order)),
  dist     = sapply(rows, function(r) r$dist),
  stringsAsFactors = FALSE
)

resp_colors <- c(
  "A" = "#2166AC", "B" = "#67A9CF", "C" = "#D1D1D1",
  "D" = "#EF8A62", "E" = "#B2182B"
)

p <- ggplot(df, aes(x = question, y = pct, fill = response)) +
  geom_bar(stat = "identity", position = "stack", width = 0.7,
           color = "grey30", size = 0.25) +
  facet_wrap(~ source, ncol = 2) +
  scale_y_continuous(
    labels = function(x) paste0(round(x * 100), "%"),
    expand = c(0, 0, 0.02, 0)
  ) +
  scale_fill_manual(values = resp_colors, name = "Response", drop = FALSE) +
  coord_flip() +
  # Add distance annotations on the right margin
  geom_text(
    data = dist_labels,
    aes(x = question, y = 1.05, label = paste0("\u0394=", dist)),
    inherit.aes = FALSE,
    hjust = 0, size = 3, fontface = "bold", color = "grey30"
  ) +
  labs(
    title    = "Worst-aligned information questions: Sonnet 4.5 vs Human",
    subtitle = "Response distributions for questions with modal distance \u2265 2. \u0394 = ordinal gap between modes.",
    x = NULL, y = "Proportion of responses"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor     = element_blank(),
    panel.grid.major.y   = element_blank(),
    strip.background     = element_rect(fill = "grey92", color = NA),
    strip.text           = element_text(face = "bold", size = 11),
    legend.position      = "bottom",
    legend.title         = element_text(face = "bold"),
    plot.title           = element_text(face = "bold", size = 13),
    plot.subtitle        = element_text(color = "grey40", size = 9.5),
    axis.text.y          = element_text(size = 9),
    panel.spacing        = unit(1.2, "lines"),
    plot.margin          = margin(10, 30, 10, 10)
  )

ggsave("../results/misalignment_worst_questions.png", p,
       width = 10, height = 5.5, dpi = 300)

cat("Saved: ../results/misalignment_worst_questions.png\n")
