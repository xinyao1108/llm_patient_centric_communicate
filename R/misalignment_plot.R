# misalignment_plot.R
# Diverging stacked bar chart for the 7 worst-aligned information questions
# (Sonnet 4.5 vs Human), showing paired response distributions.

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
})

# ---------------------------------------------------------------------------
# Data from Table 10 (hardcoded to match exact values)
# ---------------------------------------------------------------------------
rows <- list(
  list(q="DS2-Q7\nMedications", dist=3,
       hA=0.33, hB=0, hC=0, hD=0.33, hE=0.33,
       lA=0, lB=0.40, lC=0.11, lD=0.48, lE=0),
  list(q="DS4-Q7\nMedications", dist=3,
       hA=0.54, hB=0, hC=0.41, hD=0, hE=0.05,
       lA=0, lB=0, lC=0.35, lD=0.65, lE=0),
  list(q="DS1-Q5\nTest results", dist=2,
       hA=1.00, hB=0, hC=0, hD=0, hE=0,
       lA=0, lB=0.08, lC=0.92, lD=0, lE=0),
  list(q="DS3-Q5\nTest results", dist=2,
       hA=0.45, hB=0, hC=0.55, hD=0, hE=0,
       lA=1.00, lB=0, lC=0, lD=0, lE=0),
  list(q="DS4-Q5\nTest results", dist=2,
       hA=0.12, hB=0, hC=0.88, hD=0, hE=0,
       lA=1.00, lB=0, lC=0, lD=0, lE=0),
  list(q="DS1-Q8\nFollow-up", dist=2,
       hA=0.40, hB=0.40, hC=0, hD=0.20, hE=0,
       lA=0, lB=0, lC=0.35, lD=0.65, lE=0),
  list(q="DS3-Q8\nFollow-up", dist=2,
       hA=0, hB=0.58, hC=0.42, hD=0, hE=0,
       lA=0, lB=0.13, lC=0.08, lD=0.79, lE=0)
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
