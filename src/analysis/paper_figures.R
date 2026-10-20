# Figures F1, F2 and F3 for the manuscript.
#
# Every figure is drawn from the SAME analysis frame the confirmatory models
# were fitted on: this script sources e1_checkpoint_cascade.R and
# e2_prompt_manipulation.R into their own environments and calls their
# build_e1_data / build_e2_data rather than re-deriving the outcome. A figure
# that disagreed with the model it illustrates would be worse than no figure.
#
# What is plotted is the OBSERVED rate with a Wilson score interval, not the
# model's marginal probability. The two differ, and the captions say so: the
# emmeans marginals in the text average over the random effects and weight cells
# equally, while these are raw cell proportions. The figures are there to show
# shape, and the tables carry the inferential numbers.
#
# Usage (from repo root):
#   Rscript src/analysis/paper_figures.R --input data/analysis_inputs/phase_c_final \
#       --era data/analysis_inputs/phase_c_final/judgment_era.jsonl \
#       --outdir paper/figures
#
# Dependencies: ggplot2, jsonlite. lme4 and emmeans come along with the sourced
# analysis scripts but no model is fitted here.

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
})

# ---- Constants ----------------------------------------------------------------

CONF_LEVEL <- 0.95
GEN_KEYS <- c("question_id", "condition", "model_id", "trial_id")

# Stage 3 is the final preference-optimized checkpoint of each cascade. It is
# RLHF for the Tulu-3 cascades and Instruct for OLMo-2; the manuscript calls the
# ordinal position RLHF throughout and these labels follow the manuscript.
STAGE_LABELS <- c("Base", "SFT", "DPO", "RLHF")
CELL_LABELS <- c(known_real = "Known-real",
                 obscure_real = "Obscure-but-real",
                 unanswerable = "Unanswerable")
CONDITION_LABELS <- c(neutral = "Neutral",
                      citations_expected = "Citations\nexpected",
                      abstention_permitted = "Abstention\npermitted",
                      length_matched = "Length\nmatched")
# EARLY / DEGRADED / RECOVERED are the judge-drift eras, assigned by position in
# the judge's append-only output stream. The F3 numbers are computed in
# src/analysis/judge_drift_descriptives.py and only read here: they must come
# from the judge's own stream rather than from the corrected verification file,
# which carries citations the verifier-REAL trust step passed through and the
# judge never saw. Duplicating that aggregation in R is how the two would drift.
ERA_LEVELS <- c("EARLY", "DEGRADED", "RECOVERED")
ERA_LABELS <- c(EARLY = "Early", DEGRADED = "Degraded", RECOVERED = "Recovered")
# Stacked bottom to top, so the confirmed share and the fabricated share each
# sit against an axis rather than floating in the middle of the bar.
VERDICT_ORDER <- c("REAL", "UNVERIFIABLE", "FABRICATED")
VERDICT_LABELS <- c(REAL = "Real", UNVERIFIABLE = "Unverifiable",
                    FABRICATED = "Fabricated")
PALETTE_VERDICT <- c(REAL = "#4C72B0", UNVERIFIABLE = "#CCB974",
                     FABRICATED = "#C44E52")

BASE_POST_LABELS <- c("0" = "Base", "1" = "Post-trained")
FIG_DPI <- 300
PALETTE_STAGE <- c(Base = "#4C72B0", `Post-trained` = "#C44E52")

# ---- Binomial interval ---------------------------------------------------------

#' Wilson score interval. Identical to prop.test(k, n, correct = FALSE), which
#' the tests use as an independent oracle. Returns NA bounds for an empty group
#' so a missing cell plots as absent rather than as a spurious zero.
wilson_ci <- function(k, n, conf = CONF_LEVEL) {
  if (n == 0L) return(list(lo = NA_real_, hi = NA_real_))
  z <- stats::qnorm(1 - (1 - conf) / 2)
  p <- k / n
  denom <- 1 + z^2 / n
  center <- (p + z^2 / (2 * n)) / denom
  half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / denom
  list(lo = max(0, center - half), hi = min(1, center + half))
}

#' Aggregate a 0/1 outcome over the grouping columns, attaching Wilson bounds.
#' Empty groups are dropped rather than plotted, so sums over the result
#' conserve the input frame exactly.
.aggregate_rate <- function(data, group_cols, outcome = "fabricated") {
  key <- interaction(data[group_cols], drop = TRUE, sep = "\r")
  n <- tapply(data[[outcome]], key, length)
  k <- tapply(data[[outcome]], key, sum)
  parts <- do.call(rbind, strsplit(names(n), "\r", fixed = TRUE))
  out <- as.data.frame(parts, stringsAsFactors = FALSE)
  names(out) <- group_cols
  out$n <- as.integer(n)
  out$k <- as.integer(k)
  out$rate <- out$k / out$n
  bounds <- Map(wilson_ci, out$k, out$n)
  out$lo <- vapply(bounds, `[[`, numeric(1), "lo")
  out$hi <- vapply(bounds, `[[`, numeric(1), "hi")
  rownames(out) <- NULL
  out
}

# ---- F1: fabrication by stage within cell --------------------------------------

prep_stage_by_cell <- function(data) {
  out <- .aggregate_rate(data, c("cell", "stage_ordinal"))
  out$stage_ordinal <- as.integer(out$stage_ordinal)
  out$cell <- factor(out$cell, levels = names(CELL_LABELS))
  out$stage <- factor(STAGE_LABELS[out$stage_ordinal + 1L], levels = STAGE_LABELS)
  out[order(out$cell, out$stage_ordinal), ]
}

plot_f1 <- function(prepped) {
  ggplot(prepped, aes(x = stage, y = rate, group = 1)) +
    geom_line(linewidth = 0.5, color = "grey35") +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.15, color = "grey35") +
    geom_point(size = 2.2, color = "#4C72B0") +
    facet_wrap(~ cell, nrow = 1,
               labeller = labeller(cell = CELL_LABELS)) +
    scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"),
                       limits = c(0, NA)) +
    labs(x = "Post-training stage", y = "Generations with a fabricated citation") +
    theme_bw(base_size = 10, base_family = "serif") +
    theme(panel.grid.minor = element_blank(),
          strip.background = element_rect(fill = "grey92", color = NA))
}

# ---- F2: fabrication by condition, base against post-trained -------------------

prep_condition_by_stage <- function(data) {
  out <- .aggregate_rate(data, c("condition", "is_post_trained"))
  out$condition <- factor(out$condition, levels = names(CONDITION_LABELS))
  out$is_post_trained <- as.integer(out$is_post_trained)
  out$status <- factor(BASE_POST_LABELS[as.character(out$is_post_trained)],
                       levels = BASE_POST_LABELS)
  out[order(out$condition, out$is_post_trained), ]
}

plot_f2 <- function(prepped) {
  dodge <- position_dodge(width = 0.45)
  ggplot(prepped, aes(x = condition, y = rate, color = status, group = status)) +
    geom_line(position = dodge, linewidth = 0.5, alpha = 0.6) +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.18, position = dodge) +
    geom_point(size = 2.4, position = dodge) +
    scale_x_discrete(labels = CONDITION_LABELS) +
    scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"),
                       limits = c(0, NA)) +
    scale_color_manual(values = PALETTE_STAGE, name = NULL) +
    labs(x = "Prompt condition",
         y = "Generations with a fabricated citation") +
    theme_bw(base_size = 10, base_family = "serif") +
    theme(panel.grid.minor = element_blank(),
          legend.position = "top")
}

# ---- F3: judge grounding and verdicts by era -----------------------------------

#' Read the drift descriptives emitted by judge_drift_descriptives.py.
#' Fails loudly rather than plotting a stale artifact from a previous run.
load_drift_descriptives <- function(path) {
  if (!file.exists(path)) {
    stop(sprintf(paste("drift descriptives not found at %s.",
                       "Run: python -m src.analysis.judge_drift_descriptives"), path))
  }
  fromJSON(path, simplifyVector = TRUE)
}

prep_era_grounding <- function(report) {
  out <- report$grounding
  out$era <- factor(out$era, levels = ERA_LEVELS)
  out[order(out$era), ]
}

prep_era_verdict <- function(report) {
  out <- report$verdict_mix
  out$era <- factor(out$era, levels = ERA_LEVELS)
  present <- intersect(VERDICT_ORDER, unique(out$status))
  out$status <- factor(out$status, levels = present)
  out[order(out$era, out$status), ]
}

GROUNDING_PANELS <- c(mean_effort = "Mean verification effort",
                      zero_effort_pct = "Citations verified with no effort (%)")

#' Both grounding metrics in one figure. Free y scales because the two carry
#' different units, so the panel strip names the unit and the printed value
#' carries the precision.
plot_f3a <- function(prepped) {
  prepped$zero_effort_pct <- 100 * prepped$zero_effort_share
  long <- data.frame(
    era = rep(prepped$era, times = 2L),
    metric = factor(rep(names(GROUNDING_PANELS), each = nrow(prepped)),
                    levels = names(GROUNDING_PANELS)),
    value = c(prepped$mean_effort, prepped$zero_effort_pct),
    label = c(sprintf("%.2f", prepped$mean_effort),
              sprintf("%.2f%%", prepped$zero_effort_pct)),
    stringsAsFactors = FALSE)

  ggplot(long, aes(x = era, y = value)) +
    geom_col(fill = "#4C72B0", width = 0.6) +
    geom_text(aes(label = label), vjust = -0.4, size = 2.9, family = "serif") +
    facet_wrap(~ metric, nrow = 1, scales = "free_y",
               labeller = labeller(metric = GROUNDING_PANELS)) +
    scale_x_discrete(labels = ERA_LABELS) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
    labs(x = "Judge era", y = NULL) +
    theme_bw(base_size = 10, base_family = "serif") +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major.x = element_blank(),
          strip.background = element_rect(fill = "grey92", color = NA))
}

plot_f3b <- function(prepped) {
  ggplot(prepped, aes(x = era, y = share, fill = status)) +
    geom_col(width = 0.6) +
    scale_x_discrete(labels = ERA_LABELS) +
    scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"),
                       expand = expansion(mult = c(0, 0.02))) +
    scale_fill_manual(values = PALETTE_VERDICT, name = NULL,
                      labels = function(x) VERDICT_LABELS[x]) +
    labs(x = "Judge era", y = "Share of judged citations") +
    theme_bw(base_size = 10, base_family = "serif") +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major.x = element_blank(),
          legend.position = "top")
}

# ---- Input loading -------------------------------------------------------------

.source_isolated <- function(path) {
  env <- new.env(parent = globalenv())
  suppressWarnings(suppressMessages(source(path, local = env)))
  env
}

#' Rebuild the confirmatory E1/E2 frame by calling the analysis scripts' own
#' builders. Sourced into separate environments because both scripts define
#' read_jsonl, tally_citations and classify_fabricated, and a shared namespace
#' would let one silently redefine the other's.
load_analysis_frame <- function(input_dir) {
  e1 <- .source_isolated("src/analysis/e1_checkpoint_cascade.R")
  read_jsonl <- e1$read_jsonl
  parts <- lapply(c("generations", "verifications", "questions", "models"),
                  function(nm) read_jsonl(file.path(input_dir, paste0(nm, ".jsonl"))))
  names(parts) <- c("generations", "verifications", "questions", "models")
  data <- e1$build_e1_data(parts$generations, parts$verifications,
                           parts$questions, parts$models)
  data$condition <- factor(data$condition, levels = names(CONDITION_LABELS))
  list(data = data, verifications = parts$verifications)
}

# ---- Main ----------------------------------------------------------------------

parse_args <- function(args) {
  out <- list(input = "data/analysis_inputs/phase_c_final",
              drift = "data/analysis/phase_c/judge_drift_descriptives.json",
              outdir = "paper/figures")
  i <- 1L
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    if (!key %in% names(out)) stop(sprintf("Unknown argument: %s", args[i]))
    out[[key]] <- args[i + 1L]
    i <- i + 2L
  }
  out
}

.save <- function(plot, path, width, height) {
  ggsave(path, plot, width = width, height = height, units = "in",
         device = "pdf", dpi = FIG_DPI)
  cat(sprintf("  wrote %s\n", path))
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  dir.create(args$outdir, recursive = TRUE, showWarnings = FALSE)

  cat(sprintf("Loading analysis frame from %s ...\n", args$input))
  loaded <- load_analysis_frame(args$input)
  data <- loaded$data
  cat(sprintf("Analysis frame: %d rows, %d models, %d questions\n",
              nrow(data), nlevels(data$model_id), nlevels(data$question_id)))

  f1 <- prep_stage_by_cell(data)
  f2 <- prep_condition_by_stage(data)
  .save(plot_f1(f1), file.path(args$outdir, "f1_stage_by_cell.pdf"), 6.5, 2.6)
  .save(plot_f2(f2), file.path(args$outdir, "f2_condition_by_stage.pdf"), 5.2, 3.4)

  cat(sprintf("Loading judge-drift descriptives from %s ...\n", args$drift))
  report <- load_drift_descriptives(args$drift)
  cat(sprintf("Judge stream: %d judged citations\n", report$n_judged_citations))
  f3a <- prep_era_grounding(report)
  f3b <- prep_era_verdict(report)
  .save(plot_f3a(f3a), file.path(args$outdir, "f3a_era_grounding.pdf"), 4.6, 2.8)
  .save(plot_f3b(f3b), file.path(args$outdir, "f3b_era_verdict.pdf"), 3.2, 3.0)

  summary_path <- file.path(args$outdir, "figure_data.json")
  writeLines(toJSON(list(f1 = f1, f2 = f2, f3_grounding = f3a, f3_verdict = f3b),
                    auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"),
             summary_path)
  cat(sprintf("  wrote %s\n", summary_path))
  invisible(list(f1 = f1, f2 = f2, f3a = f3a, f3b = f3b))
}

if (sys.nframe() == 0L) main()
