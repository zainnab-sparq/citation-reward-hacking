# Per-domain, per-cell, per-model and per-condition breakdowns for appendix B.
#
# These are DESCRIPTIVE rates on the confirmatory frame, not model output. They
# exist so a reader can see where the pooled effects come from and, in
# particular, can check the coverage caveat for themselves.
#
# The coverage caveat is the reason the verification-status crosstab is here
# alongside the fabrication rates. Legal citations are verified through a
# different route than academic and medical ones and that route is gappier, so
# an all-domain rate can be moved by how UNVERIFIABLE is handled rather than by
# how often models fabricate. Phase B recorded the model-based form of this
# warning; what appears here is the Phase C descriptive form, computed on Phase
# C data, because the Phase B odds ratios describe a different frame.
#
# Usage (from repo root):
#   Rscript src/analysis/paper_breakdowns.R \
#       --input data/analysis_inputs/phase_c_final \
#       --output data/analysis/phase_c/paper_breakdowns.json

suppressPackageStartupMessages({ library(jsonlite) })

.figures_env <- function() {
  env <- new.env(parent = globalenv())
  suppressWarnings(suppressMessages(source("src/analysis/paper_figures.R", local = env)))
  env
}

STATUS_ORDER <- c("REAL_CONFIRMED", "EXISTENT_METADATA_ERROR", "EXISTENT_WRONG_ID",
                  "MISGROUNDED", "FABRICATED", "UNVERIFIABLE")

#' Verification-status mix per domain, over judged citations rather than
#' generations. This is the table that shows whether a domain's coverage is
#' comparable to the others'.
status_mix_by_domain <- function(verifications, questions) {
  joined <- merge(verifications, questions[, c("id", "domain")],
                  by.x = "question_id", by.y = "id")
  tab <- table(joined$domain, joined$status)
  domains <- rownames(tab)
  present <- c(intersect(STATUS_ORDER, colnames(tab)),
               setdiff(colnames(tab), STATUS_ORDER))
  out <- list()
  for (d in domains) {
    total <- sum(tab[d, ])
    row <- list(domain = d, n_citations = as.integer(total))
    for (s in present) {
      row[[paste0("n_", tolower(s))]] <- as.integer(tab[d, s])
      row[[paste0("share_", tolower(s))]] <- as.numeric(tab[d, s]) / total
    }
    out[[length(out) + 1L]] <- row
  }
  out
}

#' Mean citations emitted per generation. Kept separate from the fabrication
#' rates because a domain can look worse simply by citing more.
citations_per_generation <- function(data, group_cols) {
  agg <- aggregate(data$n_citations, by = data[group_cols], FUN = mean)
  names(agg)[ncol(agg)] <- "mean_citations"
  agg$n_generations <- aggregate(data$n_citations, by = data[group_cols],
                                 FUN = length)$x
  agg[do.call(order, agg[group_cols]), ]
}

parse_args <- function(args) {
  out <- list(input = "data/analysis_inputs/phase_c_final",
              output = "data/analysis/phase_c/paper_breakdowns.json")
  i <- 1L
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    if (!key %in% names(out)) stop(sprintf("Unknown argument: %s", args[i]))
    out[[key]] <- args[i + 1L]
    i <- i + 2L
  }
  out
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  pf <- .figures_env()

  cat(sprintf("Loading %s ...\n", args$input))
  loaded <- pf$load_analysis_frame(args$input)
  data <- loaded$data
  questions <- pf$.source_isolated("src/analysis/e1_checkpoint_cascade.R")$read_jsonl(
    file.path(args$input, "questions.jsonl"))
  cat(sprintf("Frame: %d rows\n", nrow(data)))

  out <- list(
    n_observations = nrow(data),
    by_domain_stage = pf$.aggregate_rate(data, c("domain", "stage_ordinal")),
    by_cell_condition = pf$.aggregate_rate(data, c("cell", "condition")),
    by_domain_condition = pf$.aggregate_rate(data, c("domain", "condition")),
    by_model = pf$.aggregate_rate(data, c("family", "model_id", "stage_ordinal")),
    by_model_condition = pf$.aggregate_rate(data, c("model_id", "condition")),
    status_mix_by_domain = status_mix_by_domain(loaded$verifications, questions),
    citations_per_generation = citations_per_generation(data, c("domain", "condition"))
  )

  dir.create(dirname(args$output), recursive = TRUE, showWarnings = FALSE)
  writeLines(toJSON(out, auto_unbox = TRUE, pretty = TRUE, na = "null", digits = NA),
             args$output)
  cat(sprintf("Wrote %s\n", args$output))

  cat("\nUNVERIFIABLE share by domain (the coverage caveat):\n")
  for (row in out$status_mix_by_domain) {
    share <- row[["share_unverifiable"]]
    if (!is.null(share)) {
      cat(sprintf("  %-10s %6.2f%%  of %d judged citations\n",
                  row$domain, 100 * share, row$n_citations))
    }
  }
  invisible(out)
}

if (sys.nframe() == 0L) main()
