# rePCA and singularity diagnostics for the Phase C extended random-effects fits.
#
# Why this exists separately from phase_b_repca.R: the Phase B maximal model has
# ONE grouping factor, so reporting names(rePCA(model))[1] covers it. Every
# Phase C extended-RE model has three, (1 + ... | family), (1 + ... | model_id)
# and (1 | question_id). Reporting only the first would describe one variance
# component while reading as though it covered the model.
#
# This matters because all four Phase C extended-RE fits report isSingular() as
# TRUE, and the appendix has to say WHERE the deficiency is. A singular flag on
# a three-factor model does not tell the reader whether the condition slopes are
# unsupported or whether two correlations are simply at the boundary.
#
# The E1 and E2 models are refitted here by calling the confirmatory scripts'
# own fit_*_extended_model, so the diagnosed model is the reported model.
#
# Usage (from repo root):
#   Rscript src/analysis/phase_c_repca.R \
#       --input data/analysis_inputs/phase_c_final \
#       --output data/analysis/phase_c/repca_extended.json

suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})

# A rotated-space SD at or below this fraction of its factor's leading component
# is an unsupported dimension. 1e-3 is the conventional rePCA read and matches
# phase_b_repca.R exactly; it is fixed here rather than tuned to the result.
NEGLIGIBLE_REL_SD <- 1e-3

.count_unsupported <- function(relative_sdev) {
  sum(relative_sdev <= NEGLIGIBLE_REL_SD)
}

#' One rePCA block per grouping factor, not just the first.
repca_blocks <- function(model) {
  pca <- lme4::rePCA(model)
  varcor <- lme4::VarCorr(model)
  singular <- lme4::isSingular(model)
  lapply(names(pca), function(term) {
    sdev <- as.numeric(pca[[term]]$sdev)
    rel <- if (max(sdev) > 0) sdev / max(sdev) else sdev
    re_sd <- sqrt(diag(varcor[[term]]))
    list(
      grouping_factor = term,
      n_dimensions = length(sdev),
      sdev = round(sdev, 6),
      relative_sdev = round(rel, 6),
      cumulative_variance = round(cumsum(sdev^2) / sum(sdev^2), 6),
      n_unsupported_dimensions = .count_unsupported(rel),
      model_is_singular = singular,
      re_sd_by_term = round(re_sd, 6),
      largest_re_term = names(which.max(re_sd)),
      smallest_re_term = names(which.min(re_sd))
    )
  })
}

.print_blocks <- function(label, blocks) {
  cat(sprintf("\n=== %s ===\n", label))
  for (b in blocks) {
    cat(sprintf("  %-14s dims %d, unsupported %d, singular %s\n",
                b$grouping_factor, b$n_dimensions,
                b$n_unsupported_dimensions, b$model_is_singular))
    cat(sprintf("      relative SDs : %s\n", paste(b$relative_sdev, collapse = ", ")))
    cat(sprintf("      RE SD by term: %s\n",
                paste(sprintf("%s=%s", names(b$re_sd_by_term), b$re_sd_by_term),
                      collapse = ", ")))
    cat(sprintf("      largest %s / smallest %s\n",
                b$largest_re_term, b$smallest_re_term))
  }
}

.source_isolated <- function(path) {
  env <- new.env(parent = globalenv())
  suppressWarnings(suppressMessages(source(path, local = env)))
  env
}

.load_inputs <- function(env, input_dir) {
  parts <- lapply(c("generations", "verifications", "questions", "models"),
                  function(nm) env$read_jsonl(file.path(input_dir, paste0(nm, ".jsonl"))))
  names(parts) <- c("generations", "verifications", "questions", "models")
  parts
}

parse_args <- function(args) {
  out <- list(input = "data/analysis_inputs/phase_c_final",
              output = "data/analysis/phase_c/repca_extended.json")
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
  out <- list(negligible_relative_sd_threshold = NEGLIGIBLE_REL_SD,
              input_dir = args$input, blocks = list())

  cat(sprintf("Loading %s ...\n", args$input))
  e1 <- .source_isolated("src/analysis/e1_checkpoint_cascade.R")
  p1 <- .load_inputs(e1, args$input)
  d1 <- e1$build_e1_data(p1$generations, p1$verifications, p1$questions, p1$models)
  cat(sprintf("E1 frame: %d rows. Fitting extended RE ...\n", nrow(d1)))
  m1 <- e1$fit_e1_extended_model(d1)
  out$blocks$e1_extended_re <- repca_blocks(m1)
  .print_blocks("E1 extended RE", out$blocks$e1_extended_re)

  e2 <- .source_isolated("src/analysis/e2_prompt_manipulation.R")
  p2 <- .load_inputs(e2, args$input)
  d2 <- e2$build_e2_data(p2$generations, p2$verifications, p2$questions, p2$models)
  cat(sprintf("\nE2 frame: %d rows. Fitting extended RE ...\n", nrow(d2)))
  m2 <- e2$fit_e2_extended_model(d2)
  out$blocks$e2_extended_re <- repca_blocks(m2)
  .print_blocks("E2 extended RE", out$blocks$e2_extended_re)

  dir.create(dirname(args$output), recursive = TRUE, showWarnings = FALSE)
  writeLines(toJSON(out, auto_unbox = TRUE, pretty = TRUE, na = "null", digits = NA),
             args$output)
  cat(sprintf("\nWrote %s\n", args$output))
  invisible(out)
}

if (sys.nframe() == 0L) main()
