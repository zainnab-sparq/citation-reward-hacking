# rePCA on the maximal random-effects structure (deferred Phase B item 4.3).
#
# `.summarize_re_robustness` in phase_b_descriptive.R already reports
# lme4::isSingular() and the condition-SE comparison, and the maximal model
# reports NON-singular and converged for all three scorings. That is not the
# whole story: the abstention_permitted slope's standard error inflates roughly
# 6x going from intercept-only to maximal RE (0.393 -> 2.350 on the J2a
# academic+medical fit), which is what a variance component that is technically
# estimable but carries almost no information looks like.
#
# isSingular() answers "is the RE covariance matrix on the boundary". rePCA
# answers the question actually being asked here: "how many dimensions of that
# matrix are supported by the data, and which term is the unsupported one".
# A zero (or numerically negligible) standard deviation in the rotated space
# means one RE dimension is not identified even when the fit does not trip the
# singularity flag.
#
# Diagnostic only. It re-estimates no registered contrast and changes no
# estimand: it reports the conditioning of a random-effects structure whose
# selection rule (parsimony-primary with a labelled extended-RE sensitivity) is
# already registered in pre-reg 6.
#
# Usage:
#   Rscript src/analysis/phase_b_repca.R \
#     --generations data/analysis/phase_b/generations.csv \
#     --verifications data/analysis/phase_b/verifications.csv \
#     --questions data/analysis/phase_b/questions.csv \
#     --output data/analysis/phase_b/repca_maximal.json

suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})

HERE <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(HERE) || !nzchar(HERE)) HERE <- "src/analysis"
source(file.path(HERE, "phase_b_descriptive.R"))

# A rotated-space SD at or below this fraction of the leading component is
# treated as an unsupported dimension. 1e-3 is conventional for rePCA reads and
# is fixed here rather than tuned to the result.
NEGLIGIBLE_REL_SD <- 1e-3

parse_local <- function(argv) {
  out <- list(generations = NA, verifications = NA, questions = NA,
              output = "data/analysis/phase_b/repca_maximal.json")
  i <- 1
  while (i <= length(argv)) {
    key <- sub("^--", "", argv[i])
    if (key %in% names(out)) { out[[key]] <- argv[i + 1]; i <- i + 2 } else i <- i + 1
  }
  out
}

repca_block <- function(model, label) {
  pca <- lme4::rePCA(model)
  term <- names(pca)[1]
  sdev <- as.numeric(pca[[term]]$sdev)
  rel <- if (max(sdev) > 0) sdev / max(sdev) else sdev
  n_unsupported <- sum(rel <= NEGLIGIBLE_REL_SD)
  vc <- lme4::VarCorr(model)[[term]]
  list(
    scoring = label,
    grouping_factor = term,
    n_dimensions = length(sdev),
    sdev = round(sdev, 6),
    relative_sdev = round(rel, 6),
    cumulative_variance = round(cumsum(sdev^2) / sum(sdev^2), 6),
    n_unsupported_dimensions = n_unsupported,
    is_singular_flag = lme4::isSingular(model),
    re_sd_by_term = round(sqrt(diag(vc)), 6),
    # The smallest marginal RE SD names the term carrying least information; it
    # is reported alongside, not instead of, the rotated-space read, because the
    # rotation can spread a deficiency across correlated terms.
    smallest_re_term = names(which.min(diag(vc)))
  )
}

main_repca <- function() {
  args <- parse_local(commandArgs(trailingOnly = TRUE))
  for (p in c("generations", "verifications", "questions")) {
    if (is.na(args[[p]])) stop(sprintf("Missing --%s", p))
  }

  generations <- read_csv_frame(args$generations)
  verifications <- read_csv_frame(args$verifications)
  questions <- read_csv_frame(args$questions)

  out <- list(negligible_relative_sd_threshold = NEGLIGIBLE_REL_SD, blocks = list())

  data_status <- build_descriptive_data(generations, verifications, questions, "status")
  fits <- list(
    status_academic_medical = restrict_to_domains(data_status, GOOD_COVERAGE_DOMAINS),
    status_all_domains = data_status
  )
  data_ver <- build_descriptive_data(generations, verifications, questions, "verifier_status")
  fits$verifier_status <- restrict_to_domains(data_ver, GOOD_COVERAGE_DOMAINS)

  for (label in names(fits)) {
    cat(sprintf("\n=== %s ===\n", label))
    model <- tryCatch(fit_descriptive_maximal_model(fits[[label]]),
                      error = function(e) { cat("maximal fit failed:", conditionMessage(e), "\n"); NULL })
    if (is.null(model)) { out$blocks[[label]] <- list(scoring = label, error = "maximal fit failed"); next }
    block <- repca_block(model, label)
    out$blocks[[label]] <- block
    cat(sprintf("  grouping factor      : %s\n", block$grouping_factor))
    cat(sprintf("  isSingular()         : %s\n", block$is_singular_flag))
    cat(sprintf("  rotated SDs          : %s\n", paste(block$sdev, collapse = ", ")))
    cat(sprintf("  relative SDs         : %s\n", paste(block$relative_sdev, collapse = ", ")))
    cat(sprintf("  cumulative variance  : %s\n", paste(block$cumulative_variance, collapse = ", ")))
    cat(sprintf("  unsupported dims     : %d of %d\n", block$n_unsupported_dimensions, block$n_dimensions))
    cat(sprintf("  RE SD by term        : %s\n",
                paste(sprintf("%s=%s", names(block$re_sd_by_term), block$re_sd_by_term), collapse = ", ")))
    cat(sprintf("  smallest RE term     : %s\n", block$smallest_re_term))
  }

  dir.create(dirname(args$output), recursive = TRUE, showWarnings = FALSE)
  writeLines(jsonlite::toJSON(out, auto_unbox = TRUE, pretty = TRUE, na = "null"), args$output)
  cat(sprintf("\nWrote %s\n", args$output))
  invisible(out)
}

if (sys.nframe() == 0L) main_repca()
