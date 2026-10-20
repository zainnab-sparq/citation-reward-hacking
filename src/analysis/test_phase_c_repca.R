# Tests for the Phase C rePCA diagnostic.
#
# The Phase B version reports names(rePCA(model))[1] only, which is correct
# there because its maximal model has a single grouping factor. Every Phase C
# extended-RE model has three (family, model_id, question_id), so reporting the
# first would silently describe one variance component and leave the reader to
# assume it covered all of them. These tests pin that all grouping factors come
# back, and that the unsupported-dimension rule is a relative one applied within
# each factor rather than across the pooled spectrum.
#
# Run from repo root:
#   Rscript src/analysis/test_phase_c_repca.R

suppressPackageStartupMessages({ library(lme4) })

PASS_COUNT <- 0L
FAIL_COUNT <- 0L
run_test <- function(name, body) {
  cat(sprintf("  %-64s ... ", name))
  ok <- tryCatch({ body(); TRUE },
                 error = function(e) { cat("FAIL\n      ", conditionMessage(e), "\n"); FALSE })
  if (ok) { cat("ok\n"); PASS_COUNT <<- PASS_COUNT + 1L } else { FAIL_COUNT <<- FAIL_COUNT + 1L }
}

pc <- new.env()
suppressWarnings(suppressMessages(source("src/analysis/phase_c_repca.R", local = pc)))

# A small two-grouping-factor model. Gaussian and tiny so the tests stay fast;
# rePCA reads the random-effects covariance and does not care about the family.
fit_two_factor <- function(seed = 1L) {
  set.seed(seed)
  n_g <- 12L; n_h <- 8L; reps <- 6L
  d <- expand.grid(g = factor(seq_len(n_g)), h = factor(seq_len(n_h)),
                   rep = seq_len(reps))
  d$x <- rnorm(nrow(d))
  g_int <- rnorm(n_g, sd = 1.2)[as.integer(d$g)]
  g_slope <- rnorm(n_g, sd = 0.8)[as.integer(d$g)]
  h_int <- rnorm(n_h, sd = 0.9)[as.integer(d$h)]
  d$y <- 1 + g_int + g_slope * d$x + h_int + rnorm(nrow(d), sd = 0.7)
  suppressMessages(lmer(y ~ x + (1 + x | g) + (1 | h), data = d,
                        control = lmerControl(calc.derivs = FALSE)))
}

cat("\n[repca] every grouping factor is reported ------------------------------\n")

run_test("repca_blocks returns one block per grouping factor", function() {
  blocks <- pc$repca_blocks(fit_two_factor())
  stopifnot(length(blocks) == 2L)
  stopifnot(setequal(vapply(blocks, `[[`, character(1), "grouping_factor"), c("g", "h")))
})

run_test("a two-dimensional factor reports two dimensions", function() {
  blocks <- pc$repca_blocks(fit_two_factor())
  g <- Filter(function(b) b$grouping_factor == "g", blocks)[[1]]
  h <- Filter(function(b) b$grouping_factor == "h", blocks)[[1]]
  stopifnot(g$n_dimensions == 2L, h$n_dimensions == 1L)
})

run_test("relative SDs are scaled within the factor, leading term exactly 1", function() {
  blocks <- pc$repca_blocks(fit_two_factor())
  for (b in blocks) {
    stopifnot(isTRUE(all.equal(max(b$relative_sdev), 1)))
    stopifnot(!is.unsorted(rev(b$relative_sdev)))
  }
})

run_test("cumulative variance ends at exactly 1 for every factor", function() {
  for (b in pc$repca_blocks(fit_two_factor())) {
    stopifnot(isTRUE(all.equal(b$cumulative_variance[b$n_dimensions], 1)))
  }
})

run_test("re_sd_by_term is named and matches the dimension count", function() {
  blocks <- pc$repca_blocks(fit_two_factor())
  g <- Filter(function(b) b$grouping_factor == "g", blocks)[[1]]
  stopifnot(length(g$re_sd_by_term) == 2L)
  stopifnot(!is.null(names(g$re_sd_by_term)))
  stopifnot("(Intercept)" %in% names(g$re_sd_by_term))
})

cat("\n[repca] the unsupported-dimension rule ----------------------------------\n")

run_test("threshold is the documented 1e-3 and is not tuned per call", function() {
  stopifnot(isTRUE(all.equal(pc$NEGLIGIBLE_REL_SD, 1e-3)))
})

run_test("a well-supported factor reports zero unsupported dimensions", function() {
  blocks <- pc$repca_blocks(fit_two_factor())
  g <- Filter(function(b) b$grouping_factor == "g", blocks)[[1]]
  stopifnot(g$n_unsupported_dimensions == 0L)
})

run_test(".count_unsupported counts relative SDs at or below the threshold", function() {
  stopifnot(pc$.count_unsupported(c(1, 0.5, 1e-4)) == 1L)
  stopifnot(pc$.count_unsupported(c(1, 1e-3)) == 1L)      # boundary is inclusive
  stopifnot(pc$.count_unsupported(c(1, 0.5)) == 0L)
  stopifnot(pc$.count_unsupported(c(1, 1e-4, 0)) == 2L)
})

cat(sprintf("\n%d passed / %d failed\n", PASS_COUNT, FAIL_COUNT))
if (FAIL_COUNT > 0L) quit(status = 1L)
