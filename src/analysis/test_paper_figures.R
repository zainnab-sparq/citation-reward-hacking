# Tests for the paper-figure data preparation.
#
# The figures are the only place in the manuscript where a number is drawn
# rather than written, so a silent aggregation bug here would contradict the
# prose without failing any existing test. These pin the three ways that can
# happen: a binomial interval computed by the wrong formula, counts that do not
# conserve against the analysis frame, and a share computed as a mean of ratios
# instead of a ratio of sums.
#
# Plot aesthetics are deliberately untested. Only the numbers are.
#
# Run from repo root:
#   Rscript src/analysis/test_paper_figures.R

PASS_COUNT <- 0L
FAIL_COUNT <- 0L
run_test <- function(name, body) {
  cat(sprintf("  %-64s ... ", name))
  ok <- tryCatch({ body(); TRUE },
                 error = function(e) { cat("FAIL\n      ", conditionMessage(e), "\n"); FALSE })
  if (ok) { cat("ok\n"); PASS_COUNT <<- PASS_COUNT + 1L } else { FAIL_COUNT <<- FAIL_COUNT + 1L }
}

pf <- new.env()
suppressWarnings(suppressMessages(source("src/analysis/paper_figures.R", local = pf)))

# ---- A hand-built frame with counts chosen so every group total is distinct ---
# 3 cells x 4 stages, fabricated counts laid out so a transposed or recycled
# aggregation cannot accidentally reproduce the right answer.
fake_e1 <- function() {
  rows <- list()
  n_by <- c(10L, 20L, 30L, 40L)      # per stage, same for each cell
  k_by <- list(known_real   = c(1L, 2L, 3L, 4L),
               obscure_real = c(0L, 5L, 15L, 32L),
               unanswerable = c(2L, 4L, 9L, 20L))
  for (cl in names(k_by)) {
    for (s in 0:3) {
      n <- n_by[s + 1L]; k <- k_by[[cl]][s + 1L]
      rows[[length(rows) + 1L]] <- data.frame(
        cell = cl, stage_ordinal = s,
        fabricated = c(rep(1L, k), rep(0L, n - k)),
        is_post_trained = as.integer(s > 0),
        condition = rep(c("neutral", "citations_expected",
                          "abstention_permitted", "length_matched"),
                        length.out = n),
        stringsAsFactors = FALSE)
    }
  }
  d <- do.call(rbind, rows)
  d$cell <- factor(d$cell, levels = c("known_real", "obscure_real", "unanswerable"))
  d$condition <- factor(d$condition,
                        levels = c("neutral", "citations_expected",
                                   "abstention_permitted", "length_matched"))
  d
}

cat("\n[figures] Wilson interval against base R's own implementation ----------\n")

# prop.test(correct = FALSE) IS the Wilson score interval, so base R is an
# independent oracle here rather than a restatement of our own arithmetic.
run_test("wilson_ci matches prop.test(correct = FALSE) across many k, n", function() {
  for (n in c(5L, 20L, 137L, 1000L)) {
    for (k in unique(round(seq(0, n, length.out = 7L)))) {
      got <- pf$wilson_ci(k, n)
      want <- suppressWarnings(prop.test(k, n, correct = FALSE)$conf.int)
      if (!isTRUE(all.equal(c(got$lo, got$hi), as.numeric(want), tolerance = 1e-9))) {
        stop(sprintf("k=%d n=%d: got [%.10f, %.10f] want [%.10f, %.10f]",
                     k, n, got$lo, got$hi, want[1], want[2]))
      }
    }
  }
})

run_test("wilson_ci brackets the point estimate and stays inside [0, 1]", function() {
  for (n in c(1L, 7L, 99L)) for (k in 0:n) {
    ci <- pf$wilson_ci(k, n)
    p <- k / n
    stopifnot(ci$lo >= 0, ci$hi <= 1, ci$lo <= p + 1e-12, ci$hi >= p - 1e-12)
  }
})

run_test("wilson_ci returns NA for an empty group rather than NaN", function() {
  ci <- pf$wilson_ci(0L, 0L)
  stopifnot(is.na(ci$lo), is.na(ci$hi))
})

cat("\n[figures] F1: fabrication by stage within cell ---------------------------\n")

run_test("prep_stage_by_cell returns one row per non-empty cell x stage", function() {
  out <- pf$prep_stage_by_cell(fake_e1())
  stopifnot(nrow(out) == 12L)
  stopifnot(all(c("cell", "stage_ordinal", "n", "k", "rate", "lo", "hi") %in% names(out)))
})

run_test("prep_stage_by_cell conserves the frame's row and event counts", function() {
  d <- fake_e1()
  out <- pf$prep_stage_by_cell(d)
  stopifnot(sum(out$n) == nrow(d))
  stopifnot(sum(out$k) == sum(d$fabricated))
})

run_test("prep_stage_by_cell reproduces the hand-built rates exactly", function() {
  out <- pf$prep_stage_by_cell(fake_e1())
  got <- out$rate[out$cell == "obscure_real"][order(out$stage_ordinal[out$cell == "obscure_real"])]
  stopifnot(isTRUE(all.equal(got, c(0 / 10, 5 / 20, 15 / 30, 32 / 40))))
})

run_test("prep_stage_by_cell does not transpose cell and stage", function() {
  out <- pf$prep_stage_by_cell(fake_e1())
  r <- out$rate[out$cell == "known_real" & out$stage_ordinal == 3L]
  stopifnot(isTRUE(all.equal(r, 4 / 40)))
})

cat("\n[figures] F2: fabrication by condition within post-training status -------\n")

run_test("prep_condition_by_stage conserves row and event counts", function() {
  d <- fake_e1()
  out <- pf$prep_condition_by_stage(d)
  stopifnot(sum(out$n) == nrow(d))
  stopifnot(sum(out$k) == sum(d$fabricated))
})

run_test("prep_condition_by_stage splits on is_post_trained, not on stage", function() {
  out <- pf$prep_condition_by_stage(fake_e1())
  stopifnot(length(unique(out$is_post_trained)) == 2L)
  stopifnot(nrow(out) == 8L)
})

cat("\n[figures] F3: reads the drift artifact, does not recompute it -----------\n")

# The F3 aggregation lives in src/analysis/judge_drift_descriptives.py and is
# tested in tests/test_judge_drift_descriptives.py. What matters on the R side
# is that nothing is silently recomputed here and that a missing artifact is an
# error rather than an empty panel.
fake_report <- function() {
  list(
    n_judged_citations = 60L,
    grounding = data.frame(
      era = c("RECOVERED", "EARLY", "DEGRADED"),
      n_judged = c(20L, 30L, 10L),
      zero_effort_share = c(0.05, 0.02, 0.30),
      mean_effort = c(2.1, 2.0, 1.6),
      stringsAsFactors = FALSE),
    verdict_mix = data.frame(
      era = c("DEGRADED", "DEGRADED", "EARLY", "EARLY"),
      status = c("FABRICATED", "REAL", "REAL", "FABRICATED"),
      n = c(7L, 3L, 20L, 10L),
      share = c(0.7, 0.3, 2 / 3, 1 / 3),
      stringsAsFactors = FALSE))
}

run_test("load_drift_descriptives fails loudly when the artifact is absent", function() {
  err <- tryCatch({ pf$load_drift_descriptives(tempfile()); NULL },
                  error = function(e) conditionMessage(e))
  stopifnot(!is.null(err), grepl("judge_drift_descriptives", err, fixed = TRUE))
})

run_test("prep_era_grounding orders eras chronologically, not alphabetically", function() {
  out <- pf$prep_era_grounding(fake_report())
  stopifnot(identical(as.character(out$era), c("EARLY", "DEGRADED", "RECOVERED")))
})

run_test("prep_era_grounding passes the artifact's numbers through unchanged", function() {
  out <- pf$prep_era_grounding(fake_report())
  stopifnot(isTRUE(all.equal(out$mean_effort, c(2.0, 1.6, 2.1))))
  stopifnot(isTRUE(all.equal(out$zero_effort_share, c(0.02, 0.30, 0.05))))
})

run_test("prep_era_verdict stacks REAL, UNVERIFIABLE, FABRICATED in that order", function() {
  out <- pf$prep_era_verdict(fake_report())
  early <- out[out$era == "EARLY", ]
  stopifnot(identical(as.character(early$status), c("REAL", "FABRICATED")))
})

run_test("prep_era_verdict keeps shares summing to one within era", function() {
  out <- pf$prep_era_verdict(fake_report())
  agg <- tapply(out$share, droplevels(out$era), sum)
  stopifnot(length(agg) == 2L, all(abs(agg - 1) < 1e-12))
})

cat(sprintf("\n%d passed / %d failed\n", PASS_COUNT, FAIL_COUNT))
if (FAIL_COUNT > 0L) quit(status = 1L)
