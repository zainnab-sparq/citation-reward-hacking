# Tests for the appendix B coefficient-table emitter.
#
# These tables pair the corrected and registered pipelines row by row. The
# failure that would not announce itself is a positional pairing: if the two
# artifacts ever list terms in a different order, zipping them by index puts the
# registered pipeline's number on the corrected pipeline's term and every value
# still looks plausible. The emitter therefore joins on the term name and refuses
# to emit when the term sets differ, and that refusal is what most of these tests
# check.
#
# Run from repo root:
#   Rscript src/analysis/test_coefficient_tables.R

PASS_COUNT <- 0L
FAIL_COUNT <- 0L
run_test <- function(name, body) {
  cat(sprintf("  %-64s ... ", name))
  ok <- tryCatch({ body(); TRUE },
                 error = function(e) { cat("FAIL\n      ", conditionMessage(e), "\n"); FALSE })
  if (ok) { cat("ok\n"); PASS_COUNT <<- PASS_COUNT + 1L } else { FAIL_COUNT <<- FAIL_COUNT + 1L }
}

ct <- new.env()
suppressWarnings(suppressMessages(source("src/analysis/coefficient_tables.R", local = ct)))

coef <- function(term, estimate, std_error, p_value, p_holm = NULL) {
  list(term = term, estimate = estimate, std_error = std_error,
       z = estimate / std_error, p_value = p_value, p_holm = p_holm,
       odds_ratio = exp(estimate), or_ci_lo = exp(estimate - 2 * std_error),
       or_ci_hi = exp(estimate + 2 * std_error))
}

cat("\n[coeftab] term escaping and number formatting ---------------------------\n")

run_test("underscores and colons in term names survive as LaTeX", function() {
  got <- ct$.esc_term("stage_ordinal:cellobscure_real")
  stopifnot(!grepl("(?<!\\\\)_", got, perl = TRUE))
  stopifnot(grepl("stage", got, fixed = TRUE))
})

run_test("a p-value below the print floor is shown in scientific form", function() {
  stopifnot(grepl("times 10", ct$.fmt_p(2.0e-34), fixed = TRUE))
  stopifnot(identical(ct$.fmt_p(0.3798), "0.380"))
})

run_test("Holm-adjusted p is preferred over the raw p when present", function() {
  c1 <- coef("t", 1, 0.1, p_value = 1e-9, p_holm = 0.04)
  stopifnot(identical(ct$.coef_p(c1), 0.04))
  c2 <- coef("t", 1, 0.1, p_value = 0.02)
  stopifnot(identical(ct$.coef_p(c2), 0.02))
})

run_test("estimate is printed with its standard error", function() {
  got <- ct$.fmt_est(coef("t", -2.8765, 0.3651, 1e-5))
  stopifnot(grepl("-2.88", got, fixed = TRUE), grepl("0.365", got, fixed = TRUE))
})

run_test("an absent coefficient prints a dash rather than NA", function() {
  stopifnot(identical(ct$.fmt_est(NULL), "---"))
  stopifnot(identical(ct$.fmt_p(NULL), "---"))
})

# A p-value that underflows to exactly 0 is real in these artifacts: E2's
# condition-by-post-trained interaction is far enough out that R returns 0. It
# must print as a bound, never as "0.000" (which reads as a rounded value) and
# never as 10^{-Inf}.
run_test("a p-value that underflowed to zero prints as a bound", function() {
  got <- ct$.fmt_p(0)
  stopifnot(grepl("<", got, fixed = TRUE))
  stopifnot(!grepl("Inf", got, fixed = TRUE))
  stopifnot(!identical(got, "0.000"))
})

run_test("a denormal p-value still formats without an Inf exponent", function() {
  stopifnot(!grepl("Inf", ct$.fmt_p(1e-320), fixed = TRUE))
})

cat("\n[coeftab] the two pipelines are joined by term, never by position -------\n")

run_test("rows pair on term name even when the artifacts disagree on order", function() {
  a <- list(coef("alpha", 1.0, 0.1, 0.01), coef("beta", 2.0, 0.2, 0.02))
  b <- list(coef("beta", 9.0, 0.9, 0.09), coef("alpha", 8.0, 0.8, 0.08))
  rows <- ct$pair_coefficients(a, b)
  alpha <- Filter(function(r) r$term == "alpha", rows)[[1]]
  stopifnot(isTRUE(all.equal(alpha$left$estimate, 1.0)))
  stopifnot(isTRUE(all.equal(alpha$right$estimate, 8.0)))
})

run_test("pairing refuses to emit when the term sets differ", function() {
  a <- list(coef("alpha", 1, 0.1, 0.01))
  b <- list(coef("gamma", 1, 0.1, 0.01))
  err <- tryCatch({ ct$pair_coefficients(a, b); NULL },
                  error = function(e) conditionMessage(e))
  stopifnot(!is.null(err), grepl("term", err))
})

run_test("pairing keeps the left artifact's order, which is the model's order", function() {
  a <- list(coef("z", 1, 0.1, 0.01), coef("a", 1, 0.1, 0.01))
  b <- list(coef("a", 1, 0.1, 0.01), coef("z", 1, 0.1, 0.01))
  rows <- ct$pair_coefficients(a, b)
  stopifnot(identical(vapply(rows, function(r) r$term, character(1)), c("z", "a")))
})

cat("\n[coeftab] emitted LaTeX -------------------------------------------------\n")

run_test("one body row per coefficient, each ending in a LaTeX row break", function() {
  a <- list(coef("alpha", 1, 0.1, 0.01), coef("beta", 2, 0.2, 0.02))
  body <- ct$emit_table_body(ct$pair_coefficients(a, a))
  lines <- strsplit(body, "\n", fixed = TRUE)[[1]]
  stopifnot(length(lines) == 2L)
  stopifnot(all(grepl("\\\\\\\\$", lines)))
})

run_test("emitted rows carry five ampersand-separated columns", function() {
  a <- list(coef("alpha", 1, 0.1, 0.01))
  line <- ct$emit_table_body(ct$pair_coefficients(a, a))
  stopifnot(length(gregexpr("&", line, fixed = TRUE)[[1]]) == 4L)
})

cat("\n[coeftab] the emitted file is a whole table environment ----------------\n")

# LaTeX's braced \input appends \relax after the file. A bare row fragment
# inputted inside a tabular therefore opens an alignment row after the final
# row break and the following \bottomrule becomes a misplaced \noalign. Emitting
# the complete environment is what makes the fragment safe to \input, so these
# tests pin that the environment is closed and self-contained.
run_test("emit_table produces a balanced table and tabular environment", function() {
  a <- list(coef("alpha", 1, 0.1, 0.01))
  out <- ct$emit_table(ct$pair_coefficients(a, a), "A caption.", "tab:x")
  txt <- paste(out, collapse = "\n")
  for (env in c("table", "tabular")) {
    stopifnot(lengths(regmatches(txt, gregexpr(paste0("\\\\begin\\{", env, "\\}"), txt))) == 1L)
    stopifnot(lengths(regmatches(txt, gregexpr(paste0("\\\\end\\{", env, "\\}"), txt))) == 1L)
  }
})

run_test("emit_table carries the caption and label it was given", function() {
  a <- list(coef("alpha", 1, 0.1, 0.01))
  txt <- paste(ct$emit_table(ct$pair_coefficients(a, a), "A caption.", "tab:x"),
               collapse = "\n")
  stopifnot(grepl("\\caption{A caption.}", txt, fixed = TRUE))
  stopifnot(grepl("\\label{tab:x}", txt, fixed = TRUE))
})

run_test("the column specification matches the five emitted columns", function() {
  a <- list(coef("alpha", 1, 0.1, 0.01))
  txt <- paste(ct$emit_table(ct$pair_coefficients(a, a), "c", "l"), collapse = "\n")
  stopifnot(grepl("\\begin{tabular}{lrrrr}", txt, fixed = TRUE))
})

run_test("rules are emitted in booktabs order", function() {
  a <- list(coef("alpha", 1, 0.1, 0.01))
  out <- ct$emit_table(ct$pair_coefficients(a, a), "c", "l")
  pos <- function(p) which(grepl(p, out, fixed = TRUE))[1]
  stopifnot(pos("\\toprule") < pos("\\midrule"), pos("\\midrule") < pos("\\bottomrule"))
})

cat(sprintf("\n%d passed / %d failed\n", PASS_COUNT, FAIL_COUNT))
if (FAIL_COUNT > 0L) quit(status = 1L)
