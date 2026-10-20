# Tests for the appendix B breakdown tables.
#
# The fabrication-rate aggregations reuse .aggregate_rate, which is already
# pinned in test_paper_figures.R. What is new here is the verification-status
# crosstab, and it carries the appendix's load-bearing caveat: that legal
# citations are verified through a gappier route than academic and medical ones.
# If that table silently normalized over the wrong margin, the caveat would
# either vanish or be overstated, and nothing else in the suite would notice.
#
# Run from repo root:
#   Rscript src/analysis/test_paper_breakdowns.R

PASS_COUNT <- 0L
FAIL_COUNT <- 0L
run_test <- function(name, body) {
  cat(sprintf("  %-64s ... ", name))
  ok <- tryCatch({ body(); TRUE },
                 error = function(e) { cat("FAIL\n      ", conditionMessage(e), "\n"); FALSE })
  if (ok) { cat("ok\n"); PASS_COUNT <<- PASS_COUNT + 1L } else { FAIL_COUNT <<- FAIL_COUNT + 1L }
}

pb <- new.env()
suppressWarnings(suppressMessages(source("src/analysis/paper_breakdowns.R", local = pb)))

fake_questions <- function() {
  data.frame(id = c("qa", "qb", "ql"),
             domain = c("academic", "academic", "legal"),
             cell = "known_real", stringsAsFactors = FALSE)
}

# Legal deliberately carries a high UNVERIFIABLE share on a SMALL number of
# citations, so a table normalized over the wrong margin gives a visibly
# different answer than one normalized within domain.
fake_verifications <- function() {
  data.frame(
    question_id = c(rep("qa", 6), rep("qb", 2), rep("ql", 2)),
    status = c(rep("REAL_CONFIRMED", 5), "FABRICATED",
               "FABRICATED", "FABRICATED",
               "UNVERIFIABLE", "REAL_CONFIRMED"),
    stringsAsFactors = FALSE)
}

cat("\n[breakdowns] verification-status mix by domain --------------------------\n")

run_test("shares are normalized within domain, not over the whole corpus", function() {
  out <- pb$status_mix_by_domain(fake_verifications(), fake_questions())
  legal <- Filter(function(r) r$domain == "legal", out)[[1]]
  # 1 of legal's 2 citations, not 1 of the corpus's 10.
  stopifnot(isTRUE(all.equal(legal$share_unverifiable, 0.5)))
  stopifnot(legal$n_citations == 2L)
})

run_test("shares within a domain sum to one", function() {
  out <- pb$status_mix_by_domain(fake_verifications(), fake_questions())
  for (row in out) {
    shares <- unlist(row[grepl("^share_", names(row))])
    stopifnot(abs(sum(shares) - 1) < 1e-12)
  }
})

run_test("counts conserve the input citations", function() {
  out <- pb$status_mix_by_domain(fake_verifications(), fake_questions())
  stopifnot(sum(vapply(out, function(r) r$n_citations, integer(1))) ==
              nrow(fake_verifications()))
})

run_test("a domain with no citations of a status reports zero, not absent", function() {
  out <- pb$status_mix_by_domain(fake_verifications(), fake_questions())
  academic <- Filter(function(r) r$domain == "academic", out)[[1]]
  stopifnot(!is.null(academic$n_unverifiable), academic$n_unverifiable == 0L)
  stopifnot(isTRUE(all.equal(academic$share_unverifiable, 0)))
})

run_test("citations are attributed by question, not recycled across domains", function() {
  out <- pb$status_mix_by_domain(fake_verifications(), fake_questions())
  academic <- Filter(function(r) r$domain == "academic", out)[[1]]
  stopifnot(academic$n_citations == 8L)          # qa 6 + qb 2
  stopifnot(academic$n_fabricated == 3L)         # 1 from qa, 2 from qb
})

cat("\n[breakdowns] citations emitted per generation ---------------------------\n")

run_test("mean citations is a per-generation mean within each group", function() {
  d <- data.frame(domain = c("a", "a", "b", "b"),
                  condition = "neutral",
                  n_citations = c(0L, 4L, 3L, 3L),
                  stringsAsFactors = FALSE)
  out <- pb$citations_per_generation(d, c("domain", "condition"))
  stopifnot(isTRUE(all.equal(out$mean_citations[out$domain == "a"], 2)))
  stopifnot(isTRUE(all.equal(out$mean_citations[out$domain == "b"], 3)))
})

run_test("generation counts accompany the means so a thin cell is visible", function() {
  d <- data.frame(domain = c("a", "b", "b"), condition = "neutral",
                  n_citations = c(1L, 2L, 4L), stringsAsFactors = FALSE)
  out <- pb$citations_per_generation(d, c("domain", "condition"))
  stopifnot(out$n_generations[out$domain == "a"] == 1L)
  stopifnot(out$n_generations[out$domain == "b"] == 2L)
})

cat(sprintf("\n%d passed / %d failed\n", PASS_COUNT, FAIL_COUNT))
if (FAIL_COUNT > 0L) quit(status = 1L)
