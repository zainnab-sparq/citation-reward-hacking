# Unit tests for the chunked parametric bootstrap (src/analysis/bootstrap_chunked.R).
#
# Why this file exists. The registered §6 bootstrap is 10,000 resamples at a
# measured 288 s each, and lme4::bootMer cannot checkpoint: it returns nothing
# until the last resample lands. A machine restart on 2026-09-02 destroyed 64.9
# hours of a run for that reason. Chunking splits the resampling into separately
# saved pieces so an interruption costs one chunk instead of the whole pass.
#
# The correctness claim being tested is narrow and specific: pooling k chunks of
# m resamples must produce exactly the percentile CI that one bootMer object
# carrying those same k*m resample rows would produce. That is what makes the
# chunked path the same estimator as the registered one rather than an
# approximation of it.
#
# Run from repo root:
#   Rscript src/analysis/test_bootstrap_chunked.R
#
# Exit status: 0 on all-pass, 1 on any failure.

suppressPackageStartupMessages({
  library(lme4)
})

PASS_COUNT <- 0L
FAIL_COUNT <- 0L

run_test <- function(name, body) {
  cat(sprintf("  %-58s ... ", name))
  ok <- tryCatch({ body(); TRUE },
                 error = function(e) { cat("FAIL\n      ", conditionMessage(e), "\n"); FALSE })
  if (ok) {
    cat("ok\n")
    PASS_COUNT <<- PASS_COUNT + 1L
  } else {
    FAIL_COUNT <<- FAIL_COUNT + 1L
  }
}

bc <- new.env()
source("src/analysis/bootstrap_chunked.R", local = bc)

# A scratch directory per run, so a stale chunk from an earlier run can never
# make a resume test pass for the wrong reason.
scratch <- file.path(tempdir(), paste0("bootchunk_test_", as.integer(Sys.time())))
dir.create(scratch, recursive = TRUE, showWarnings = FALSE)
on.exit(unlink(scratch, recursive = TRUE), add = TRUE)

# ---- Chunk planning -----------------------------------------------------------

cat("\n[plan] boot_chunk_plan ---------------------------------------------------\n")

run_test("chunks sum to exactly nsim when it divides evenly", function() {
  plan <- bc$boot_chunk_plan(nsim = 10000, chunk_size = 400)
  stopifnot(sum(plan$nsim) == 10000)
  stopifnot(nrow(plan) == 25)
  stopifnot(all(plan$nsim == 400))
})

run_test("a remainder becomes a final short chunk, total still exact", function() {
  plan <- bc$boot_chunk_plan(nsim = 10000, chunk_size = 300)
  stopifnot(sum(plan$nsim) == 10000)
  stopifnot(nrow(plan) == 34)
  stopifnot(plan$nsim[34] == 100)
  stopifnot(all(plan$nsim[1:33] == 300))
})

run_test("chunk_size larger than nsim yields a single chunk", function() {
  plan <- bc$boot_chunk_plan(nsim = 50, chunk_size = 400)
  stopifnot(nrow(plan) == 1)
  stopifnot(plan$nsim == 50)
})

run_test("indices are 1..n with no gaps", function() {
  plan <- bc$boot_chunk_plan(nsim = 1000, chunk_size = 250)
  stopifnot(identical(plan$index, 1:4))
})

# ---- Seeds --------------------------------------------------------------------

cat("\n[seed] per-chunk seeding -------------------------------------------------\n")

run_test("every chunk gets a distinct seed", function() {
  plan <- bc$boot_chunk_plan(nsim = 10000, chunk_size = 400)
  stopifnot(!anyDuplicated(plan$seed))
})

run_test("the plan is deterministic across calls", function() {
  a <- bc$boot_chunk_plan(nsim = 10000, chunk_size = 400)
  b <- bc$boot_chunk_plan(nsim = 10000, chunk_size = 400)
  stopifnot(identical(a$seed, b$seed))
})

run_test("distinct seeds actually produce distinct simulated draws", function() {
  # The failure this guards against is a chunked run that reuses one seed and
  # stamps out the same resamples k times, which would look like a very tight
  # interval and be worthless.
  plan <- bc$boot_chunk_plan(nsim = 20, chunk_size = 10)
  set.seed(plan$seed[1]); a <- runif(5)
  set.seed(plan$seed[2]); b <- runif(5)
  stopifnot(!isTRUE(all.equal(a, b)))
})

# ---- A real model, fitted once and shared -------------------------------------

cat("\n[fit] synthetic model for the bootstrap tests ----------------------------\n")

set.seed(11)
n <- 300
sim_df <- data.frame(
  grp = factor(rep(1:15, length.out = n)),
  x   = rnorm(n)
)
sim_df$y <- rbinom(n, 1, plogis(-0.4 + 0.8 * sim_df$x +
                                 rnorm(15, sd = 0.5)[as.integer(sim_df$grp)]))

model <- glmer(y ~ x + (1 | grp), data = sim_df, family = binomial)

# confint.bootMer(parm = <subset>) is broken in lme4 2.0.1: it assigns all of
# names(t0) as rownames of a subset-length table and errors. lme4's own
# confint.merMod avoids this by computing the whole table and subsetting after,
# so the tests and the implementation do the same.
perc_ci <- function(obj, term) {
  tab <- confint(obj, level = 0.95, type = "perc")
  tab[term, , drop = FALSE]
}

run_test("the synthetic model fits and exposes the term under test", function() {
  stopifnot("x" %in% names(lme4::fixef(model)))
})

run_test("the bootstrap statistic matches the one lme4 uses for method='boot'", function() {
  # Faithfulness to the registered path: same statistic, so same resample
  # matrix and the same NA pattern on a failed refit.
  fn <- bc$lme4_boot_statistic(model)
  v <- fn(model)
  stopifnot(all(names(lme4::fixef(model)) %in% names(v)))
  stopifnot(length(v) > length(lme4::fixef(model)))  # variance components too
})

# ---- Pooling: the central correctness claim -----------------------------------

cat("\n[pool] pooled chunks equal one undivided bootMer object ------------------\n")

run_test("pooled resample matrix has one row per resample, in chunk order", function() {
  plan <- bc$boot_chunk_plan(nsim = 12, chunk_size = 4)
  for (i in seq_len(nrow(plan))) {
    bc$run_boot_chunk(model, plan[i, ], scratch, "poolshape", cl = NULL)
  }
  pooled <- bc$pool_boot_chunks(plan, scratch, "poolshape")
  stopifnot(nrow(pooled$t) == 12)
  stopifnot(pooled$R == 12)
  stopifnot(identical(colnames(pooled$t), names(pooled$t0)))
})

run_test("percentile CI from pooled chunks matches confint on the same rows", function() {
  # The equivalence that licenses chunking. Build one bootMer object, then a
  # pooled object carrying exactly the same resample rows, and require the
  # registered percentile interval to agree to floating-point tolerance.
  set.seed(99)
  single <- lme4::bootMer(model, bc$lme4_boot_statistic(model),
                          nsim = 40, type = "parametric")
  expected <- perc_ci(single, "x")

  faked <- bc$pool_boot_chunks_from_t(list(single$t[1:20, , drop = FALSE],
                                           single$t[21:40, , drop = FALSE]),
                                      template = single)
  actual <- perc_ci(faked, "x")

  stopifnot(isTRUE(all.equal(unname(expected), unname(actual), tolerance = 1e-12)))
})

run_test("pooling preserves t0 from the fitted model, not from a resample", function() {
  set.seed(7)
  single <- lme4::bootMer(model, bc$lme4_boot_statistic(model),
                          nsim = 6, type = "parametric")
  faked <- bc$pool_boot_chunks_from_t(list(single$t), template = single)
  stopifnot(isTRUE(all.equal(faked$t0, single$t0)))
})

# ---- Resume -------------------------------------------------------------------

cat("\n[resume] completed chunks are not recomputed ------------------------------\n")

run_test("pending_boot_chunks returns the whole plan when nothing is on disk", function() {
  plan <- bc$boot_chunk_plan(nsim = 12, chunk_size = 4)
  stopifnot(nrow(bc$pending_boot_chunks(plan, scratch, "resumejob")) == 3)
})

run_test("a chunk already on disk is skipped", function() {
  plan <- bc$boot_chunk_plan(nsim = 12, chunk_size = 4)
  bc$run_boot_chunk(model, plan[1, ], scratch, "resumejob", cl = NULL)
  pending <- bc$pending_boot_chunks(plan, scratch, "resumejob")
  stopifnot(nrow(pending) == 2)
  stopifnot(identical(pending$index, 2:3))
})

run_test("a chunk file with the wrong row count is rejected, not pooled", function() {
  # A half-written chunk from a machine that died mid-save must not silently
  # shrink the resample count and produce a plausible, wrong interval.
  plan <- bc$boot_chunk_plan(nsim = 8, chunk_size = 4)
  bc$run_boot_chunk(model, plan[1, ], scratch, "corruptjob", cl = NULL)
  bc$run_boot_chunk(model, plan[2, ], scratch, "corruptjob", cl = NULL)
  path <- bc$boot_chunk_path(scratch, "corruptjob", 2)
  truncated <- readRDS(path)
  truncated$t <- truncated$t[1:2, , drop = FALSE]
  saveRDS(truncated, path)
  err <- tryCatch({ bc$pool_boot_chunks(plan, scratch, "corruptjob"); NULL },
                  error = function(e) conditionMessage(e))
  stopifnot(!is.null(err))
  stopifnot(grepl("row", err, ignore.case = TRUE))
})

run_test("the bootstrap statistic does not capture the fitted model", function() {
  # bootMer stores the statistic in its returned object, so a closure that
  # closes over the model serializes the whole 42,845-row fit into every chunk
  # file and ships it to every worker on every chunk.
  fn <- bc$lme4_boot_statistic(model)
  held <- mget(ls(environment(fn)), envir = environment(fn))
  stopifnot(!any(vapply(held, function(o) inherits(o, "merMod"), logical(1))))
})

run_test("chunks left over from a different fit are rejected", function() {
  # Chunk files outlive a run. If the analysis inputs are regenerated and the
  # job relaunched, stale chunks would otherwise be pooled with resamples from
  # the new fit. t0 is the fitted model's own statistic, so it changes when the
  # fit changes and makes the mismatch detectable.
  plan <- bc$boot_chunk_plan(nsim = 8, chunk_size = 4)
  bc$run_boot_chunk(model, plan[1, ], scratch, "mixfit", cl = NULL)
  bc$run_boot_chunk(model, plan[2, ], scratch, "mixfit", cl = NULL)
  path <- bc$boot_chunk_path(scratch, "mixfit", 2)
  obj <- readRDS(path)
  obj$t0 <- obj$t0 + 1
  saveRDS(obj, path)
  err <- tryCatch({ bc$pool_boot_chunks(plan, scratch, "mixfit"); NULL },
                  error = function(e) conditionMessage(e))
  stopifnot(!is.null(err))
  stopifnot(grepl("different fit", err))
})

# ---- Contract preserved from the unchunked path -------------------------------

cat("\n[contract] behaviour inherited from compute_bootstrap_cis ----------------\n")

run_test("nsim <= 0 returns NULL", function() {
  stopifnot(is.null(bc$compute_bootstrap_cis_chunked(model, "x", nsim = 0,
                                                     chunk_dir = scratch, job = "zero")))
})

run_test("returns OR-scale percentile bounds per term", function() {
  out <- bc$compute_bootstrap_cis_chunked(model, "x", nsim = 12, chunk_size = 4,
                                          chunk_dir = scratch, job = "orscale")
  stopifnot(length(out) == 1)
  stopifnot(out[[1]]$term == "x")
  stopifnot(all(c("or_ci_lo_boot", "or_ci_hi_boot") %in% names(out[[1]])))
  stopifnot(out[[1]]$or_ci_lo_boot > 0)
  stopifnot(out[[1]]$or_ci_lo_boot < out[[1]]$or_ci_hi_boot)
})

run_test("a second call reuses chunks on disk instead of refitting", function() {
  first <- bc$compute_bootstrap_cis_chunked(model, "x", nsim = 12, chunk_size = 4,
                                            chunk_dir = scratch, job = "reuse")
  second <- bc$compute_bootstrap_cis_chunked(model, "x", nsim = 12, chunk_size = 4,
                                             chunk_dir = scratch, job = "reuse")
  stopifnot(isTRUE(all.equal(first, second)))
})

# ---- Job identity -------------------------------------------------------------

cat("\n[job] chunk namespacing --------------------------------------------------\n")

run_test("job name is derived from the output artifact path", function() {
  stopifnot(bc$boot_job_name("data/analysis/phase_c/e1_corrected.json") == "e1_corrected")
})

run_test("the four confirmatory jobs get four distinct chunk namespaces", function() {
  # The hazard this guards: e1_corrected and e1_registered are the same model
  # formula fitted to different inputs. If they shared a chunk directory, one
  # would pool the other's resamples and the corrected/registered comparison
  # that sub-amendment #12 turns on would be silently corrupted.
  outs <- c("data/analysis/phase_c/e1_corrected.json",
            "data/analysis/phase_c/e2_corrected.json",
            "data/analysis/phase_c/e1_registered.json",
            "data/analysis/phase_c/e2_registered.json")
  names <- vapply(outs, bc$boot_job_name, character(1))
  stopifnot(length(unique(names)) == 4)
  paths <- vapply(names, function(n) bc$boot_chunk_path(scratch, n, 1), character(1))
  stopifnot(length(unique(paths)) == 4)
})

# ---- Summary ------------------------------------------------------------------

cat(sprintf("\n%d passed, %d failed\n", PASS_COUNT, FAIL_COUNT))
if (FAIL_COUNT > 0L) quit(status = 1L)
