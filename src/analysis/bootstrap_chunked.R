# Chunked parametric bootstrap for the pre-registered §6 percentile CIs.
#
# WHY THIS EXISTS
#
# lme4::bootMer cannot checkpoint. It returns nothing until the last of its nsim
# resamples completes, so an interruption at hour 200 of a 200-hour run yields
# exactly as much as an interruption at hour 1. A machine restart on 2026-09-02
# destroyed 64.9 hours of the registered §6 run for that reason.
#
# This module splits the resampling into separately saved chunks. An interrupted
# run resumes from the last completed chunk instead of from zero.
#
# WHY THIS IS NOT A PRE-REGISTRATION DEVIATION
#
# A parametric bootstrap resample is an iid draw from the fitted model: simulate
# a response, refit, record the statistic. The percentile CI is a pair of
# quantiles of the resulting pile of draws. Nothing in that estimator depends on
# which process produced which draw, so 25 chunks of 400 pooled is the same
# estimator as one run of 10,000, not an approximation of it. The equivalence is
# asserted directly in test_bootstrap_chunked.R against an undivided bootMer
# object carrying the same resample rows.
#
# Two details keep that claim honest:
#
#   1. Each chunk gets its own seed. Reusing one seed would stamp out identical
#      draws k times, producing a spuriously tight interval. Seeds are
#      base_seed + chunk index; R's set.seed runs an integer seed through 50
#      rounds of scrambling before seeding the Mersenne-Twister state, so
#      adjacent integers give unrelated streams.
#   2. The statistic is the one lme4 itself uses for method = "boot" (variance
#      components on the sdcor scale, then the fixed effects), not a
#      hand-rolled fixef extractor. A cheaper statistic would differ in its NA
#      pattern when a refit fails, which would change the effective resample
#      count.
#
# A NOTE ON bootMer AND WORKER COUNT
#
# bootMer simulates all nsim response vectors in the PARENT before dispatching
# anything (see the ss <- do.call(simulate, argList) line in its body). Workers
# only refit. So the draws depend on the parent's seed alone and the worker
# count cannot change the result. It does drive memory: that simulated-response
# list is serialized into every worker, roughly 340 KB per resample on the real
# 42,845-row frame, which is 3.4 GB per worker at nsim = 10,000 and the reason
# the unchunked path could not safely be widened past four workers per job.
# Chunking at 400 caps the live copy near 140 MB.

suppressPackageStartupMessages({
  library(lme4)
})

DEFAULT_CHUNK_SIZE <- 400L
DEFAULT_BASE_SEED <- 42L
DEFAULT_CHUNK_DIR <- "data/analysis/phase_c/boot_chunks"

#' Where chunk files live, from CITATION_BOOT_CHUNK_DIR.
boot_chunk_dir <- function() {
  raw <- Sys.getenv("CITATION_BOOT_CHUNK_DIR", "")
  if (nzchar(raw)) raw else DEFAULT_CHUNK_DIR
}

#' Resamples per chunk, from CITATION_BOOT_CHUNK. Chunk size trades checkpoint
#' granularity against per-chunk overhead; it cannot change the estimator, since
#' the pooled resample count is fixed by nsim regardless of how it is divided.
boot_chunk_size <- function() {
  n <- suppressWarnings(as.integer(Sys.getenv("CITATION_BOOT_CHUNK", "")))
  if (is.na(n) || n < 1L) DEFAULT_CHUNK_SIZE else n
}

#' Split `nsim` resamples into chunks, each with its own seed.
#' Returns a data.frame of index, nsim and seed whose nsim column sums to
#' exactly `nsim`; a remainder becomes a final short chunk.
boot_chunk_plan <- function(nsim, chunk_size = DEFAULT_CHUNK_SIZE,
                            base_seed = DEFAULT_BASE_SEED) {
  nsim <- as.integer(nsim)
  chunk_size <- as.integer(chunk_size)
  if (is.na(nsim) || nsim <= 0L) {
    return(data.frame(index = integer(0), nsim = integer(0), seed = integer(0)))
  }
  if (is.na(chunk_size) || chunk_size < 1L) chunk_size <- DEFAULT_CHUNK_SIZE
  n_full <- nsim %/% chunk_size
  remainder <- nsim %% chunk_size
  sizes <- c(rep(chunk_size, n_full), if (remainder > 0L) remainder else NULL)
  data.frame(
    index = seq_along(sizes),
    nsim = as.integer(sizes),
    seed = as.integer(base_seed + seq_along(sizes)),
    stringsAsFactors = FALSE)
}

#' The statistic lme4 uses for confint(method = "boot"): variance components
#' followed by fixed effects. Reproduced here so a chunked run bootstraps
#' exactly what the registered path bootstraps. Fails loudly rather than
#' silently substituting a different statistic if lme4's internals move.
lme4_boot_statistic <- function(model, boot_scale = "sdcor", signames = TRUE) {
  ns <- asNamespace("lme4")
  needed <- c("profnames", "hasNoScale", "convParToProfPar")
  absent <- needed[!vapply(needed, exists, logical(1), envir = ns, inherits = FALSE)]
  if (length(absent)) {
    stop(sprintf(paste0("lme4 internals not found: %s. The bootstrap statistic must ",
                        "match confint(method='boot'); refusing to substitute one."),
                 paste(absent, collapse = ", ")))
  }
  profnames <- get("profnames", envir = ns)
  has_no_scale <- get("hasNoScale", envir = ns)
  conv_par <- get("convParToProfPar", envir = ns)

  use_sc0 <- !(lme4::isGLMM(model) && has_no_scale(stats::family(model)))
  vn <- profnames(model, signames, useSc = use_sc0)

  # Give the closure an environment holding only what it needs. Closing over the
  # calling frame would capture `model`, and bootMer stores the statistic in its
  # returned object: the whole fit would then be serialized into every chunk file
  # and shipped to every worker on every chunk.
  fn_env <- new.env(parent = globalenv())
  fn_env$vn <- vn
  fn_env$boot_scale <- boot_scale
  fn_env$has_no_scale <- has_no_scale
  fn_env$conv_par <- conv_par

  statistic <- function(x) {
    use_sc <- !(lme4::isGLMM(x) && has_no_scale(stats::family(x)))
    sc <- if (use_sc) lme4::sigma(x) else NULL
    ss <- conv_par(x@optinfo$val, x, profscale = boot_scale, sc = sc)
    c(stats::setNames(ss, vn), lme4::fixef(x))
  }
  environment(statistic) <- fn_env
  statistic
}

#' Chunk namespace for a run, derived from its output artifact path.
#' The launcher already guarantees one output path per job, so this inherits
#' that uniqueness rather than introducing a second naming scheme that could
#' drift out of step with it. e1_corrected and e1_registered are the same model
#' formula on different inputs; sharing chunks between them would corrupt the
#' comparison sub-amendment #12 depends on.
boot_job_name <- function(output_path) {
  tools::file_path_sans_ext(basename(output_path))
}

boot_chunk_path <- function(chunk_dir, job, index) {
  file.path(chunk_dir, sprintf("%s_chunk%03d.rds", job, as.integer(index)))
}

#' Run one chunk and save it. Writes to a .partial file and renames, so a
#' process killed mid-save leaves an obviously incomplete file rather than a
#' short .rds that would silently shrink the pooled resample count.
run_boot_chunk <- function(model, chunk, chunk_dir, job, cl = NULL, ncpus = 1L) {
  dir.create(chunk_dir, recursive = TRUE, showWarnings = FALSE)
  path <- boot_chunk_path(chunk_dir, job, chunk$index)
  statistic <- lme4_boot_statistic(model)

  set.seed(chunk$seed)
  bb <- if (!is.null(cl)) {
    lme4::bootMer(model, statistic, nsim = chunk$nsim, type = "parametric",
                  parallel = "snow", ncpus = length(cl), cl = cl)
  } else if (ncpus > 1L) {
    lme4::bootMer(model, statistic, nsim = chunk$nsim, type = "parametric",
                  parallel = "snow", ncpus = ncpus)
  } else {
    lme4::bootMer(model, statistic, nsim = chunk$nsim, type = "parametric")
  }

  partial <- paste0(path, ".partial")
  saveRDS(bb, partial)
  if (!file.rename(partial, path)) {
    unlink(partial)
    stop(sprintf("could not finalize chunk file %s", path))
  }
  invisible(path)
}

#' The chunks in `plan` that have no completed file on disk yet.
pending_boot_chunks <- function(plan, chunk_dir, job) {
  if (nrow(plan) == 0L) return(plan)
  paths <- vapply(plan$index, function(i) boot_chunk_path(chunk_dir, job, i), character(1))
  plan[!file.exists(paths), , drop = FALSE]
}

#' Combine resample matrices into one bootMer object. t0 and every other field
#' come from the template, so the point estimate stays the fitted model's and
#' never becomes a resample's.
pool_boot_chunks_from_t <- function(t_list, template) {
  expected_cols <- colnames(template$t)
  for (k in seq_along(t_list)) {
    if (!identical(colnames(t_list[[k]]), expected_cols)) {
      stop(sprintf("chunk %d has different columns from the template", k))
    }
  }
  pooled <- do.call(rbind, t_list)
  out <- template
  out$t <- pooled
  out$R <- nrow(pooled)
  out
}

#' Load every chunk in `plan` and pool it. Refuses to pool a chunk whose row
#' count does not match the plan, which is how a half-written or hand-edited
#' chunk is caught instead of quietly reducing the resample count.
pool_boot_chunks <- function(plan, chunk_dir, job) {
  paths <- vapply(plan$index, function(i) boot_chunk_path(chunk_dir, job, i), character(1))
  absent <- paths[!file.exists(paths)]
  if (length(absent)) {
    stop(sprintf("missing %d chunk file(s): %s", length(absent),
                 paste(basename(absent), collapse = ", ")))
  }
  objs <- lapply(paths, readRDS)
  reference_t0 <- objs[[1]]$t0
  for (k in seq_along(objs)) {
    got <- nrow(objs[[k]]$t)
    want <- plan$nsim[k]
    if (!identical(as.integer(got), as.integer(want))) {
      stop(sprintf("chunk %d has %d resample rows, expected %d",
                   plan$index[k], got, want))
    }
    # Chunk files outlive a run. If the analysis inputs are regenerated and the
    # job relaunched into the same namespace, stale chunks would otherwise be
    # pooled with resamples from a different fit. t0 is the fitted model's own
    # statistic, so it moves when the fit moves.
    if (!isTRUE(all.equal(objs[[k]]$t0, reference_t0))) {
      stop(sprintf(paste0("chunk %d came from a different fit than chunk %d ",
                          "(point estimates disagree). Delete the chunk ",
                          "directory for this job and rerun."),
                   plan$index[k], plan$index[1]))
    }
  }
  pool_boot_chunks_from_t(lapply(objs, `[[`, "t"), template = objs[[1]])
}

#' Chunked, resumable replacement for compute_bootstrap_cis. Same contract:
#' OR-scale percentile bounds per term, NULL when nsim <= 0 or the bootstrap
#' cannot be completed.
compute_bootstrap_cis_chunked <- function(model, terms, nsim,
                                          chunk_size = DEFAULT_CHUNK_SIZE,
                                          chunk_dir, job, ncpus = 1L,
                                          level = 0.95) {
  if (nsim <= 0) return(NULL)
  plan <- boot_chunk_plan(nsim, chunk_size)
  pending <- pending_boot_chunks(plan, chunk_dir, job)

  cat(sprintf("Bootstrap: %d resamples in %d chunk(s) of up to %d, %d already done, %d worker(s)\n",
              nsim, nrow(plan), chunk_size, nrow(plan) - nrow(pending), ncpus))

  cl <- NULL
  if (ncpus > 1L && nrow(pending) > 0L) {
    # One cluster for every chunk. bootMer would otherwise spawn and tear down
    # workers per chunk, paying the process-start and package-load cost 25 times.
    cl <- parallel::makeCluster(ncpus)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::clusterEvalQ(cl, suppressPackageStartupMessages(library(lme4)))
  }

  for (i in seq_len(nrow(pending))) {
    chunk <- pending[i, , drop = FALSE]
    started <- Sys.time()
    run_boot_chunk(model, chunk, chunk_dir, job, cl = cl, ncpus = ncpus)
    cat(sprintf("  chunk %d/%d done (%d resamples, seed %d, %.1f min)\n",
                chunk$index, nrow(plan), chunk$nsim, chunk$seed,
                as.numeric(difftime(Sys.time(), started, units = "mins"))))
    utils::flush.console()
  }

  pooled <- tryCatch(pool_boot_chunks(plan, chunk_dir, job),
                     error = function(e) {
                       cat("bootstrap pooling failed:", conditionMessage(e), "\n"); NULL })
  if (is.null(pooled)) return(NULL)

  # confint.bootMer(parm = <subset>) is broken in lme4 2.0.1: it assigns all of
  # names(t0) as rownames of a subset-length table. lme4's own confint.merMod
  # sidesteps it by computing the full table and subsetting after, so we do too.
  tab <- tryCatch(stats::confint(pooled, level = level, type = "perc"),
                  error = function(e) {
                    cat("bootstrap CI failed:", conditionMessage(e), "\n"); NULL })
  if (is.null(tab)) return(NULL)

  keep <- intersect(terms, rownames(tab))
  lapply(keep, function(tm) list(
    term = tm,
    or_ci_lo_boot = unname(exp(tab[tm, 1])),
    or_ci_hi_boot = unname(exp(tab[tm, 2]))
  ))
}
