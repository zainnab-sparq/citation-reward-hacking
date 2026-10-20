# Synthetic-data unit tests for E1 + E2 analysis scripts.
#
# Pre-reg §13 commits to "synthetic-data unit tests" alongside the frozen
# analysis scripts. This harness uses base R stopifnot() (no testthat
# dependency, keeping the lockfile minimal) and asserts the schema /
# correctness of the pure-function layer + one end-to-end glmer fit per
# script on a minimal synthetic design.
#
# Run from repo root:
#   Rscript src/analysis/test_analysis.R
#
# Exit status: 0 on all-pass, 1 on any failure. Each assertion is announced
# before it runs so failures are localizable.

suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
  library(jsonlite)
})

# ---- Test scaffolding ---------------------------------------------------------

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

# Source each analysis script into its own environment. The scripts guard their
# main block with sys.nframe() == 0L so sourcing does NOT re-run the CLI entry.
e1_env <- new.env()
source("src/analysis/e1_checkpoint_cascade.R", local = e1_env)

e2_env <- new.env()
source("src/analysis/e2_prompt_manipulation.R", local = e2_env)

# ---- E1: pure-function tests (no model fit) -----------------------------------

cat("\n[E1] generate_synthetic_e1_inputs ---------------------------------------\n")

run_test("returns list with $generations, $verifications, $questions, $models", function() {
  syn <- e1_env$generate_synthetic_e1_inputs(n_per_cell_domain = 5)
  stopifnot(is.list(syn))
  stopifnot(all(c("generations", "verifications", "questions", "models") %in% names(syn)))
})

run_test("questions cover 3 cells x 3 domains", function() {
  syn <- e1_env$generate_synthetic_e1_inputs(n_per_cell_domain = 5)
  stopifnot(all(c("known_real", "obscure_real", "unanswerable") %in% syn$questions$cell))
  stopifnot(all(c("academic", "medical", "legal") %in% syn$questions$domain))
})

run_test("models have 4 stage_ordinal levels (0-3) and is_post_trained flag", function() {
  syn <- e1_env$generate_synthetic_e1_inputs(n_per_cell_domain = 5)
  stopifnot(setequal(unique(syn$models$stage_ordinal), 0:3))
  stopifnot("is_post_trained" %in% names(syn$models))
  stopifnot(all(syn$models$is_post_trained[syn$models$stage_ordinal == 0] == 0))
  stopifnot(all(syn$models$is_post_trained[syn$models$stage_ordinal > 0] == 1))
})

run_test("verifications use status taxonomy values", function() {
  syn <- e1_env$generate_synthetic_e1_inputs(n_per_cell_domain = 5)
  stopifnot(all(syn$verifications$status %in%
                c("FABRICATED", "MISGROUNDED", "UNVERIFIABLE",
                  "REAL_CONFIRMED", "EXISTENT_METADATA_ERROR", "EXISTENT_WRONG_ID")))
})

cat("\n[E1] build_e1_data ------------------------------------------------------\n")

run_test("output has expected columns", function() {
  syn <- e1_env$generate_synthetic_e1_inputs(n_per_cell_domain = 5)
  data <- e1_env$build_e1_data(syn$generations, syn$verifications,
                               syn$questions, syn$models)
  required <- c("question_id", "condition", "model_id", "trial_id", "fabricated",
                "cell", "domain", "family", "stage_ordinal", "is_post_trained")
  stopifnot(all(required %in% names(data)))
})

run_test("FABRICATED -> 1, REAL_CONFIRMED -> 0", function() {
  # Construct a tiny hand-built dataset to verify the mapping cleanly.
  q <- data.frame(id = c("q1", "q2"), cell = c("known_real", "obscure_real"),
                  domain = c("academic", "academic"), stringsAsFactors = FALSE)
  m <- data.frame(model_id = c("m_base", "m_rlhf"), family = c("f1", "f1"),
                  stage_ordinal = c(0L, 3L), is_post_trained = c(0L, 1L),
                  stringsAsFactors = FALSE)
  g <- expand.grid(question_id = q$id, model_id = m$model_id,
                   condition = "neutral", trial_id = 0L, stringsAsFactors = FALSE)
  v <- g
  v$status <- c("REAL_CONFIRMED", "FABRICATED", "REAL_CONFIRMED", "FABRICATED")
  data <- e1_env$build_e1_data(g, v, q, m)
  data <- data[order(data$model_id, data$question_id), ]
  stopifnot(all(data$fabricated == c(0, 1, 0, 1)))
})

run_test("UNVERIFIABLE excluded from primary (default)", function() {
  q <- data.frame(id = "q1", cell = "obscure_real", domain = "academic",
                  stringsAsFactors = FALSE)
  m <- data.frame(model_id = "m_base", family = "f1",
                  stage_ordinal = 0L, is_post_trained = 0L,
                  stringsAsFactors = FALSE)
  g <- data.frame(question_id = "q1", model_id = "m_base",
                  condition = "neutral", trial_id = 0L,
                  stringsAsFactors = FALSE)
  v <- data.frame(question_id = "q1", model_id = "m_base",
                  condition = "neutral", trial_id = 0L,
                  status = "UNVERIFIABLE", stringsAsFactors = FALSE)
  data <- e1_env$build_e1_data(g, v, q, m)
  stopifnot(nrow(data) == 0)
})

run_test("UNVERIFIABLE coded as fabricated when sensitivity flag set", function() {
  q <- data.frame(id = "q1", cell = "obscure_real", domain = "academic",
                  stringsAsFactors = FALSE)
  m <- data.frame(model_id = "m_base", family = "f1",
                  stage_ordinal = 0L, is_post_trained = 0L,
                  stringsAsFactors = FALSE)
  g <- data.frame(question_id = "q1", model_id = "m_base",
                  condition = "neutral", trial_id = 0L,
                  stringsAsFactors = FALSE)
  v <- data.frame(question_id = "q1", model_id = "m_base",
                  condition = "neutral", trial_id = 0L,
                  status = "UNVERIFIABLE", stringsAsFactors = FALSE)
  data <- e1_env$build_e1_data(g, v, q, m,
                               include_unverifiable_as_fabricated = TRUE)
  stopifnot(nrow(data) == 1, data$fabricated == 1)
})

run_test("abstention (generation with NO citation) coded as fabricated=0", function() {
  # The reward-hacking thesis predicts honest models ABSTAIN on obscure/unanswerable
  # questions. An abstaining response produces zero citation rows. Pre-reg codes it
  # as fabricated=0 (honest non-fabrication), NOT as a dropped row.
  q <- data.frame(id = c("q1", "q2"), cell = c("obscure_real", "obscure_real"),
                  domain = c("academic", "academic"), stringsAsFactors = FALSE)
  m <- data.frame(model_id = "m_rlhf", family = "f1",
                  stage_ordinal = 3L, is_post_trained = 1L,
                  stringsAsFactors = FALSE)
  g <- expand.grid(question_id = q$id, model_id = m$model_id,
                   condition = "neutral", trial_id = 0L, stringsAsFactors = FALSE)
  # q1 fabricates; q2 abstains (no verification row at all).
  v <- data.frame(question_id = "q1", model_id = "m_rlhf", condition = "neutral",
                  trial_id = 0L, status = "FABRICATED", stringsAsFactors = FALSE)
  data <- e1_env$build_e1_data(g, v, q, m)
  stopifnot(nrow(data) == 2)  # both generations present, abstention NOT dropped
  by_q <- setNames(data$fabricated, data$question_id)
  stopifnot(by_q[["q1"]] == 1, by_q[["q2"]] == 0)
})

run_test("n_citations covariate present and counts citations per generation", function() {
  q <- data.frame(id = c("q1", "q2"), cell = c("obscure_real", "obscure_real"),
                  domain = c("academic", "academic"), stringsAsFactors = FALSE)
  m <- data.frame(model_id = "m_rlhf", family = "f1",
                  stage_ordinal = 3L, is_post_trained = 1L,
                  stringsAsFactors = FALSE)
  g <- expand.grid(question_id = q$id, model_id = m$model_id,
                   condition = "neutral", trial_id = 0L, stringsAsFactors = FALSE)
  # q1 has two citations; q2 abstains (zero).
  v <- data.frame(
    question_id = c("q1", "q1"), model_id = "m_rlhf", condition = "neutral",
    trial_id = 0L, status = c("FABRICATED", "REAL_CONFIRMED"),
    stringsAsFactors = FALSE)
  data <- e1_env$build_e1_data(g, v, q, m)
  stopifnot("n_citations" %in% names(data))
  by_q <- setNames(data$n_citations, data$question_id)
  stopifnot(by_q[["q1"]] == 2, by_q[["q2"]] == 0)
})

run_test("any_fabricated and any_misgrounded reported separately", function() {
  q <- data.frame(id = c("q1", "q2"), cell = c("obscure_real", "obscure_real"),
                  domain = c("academic", "academic"), stringsAsFactors = FALSE)
  m <- data.frame(model_id = "m_rlhf", family = "f1",
                  stage_ordinal = 3L, is_post_trained = 1L,
                  stringsAsFactors = FALSE)
  g <- expand.grid(question_id = q$id, model_id = m$model_id,
                   condition = "neutral", trial_id = 0L, stringsAsFactors = FALSE)
  v <- data.frame(question_id = c("q1", "q2"), model_id = "m_rlhf",
                  condition = "neutral", trial_id = 0L,
                  status = c("FABRICATED", "MISGROUNDED"), stringsAsFactors = FALSE)
  data <- e1_env$build_e1_data(g, v, q, m)
  stopifnot(all(c("any_fabricated", "any_misgrounded") %in% names(data)))
  by_q_fab <- setNames(data$any_fabricated, data$question_id)
  by_q_mis <- setNames(data$any_misgrounded, data$question_id)
  stopifnot(by_q_fab[["q1"]] == 1, by_q_mis[["q1"]] == 0)
  stopifnot(by_q_fab[["q2"]] == 0, by_q_mis[["q2"]] == 1)
  # combined fabricated outcome (the modeled one) is 1 for both
  stopifnot(all(data$fabricated == 1))
})

run_test("fit_e1_model is random-INTERCEPTS confirmatory (no stage_ordinal slope)", function() {
  src <- gsub("\\s+", " ", paste(deparse(body(e1_env$fit_e1_model)), collapse = " "))
  stopifnot(grepl("(1 | family/model_id)", src, fixed = TRUE))
  stopifnot(grepl("(1 | question_id)", src, fixed = TRUE))
  stopifnot(!grepl("1 + stage_ordinal | family", src, fixed = TRUE))  # no random slope
})

run_test("fit_e1_extended_model exists with identifiability-repaired slopes", function() {
  stopifnot(is.function(e1_env$fit_e1_extended_model))
  src <- gsub("\\s+", " ", paste(deparse(body(e1_env$fit_e1_extended_model)), collapse = " "))
  # stage slope only at family level (constant within model_id)
  stopifnot(grepl("1 + stage_ordinal + cell | family", src, fixed = TRUE))
  stopifnot(grepl("1 + cell | model_id", src, fixed = TRUE))
})

cat("\n[E1] apply_holm (no model fit, hand-built records) ---------------------\n")

run_test("all records get p_holm and significant_holm fields", function() {
  recs <- list(
    list(term = "(Intercept)",    p_value = 0.5),
    list(term = "stage_ordinal",  p_value = 0.01),
    list(term = "domainmedical",  p_value = 0.04),
    list(term = "stage_ordinal:cellobscure_real", p_value = 0.001)
  )
  out <- e1_env$apply_holm(recs, family_terms = c("stage_ordinal",
                                                  "stage_ordinal:cellobscure_real"))
  stopifnot(all(vapply(out, function(r) "p_holm" %in% names(r), logical(1))))
  stopifnot(all(vapply(out, function(r) "significant_holm" %in% names(r), logical(1))))
})

run_test("p_holm is NA for terms outside the family", function() {
  recs <- list(
    list(term = "(Intercept)",    p_value = 0.5),
    list(term = "stage_ordinal",  p_value = 0.01),
    list(term = "domainmedical",  p_value = 0.04)
  )
  out <- e1_env$apply_holm(recs, family_terms = c("stage_ordinal"))
  by_term <- setNames(out, vapply(out, `[[`, character(1), "term"))
  stopifnot(is.na(by_term[["(Intercept)"]]$p_holm))
  stopifnot(is.na(by_term[["domainmedical"]]$p_holm))
  stopifnot(!is.na(by_term[["stage_ordinal"]]$p_holm))
})

run_test("Holm correction inflates raw p-values monotonically", function() {
  recs <- list(
    list(term = "a", p_value = 0.01),
    list(term = "b", p_value = 0.02),
    list(term = "c", p_value = 0.03)
  )
  out <- e1_env$apply_holm(recs, family_terms = c("a", "b", "c"))
  for (r in out) stopifnot(r$p_holm >= r$p_value)
})

# ---- E1: end-to-end model fit (one slow path, reused for multiple assertions) -

cat("\n[E1] full pipeline on minimal synthetic design (one glmer fit) --------\n")

cat("  fitting model for downstream tests ...\n")
syn_e1 <- e1_env$generate_synthetic_e1_inputs(n_per_cell_domain = 8)
data_e1 <- e1_env$build_e1_data(syn_e1$generations, syn_e1$verifications,
                                syn_e1$questions, syn_e1$models)
model_e1 <- e1_env$fit_e1_model(data_e1)

run_test("fit_e1_model returns glmerMod object", function() {
  stopifnot(inherits(model_e1, "glmerMod"))
})

run_test("extract_coefficients records have 8 base fields", function() {
  coefs <- e1_env$extract_coefficients(model_e1)
  required <- c("term", "estimate", "std_error", "z", "p_value",
                "odds_ratio", "or_ci_lo", "or_ci_hi")
  stopifnot(length(coefs) > 0)
  for (r in coefs) stopifnot(all(required %in% names(r)))
})

run_test("odds_ratio == exp(estimate); CI bounds frame estimate", function() {
  coefs <- e1_env$extract_coefficients(model_e1)
  for (r in coefs) {
    stopifnot(abs(r$odds_ratio - exp(r$estimate)) < 1e-9)
    stopifnot(r$or_ci_lo <= r$odds_ratio + 1e-9)
    stopifnot(r$or_ci_hi >= r$odds_ratio - 1e-9)
  }
})

run_test("check_directional_h1_1 returns expected schema", function() {
  d <- e1_env$check_directional_h1_1(model_e1, data_e1)
  stopifnot(all(c("observed_rise_pp", "passes_threshold",
                  "base_prob", "rlhf_prob") %in% names(d)))
  stopifnot(is.logical(d$passes_threshold))
  stopifnot(d$base_prob >= 0 && d$base_prob <= 1)
  stopifnot(d$rlhf_prob >= 0 && d$rlhf_prob <= 1)
})

run_test("run_e1_analysis returns nested structure with metadata", function() {
  results <- e1_env$run_e1_analysis(data_e1)
  stopifnot(all(c("primary", "sensitivity_stage_as_factor", "metadata") %in% names(results)))
  stopifnot(all(c("formula", "coefficients", "directional_h1_1",
                  "n_observations") %in% names(results$primary)))
  stopifnot(all(c("alpha", "directional_threshold_pp", "primary_interaction_term",
                  "r_version", "lme4_version", "timestamp_utc") %in%
                names(results$metadata)))
})

run_test("bootstrap CIs are produced from resumable chunks on disk", function() {
  # The registered §6 bootstrap runs for days and cannot be restarted from
  # scratch every time the machine reboots. This asserts the analysis actually
  # goes through the chunked path, not merely that the chunked module exists.
  chunk_dir <- file.path(tempdir(), paste0("e1_itest_", as.integer(Sys.time())))
  Sys.setenv(CITATION_BOOT_CHUNK_DIR = chunk_dir)
  on.exit({ Sys.unsetenv("CITATION_BOOT_CHUNK_DIR")
            unlink(chunk_dir, recursive = TRUE) }, add = TRUE)

  results <- e1_env$run_e1_analysis(data_e1, bootstrap_nsim = 4,
                                    chunk_job = "e1_itest")
  cis <- results$primary$bootstrap_cis
  stopifnot(!is.null(cis), length(cis) >= 1L)
  stopifnot(all(c("term", "or_ci_lo_boot", "or_ci_hi_boot") %in% names(cis[[1]])))
  stopifnot(length(list.files(chunk_dir, pattern = "^e1_itest_chunk")) >= 1L)
})

run_test("a completed chunk is reused rather than refitted", function() {
  chunk_dir <- file.path(tempdir(), paste0("e1_resume_", as.integer(Sys.time())))
  Sys.setenv(CITATION_BOOT_CHUNK_DIR = chunk_dir)
  on.exit({ Sys.unsetenv("CITATION_BOOT_CHUNK_DIR")
            unlink(chunk_dir, recursive = TRUE) }, add = TRUE)

  first <- e1_env$run_e1_analysis(data_e1, bootstrap_nsim = 4,
                                  chunk_job = "e1_resume")$primary$bootstrap_cis
  stamp <- file.info(list.files(chunk_dir, full.names = TRUE))$mtime
  second <- e1_env$run_e1_analysis(data_e1, bootstrap_nsim = 4,
                                   chunk_job = "e1_resume")$primary$bootstrap_cis
  stopifnot(isTRUE(all.equal(first, second)))
  stopifnot(identical(stamp, file.info(list.files(chunk_dir, full.names = TRUE))$mtime))
})

cat("\n[E1] round-3 panel additions -------------------------------------------\n")

run_test("summarize_status_split surfaces known-real global-abstention placebo", function() {
  # pre-reg §6 #11: known-real abstention by stage. If it rises/falls with stage
  # like obscure-real, the stage effect is global (not citation-specific).
  df <- data.frame(
    cell = rep(c("known_real", "obscure_real"), each = 4),
    stage_ordinal = rep(c(0L, 1L, 2L, 3L), times = 2),
    any_fabricated = 0, any_misgrounded = 0, fabricated = 0,
    n_citations = c(1, 1, 0, 0, 0, 0, 0, 0), stringsAsFactors = FALSE)
  sp <- e1_env$summarize_status_split(df)
  stopifnot("abstention_placebo" %in% names(sp))
  stopifnot(all(c("by_stage", "slope_base_to_max_pp") %in% names(sp$abstention_placebo)))
  kr <- sp$abstention_placebo$by_stage
  stopifnot(all(kr$cell == "known_real"))           # placebo isolates known_real
  stopifnot(is.numeric(sp$abstention_placebo$slope_base_to_max_pp))
})

run_test(".profile_escalation_triggered fires only within 2pp of the threshold", function() {
  trg <- e1_env$.profile_escalation_triggered
  stopifnot(isTRUE(trg(0.10)))            # exactly at 10pp threshold
  stopifnot(isTRUE(trg(0.115)))           # within 2pp
  stopifnot(isFALSE(trg(0.20)))           # far above
  stopifnot(isFALSE(trg(NA_real_)))       # missing -> no escalation
})

run_test("compute_profile_cis returns OR-scale profile CI schema or NULL", function() {
  stopifnot(is.function(e1_env$compute_profile_cis))
  ci <- e1_env$compute_profile_cis(model_e1, "stage_ordinal:cellobscure_real")
  stopifnot(is.null(ci) ||
            all(c("term", "or_ci_lo_profile", "or_ci_hi_profile") %in% names(ci[[1]])))
})

run_test("check_directional_h1_1 reports per-domain rise (3-way de-marginalization)", function() {
  d <- e1_env$check_directional_h1_1(model_e1, data_e1)
  stopifnot("by_domain" %in% names(d))
  stopifnot(is.null(d$by_domain) ||
            all(vapply(d$by_domain, function(x)
              all(c("domain", "rise_pp") %in% names(x)), logical(1))))
})

# ---- E2: pure-function tests --------------------------------------------------

cat("\n[E2] generate_synthetic_e2_inputs --------------------------------------\n")

run_test("returns list with $generations, $verifications, $questions, $models", function() {
  syn <- e2_env$generate_synthetic_e2_inputs(n_per_cell_domain = 5)
  stopifnot(all(c("generations", "verifications", "questions", "models") %in% names(syn)))
})

run_test("models include both open-weights and closed-API", function() {
  syn <- e2_env$generate_synthetic_e2_inputs(n_per_cell_domain = 5)
  stopifnot(any(syn$models$is_open_weights == 1))
  stopifnot(any(syn$models$is_open_weights == 0))
})

run_test("open-weights models include both base and post-trained", function() {
  syn <- e2_env$generate_synthetic_e2_inputs(n_per_cell_domain = 5)
  open <- syn$models[syn$models$is_open_weights == 1, ]
  stopifnot(any(open$is_post_trained == 0))
  stopifnot(any(open$is_post_trained == 1))
})

run_test("conditions include all four E2 levels", function() {
  syn <- e2_env$generate_synthetic_e2_inputs(n_per_cell_domain = 5)
  stopifnot(all(c("neutral", "citations_expected", "abstention_permitted",
                  "length_matched")
                %in% unique(syn$generations$condition)))
})

cat("\n[E2] build_e2_data -----------------------------------------------------\n")

run_test("output has expected columns including is_open_weights", function() {
  syn <- e2_env$generate_synthetic_e2_inputs(n_per_cell_domain = 5)
  data <- e2_env$build_e2_data(syn$generations, syn$verifications,
                               syn$questions, syn$models)
  required <- c("question_id", "condition", "model_id", "trial_id", "fabricated",
                "cell", "domain", "family", "is_open_weights", "is_post_trained")
  stopifnot(all(required %in% names(data)))
})

run_test("condition factor uses neutral as reference level", function() {
  syn <- e2_env$generate_synthetic_e2_inputs(n_per_cell_domain = 5)
  data <- e2_env$build_e2_data(syn$generations, syn$verifications,
                               syn$questions, syn$models)
  stopifnot(levels(data$condition)[1] == "neutral")
})

run_test("E2 includes 4th length_matched control condition", function() {
  syn <- e2_env$generate_synthetic_e2_inputs(n_per_cell_domain = 5)
  stopifnot("length_matched" %in% unique(syn$generations$condition))
  data <- e2_env$build_e2_data(syn$generations, syn$verifications,
                               syn$questions, syn$models)
  stopifnot("length_matched" %in% levels(data$condition))
})

run_test("E2 abstention (no citation) coded as fabricated=0", function() {
  q <- data.frame(id = c("q1", "q2"), cell = c("obscure_real", "obscure_real"),
                  domain = c("academic", "academic"), stringsAsFactors = FALSE)
  m <- data.frame(model_id = "m_open_post", family = "f1", is_post_trained = 1L,
                  is_open_weights = 1L, stringsAsFactors = FALSE)
  g <- expand.grid(question_id = q$id, model_id = m$model_id,
                   condition = "neutral", trial_id = 0L, stringsAsFactors = FALSE)
  v <- data.frame(question_id = "q1", model_id = "m_open_post", condition = "neutral",
                  trial_id = 0L, status = "MISGROUNDED", stringsAsFactors = FALSE)
  data <- e2_env$build_e2_data(g, v, q, m)
  stopifnot(nrow(data) == 2)
  by_q <- setNames(data$fabricated, data$question_id)
  stopifnot(by_q[["q1"]] == 1, by_q[["q2"]] == 0)
})

run_test("fit_e2_primary_model is random-INTERCEPTS confirmatory (no condition slope)", function() {
  src <- gsub("\\s+", " ", paste(deparse(body(e2_env$fit_e2_primary_model)), collapse = " "))
  stopifnot(grepl("(1 | family/model_id)", src, fixed = TRUE))
  stopifnot(grepl("(1 | question_id)", src, fixed = TRUE))
  stopifnot(!grepl("1 + condition | family", src, fixed = TRUE))
})

run_test("fit_e2_extended_model exists with condition slopes at both levels", function() {
  stopifnot(is.function(e2_env$fit_e2_extended_model))
  src <- gsub("\\s+", " ", paste(deparse(body(e2_env$fit_e2_extended_model)), collapse = " "))
  stopifnot(grepl("1 + condition + cell | family", src, fixed = TRUE))
  stopifnot(grepl("1 + condition + cell | model_id", src, fixed = TRUE))
})

# ---- E2: end-to-end model fit (one slow path) ---------------------------------

cat("\n[E2] full pipeline on minimal synthetic design (two glmer fits) -------\n")

cat("  fitting primary + secondary models for downstream tests ...\n")
syn_e2 <- e2_env$generate_synthetic_e2_inputs(n_per_cell_domain = 6)
data_e2 <- e2_env$build_e2_data(syn_e2$generations, syn_e2$verifications,
                                syn_e2$questions, syn_e2$models)
model_primary_e2 <- e2_env$fit_e2_primary_model(data_e2)
model_secondary_e2 <- e2_env$fit_e2_secondary_model(data_e2)

run_test("fit_e2_primary_model returns glmerMod (open-weights subsample)", function() {
  stopifnot(inherits(model_primary_e2, "glmerMod"))
})

run_test("fit_e2_secondary_model returns glmerMod (closed-API post only)", function() {
  stopifnot(inherits(model_secondary_e2, "glmerMod"))
})

run_test("check_directional_h1_2 returns interaction-delta schema", function() {
  d <- e2_env$check_directional_h1_2(model_primary_e2, data_e2)
  stopifnot(all(c("p_neutral_base", "p_neutral_post", "p_cite_base",
                  "p_cite_post", "delta_base", "delta_post",
                  "interaction_delta_pp", "passes_threshold") %in% names(d)))
  stopifnot(is.logical(d$passes_threshold))
})

run_test("check_directional_h1_2b returns rise schema", function() {
  d <- e2_env$check_directional_h1_2b(model_secondary_e2)
  stopifnot(all(c("p_neutral", "p_cite", "rise_pp",
                  "passes_threshold") %in% names(d)))
  stopifnot(is.logical(d$passes_threshold))
})

# Fitted once and reused by the gate tests below; run_e2_analysis fits four
# models per call, so re-running it per assertion dominates the suite runtime.
results_e2_default <- e2_env$run_e2_analysis(data_e2)

run_test("run_e2_analysis returns nested structure with both blocks", function() {
  results <- results_e2_default
  stopifnot(all(c("primary_open_weights", "secondary_closed_api", "metadata")
                %in% names(results)))
  stopifnot(all(c("formula", "coefficients", "directional_h1_2")
                %in% names(results$primary_open_weights)))
  stopifnot(all(c("formula", "coefficients", "directional_h1_2b")
                %in% names(results$secondary_closed_api)))
})

cat("\n[E2] round-3 panel additions -------------------------------------------\n")

run_test("fit_e2_primary_model rejects non-numeric is_post_trained (type guard)", function() {
  # is_post_trained must enter as numeric so the Holm family term
  # 'conditioncitations_expected:is_post_trained' is named correctly; a factor
  # encoding silently renames the coefficient and breaks the Holm lookup.
  bad <- data_e2
  bad$is_post_trained <- factor(bad$is_post_trained)
  err <- tryCatch({ e2_env$fit_e2_primary_model(bad); NULL },
                  error = function(e) conditionMessage(e))
  stopifnot(!is.null(err))
})

run_test("check_directional_h1_2 reports per-domain interaction delta", function() {
  d <- e2_env$check_directional_h1_2(model_primary_e2, data_e2)
  stopifnot("by_domain" %in% names(d))
  stopifnot(is.null(d$by_domain) ||
            all(vapply(d$by_domain, function(x)
              all(c("domain", "interaction_delta_pp") %in% names(x)), logical(1))))
})

run_test("compute_profile_cis (E2) returns OR-scale profile CI schema or NULL", function() {
  stopifnot(is.function(e2_env$compute_profile_cis))
  ci <- e2_env$compute_profile_cis(model_primary_e2,
                                   "conditioncitations_expected:is_post_trained")
  stopifnot(is.null(ci) ||
            all(c("term", "or_ci_lo_profile", "or_ci_hi_profile") %in% names(ci[[1]])))
})

cat("\n[E2] amendment #10 de-confounding contrast -----------------------------\n")

# The bin rule is hand-tested with literal deltas and CIs because it is the part
# the pre-registration fixes in advance: one computed delta -> one decision. The
# emmeans plumbing can move; these boundaries cannot. Units are proportions, so
# the +/-5pp boundary is 0.05 and the 2pp boundary tolerance is 0.02.

run_test("bin: delta >= +5pp with CI clear of the boundary is citation-cue", function() {
  b <- e2_env$.amendment10_bin(0.09, 0.070, 0.110)
  stopifnot(identical(b, "citation_cue_supported"))
})

run_test("bin: |delta| < 5pp is not_separable", function() {
  stopifnot(identical(e2_env$.amendment10_bin(0.01, -0.010, 0.030), "not_separable"))
  stopifnot(identical(e2_env$.amendment10_bin(-0.02, -0.040, 0.000), "not_separable"))
})

run_test("bin: delta <= -5pp with CI clear of the boundary is length-dominant", function() {
  b <- e2_env$.amendment10_bin(-0.09, -0.110, -0.070)
  stopifnot(identical(b, "length_demand_dominant"))
})

run_test("bin: within 2pp of +5pp AND CI straddling it is boundary_ambiguous", function() {
  # point estimate 6pp (within 2pp of the +5pp boundary), CI 3pp..9pp crosses it.
  b <- e2_env$.amendment10_bin(0.06, 0.030, 0.090)
  stopifnot(identical(b, "boundary_ambiguous"))
})

run_test("bin: within 2pp of -5pp AND CI straddling it is boundary_ambiguous", function() {
  b <- e2_env$.amendment10_bin(-0.06, -0.090, -0.030)
  stopifnot(identical(b, "boundary_ambiguous"))
})

run_test("bin: near the boundary but CI NOT crossing keeps the definite bin", function() {
  # 6pp point estimate, CI 5.5pp..6.5pp does not cross +5pp -> definite bin.
  stopifnot(identical(e2_env$.amendment10_bin(0.06, 0.055, 0.065),
                      "citation_cue_supported"))
  # 4pp point estimate, CI 3.5pp..4.5pp does not cross +5pp -> not_separable.
  stopifnot(identical(e2_env$.amendment10_bin(0.04, 0.035, 0.045), "not_separable"))
})

run_test("bin: far from the boundary is unaffected by a wide CI", function() {
  # The boundary rule requires the POINT ESTIMATE within 2pp; a wide CI alone
  # must not trigger it, or every imprecise estimate becomes ambiguous.
  stopifnot(identical(e2_env$.amendment10_bin(0.00, -0.200, 0.200), "not_separable"))
})

run_test("bin: NA delta returns NA rather than a bin", function() {
  stopifnot(is.na(e2_env$.amendment10_bin(NA_real_, NA_real_, NA_real_)))
})

run_test("compute_amendment10_contrast returns the registered schema", function() {
  a <- e2_env$compute_amendment10_contrast(model_primary_e2, data_e2)
  stopifnot(all(c("delta_pp", "ci_lo_pp", "ci_hi_pp", "bin",
                  "p_citations_expected", "p_length_matched",
                  "odds_ratio_ordering", "secondary_all_open_weights")
                %in% names(a)))
})

run_test("amendment10 delta equals the difference of the two marginal probs", function() {
  # The bin is determined by the pp-scale marginal contrast, so the reported
  # delta must be exactly P(cite) - P(length) on the probability scale, not an
  # odds ratio and not a link-scale difference.
  a <- e2_env$compute_amendment10_contrast(model_primary_e2, data_e2)
  stopifnot(isTRUE(all.equal(a$delta_pp,
                             a$p_citations_expected - a$p_length_matched,
                             tolerance = 1e-8)))
})

run_test("amendment10 CI brackets the point estimate", function() {
  a <- e2_env$compute_amendment10_contrast(model_primary_e2, data_e2)
  stopifnot(a$ci_lo_pp <= a$delta_pp, a$delta_pp <= a$ci_hi_pp)
})

run_test("amendment10 secondary summary carries a real CI, not NULLs", function() {
  # emmeans labels confidence limits asymp.LCL/asymp.UCL on an asymptotic
  # contrast and lower.CL/upper.CL otherwise, and which you get depends on the
  # reference grid. Reading one pair by name returned NULL limits here while the
  # primary contrast looked fine, so assert the bounds exist and bracket.
  a <- e2_env$compute_amendment10_contrast(model_primary_e2, data_e2)
  s <- a$secondary_all_open_weights
  stopifnot(!is.null(s))
  stopifnot(is.numeric(s$ci_lo_pp), length(s$ci_lo_pp) == 1, !is.na(s$ci_lo_pp))
  stopifnot(is.numeric(s$ci_hi_pp), length(s$ci_hi_pp) == 1, !is.na(s$ci_hi_pp))
  stopifnot(s$ci_lo_pp <= s$delta_pp, s$delta_pp <= s$ci_hi_pp)
  stopifnot(s$share_post_trained > 0, s$share_post_trained <= 1)
})

run_test(".emm_ci_bounds reads both emmeans confidence-limit namings", function() {
  finite <- data.frame(estimate = 0.1, lower.CL = 0.05, upper.CL = 0.15)
  asymp <- data.frame(estimate = 0.1, asymp.LCL = 0.05, asymp.UCL = 0.15)
  for (df in list(finite, asymp)) {
    b <- e2_env$.emm_ci_bounds(df)
    stopifnot(isTRUE(all.equal(b$lo, 0.05)), isTRUE(all.equal(b$hi, 0.15)))
  }
  err <- tryCatch({ e2_env$.emm_ci_bounds(data.frame(estimate = 1)); NULL },
                  error = function(e) conditionMessage(e))
  stopifnot(!is.null(err))  # must fail loudly, never return NULL limits
})

run_test("amendment10 is evaluated at is_post_trained = 1, not at its mean", function() {
  # The registered restriction: base checkpoints do not follow the prompt
  # conditions, so the contrast is only defined where the model responds. If the
  # covariate were left at its mean this would silently return a different
  # estimand, so pin it and compare against an explicit at= grid.
  a <- e2_env$compute_amendment10_contrast(model_primary_e2, data_e2)
  emm <- emmeans::emmeans(model_primary_e2, ~ condition,
                          at = list(is_post_trained = 1),
                          weights = "proportional")
  pr <- as.data.frame(emmeans::regrid(emm))
  est <- pr[[e2_env$.emm_estimate_column(pr)]]
  expected <- est[pr$condition == "citations_expected"] -
              est[pr$condition == "length_matched"]
  stopifnot(isTRUE(all.equal(a$delta_pp, unname(expected), tolerance = 1e-8)))
})

run_test("amendment10 delta differs from the odds-ratio ordering it reports", function() {
  # Guards the regrid() step. If the contrast were taken on a type="response"
  # grid it would come back as an odds ratio, and delta_pp would stop being a
  # probability difference; a pp-scale delta must stay inside [-1, 1].
  a <- e2_env$compute_amendment10_contrast(model_primary_e2, data_e2)
  stopifnot(a$delta_pp >= -1, a$delta_pp <= 1)
  stopifnot(a$p_citations_expected >= 0, a$p_citations_expected <= 1)
  stopifnot(a$p_length_matched >= 0, a$p_length_matched <= 1)
})

run_test("run_e2_analysis gates amendment10 off by default and on when asked", function() {
  # One gated run, asserted from both sides. run_e2_analysis fits four models
  # including the maximal random-slopes structure, so this deliberately does not
  # spend a second full fit just to check the default; the default-off half is
  # asserted on results_e2_default, fitted once above.
  stopifnot(is.null(results_e2_default$primary_open_weights$amendment10_contrast))
  results_on <- e2_env$run_e2_analysis(data_e2, amendment10 = TRUE)
  a <- results_on$primary_open_weights$amendment10_contrast
  stopifnot(!is.null(a), "bin" %in% names(a))
  stopifnot(isTRUE(results_on$metadata$amendment10_computed))
})

run_test("parse_args exposes --amendment10 and --unverifiable-as-fabricated", function() {
  a <- e2_env$parse_args(c("--amendment10", "--unverifiable-as-fabricated"))
  stopifnot(isTRUE(a$amendment10))
  stopifnot(isTRUE(a$unverifiable_as_fabricated))
  b <- e2_env$parse_args(character(0))
  stopifnot(identical(b$amendment10, FALSE))
  stopifnot(identical(b$unverifiable_as_fabricated, FALSE))
})

cat("\n[sensitivity #1] UNVERIFIABLE-as-fabricated plumbing ------------------\n")

run_test("E1 parse_args exposes --unverifiable-as-fabricated, default FALSE", function() {
  stopifnot(isTRUE(e1_env$parse_args(c("--unverifiable-as-fabricated"))$unverifiable_as_fabricated))
  stopifnot(identical(e1_env$parse_args(character(0))$unverifiable_as_fabricated, FALSE))
})

run_test("build_e1_data tags the frame with its outcome coding", function() {
  syn <- e1_env$generate_synthetic_e1_inputs(n_per_cell_domain = 5)
  d0 <- e1_env$build_e1_data(syn$generations, syn$verifications,
                             syn$questions, syn$models)
  d1 <- e1_env$build_e1_data(syn$generations, syn$verifications,
                             syn$questions, syn$models,
                             include_unverifiable_as_fabricated = TRUE)
  stopifnot(identical(attr(d0, "unverifiable_as_fabricated"), FALSE))
  stopifnot(identical(attr(d1, "unverifiable_as_fabricated"), TRUE))
})

run_test("build_e2_data tags the frame with its outcome coding", function() {
  d0 <- e2_env$build_e2_data(syn_e2$generations, syn_e2$verifications,
                             syn_e2$questions, syn_e2$models)
  d1 <- e2_env$build_e2_data(syn_e2$generations, syn_e2$verifications,
                             syn_e2$questions, syn_e2$models,
                             include_unverifiable_as_fabricated = TRUE)
  stopifnot(identical(attr(d0, "unverifiable_as_fabricated"), FALSE))
  stopifnot(identical(attr(d1, "unverifiable_as_fabricated"), TRUE))
})

run_test("E2 metadata records the outcome coding, so artifacts are tellable apart", function() {
  # Two runs that differ only in this coding must not produce indistinguishable
  # JSON; the sensitivity artifact has to declare itself.
  stopifnot(identical(results_e2_default$metadata$unverifiable_as_fabricated, FALSE))
  stopifnot(identical(results_e2_default$metadata$amendment10_computed, FALSE))
})

run_test("recoding UNVERIFIABLE as fabricated never drops rows", function() {
  # The registered rationale for sensitivity #1: exclusion conditions on "the
  # model cited something verifiable", which is post-treatment. Under the recode
  # nothing is excluded, so the frame must be at least as large.
  d0 <- e2_env$build_e2_data(syn_e2$generations, syn_e2$verifications,
                             syn_e2$questions, syn_e2$models)
  d1 <- e2_env$build_e2_data(syn_e2$generations, syn_e2$verifications,
                             syn_e2$questions, syn_e2$models,
                             include_unverifiable_as_fabricated = TRUE)
  stopifnot(nrow(d1) >= nrow(d0))
})

# ---- Cross-script invariants --------------------------------------------------

cat("\n[cross] invariants ------------------------------------------------------\n")

run_test("E1 and E2 apply_holm both treat all records uniformly", function() {
  recs <- list(
    list(term = "a", p_value = 0.01),
    list(term = "b", p_value = 0.5)
  )
  e1_out <- e1_env$apply_holm(recs, family_terms = "a")
  e2_out <- e2_env$apply_holm(recs, family_terms = "a")
  stopifnot(all(c("p_holm", "significant_holm") %in% names(e1_out[[1]])))
  stopifnot(all(c("p_holm", "significant_holm") %in% names(e1_out[[2]])))
  stopifnot(all(c("p_holm", "significant_holm") %in% names(e2_out[[1]])))
  stopifnot(all(c("p_holm", "significant_holm") %in% names(e2_out[[2]])))
})

run_test("E1 and E2 extract_coefficients produce identical field sets", function() {
  e1_coefs <- e1_env$extract_coefficients(model_e1)
  e2_coefs <- e2_env$extract_coefficients(model_primary_e2)
  stopifnot(setequal(names(e1_coefs[[1]]), names(e2_coefs[[1]])))
})

# ---- Summary ------------------------------------------------------------------

cat(sprintf("\nTests passed: %d, failed: %d\n", PASS_COUNT, FAIL_COUNT))
if (FAIL_COUNT > 0L) quit(status = 1, save = "no")
