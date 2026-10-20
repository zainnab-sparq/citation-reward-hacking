# Synthetic-data unit tests for the Phase B descriptive analysis script.
#
# Phase B is the Haiku-4.5-only pilot (18,000 generations). The confirmatory
# E1/E2 models need the open-weights roster (base vs post checkpoints), which
# does not exist in Phase B, so this is a DESCRIPTIVE (exploratory) single-model
# GLMM: fabricated ~ condition * cell + domain + (1 | question_id). It quantifies
# the prompt-condition reward-hacking gradient within Haiku and is the $0 gate on
# whether to commit GPU spend to the open-weights confirmatory run.
#
# Same base-R stopifnot() harness as test_analysis.R (no testthat). Run from repo
# root:  Rscript src/analysis/test_phase_b_descriptive.R
# Exit 0 on all-pass, 1 on any failure.

suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
  library(jsonlite)
})

PASS_COUNT <- 0L
FAIL_COUNT <- 0L

run_test <- function(name, body) {
  cat(sprintf("  %-60s ... ", name))
  ok <- tryCatch({ body(); TRUE },
                 error = function(e) { cat("FAIL\n      ", conditionMessage(e), "\n"); FALSE })
  if (ok) { cat("ok\n"); PASS_COUNT <<- PASS_COUNT + 1L }
  else FAIL_COUNT <<- FAIL_COUNT + 1L
}

env <- new.env()
source("src/analysis/phase_b_descriptive.R", local = env)

# ---- synthetic generator ------------------------------------------------------

cat("\n[desc] generate_synthetic_inputs ----------------------------------------\n")

run_test("returns list with $generations, $verifications, $questions", function() {
  syn <- env$generate_synthetic_inputs(n_per_cell_domain = 5)
  stopifnot(all(c("generations", "verifications", "questions") %in% names(syn)))
})

run_test("single model only (Phase B is Haiku-only)", function() {
  syn <- env$generate_synthetic_inputs(n_per_cell_domain = 5)
  stopifnot(length(unique(syn$generations$model_id)) == 1)
})

run_test("questions cover 3 cells x 3 domains; all four conditions present", function() {
  syn <- env$generate_synthetic_inputs(n_per_cell_domain = 5)
  stopifnot(all(c("known_real", "obscure_real", "unanswerable") %in% syn$questions$cell))
  stopifnot(all(c("academic", "medical", "legal") %in% syn$questions$domain))
  stopifnot(all(c("neutral", "citations_expected", "abstention_permitted",
                  "length_matched") %in% unique(syn$generations$condition)))
})

run_test("verifications carry both status (j2a) and verifier_status columns", function() {
  syn <- env$generate_synthetic_inputs(n_per_cell_domain = 5)
  stopifnot(all(c("status", "verifier_status") %in% names(syn$verifications)))
})

# ---- build_descriptive_data ---------------------------------------------------

cat("\n[desc] build_descriptive_data -------------------------------------------\n")

run_test("output has the modeling columns", function() {
  syn <- env$generate_synthetic_inputs(n_per_cell_domain = 5)
  data <- env$build_descriptive_data(syn$generations, syn$verifications, syn$questions)
  required <- c("question_id", "condition", "cell", "domain", "fabricated", "n_citations")
  stopifnot(all(required %in% names(data)))
})

run_test("FABRICATED -> 1, REAL_CONFIRMED -> 0", function() {
  q <- data.frame(id = c("q1", "q2"), cell = c("known_real", "obscure_real"),
                  domain = c("academic", "academic"), stringsAsFactors = FALSE)
  g <- expand.grid(question_id = q$id, model_id = "haiku", condition = "neutral",
                   trial_id = 0L, stringsAsFactors = FALSE)
  v <- g
  v$status <- c("REAL_CONFIRMED", "FABRICATED")
  v$verifier_status <- c("REAL_CONFIRMED", "FABRICATED")
  data <- env$build_descriptive_data(g, v, q)
  data <- data[order(data$question_id), ]
  stopifnot(all(data$fabricated == c(0, 1)))
})

run_test("abstention (generation with NO citation) coded fabricated=0", function() {
  q <- data.frame(id = c("q1", "q2"), cell = "obscure_real", domain = "academic",
                  stringsAsFactors = FALSE)
  g <- expand.grid(question_id = q$id, model_id = "haiku", condition = "neutral",
                   trial_id = 0L, stringsAsFactors = FALSE)
  v <- data.frame(question_id = "q1", model_id = "haiku", condition = "neutral",
                  trial_id = 0L, status = "FABRICATED", verifier_status = "FABRICATED",
                  stringsAsFactors = FALSE)
  data <- env$build_descriptive_data(g, v, q)
  stopifnot(nrow(data) == 2)
  by_q <- setNames(data$fabricated, data$question_id)
  stopifnot(by_q[["q1"]] == 1, by_q[["q2"]] == 0)
})

run_test("cited-only-UNVERIFIABLE excluded from the modeled outcome", function() {
  q <- data.frame(id = "q1", cell = "obscure_real", domain = "academic",
                  stringsAsFactors = FALSE)
  g <- data.frame(question_id = "q1", model_id = "haiku", condition = "neutral",
                  trial_id = 0L, stringsAsFactors = FALSE)
  v <- data.frame(question_id = "q1", model_id = "haiku", condition = "neutral",
                  trial_id = 0L, status = "UNVERIFIABLE", verifier_status = "UNVERIFIABLE",
                  stringsAsFactors = FALSE)
  data <- env$build_descriptive_data(g, v, q)
  stopifnot(nrow(data) == 0)
})

run_test("status_col switches the modeled outcome (j2a vs verifier-only)", function() {
  # A citation the verifier called FABRICATED but J2a rescued to REAL_CONFIRMED:
  # under status='status' (j2a) the generation is fabricated=0; under
  # status='verifier_status' it is fabricated=1.
  q <- data.frame(id = "q1", cell = "known_real", domain = "academic",
                  stringsAsFactors = FALSE)
  g <- data.frame(question_id = "q1", model_id = "haiku", condition = "neutral",
                  trial_id = 0L, stringsAsFactors = FALSE)
  v <- data.frame(question_id = "q1", model_id = "haiku", condition = "neutral",
                  trial_id = 0L, status = "REAL_CONFIRMED", verifier_status = "FABRICATED",
                  stringsAsFactors = FALSE)
  d_j2a <- env$build_descriptive_data(g, v, q, status_col = "status")
  d_ver <- env$build_descriptive_data(g, v, q, status_col = "verifier_status")
  stopifnot(d_j2a$fabricated == 0, d_ver$fabricated == 1)
})

run_test("condition factor uses neutral as reference level", function() {
  syn <- env$generate_synthetic_inputs(n_per_cell_domain = 5)
  data <- env$build_descriptive_data(syn$generations, syn$verifications, syn$questions)
  stopifnot(levels(data$condition)[1] == "neutral")
})

# ---- model structure ----------------------------------------------------------

cat("\n[desc] model structure --------------------------------------------------\n")

run_test("fit_descriptive_model has condition*cell + domain + (1|question_id) and NO model RE", function() {
  src <- gsub("\\s+", " ", paste(deparse(body(env$fit_descriptive_model)), collapse = " "))
  stopifnot(grepl("condition * cell", src, fixed = TRUE))
  stopifnot(grepl("(1 | question_id)", src, fixed = TRUE))
  # single model -> no family/model_id grouping factor
  stopifnot(!grepl("family/model_id", src, fixed = TRUE))
  stopifnot(!grepl("| model_id", src, fixed = TRUE))
})

# ---- end-to-end fit -----------------------------------------------------------

cat("\n[desc] full pipeline on synthetic design (one glmer fit) ---------------\n")

cat("  fitting model for downstream tests ...\n")
syn <- env$generate_synthetic_inputs(n_per_cell_domain = 12)
data <- env$build_descriptive_data(syn$generations, syn$verifications, syn$questions)
model <- env$fit_descriptive_model(data)

run_test("fit_descriptive_model returns glmerMod", function() {
  stopifnot(inherits(model, "glmerMod"))
})

run_test("extract_coefficients records have the 8 base fields", function() {
  coefs <- env$extract_coefficients(model)
  required <- c("term", "estimate", "std_error", "z", "p_value",
                "odds_ratio", "or_ci_lo", "or_ci_hi")
  stopifnot(length(coefs) > 0)
  for (r in coefs) stopifnot(all(required %in% names(r)))
})

run_test("condition_contrasts returns one row per non-reference condition vs neutral", function() {
  cc <- env$condition_contrasts(model)
  conds <- vapply(cc, function(r) r$condition, character(1))
  stopifnot(all(c("citations_expected", "abstention_permitted", "length_matched") %in% conds))
  for (r in cc) stopifnot(all(c("condition", "prob", "rise_pp_vs_neutral") %in% names(r)))
})

run_test("synthetic gradient recovered: length_matched prob > abstention_permitted prob", function() {
  # The synthetic DGP bakes in pressure > neutral > abstention, mirroring the
  # observed Phase B gradient; the fitted marginal must recover that ordering.
  cc <- env$condition_contrasts(model)
  by_cond <- setNames(vapply(cc, function(r) r$prob, numeric(1)),
                      vapply(cc, function(r) r$condition, character(1)))
  neutral_p <- env$marginal_prob_neutral(model)
  stopifnot(by_cond[["length_matched"]] > by_cond[["abstention_permitted"]])
  stopifnot(by_cond[["length_matched"]] > neutral_p)
  stopifnot(by_cond[["abstention_permitted"]] < neutral_p)
})

run_test("run_descriptive_analysis returns nested structure with metadata", function() {
  res <- env$run_descriptive_analysis(data, syn$generations)
  stopifnot(all(c("coefficients", "condition_contrasts", "observed_rates",
                  "exclusion_table", "re_robustness", "metadata") %in% names(res)))
  stopifnot(all(c("alpha", "reference_condition", "status_col", "n_observations",
                  "n_generations_total", "r_version", "lme4_version",
                  "timestamp_utc") %in% names(res$metadata)))
})

# ---- hardening: reviewer-driven fixes -----------------------------------------

cat("\n[desc] hardening fixes (reviewer pass) ---------------------------------\n")

run_test("abstention_rate uses the FULL generation denominator, not the modeled frame", function() {
  # 3 neutral generations for q1: one cites a REAL, one abstains (0 citations),
  # one cites only an UNVERIFIABLE (excluded from the model frame but NOT an
  # abstention). Correct abstention rate = 1/3, NOT 1/2 (the modeled-frame bug).
  q <- data.frame(id = "q1", cell = "obscure_real", domain = "academic",
                  stringsAsFactors = FALSE)
  g <- data.frame(question_id = "q1", model_id = "haiku", condition = "neutral",
                  trial_id = 0:2, stringsAsFactors = FALSE)
  v <- data.frame(question_id = "q1", model_id = "haiku", condition = "neutral",
                  trial_id = c(0L, 2L),
                  status = c("REAL_CONFIRMED", "UNVERIFIABLE"),
                  verifier_status = c("REAL_CONFIRMED", "UNVERIFIABLE"),
                  stringsAsFactors = FALSE)
  data_hb <- env$build_descriptive_data(g, v, q)        # trial 1 abstains, trial 2 dropped
  rates <- env$summarize_observed_rates(data_hb, g)
  neutral <- Filter(function(r) r$condition == "neutral", rates)[[1]]
  stopifnot(abs(neutral$abstention_rate_all_gens - (1/3)) < 1e-9)
})

run_test("conditional_fabrication_rate is fabrications / citing (non-abstaining) responses", function() {
  # q1 neutral: trial0 fabricates (cited), trial1 abstains. Unconditional fab over
  # all gens = 1/2; conditional (among the 1 citing response) = 1/1 = 1.0.
  q <- data.frame(id = "q1", cell = "known_real", domain = "academic",
                  stringsAsFactors = FALSE)
  g <- data.frame(question_id = "q1", model_id = "haiku", condition = "neutral",
                  trial_id = 0:1, stringsAsFactors = FALSE)
  v <- data.frame(question_id = "q1", model_id = "haiku", condition = "neutral",
                  trial_id = 0L, status = "FABRICATED", verifier_status = "FABRICATED",
                  stringsAsFactors = FALSE)
  data_hb <- env$build_descriptive_data(g, v, q)
  rates <- env$summarize_observed_rates(data_hb, g)
  neutral <- Filter(function(r) r$condition == "neutral", rates)[[1]]
  stopifnot(all(c("fabrication_rate_all_gens", "conditional_fabrication_rate",
                  "mean_citations_when_citing", "abstention_rate_all_gens")
                %in% names(neutral)))
  stopifnot(abs(neutral$conditional_fabrication_rate - 1.0) < 1e-9)
  stopifnot(abs(neutral$fabrication_rate_all_gens - 0.5) < 1e-9)
  stopifnot(abs(neutral$mean_citations_when_citing - 1.0) < 1e-9)
})

run_test("restrict_to_domains drops a domain and droplevels the factor", function() {
  syn5 <- env$generate_synthetic_inputs(n_per_cell_domain = 5)
  d <- env$build_descriptive_data(syn5$generations, syn5$verifications, syn5$questions)
  d2 <- env$restrict_to_domains(d, c("academic", "medical"))
  stopifnot(!("legal" %in% levels(d2$domain)))
  stopifnot(all(as.character(d2$domain) %in% c("academic", "medical")))
})

run_test("fit_descriptive_maximal_model carries a (1 + condition | question_id) slope", function() {
  src <- gsub("\\s+", " ", paste(deparse(body(env$fit_descriptive_maximal_model)), collapse = " "))
  stopifnot(grepl("1 + condition | question_id", src, fixed = TRUE))
})

run_test("p-value underflow is flagged, never serialized as a bare 0", function() {
  recs <- env$extract_coefficients(model)
  for (r in recs) {
    stopifnot("p_underflow" %in% names(r))
    stopifnot(r$p_value > 0)            # floored away from exact 0
  }
})

run_test("exclusion_table reports modeled vs total N per domain x condition", function() {
  res <- env$run_descriptive_analysis(data, syn$generations)
  et <- res$exclusion_table
  stopifnot(length(et) > 0)
  for (r in et) stopifnot(all(c("domain", "condition", "n_total", "n_modeled",
                               "n_excluded") %in% names(r)))
})

run_test("re_robustness reports maximal-model singularity + condition SE comparison", function() {
  res <- env$run_descriptive_analysis(data, syn$generations)
  rr <- res$re_robustness
  stopifnot(all(c("maximal_formula", "maximal_singular", "maximal_converged",
                  "intercept_only_used") %in% names(rr)))
  stopifnot(is.logical(rr$intercept_only_used))
})

run_test("condition_effect_table returns marginal + per-cell OR/CI/p contrasts vs neutral", function() {
  et <- env$condition_effect_table(model)
  stopifnot(all(c("marginal", "by_cell") %in% names(et)))
  stopifnot(length(et$marginal) == 3)   # 3 non-reference conditions
  for (r in et$marginal) {
    stopifnot(all(c("contrast", "odds_ratio", "or_ci_lo", "or_ci_hi", "z", "p_value")
                  %in% names(r)))
    stopifnot(r$or_ci_lo <= r$odds_ratio + 1e-9, r$or_ci_hi >= r$odds_ratio - 1e-9)
  }
  for (r in et$by_cell) stopifnot("cell" %in% names(r))
})

run_test("run_descriptive_analysis carries condition_effects from the chosen primary model", function() {
  res <- env$run_descriptive_analysis(data, syn$generations)
  stopifnot("condition_effects" %in% names(res))
  stopifnot(all(c("marginal", "by_cell") %in% names(res$condition_effects)))
  stopifnot("primary_model" %in% names(res$metadata))  # "maximal" or "intercept_only"
})

# ---- hardening pass 2 (reviewer verification) --------------------------------

cat("\n[desc] hardening pass 2 ------------------------------------------------\n")

run_test("coefficients + condition_effects records carry separation_flag, p_value>0", function() {
  for (r in env$extract_coefficients(model)) {
    stopifnot("separation_flag" %in% names(r), is.logical(r$separation_flag))
  }
  et <- env$condition_effect_table(model)
  for (r in c(et$marginal, et$by_cell)) {
    stopifnot(all(c("separation_flag", "p_underflow") %in% names(r)))
    stopifnot(r$p_value > 0)   # contrast p-values floored away from a bare 0
  }
})

run_test("observed-rate denominator follows rates_generations (domain-restricted)", function() {
  res_all <- env$run_descriptive_analysis(data, syn$generations, fit_maximal = FALSE)
  am <- c("academic", "medical")
  gen_am <- syn$generations[grepl("^(academic|medical)", syn$generations$question_id), ]
  res_am <- env$run_descriptive_analysis(
    env$restrict_to_domains(data, am), syn$generations,
    fit_maximal = FALSE, rates_generations = gen_am)
  stopifnot(res_am$metadata$n_generations_total < res_all$metadata$n_generations_total)
  # per-condition n_total must follow the restricted denominator, not the full one
  full_total <- res_all$observed_rates[[1]]$n_total
  for (r in res_am$observed_rates) stopifnot(r$n_total < full_total)
})

cat(sprintf("\nTests passed: %d, failed: %d\n", PASS_COUNT, FAIL_COUNT))
if (FAIL_COUNT > 0L) quit(status = 1, save = "no")
