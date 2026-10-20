# Synthetic-data unit tests for the Phase C drift-interaction analysis.
#
# Same base-R stopifnot() harness as test_analysis.R / test_phase_b_descriptive.R
# (no testthat). Run from repo root:
#   Rscript src/analysis/test_phase_c_drift_interaction.R
# Exit 0 on all-pass, 1 on any failure.
#
# Two tests carry most of the weight:
#
#   * OUTCOME EQUIVALENCE. The drift test diagnoses the registered analysis, so
#     it must score the SAME outcome. Rather than trusting that the copied
#     tally/classify logic stayed in sync with phase_b_descriptive.R, this asserts
#     the two produce identical `fabricated` values on identical input.
#
#   * POSITIVE AND NEGATIVE CONTROL. A differentiality test that cannot detect
#     planted differentiality is worthless, and one that fires on null data is
#     worse. Both directions are checked on synthetic data with a known truth.

suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})

PASS_COUNT <- 0L
FAIL_COUNT <- 0L

run_test <- function(name, body) {
  cat(sprintf("  %-62s ... ", name))
  ok <- tryCatch({ body(); TRUE },
                 error = function(e) { cat("FAIL\n      ", conditionMessage(e), "\n"); FALSE })
  if (ok) { cat("ok\n"); PASS_COUNT <<- PASS_COUNT + 1L }
  else FAIL_COUNT <<- FAIL_COUNT + 1L
}

env <- new.env()
source("src/analysis/phase_c_drift_interaction.R", local = env)

# ---- synthetic fixtures -------------------------------------------------------

cat("\n[drift] generate_synthetic_inputs ---------------------------------------\n")

run_test("returns generations, verifications, questions, models, era", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 4)
  stopifnot(all(c("generations", "verifications", "questions", "models", "era")
                %in% names(syn)))
})

run_test("era frame carries the four generation merge keys", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 4)
  stopifnot(all(env$GEN_KEYS %in% names(syn$era)))
})

run_test("all three eras present and all four conditions present", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 4)
  stopifnot(all(env$ERA_LEVELS %in% unique(syn$era$era)))
  stopifnot(all(env$CONDITION_LEVELS %in% unique(syn$generations$condition)))
})

# ---- outcome construction -----------------------------------------------------

cat("\n[drift] outcome construction --------------------------------------------\n")

run_test("classify_fabricated: any bad citation -> 1", function() {
  stopifnot(env$classify_fabricated(n_bad = 2, n_good = 1, n_citations = 3) == 1L)
})

run_test("classify_fabricated: no citations -> 0 (an abstention is not a fabrication)", function() {
  stopifnot(env$classify_fabricated(n_bad = 0, n_good = 0, n_citations = 0) == 0L)
})

run_test("classify_fabricated: cited but all UNVERIFIABLE -> NA (pre-reg §5 exclusion)", function() {
  stopifnot(is.na(env$classify_fabricated(n_bad = 0, n_good = 0, n_citations = 2)))
})

run_test("OUTCOME EQUIVALENCE with the registered phase_b_descriptive.R logic", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 6)
  env_b <- new.env()
  source("src/analysis/phase_b_descriptive.R", local = env_b)
  b <- env_b$build_descriptive_data(syn$generations, syn$verifications, syn$questions)
  d <- env$modelable(env$build_drift_frame(syn$generations, syn$verifications,
                                           syn$questions, syn$models, syn$era))
  key <- function(f) paste(f$question_id, f$condition, f$model_id, f$trial_id, sep = "|")
  b_map <- setNames(b$fabricated, key(b))
  d_map <- setNames(d$fabricated, key(d))
  shared <- intersect(names(b_map), names(d_map))
  stopifnot(length(shared) > 0)
  stopifnot(length(shared) == length(d_map))          # drift frame adds no rows
  stopifnot(identical(unname(b_map[shared]), unname(d_map[shared])))
})

# ---- era attachment -----------------------------------------------------------

cat("\n[drift] era attachment --------------------------------------------------\n")

run_test("generations with no judged citation are excluded (no era, drift-immune)", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 4)
  syn$era <- syn$era[1:50, , drop = FALSE]
  frame <- env$build_drift_frame(syn$generations, syn$verifications, syn$questions,
                                 syn$models, syn$era)
  stopifnot(nrow(frame) == 50)
})

run_test("MIXED-era generations are dropped and the count is recorded", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 4)
  syn$era$era[1:7] <- env$MIXED_ERA
  frame <- env$build_drift_frame(syn$generations, syn$verifications, syn$questions,
                                 syn$models, syn$era)
  stopifnot(attr(frame, "n_mixed_era_dropped") == 7)
  stopifnot(!any(frame$era == env$MIXED_ERA))
})

run_test("era is a factor referenced at EARLY", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 4)
  frame <- env$build_drift_frame(syn$generations, syn$verifications, syn$questions,
                                 syn$models, syn$era)
  stopifnot(is.factor(frame$era))
  stopifnot(levels(frame$era)[1] == "EARLY")
})

run_test("condition is a factor referenced at neutral (matches the registered fits)", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 4)
  frame <- env$build_drift_frame(syn$generations, syn$verifications, syn$questions,
                                 syn$models, syn$era)
  stopifnot(levels(frame$condition)[1] == "neutral")
})

run_test("grounding_c is mean-centred", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 4)
  frame <- env$build_drift_frame(syn$generations, syn$verifications, syn$questions,
                                 syn$models, syn$era)
  stopifnot(abs(mean(frame$grounding_c, na.rm = TRUE)) < 1e-9)
})

run_test("stage_ordinal and family arrive from the model registry", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 4)
  frame <- env$build_drift_frame(syn$generations, syn$verifications, syn$questions,
                                 syn$models, syn$era)
  stopifnot(all(c("stage_ordinal", "family", "is_post_trained") %in% names(frame)))
  stopifnot(nlevels(frame$family) == 2)
})

# ---- exclusion reporting ------------------------------------------------------

cat("\n[drift] exclusion reporting ---------------------------------------------\n")

run_test("exclusion_rate_by era returns one row per era with a rate in [0,1]", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 6)
  frame <- env$build_drift_frame(syn$generations, syn$verifications, syn$questions,
                                 syn$models, syn$era)
  tab <- env$exclusion_rate_by(frame, "era")
  stopifnot(nrow(tab) == 3)
  stopifnot(all(tab$exclusion_rate >= 0 & tab$exclusion_rate <= 1))
  stopifnot(all(tab$n_excluded <= tab$n))
})

run_test("exclusion_rate_by supports the two registered stratifications", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 6)
  frame <- env$build_drift_frame(syn$generations, syn$verifications, syn$questions,
                                 syn$models, syn$era)
  stopifnot(nrow(env$exclusion_rate_by(frame, c("cell", "stage_ordinal"))) > 0)
  stopifnot(nrow(env$exclusion_rate_by(frame, c("condition", "is_post_trained"))) > 0)
})

run_test("Sensitivity #1 (UNVERIFIABLE -> fabricated) excludes nothing", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 6)
  frame <- env$build_drift_frame(syn$generations, syn$verifications, syn$questions,
                                 syn$models, syn$era,
                                 include_unverifiable_as_fabricated = TRUE)
  stopifnot(sum(is.na(frame$fabricated)) == 0)
})

# ---- the interaction test itself ----------------------------------------------

cat("\n[drift] interaction test (fits GLMMs, slower) ---------------------------\n")

SIMPLE_RHS <- paste("condition + era + is_post_trained + (1 | question_id)")

fit_frame <- function(differential_effect, n_questions = 8) {
  syn <- env$generate_synthetic_inputs(n_questions = n_questions,
                                       differential_effect = differential_effect)
  env$open_weights_only(env$modelable(
    env$build_drift_frame(syn$generations, syn$verifications, syn$questions,
                          syn$models, syn$era)))
}

run_test("NEGATIVE control: no planted differentiality -> interaction not significant", function() {
  res <- env$run_interaction_test(fit_frame(0), SIMPLE_RHS, "condition:era", "neg")
  stopifnot(isTRUE(res$fitted))
  stopifnot(res$df == 6)          # (4 conditions - 1) x (3 eras - 1)
  stopifnot(!res$significant)
})

run_test("POSITIVE control: planted condition-specific era shift -> significant", function() {
  res <- env$run_interaction_test(fit_frame(1.6), SIMPLE_RHS, "condition:era", "pos")
  stopifnot(isTRUE(res$fitted))
  stopifnot(res$significant)
  stopifnot(res$p_value < env$ALPHA)
})

run_test("reports the interaction coefficients so a hit can be read", function() {
  res <- env$run_interaction_test(fit_frame(1.6), SIMPLE_RHS, "condition:era", "pos")
  stopifnot(length(res$interaction_coefficients) == 6)
  terms <- vapply(res$interaction_coefficients, function(x) x$term, character(1))
  stopifnot(all(grepl("condition", terms) & grepl("era", terms)))
})

run_test("a model that cannot be fitted returns fitted=FALSE, not an abort", function() {
  res <- env$run_interaction_test(fit_frame(0), "nonexistent_column + (1 | question_id)",
                                  "condition:era", "broken")
  stopifnot(identical(res$fitted, FALSE))
})

run_test("records both formulas so the report is self-documenting", function() {
  res <- env$run_interaction_test(fit_frame(0), SIMPLE_RHS, "condition:era", "neg")
  stopifnot(grepl("condition:era", res$alt_formula, fixed = TRUE))
  stopifnot(!grepl("condition:era", res$null_formula, fixed = TRUE))
})

# ---- registered-model structure -----------------------------------------------

cat("\n[drift] registered-model structure --------------------------------------\n")

run_test("D1 base mirrors the registered E2 fixed and random structure plus era", function() {
  stopifnot(grepl("condition * is_post_trained * cell + domain", env$D1_BASE_RHS, fixed = TRUE))
  stopifnot(grepl("(1 | family/model_id)", env$D1_BASE_RHS, fixed = TRUE))
  stopifnot(grepl("(1 | question_id)", env$D1_BASE_RHS, fixed = TRUE))
  stopifnot(grepl("+ era +", env$D1_BASE_RHS, fixed = TRUE))
})

run_test("D2 base mirrors the registered E1 fixed and random structure plus era", function() {
  stopifnot(grepl("stage_ordinal * cell * domain", env$D2_BASE_RHS, fixed = TRUE))
  stopifnot(grepl("(1 | family/model_id)", env$D2_BASE_RHS, fixed = TRUE))
  stopifnot(grepl("+ era +", env$D2_BASE_RHS, fixed = TRUE))
})

run_test("D3 swaps the era factor for continuous grounding (no boundaries drawn)", function() {
  stopifnot(grepl("grounding_c", env$D3_BASE_RHS, fixed = TRUE))
  stopifnot(!grepl("era", env$D3_BASE_RHS, fixed = TRUE))
})

run_test("era boundaries are the pre-committed block edges", function() {
  stopifnot(identical(env$ERA_LEVELS, c("EARLY", "DEGRADED", "RECOVERED")))
  stopifnot(env$REFERENCE_ERA == "EARLY")
})

# ---- robustness panel ---------------------------------------------------------

cat("\n[drift] robustness panel ------------------------------------------------\n")

run_test("robustness panel is OFF by default", function() {
  # It re-estimates the REGISTERED condition contrasts. Running it before the
  # one-shot confirmatory fit would be peeking at the registered result, so it
  # must never fire as a side effect of running the drift tests.
  stopifnot(identical(env$parse_args(c("--output", "x.json"))$robustness_panel, FALSE))
})

run_test("robustness panel requires an explicit opt-in flag", function() {
  args <- env$parse_args(c("--output", "x.json", "--robustness-panel"))
  stopifnot(isTRUE(args$robustness_panel))
})

run_test("drop_zero_effort removes generations with any evidence-free judgment", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 4)
  syn$era$n_search_zero[1:20] <- 1L
  frame <- env$build_drift_frame(syn$generations, syn$verifications, syn$questions,
                                 syn$models, syn$era)
  kept <- env$drop_zero_effort(frame)
  stopifnot(nrow(kept) == nrow(frame) - 20)
  stopifnot(all(kept$n_search_zero == 0))
})

run_test("condition_contrasts returns one odds ratio per non-reference condition", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 6)
  data <- env$modelable(env$build_drift_frame(syn$generations, syn$verifications,
                                              syn$questions, syn$models, syn$era))
  fit <- env$fit_glmm(fabricated ~ condition + (1 | question_id), data)
  contrasts <- env$condition_contrasts(fit)
  stopifnot(length(contrasts) == 3)
  for (c in contrasts) {
    stopifnot(all(c("contrast", "odds_ratio", "or_ci_lo", "or_ci_hi", "p_value")
                  %in% names(c)))
    stopifnot(c$or_ci_lo <= c$odds_ratio, c$odds_ratio <= c$or_ci_hi)
  }
})

run_test("era_stratified_panel fits one model per era and reports n", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 6)
  data <- env$modelable(env$build_drift_frame(syn$generations, syn$verifications,
                                              syn$questions, syn$models, syn$era))
  cheap <- function(d) env$fit_glmm(fabricated ~ condition + (1 | question_id), d)
  panel <- env$era_stratified_panel(data, fitter = cheap)
  stopifnot(setequal(names(panel), env$ERA_LEVELS))
  for (era in env$ERA_LEVELS) {
    stopifnot(panel[[era]]$n == sum(data$era == era))
    stopifnot(length(panel[[era]]$condition_contrasts) == 3)
  }
})

run_test("era_stratified_panel survives an era whose model will not fit", function() {
  syn <- env$generate_synthetic_inputs(n_questions = 4)
  data <- env$modelable(env$build_drift_frame(syn$generations, syn$verifications,
                                              syn$questions, syn$models, syn$era))
  broken <- function(d) stop("synthetic fit failure")
  panel <- env$era_stratified_panel(data, fitter = broken)
  stopifnot(identical(panel[["EARLY"]]$fitted, FALSE))
})

# ---- summary ------------------------------------------------------------------

cat(sprintf("\n%d passed, %d failed\n", PASS_COUNT, FAIL_COUNT))
if (FAIL_COUNT > 0L) quit(status = 1L) else quit(status = 0L)
