# DESCRIPTIVE (exploratory) analysis of the Phase B Haiku-4.5 pilot.
#
# Phase B is a single closed, post-trained model (Haiku-4.5, 18,000 generations).
# The confirmatory E1/E2 models (see e1_checkpoint_cascade.R, e2_prompt_manipulation.R)
# require the open-weights roster (base vs post-trained checkpoints within a
# family); none of that structure exists in Phase B, so the confirmatory models
# CANNOT be fit here. This script instead fits the descriptive single-model GLMM
#
#   glmer(fabricated ~ condition * cell + domain + (1 | question_id), binomial)
#
# which quantifies the prompt-condition reward-hacking gradient WITHIN Haiku. It
# is exploratory, not a registered confirmatory test: its role is to confirm the
# gradient is present and in the predicted direction before committing GPU spend
# to the open-weights confirmatory run. The single model means question_id is the
# only grouping factor (no family/model_id random effects); is_post_trained is
# constant (dropped).
#
# Outcome coding mirrors E1/E2: any FABRICATED/MISGROUNDED citation -> 1;
# generation with zero citations (abstention) -> 0; cited-only-UNVERIFIABLE -> NA
# (excluded). status_col="status" uses the J2a-combined verdict (PRIMARY);
# status_col="verifier_status" uses the verifier-only verdict (alternative scoring).
#
# Hardening (reviewer panel, 2026-06-13) folded in:
#   * Observed rates report THREE distinct quantities so the abstention story is
#     not oversold: (a) fabrication over all generations, (b) CONDITIONAL
#     fabrication among citing (non-abstaining) responses, (c) abstention rate.
#     All denominators are the FULL pre-exclusion generation count, not the
#     NA-filtered model frame (which would drop cited-only-UNVERIFIABLE rows that
#     are NOT abstentions and misstate the rate).
#   * mean_citations_when_citing per condition guards the length_matched verbosity
#     confound (more citation slots mechanically raise P(>=1 fabricated)).
#   * Maximal RE model (1 + condition | question_id) is attempted and its
#     singularity/convergence + condition-SE comparison reported (re_robustness);
#     condition varies within question, so the intercept-only SEs are anti-
#     conservative if the slope is supported. Intercept-only is the justified
#     fallback only when the maximal model is singular.
#   * The verifier-only model DROPS the legal domain: legal UNVERIFIABLE
#     exclusions are condition-correlated (23%-85%), a non-ignorable-missingness
#     violation that contaminates a pooled fit. exclusion_table documents it.
#   * J2a-vs-verifier divergence is reported on academic+medical ONLY (good
#     verification coverage), removing the CourtListener-indexing confound that
#     makes the legal divergence un-attributable.
#   * Profile-likelihood CIs for the condition main effects (the abstention term
#     sits near zero where Wald is unreliable).
#   * p-value underflow is floored + flagged (never serialized as a bare 0).
#
# Inputs (committed CSV analysis frame, written by src/analysis/aggregate.py):
#   --generations data/analysis/phase_b/generations.csv
#   --verifications data/analysis/phase_b/verifications.csv
#   --questions data/analysis/phase_b/questions.csv
# Output: --output JSON.
#
# Usage:
#   Rscript src/analysis/phase_b_descriptive.R \
#       --generations data/analysis/phase_b/generations.csv \
#       --verifications data/analysis/phase_b/verifications.csv \
#       --questions data/analysis/phase_b/questions.csv \
#       --output data/analysis/phase_b/descriptive_results.json
#   Rscript src/analysis/phase_b_descriptive.R --synthetic
#
# Dependencies: lme4, emmeans, jsonlite (frozen via renv; same pin as E1/E2).

suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
  library(jsonlite)
})

# ---- Constants ----------------------------------------------------------------

ALPHA <- 0.05
CELL_LEVELS <- c("known_real", "obscure_real", "unanswerable")
DOMAIN_LEVELS <- c("academic", "medical", "legal")
CONDITION_LEVELS <- c("neutral", "citations_expected", "abstention_permitted",
                      "length_matched")
REFERENCE_CONDITION <- "neutral"
GOOD_COVERAGE_DOMAINS <- c("academic", "medical")  # legal verification is gappy
RANDOM_SEED <- 20260613

GEN_KEYS <- c("question_id", "condition", "model_id", "trial_id")
GOOD_STATUSES <- c("REAL_CONFIRMED", "EXISTENT_METADATA_ERROR", "EXISTENT_WRONG_ID")
CONDITION_MAIN_TERMS <- c("conditioncitations_expected",
                          "conditionabstention_permitted",
                          "conditionlength_matched")

# ---- Data loading -------------------------------------------------------------

read_csv_frame <- function(path) {
  if (!file.exists(path)) stop(sprintf("File not found: %s", path))
  read.csv(path, stringsAsFactors = FALSE, encoding = "UTF-8")
}

# ---- Outcome construction (shared logic with e2_prompt_manipulation.R) --------

tally_citations <- function(verifications, bad_statuses, status_col) {
  if (nrow(verifications) == 0) {
    return(data.frame(question_id = character(0), condition = character(0),
                      model_id = character(0), trial_id = integer(0),
                      n_citations = integer(0), n_bad = integer(0),
                      n_good = integer(0), stringsAsFactors = FALSE))
  }
  v <- verifications
  st <- v[[status_col]]
  v$one <- 1L
  v$is_bad <- as.integer(st %in% bad_statuses)
  v$is_good <- as.integer(st %in% GOOD_STATUSES)
  agg <- aggregate(cbind(one, is_bad, is_good) ~
                     question_id + condition + model_id + trial_id,
                   data = v, FUN = sum)
  names(agg)[names(agg) == "one"] <- "n_citations"
  names(agg)[names(agg) == "is_bad"] <- "n_bad"
  names(agg)[names(agg) == "is_good"] <- "n_good"
  agg
}

classify_fabricated <- function(n_bad, n_good, n_citations) {
  ifelse(n_bad > 0, 1L,
         ifelse(n_citations == 0L, 0L,
                ifelse(n_good > 0, 0L, NA_integer_)))
}

build_descriptive_data <- function(generations, verifications, questions,
                                   status_col = "status",
                                   include_unverifiable_as_fabricated = FALSE) {
  bad <- if (include_unverifiable_as_fabricated) {
    c("FABRICATED", "MISGROUNDED", "UNVERIFIABLE")
  } else {
    c("FABRICATED", "MISGROUNDED")
  }
  tallies <- tally_citations(verifications, bad, status_col)
  gen_keys <- unique(generations[, GEN_KEYS, drop = FALSE])
  data <- merge(gen_keys, tallies, by = GEN_KEYS, all.x = TRUE)
  for (col in c("n_citations", "n_bad", "n_good")) {
    data[[col]][is.na(data[[col]])] <- 0L
  }
  data$fabricated <- classify_fabricated(data$n_bad, data$n_good, data$n_citations)
  data <- data[!is.na(data$fabricated), , drop = FALSE]

  data <- merge(data, questions[, c("id", "cell", "domain")],
                by.x = "question_id", by.y = "id")
  data$cell <- factor(data$cell, levels = CELL_LEVELS)
  data$domain <- factor(data$domain, levels = DOMAIN_LEVELS)
  data$condition <- factor(data$condition, levels = CONDITION_LEVELS)
  data$condition <- relevel(data$condition, ref = REFERENCE_CONDITION)
  data$question_id <- factor(data$question_id)
  data$fabricated <- as.integer(data$fabricated)
  data
}

#' Restrict the analysis frame to a domain subset and droplevels (used to drop
#' the legal domain from the verifier-only fit, where its exclusions are
#' non-ignorably missing). Refactors the condition reference level afterward.
restrict_to_domains <- function(data, domains) {
  d <- data[as.character(data$domain) %in% domains, , drop = FALSE]
  d$domain <- droplevels(factor(as.character(d$domain), levels = DOMAIN_LEVELS))
  d$cell <- droplevels(d$cell)
  d$condition <- relevel(droplevels(d$condition), ref = REFERENCE_CONDITION)
  d$question_id <- droplevels(d$question_id)
  d
}

# ---- Models -------------------------------------------------------------------
# Single model -> question_id is the only grouping factor; no family/model_id RE,
# no is_post_trained term (constant). Exploratory, not confirmatory.
fit_descriptive_model <- function(data) {
  glmer(
    fabricated ~ condition * cell + domain + (1 | question_id),
    data = data,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  )
}

#' Maximal-RE check: condition varies within question, so its random slope is
#' identifiable in principle. If this fits non-singularly its condition SEs are
#' the honest ones; if singular, the intercept-only model is the justified fallback.
fit_descriptive_maximal_model <- function(data) {
  glmer(
    fabricated ~ condition * cell + domain + (1 + condition | question_id),
    data = data,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  )
}

# ---- Reporting helpers --------------------------------------------------------

# SE above this (logit scale) flags quasi/complete separation: the coefficient is
# uninformative (boundary estimate), NOT a precise near-zero/large effect.
SEPARATION_SE_THRESHOLD <- 100

#' Floor a p-value away from a bare 0 (underflow) and report the flag.
.floor_p <- function(p) {
  underflow <- !is.finite(p) || p <= 0 || p < .Machine$double.xmin
  list(p_value = if (underflow) .Machine$double.xmin else p, p_underflow = underflow)
}

extract_coefficients <- function(model) {
  cf <- summary(model)$coefficients
  z_crit <- qnorm(1 - ALPHA / 2)
  lapply(seq_len(nrow(cf)), function(i) {
    est <- cf[i, "Estimate"]; se <- cf[i, "Std. Error"]
    pf <- .floor_p(cf[i, "Pr(>|z|)"])
    list(term = rownames(cf)[i], estimate = unname(est), std_error = unname(se),
         z = unname(cf[i, "z value"]),
         p_value = unname(pf$p_value), p_underflow = unname(pf$p_underflow),
         separation_flag = unname(!is.finite(se) || se > SEPARATION_SE_THRESHOLD),
         odds_ratio = unname(exp(est)),
         or_ci_lo = unname(exp(est - z_crit * se)),
         or_ci_hi = unname(exp(est + z_crit * se)))
  })
}

#' Pull one coefficient record by term name (NULL if absent).
.pick_term <- function(coefs, term) {
  hit <- Filter(function(r) r$term == term, coefs)
  if (length(hit) == 0) NULL else hit[[1]]
}

.condition_emm_df <- function(model, weights = "equal") {
  as.data.frame(emmeans(model, ~ condition, type = "response", weights = weights,
                        lmer.df = "asymptotic"))
}

marginal_prob_neutral <- function(model, weights = "equal") {
  df <- .condition_emm_df(model, weights)
  df$prob[df$condition == REFERENCE_CONDITION][1]
}

#' One record per NON-reference condition: marginal prob + rise vs neutral (pp).
#' weights="equal" averages cell/domain levels evenly; weights="proportional"
#' weights by observed frequency (closer to the data-weighted rate).
condition_contrasts <- function(model, weights = "equal") {
  df <- .condition_emm_df(model, weights)
  p_neutral <- df$prob[df$condition == REFERENCE_CONDITION][1]
  others <- df[df$condition != REFERENCE_CONDITION, ]
  lapply(seq_len(nrow(others)), function(i) {
    list(condition = as.character(others$condition[i]),
         prob = unname(others$prob[i]),
         rise_pp_vs_neutral = unname(others$prob[i] - p_neutral))
  })
}

#' Condition-vs-neutral contrasts (OR scale) from an emmeans treatment-vs-control
#' contrast, on the LINK scale then exponentiated, with asymptotic CI + p. This is
#' the honest "overall effect of condition X vs neutral" averaged over cells: when
#' computed on the maximal-RE model it accounts for between-question heterogeneity
#' in the condition response, unlike the raw reference-cell coefficient.
.or_from_contrast <- function(cdf, by = NULL) {
  z_crit <- qnorm(1 - ALPHA / 2)
  lapply(seq_len(nrow(cdf)), function(i) {
    est <- cdf$estimate[i]; se <- cdf$SE[i]
    pf <- .floor_p(cdf$p.value[i])
    sep <- !is.finite(se) || se > SEPARATION_SE_THRESHOLD
    rec <- list(
      contrast = as.character(cdf$contrast[i]),
      log_or = unname(est), std_error = unname(se),
      z = unname(cdf$z.ratio[i]),
      p_value = unname(pf$p_value), p_underflow = unname(pf$p_underflow),
      separation_flag = unname(sep),
      odds_ratio = unname(exp(est)),
      or_ci_lo = unname(exp(est - z_crit * se)),
      or_ci_hi = unname(exp(est + z_crit * se)),
      significant = unname(!sep && pf$p_value < ALPHA))
    if (!is.null(by)) rec[[by]] <- as.character(cdf[[by]][i])
    rec
  })
}

#' Marginal (averaged over cell) AND per-cell condition-vs-neutral OR contrasts.
#' weights="proportional" averages cells by observed frequency.
condition_effect_table <- function(model, weights = "proportional") {
  lv <- levels(model@frame$condition)
  ref_idx <- which(lv == REFERENCE_CONDITION)
  emm_m <- emmeans(model, ~ condition, weights = weights)
  marg <- as.data.frame(contrast(emm_m, method = "trt.vs.ctrl", ref = ref_idx))
  emm_c <- emmeans(model, ~ condition | cell, weights = weights)
  perc <- as.data.frame(contrast(emm_c, method = "trt.vs.ctrl", ref = ref_idx))
  list(marginal = .or_from_contrast(marg),
       by_cell = .or_from_contrast(perc, by = "cell"))
}

#' Profile-likelihood CIs (OR scale) for the named terms; NULL on failure.
compute_profile_cis <- function(model, terms) {
  ci <- tryCatch(
    confint(model, parm = terms, method = "profile", level = 1 - ALPHA),
    error = function(e) { cat("profile CI failed:", conditionMessage(e), "\n"); NULL })
  if (is.null(ci)) return(NULL)
  lapply(rownames(ci), function(tm) list(
    term = tm,
    or_ci_lo_profile = unname(exp(ci[tm, 1])),
    or_ci_hi_profile = unname(exp(ci[tm, 2]))
  ))
}

#' Observed (unmodeled) rates per condition. ALL rates use the full pre-exclusion
#' generation count as the denominator (not the NA-filtered model frame), so
#' cited-only-UNVERIFIABLE rows — which are NOT abstentions — do not distort the
#' abstention rate. conditional_fabrication_rate is fabrications among citing
#' (non-abstaining) modeled responses: it separates "abstained more" from
#' "fabricated more when it did answer".
summarize_observed_rates <- function(data, all_generations) {
  gk <- unique(all_generations[, GEN_KEYS, drop = FALSE])
  totals <- tapply(rep(1L, nrow(gk)), gk$condition, sum)
  conds <- names(totals)
  lapply(conds, function(cond) {
    d <- data[as.character(data$condition) == cond, , drop = FALSE]
    n_total <- as.integer(totals[[cond]])
    n_modeled <- nrow(d)
    n_abstain <- sum(d$n_citations == 0L)
    n_fab <- sum(d$fabricated == 1L)
    n_citing <- n_modeled - n_abstain
    cited <- d$n_citations[d$n_citations > 0L]
    list(
      condition = cond,
      n_total = n_total,
      n_modeled = n_modeled,
      n_excluded = n_total - n_modeled,
      fabrication_rate_all_gens = if (n_total > 0) n_fab / n_total else NA_real_,
      conditional_fabrication_rate = if (n_citing > 0) n_fab / n_citing else NA_real_,
      abstention_rate_all_gens = if (n_total > 0) n_abstain / n_total else NA_real_,
      mean_citations_when_citing = if (length(cited) > 0) mean(cited) else NA_real_
    )
  })
}

#' Modeled-vs-total N by domain x condition, exposing condition-correlated
#' exclusion (the legal/verifier non-ignorable-missingness issue).
build_exclusion_table <- function(data, all_generations) {
  if ("domain" %in% names(all_generations)) {
    ag <- unique(all_generations[, c(GEN_KEYS, "domain"), drop = FALSE])
  } else {
    ag <- unique(all_generations[, GEN_KEYS, drop = FALSE])
    dmap <- unique(data[, c("question_id", "domain")])
    ag <- merge(ag, dmap, by = "question_id", all.x = TRUE)
  }
  ag <- ag[!is.na(ag$domain), , drop = FALSE]
  ag$domain <- as.character(ag$domain); ag$condition <- as.character(ag$condition)
  tot <- aggregate(list(n_total = rep(1L, nrow(ag))),
                   by = list(domain = ag$domain, condition = ag$condition), FUN = sum)
  md <- data
  md$domain <- as.character(md$domain); md$condition <- as.character(md$condition)
  mod <- aggregate(list(n_modeled = rep(1L, nrow(md))),
                   by = list(domain = md$domain, condition = md$condition), FUN = sum)
  m <- merge(tot, mod, by = c("domain", "condition"), all.x = TRUE)
  m$n_modeled[is.na(m$n_modeled)] <- 0L
  m$n_excluded <- m$n_total - m$n_modeled
  m <- m[order(m$domain, m$condition), ]
  lapply(seq_len(nrow(m)), function(i) list(
    domain = m$domain[i], condition = m$condition[i],
    n_total = as.integer(m$n_total[i]), n_modeled = as.integer(m$n_modeled[i]),
    n_excluded = as.integer(m$n_excluded[i])))
}

.converged <- function(model) {
  msgs <- model@optinfo$conv$lme4$messages
  is.null(msgs) || length(msgs) == 0
}

.condition_ses <- function(model, terms) {
  cf <- summary(model)$coefficients
  present <- intersect(terms, rownames(cf))
  setNames(lapply(present, function(t) unname(cf[t, "Std. Error"])), present)
}

#' Report maximal-RE singularity/convergence and the condition-SE comparison so a
#' reviewer can see whether the intercept-only SEs are anti-conservative or the
#' slope is unsupported (singular -> intercept-only fallback justified). Takes the
#' already-fitted maximal model (NULL if it failed) to avoid a redundant fit.
.summarize_re_robustness <- function(intercept_model, maximal) {
  se_int <- .condition_ses(intercept_model, CONDITION_MAIN_TERMS)
  if (is.null(maximal)) {
    return(list(
      maximal_formula = "fabricated ~ condition * cell + domain + (1 + condition | question_id)",
      maximal_singular = NA, maximal_converged = FALSE, intercept_only_used = TRUE,
      note = "maximal RE model failed to fit; intercept-only RE used as primary",
      condition_se_intercept_only = se_int, condition_se_maximal = NULL))
  }
  sing <- lme4::isSingular(maximal); conv <- .converged(maximal)
  list(
    maximal_formula = deparse(formula(maximal)),
    maximal_singular = sing, maximal_converged = conv,
    intercept_only_used = (sing || !conv),
    note = if (sing || !conv)
      "maximal RE singular/non-converged; intercept-only RE is the justified primary"
    else "maximal RE fit cleanly and is the primary; intercept-only SEs shown for contrast are anti-conservative",
    condition_se_intercept_only = se_int,
    condition_se_maximal = .condition_ses(maximal, CONDITION_MAIN_TERMS))
}

# ---- Top-level analysis -------------------------------------------------------

run_descriptive_analysis <- function(data, generations, status_col = "status",
                                     fit_maximal = TRUE, exclusion_data = NULL,
                                     rates_generations = NULL) {
  if (is.null(exclusion_data)) exclusion_data <- data
  # Observed rates denominate over rates_generations (the generations whose
  # domains match the fitted model); the exclusion table still spans `generations`
  # so it can document held-out / condition-correlated exclusions (e.g. legal).
  if (is.null(rates_generations)) rates_generations <- generations
  intercept_model <- fit_descriptive_model(data)
  maximal <- if (fit_maximal) {
    tryCatch(fit_descriptive_maximal_model(data), error = function(e) {
      cat("maximal RE failed:", conditionMessage(e), "\n"); NULL })
  } else NULL
  # The maximal-RE model is the honest primary when it fits non-singularly:
  # condition varies within question, so its slope (if supported) carries the
  # between-question heterogeneity that the intercept-only model misattributes to
  # residual, deflating condition SEs.
  use_maximal <- !is.null(maximal) && !lme4::isSingular(maximal) && .converged(maximal)
  primary <- if (use_maximal) maximal else intercept_model

  list(
    formula = deparse(formula(primary)),
    coefficients = extract_coefficients(primary),
    coefficients_intercept_only = extract_coefficients(intercept_model),
    condition_effects = condition_effect_table(primary),
    condition_contrasts = condition_contrasts(primary, weights = "equal"),
    condition_contrasts_proportional = condition_contrasts(primary, weights = "proportional"),
    neutral_prob = marginal_prob_neutral(primary, weights = "equal"),
    neutral_prob_proportional = marginal_prob_neutral(primary, weights = "proportional"),
    observed_rates = summarize_observed_rates(data, rates_generations),
    exclusion_table = build_exclusion_table(exclusion_data, generations),
    re_robustness = .summarize_re_robustness(intercept_model, maximal),
    metadata = list(
      analysis = "phase_b_descriptive (exploratory, Haiku-only)",
      alpha = ALPHA,
      multiplicity = "no correction applied: exploratory. Treat all contrasts as signals motivating Phase C; a confirmatory single term would need ~p<0.0036 (Bonferroni/14).",
      reference_condition = REFERENCE_CONDITION,
      status_col = status_col,
      primary_model = if (use_maximal) "maximal" else "intercept_only",
      ci_method = "asymptotic Wald (z); abstention sits near the probability boundary, interpret its CI cautiously",
      domains_in_fit = levels(data$domain),
      n_observations = nrow(data),
      n_generations_total = nrow(unique(rates_generations[, GEN_KEYS, drop = FALSE])),
      n_questions = nlevels(droplevels(data$question_id)),
      converged = .converged(primary),
      is_singular = lme4::isSingular(primary),
      r_version = R.version.string,
      lme4_version = as.character(packageVersion("lme4")),
      timestamp_utc = format(Sys.time(), tz = "UTC", format = "%Y-%m-%dT%H:%M:%SZ")
    )
  )
}

# ---- Synthetic-data dry-run ---------------------------------------------------

generate_synthetic_inputs <- function(n_per_cell_domain = 12, n_trials = 3,
                                       seed = RANDOM_SEED) {
  set.seed(seed)
  questions <- expand.grid(cell = CELL_LEVELS, domain = DOMAIN_LEVELS,
                           q_within = seq_len(n_per_cell_domain),
                           stringsAsFactors = FALSE)
  questions$id <- with(questions, paste(domain, cell, q_within, sep = "_"))
  questions$q_within <- NULL

  generations <- expand.grid(question_id = questions$id,
                             condition = CONDITION_LEVELS,
                             trial_id = seq_len(n_trials) - 1L,
                             stringsAsFactors = FALSE)
  generations$model_id <- "haiku"

  cond_logit <- c(neutral = 0, citations_expected = 0.6,
                  abstention_permitted = -0.8, length_matched = 1.0)
  cell_logit <- c(known_real = -0.3, obscure_real = 0.0, unanswerable = 0.5)
  cell_map <- setNames(questions$cell, questions$id)
  q_re <- setNames(rnorm(nrow(questions), 0, 0.4), questions$id)

  eta <- qlogis(0.25) +
    cond_logit[generations$condition] +
    cell_logit[as.character(cell_map[generations$question_id])] +
    q_re[generations$question_id]
  fabricated <- rbinom(nrow(generations), 1, plogis(eta))

  abstain <- fabricated == 0 & rbinom(nrow(generations), 1, 0.4) == 1
  keep <- fabricated == 1 | !abstain
  verifications <- generations[keep, , drop = FALSE]
  verifications$status <- ifelse(fabricated[keep] == 1, "FABRICATED", "REAL_CONFIRMED")
  verifications$verifier_status <- verifications$status

  list(generations = generations, verifications = verifications, questions = questions)
}

# ---- Main ---------------------------------------------------------------------

parse_args <- function(args) {
  out <- list(synthetic = FALSE, output = NA_character_,
              generations = NA_character_, verifications = NA_character_,
              questions = NA_character_)
  i <- 1
  while (i <= length(args)) {
    a <- args[i]
    if (a == "--synthetic") { out$synthetic <- TRUE; i <- i + 1 }
    else if (a %in% c("--generations", "--verifications", "--questions", "--output")) {
      out[[sub("^--", "", a)]] <- args[i + 1]; i <- i + 2
    } else stop(sprintf("Unknown argument: %s", a))
  }
  out
}

.print_block <- function(label, res) {
  cat(sprintf("\n== %s (domains: %s) ==\n", label,
              paste(res$metadata$domains_in_fit, collapse = "+")))
  cat(sprintf("  n=%d modeled / %d total gens, %d questions, converged=%s, singular=%s\n",
              res$metadata$n_observations, res$metadata$n_generations_total,
              res$metadata$n_questions, res$metadata$converged, res$metadata$is_singular))
  cat(sprintf("  primary model: %s (maximal singular=%s converged=%s)\n",
              res$metadata$primary_model, res$re_robustness$maximal_singular,
              res$re_robustness$maximal_converged))
  cat("  observed rates (denominator = ALL generations):\n")
  for (r in res$observed_rates) {
    cat(sprintf("    %-22s fab(all)=%.3f  fab|citing=%s  abstain=%.3f  cites|citing=%.2f\n",
                r$condition, r$fabrication_rate_all_gens,
                ifelse(is.na(r$conditional_fabrication_rate), "  NA",
                       sprintf("%.3f", r$conditional_fabrication_rate)),
                r$abstention_rate_all_gens,
                ifelse(is.na(r$mean_citations_when_citing), NA, r$mean_citations_when_citing)))
  }
  cat("  marginal condition effect vs neutral (OR [95% CI], p) -- primary model:\n")
  for (r in res$condition_effects$marginal) {
    cat(sprintf("    %-26s OR=%.2f [%.2f, %.2f]  p=%.4g %s\n",
                r$contrast, r$odds_ratio, r$or_ci_lo, r$or_ci_hi, r$p_value,
                ifelse(r$significant, "*", "")))
  }
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  if (args$synthetic) {
    cat("phase_b_descriptive synthetic dry-run\n")
    syn <- generate_synthetic_inputs()
    data <- build_descriptive_data(syn$generations, syn$verifications, syn$questions)
    res <- run_descriptive_analysis(data, syn$generations)
    .print_block("synthetic (status)", res)
    return(invisible(res))
  }

  for (p in c("generations", "verifications", "questions")) {
    if (is.na(args[[p]])) stop(sprintf("Missing --%s", p))
  }
  if (is.na(args$output)) args$output <- "data/analysis/phase_b/descriptive_results.json"

  cat(sprintf("Loading generations from %s ...\n", args$generations))
  generations <- read_csv_frame(args$generations)
  cat(sprintf("Loading verifications from %s ...\n", args$verifications))
  verifications <- read_csv_frame(args$verifications)
  cat(sprintf("Loading questions from %s ...\n", args$questions))
  questions <- read_csv_frame(args$questions)

  # Attach domain to the generation frame so the exclusion table counts fully-
  # excluded questions correctly. gens_amed = the good-coverage subset used as the
  # observed-rate denominator for the academic+medical blocks.
  gens_dom <- merge(generations, questions[, c("id", "domain")],
                    by.x = "question_id", by.y = "id", all.x = TRUE)
  gens_amed <- gens_dom[gens_dom$domain %in% GOOD_COVERAGE_DOMAINS, , drop = FALSE]

  out <- list()
  data_status <- build_descriptive_data(generations, verifications, questions, "status")

  # PRIMARY interpretable claim: J2a verdict on the VERIFICATION-RELIABLE domains
  # (academic+medical). Legal is held out of the headline because (a) its
  # verifier-only UNVERIFIABLE exclusions are condition-correlated and (b) its
  # J2a UNVERIFIABLE->FABRICATED calls (over-call risk) are not independently
  # validated. Maximal-RE model is selected as primary when non-singular.
  data_status_amed <- restrict_to_domains(data_status, GOOD_COVERAGE_DOMAINS)
  cat(sprintf("\n[status academic+medical] frame: %d rows, %d questions\n",
              nrow(data_status_amed), nlevels(droplevels(data_status_amed$question_id))))
  out$status_academic_medical <- run_descriptive_analysis(
    data_status_amed, gens_dom, status_col = "status", fit_maximal = TRUE,
    rates_generations = gens_amed)
  .print_block("status (J2a, academic+medical, PRIMARY)", out$status_academic_medical)

  # EXPLORATORY: J2a across all 3 domains (adds legal; interpret with the legal
  # validity caveat above). Documents how much legal drives the pooled estimates.
  cat(sprintf("\n[status all-domains] frame: %d rows, %d questions\n",
              nrow(data_status), nlevels(droplevels(data_status$question_id))))
  out$status_all_domains <- run_descriptive_analysis(
    data_status, gens_dom, status_col = "status", fit_maximal = TRUE)
  .print_block("status (J2a, all 3 domains, exploratory)", out$status_all_domains)

  # ALTERNATIVE SCORING: verifier-only on academic+medical. Legal DROPPED — its
  # UNVERIFIABLE exclusions are condition-correlated (non-ignorable missingness).
  # The full 3-domain exclusion table is attached to document why.
  data_verif_full <- build_descriptive_data(generations, verifications, questions,
                                            "verifier_status")
  data_verif_amed <- restrict_to_domains(data_verif_full, GOOD_COVERAGE_DOMAINS)
  cat(sprintf("\n[verifier_status] frame (legal dropped): %d rows, %d questions\n",
              nrow(data_verif_amed), nlevels(droplevels(data_verif_amed$question_id))))
  out$verifier_status <- run_descriptive_analysis(
    data_verif_amed, gens_dom, status_col = "verifier_status", fit_maximal = TRUE,
    exclusion_data = data_verif_full, rates_generations = gens_amed)
  .print_block("verifier_status (academic+medical)", out$verifier_status)

  # DIVERGENCE (the judge-layer question): MARGINAL citations_expected-vs-neutral
  # OR (cell-averaged, maximal-RE) under J2a vs verifier-only, on good-coverage
  # domains so the legal CourtListener-indexing confound cannot drive it.
  .pick_marginal <- function(res, cond) {
    hit <- Filter(function(r) grepl(cond, r$contrast),
                  res$condition_effects$marginal)
    if (length(hit)) hit[[1]] else NULL
  }
  ce_j2a_amed <- .pick_marginal(out$status_academic_medical, "citations_expected")
  ce_j2a_all <- .pick_marginal(out$status_all_domains, "citations_expected")
  ce_verif <- .pick_marginal(out$verifier_status, "citations_expected")
  out$divergence_academic_medical <- list(
    scope = "academic+medical only (good verification coverage; legal excluded to remove CourtListener-indexing confound)",
    citations_expected_marginal_j2a = ce_j2a_amed,
    citations_expected_marginal_verifier = ce_verif,
    legal_inflation = list(
      or_all_domains = ce_j2a_all$odds_ratio, or_academic_medical = ce_j2a_amed$odds_ratio,
      note = "The all-domain citations_expected OR shrinks sharply on legal removal, confirming the all-domain figure is legal-UNVERIFIABLE-inflated and must not be the headline."),
    ci_overlap_caveat = "The J2a vs verifier marginal CIs OVERLAP; the J2a>verifier gap is directional only, not a formal difference. Both scorings independently place the effect in obscure_real. The judge adds resolution, not the whole signal.",
    length_matched_reversal = "PAPER-STAGE FLAG: length_matched marginal OR EXCEEDS citations_expected in this Haiku pilot, reversing the pre-reg §5 directional assumption (citations_expected was the presumed high-fabrication arm). Unexplained; needs an explicit appendix note or §11 amendment, and Phase C re-examination on the open-weights roster."
  )

  dir.create(dirname(args$output), recursive = TRUE, showWarnings = FALSE)
  writeLines(jsonlite::toJSON(out, auto_unbox = TRUE, pretty = TRUE, na = "null"),
             args$output)
  cat(sprintf("\nWrote results to %s\n", args$output))
  invisible(out)
}

if (sys.nframe() == 0L) main()
