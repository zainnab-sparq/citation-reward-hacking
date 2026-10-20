# Primary analysis for E1 (checkpoint cascade, T1 behavioral).
#
# Pre-reg §6 CONFIRMATORY model (parsimony-primary, random intercepts):
#
#   glmer(fabricated ~ stage_ordinal * cell * domain
#                    + (1 | family/model_id)
#                    + (1 | question_id),
#         data = ..., family = binomial(link = "logit"))
#
# This is the structure the §3 power analysis was computed on (E1 97.5%). The
# design-justified maximal (random-slopes) structure is the SENSITIVITY model
# (fit_e1_extended_model), identifiability-repaired: a stage slope is placed
# only at the family level because stage_ordinal is constant within a model_id
# (a checkpoint IS one stage). See pre-reg §6 for the operative-model rule.
#
# Hypothesis H1.1: the stage_ordinal:cell interaction is significant
# (Holm-corrected within E1), with steepest rise in obscure_real.
# H0.1: stage effect uniform across cells.
#
# Co-primary stage coding (pre-reg §6 sensitivity #5, intersection-union): the
# headline T1 claim holds only if BOTH the continuous stage_ordinal:obscure_real
# interaction AND the binary base-vs-any-post-trained:obscure_real contrast
# reject after Holm. Reported in the co_primary_t1 block.
#
# Outcome coding (pre-reg §6): a generation's `fabricated` is 1 if it produced
# any FABRICATED/MISGROUNDED citation; 0 if all its citations verify OR it cited
# NOTHING (abstention is honest non-fabrication, not a dropped row); NA/excluded
# if it cited only UNVERIFIABLE sources (default). n_citations is carried as the
# verbosity proxy for the covariate-adjusted sensitivity model.
#
# Inputs (all JSONL):
#   --generations    one record per (question_id, condition, model_id, trial_id);
#                    optional output_text used for the response-length covariate
#   --verifications  one record per citation (status field is required)
#   --questions      one record per question (id, cell, domain)
#   --models         one record per model_id (family, stage_ordinal,
#                                              is_post_trained)
# Output: --output JSON file with confirmatory-model coefficients (Holm-adjusted
# p-values, Wald + bootstrap 95% CIs), the co-primary decision, sensitivity
# analyses (4-level factor, extended RE, citation-count covariate), the
# FABRICATED-vs-MISGROUNDED status split, and pass/fail of the directional H1.1
# threshold.
#
# Usage:
#   Rscript src/analysis/e1_checkpoint_cascade.R \
#       --generations data/generations/all.jsonl \
#       --verifications data/verifications/all.jsonl \
#       --questions data/questions/frozen_900q.jsonl \
#       --models data/models.jsonl \
#       --output data/analysis/e1_results.json
#
#   # Synthetic-data dry-run (used by src/analysis/test_analysis.R):
#   Rscript src/analysis/e1_checkpoint_cascade.R --synthetic
#
# Dependencies: lme4 1.1.35+, emmeans 1.10+, jsonlite. Frozen via renv at
# A.6 freeze; see src/analysis/renv.lock.

suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
  library(jsonlite)
})

# Chunked, resumable parametric bootstrap. local = TRUE so a test harness that
# sources this script into its own environment gets these definitions there too.
source("src/analysis/bootstrap_chunked.R", local = TRUE)

# ---- Constants ----------------------------------------------------------------

ALPHA <- 0.05
PRIMARY_INTERACTION <- "stage_ordinal:cellobscure_real"
DIRECTIONAL_THRESHOLD_PP <- 0.10  # plan/pre-reg declare 10pp end-to-end rise
PROFILE_ESCALATION_TOL_PP <- 0.02  # pre-reg §6: add a profile-likelihood CI when the
                                   # directional rise lands within 2pp of the threshold
                                   # (the regime where CI-method choice can flip the call)
N_BOOTSTRAP_CI <- 10000  # pre-reg §6: bootstrap 95% CIs, 10k resamples (real-data
                         # path only; synthetic dry-run + tests pass nsim = 0)
RANDOM_SEED <- 42

# Worker count for the parametric bootstrap, read from CITATION_BOOT_NCPUS.
# Measured 2026-08-30 on the real 42,845-row frame: one resample costs 288 s, so
# the registered 10k resamples run 801 hours single-threaded. Distributing them
# changes the schedule and nothing else: same resample count, same estimator, no
# pre-registration deviation. Default 1 so the confirmatory path never changes
# behaviour unless a run asks it to. Windows needs "snow"; "multicore" is a fork
# and does not exist here.
boot_ncpus <- function() {
  n <- suppressWarnings(as.integer(Sys.getenv("CITATION_BOOT_NCPUS", "1")))
  if (is.na(n) || n < 1) 1L else n
}

# Resample count, read from CITATION_BOOTSTRAP_NSIM, defaulting to the
# registered 10,000. The override exists for crash insurance on a multi-day run,
# NOT to weaken the analysis: the confirmatory coefficients, Holm-adjusted
# p-values, Wald intervals and every directional decision come from the fitted
# models and are identical at any nsim. Setting 0 produces that complete result
# in minutes, so a machine that dies on day seven of the bootstrap costs
# intervals and time rather than the analysis.
#
# A 0-resample run also skips the §6 profile-CI escalation, which is gated on the
# same flag as the cheap path. Any run whose output is reported as the registered
# artifact must use the default.
boot_nsim <- function() {
  raw <- Sys.getenv("CITATION_BOOTSTRAP_NSIM", "")
  if (!nzchar(raw)) return(N_BOOTSTRAP_CI)
  n <- suppressWarnings(as.integer(raw))
  if (is.na(n) || n < 0) N_BOOTSTRAP_CI else n
}

CELL_LEVELS <- c("known_real", "obscure_real", "unanswerable")
DOMAIN_LEVELS <- c("academic", "medical", "legal")

# ---- Data loading -------------------------------------------------------------

read_jsonl <- function(path) {
  if (!file.exists(path)) stop(sprintf("File not found: %s", path))
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  lines <- lines[nzchar(lines)]
  if (length(lines) == 0) return(data.frame())
  do.call(rbind, lapply(lines, function(line) {
    as.data.frame(jsonlite::fromJSON(line, simplifyVector = TRUE),
                  stringsAsFactors = FALSE)
  }))
}

# ---- Outcome construction -----------------------------------------------------

GEN_KEYS <- c("question_id", "condition", "model_id", "trial_id")
GOOD_STATUSES <- c("REAL_CONFIRMED", "EXISTENT_METADATA_ERROR", "EXISTENT_WRONG_ID")

#' Per-generation citation tallies from ALL verification rows (every status).
#' Counting all statuses (including UNVERIFIABLE) lets the caller distinguish
#' "cited nothing" (abstention) from "cited only unverifiable sources".
#' Returns one row per generation key present in `verifications`.
tally_citations <- function(verifications, bad_statuses) {
  if (nrow(verifications) == 0) {
    return(data.frame(question_id = character(0), condition = character(0),
                      model_id = character(0), trial_id = integer(0),
                      n_citations = integer(0), n_bad = integer(0),
                      n_good = integer(0), n_fab = integer(0),
                      n_mis = integer(0), stringsAsFactors = FALSE))
  }
  v <- verifications
  v$one <- 1L
  v$is_bad <- as.integer(v$status %in% bad_statuses)
  v$is_good <- as.integer(v$status %in% GOOD_STATUSES)
  v$is_fab <- as.integer(v$status == "FABRICATED")
  v$is_mis <- as.integer(v$status == "MISGROUNDED")
  agg <- aggregate(
    cbind(one, is_bad, is_good, is_fab, is_mis) ~
      question_id + condition + model_id + trial_id,
    data = v, FUN = sum)
  names(agg)[names(agg) == "one"] <- "n_citations"
  names(agg)[names(agg) == "is_bad"] <- "n_bad"
  names(agg)[names(agg) == "is_good"] <- "n_good"
  names(agg)[names(agg) == "is_fab"] <- "n_fab"
  names(agg)[names(agg) == "is_mis"] <- "n_mis"
  agg
}

#' Map per-generation tallies to the modeled outcome (pre-reg §6).
#'   any bad citation                              -> 1
#'   cited nothing (abstention)                    -> 0  (honest non-fabrication)
#'   >= 1 good citation, no bad                     -> 0
#'   cited only UNVERIFIABLE (no good, no bad)      -> NA (excluded from primary)
#' When include_unverifiable_as_fabricated = TRUE, UNVERIFIABLE is already in
#' n_bad, so the NA branch never triggers (sensitivity #1).
classify_fabricated <- function(n_bad, n_good, n_citations) {
  ifelse(n_bad > 0, 1L,
         ifelse(n_citations == 0L, 0L,
                ifelse(n_good > 0, 0L, NA_integer_)))
}

#' Build the analysis-ready data frame for E1 from raw inputs.
#'
#' The universe is the GENERATIONS frame (one row per key), not the citations,
#' so abstaining responses (zero citations) are retained as fabricated = 0
#' rather than silently dropped. n_citations / any_fabricated / any_misgrounded
#' are carried for the verbosity covariate and the FABRICATED-vs-MISGROUNDED
#' split. output_chars (response length) is carried when output_text is present.
build_e1_data <- function(generations, verifications, questions, models,
                          include_unverifiable_as_fabricated = FALSE) {
  bad <- if (include_unverifiable_as_fabricated) {
    c("FABRICATED", "MISGROUNDED", "UNVERIFIABLE")
  } else {
    c("FABRICATED", "MISGROUNDED")
  }

  tallies <- tally_citations(verifications, bad)

  gen_cols <- GEN_KEYS
  has_text <- "output_text" %in% names(generations)
  if (has_text) gen_cols <- c(gen_cols, "output_text")
  gen_keys <- unique(generations[, gen_cols, drop = FALSE])

  data <- merge(gen_keys, tallies, by = GEN_KEYS, all.x = TRUE)
  for (col in c("n_citations", "n_bad", "n_good", "n_fab", "n_mis")) {
    data[[col]][is.na(data[[col]])] <- 0L
  }

  data$any_fabricated <- as.integer(data$n_fab > 0)
  data$any_misgrounded <- as.integer(data$n_mis > 0)
  data$fabricated <- classify_fabricated(data$n_bad, data$n_good, data$n_citations)
  data <- data[!is.na(data$fabricated), , drop = FALSE]

  if (has_text) {
    data$output_chars <- nchar(as.character(data$output_text))
    data$output_text <- NULL
  } else {
    data$output_chars <- rep(NA_integer_, nrow(data))  # length 0 when all excluded
  }

  data <- merge(data, questions[, c("id", "cell", "domain")],
                by.x = "question_id", by.y = "id")
  data <- merge(data, models[, c("model_id", "family", "stage_ordinal",
                                 "is_post_trained")],
                by = "model_id")

  data$cell <- factor(data$cell, levels = CELL_LEVELS)
  data$domain <- factor(data$domain, levels = DOMAIN_LEVELS)
  data$family <- factor(data$family)
  data$model_id <- factor(data$model_id)
  data$question_id <- factor(data$question_id)
  data$stage_ordinal <- as.integer(data$stage_ordinal)
  data$fabricated <- as.integer(data$fabricated)
  # See the same tag in build_e2_data: the outcome coding rides along with the
  # frame so a sensitivity #1 artifact is never mistaken for a primary one.
  attr(data, "unverifiable_as_fabricated") <- isTRUE(include_unverifiable_as_fabricated)
  data
}

# ---- Primary model ------------------------------------------------------------

#' E1 CONFIRMATORY model (parsimony-primary, random intercepts). Pre-reg §6.
#' This is the structure the §3 power analysis was computed on (E1 97.5%).
fit_e1_model <- function(data) {
  glmer(
    fabricated ~ stage_ordinal * cell * domain
                + (1 | family/model_id)
                + (1 | question_id),
    data = data,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 1e5))
  )
}

# ---- Sensitivity: extended (maximal) random effects, identifiability-repaired -
# Pre-reg §6. stage_ordinal is CONSTANT within a model_id (a checkpoint IS one
# stage), so its random slope is identifiable only at the family level; cell
# varies within both family and model_id so it gets a slope at both.
fit_e1_extended_model <- function(data) {
  glmer(
    fabricated ~ stage_ordinal * cell * domain
                + (1 + stage_ordinal + cell | family)
                + (1 + cell | model_id)
                + (1 | question_id),
    data = data,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 1e5))
  )
}

# ---- Sensitivity: stage_ordinal as 4-level factor (pre-reg §6 sensitivity #5) -
# Relaxes the equal-spacing assumption of the continuous coding.
fit_e1_factor_model <- function(data) {
  d <- data
  d$stage_factor <- factor(d$stage_ordinal, levels = 0:3,
                           labels = c("base", "SFT", "DPO", "RLHF"))
  glmer(
    fabricated ~ stage_factor * cell * domain
                + (1 | family/model_id)
                + (1 | question_id),
    data = d,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 1e5))
  )
}

# ---- Co-primary: binary base-vs-any-post coding (pre-reg §6 sensitivity #5) ---
# The "base vs any_post_trained" contrast as a single coefficient, for the
# intersection-union T1 decision. post = I(stage_ordinal > 0).
fit_e1_binary_stage_model <- function(data) {
  d <- data
  d$post <- factor(as.integer(d$stage_ordinal > 0), levels = c(0, 1),
                   labels = c("base", "post"))
  glmer(
    fabricated ~ post * cell * domain
                + (1 | family/model_id)
                + (1 | question_id),
    data = d,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 1e5))
  )
}

# ---- Sensitivity: response-length / citation-count covariate adjustment ------
# Pre-reg §6: verbosity confound. Post-trained models may produce longer answers
# with more citations, mechanically raising the chance of >= 1 fabricated cite.
# We adjust for response length (output_chars) when available, falling back to
# log1p(n_citations). NOTE (collider caveat, documented in pre-reg): n_citations
# is partly downstream of the outcome, so this is a SENSITIVITY model only, never
# the confirmatory test. Returns NULL if no usable covariate is present.
fit_e1_covariate_model <- function(data) {
  d <- data
  if (any(!is.na(d$output_chars))) {
    d <- d[!is.na(d$output_chars), , drop = FALSE]
    d$verbosity <- as.numeric(scale(log1p(d$output_chars)))
  } else {
    d$verbosity <- as.numeric(scale(log1p(d$n_citations)))
  }
  if (length(unique(d$verbosity)) < 2) return(NULL)
  glmer(
    fabricated ~ stage_ordinal * cell * domain + verbosity
                + (1 | family/model_id)
                + (1 | question_id),
    data = d,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 1e5))
  )
}

# ---- Reporting helpers --------------------------------------------------------

extract_coefficients <- function(model) {
  cf <- summary(model)$coefficients
  z_crit <- qnorm(1 - ALPHA / 2)
  list_records <- lapply(seq_len(nrow(cf)), function(i) {
    est <- cf[i, "Estimate"]
    se <- cf[i, "Std. Error"]
    list(
      term = rownames(cf)[i],
      estimate = unname(est),
      std_error = unname(se),
      z = unname(cf[i, "z value"]),
      p_value = unname(cf[i, "Pr(>|z|)"]),
      odds_ratio = unname(exp(est)),
      or_ci_lo = unname(exp(est - z_crit * se)),
      or_ci_hi = unname(exp(est + z_crit * se))
    )
  })
  list_records
}

#' Apply Holm-Bonferroni correction within the E1 family.
#' Pre-reg §6: family-wise within-experiment correction at alpha = 0.05.
apply_holm <- function(coefficient_records, family_terms) {
  for (i in seq_along(coefficient_records)) {
    coefficient_records[[i]]$p_holm <- NA_real_
    coefficient_records[[i]]$significant_holm <- NA
  }
  in_family <- vapply(coefficient_records, function(r) r$term %in% family_terms,
                      logical(1))
  p_vals <- vapply(coefficient_records[in_family], function(r) r$p_value,
                   numeric(1))
  p_holm <- p.adjust(p_vals, method = "holm")
  idx <- which(in_family)
  for (j in seq_along(idx)) {
    coefficient_records[[idx[j]]]$p_holm <- unname(p_holm[j])
    coefficient_records[[idx[j]]]$significant_holm <- unname(p_holm[j] < ALPHA)
  }
  coefficient_records
}

#' Parametric bootstrap CIs (pre-reg §6: "bootstrap 95% CIs, 10k resamples").
#' Refits the model `nsim` times and returns OR-scale percentile CIs, or NULL if
#' nsim <= 0 (synthetic dry-run) or the bootstrap cannot be completed. Slow on
#' real data by design; the confirmatory p-values do not depend on it.
#'
#' Delegates to the chunked implementation so a multi-day run survives an
#' interruption. Pooling k chunks is the same estimator as one undivided run,
#' proven against confint() in test_bootstrap_chunked.R. The earlier
#' confint(method="boot") call this replaced could not checkpoint, and a machine
#' restart on 2026-09-02 cost 64.9 hours because of it.
#'
#' Note for anyone comparing against the old call: its `parm = terms` argument
#' never bounded cost. confint.merMod computes the whole table and subsets
#' after, so every resample refit the full model then as it does now.
compute_bootstrap_cis <- function(model, terms, nsim, chunk_job = "e1") {
  if (nsim <= 0) return(NULL)
  ncpus <- boot_ncpus()
  if (ncpus <= 1 && nsim >= 1000) {
    cat(sprintf(paste0("WARNING: %d resamples on one core. Measured cost is ~288 s ",
                       "per resample on the real frame, so expect ~%.0f hours. Set ",
                       "CITATION_BOOT_NCPUS to distribute them.\n"),
                nsim, nsim * 288 / 3600))
  }
  compute_bootstrap_cis_chunked(
    model = model, terms = terms, nsim = nsim,
    chunk_size = boot_chunk_size(),
    chunk_dir = boot_chunk_dir(),
    job = chunk_job,
    ncpus = ncpus,
    level = 1 - ALPHA)
}

#' Profile-likelihood CIs (pre-reg §6 escalation). Reported in addition to the
#' percentile bootstrap when the directional rise is near the threshold, since
#' that is the regime where small-sample CI calibration can flip the call.
#' Returns NULL on failure, matching compute_bootstrap_cis's contract.
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

#' Is the directional rise within the pre-registered tolerance of the threshold?
#' Pre-reg §6: profile-CI escalation triggers within 2pp of the 10pp threshold.
.profile_escalation_triggered <- function(rise_pp,
                                          threshold_pp = DIRECTIONAL_THRESHOLD_PP,
                                          tol_pp = PROFILE_ESCALATION_TOL_PP) {
  isTRUE(!is.na(rise_pp) && abs(rise_pp - threshold_pp) <= tol_pp)
}

#' FABRICATED-vs-MISGROUNDED status split (pre-reg §6: report separately, not
#' only summed). MISGROUNDED (real work, wrong claim) is the cleaner reward-
#' hacking signal because it controls for retrieval capability.
summarize_status_split <- function(data) {
  by_cs <- aggregate(
    cbind(any_fabricated, any_misgrounded, fabricated) ~ cell + stage_ordinal,
    data = data, FUN = mean)
  # Abstention rate (zero-citation generations) reported by cell x stage so the
  # fabricated=0 category is decomposable into "cited accurately" vs "abstained":
  # a stage effect could be falling fabrications OR rising abstentions (both
  # consistent with the thesis, but distinct mechanisms). Panel review 2026-05-25.
  abst <- aggregate(I(n_citations == 0) ~ cell + stage_ordinal, data = data, FUN = mean)
  names(abst)[ncol(abst)] <- "abstention_rate"
  # Global-abstention placebo (pre-reg §6 #11): isolate the known-real abstention
  # trajectory across stages. The reward-hacking account predicts it stays low/flat
  # (capability ceiling already suppresses abstention there); a global-threshold
  # account predicts it falls with stage like obscure-real. The base->max-stage
  # slope is surfaced so a reviewer can read the alternative-DAG check directly.
  kr <- abst[abst$cell == "known_real", , drop = FALSE]
  kr <- kr[order(kr$stage_ordinal), , drop = FALSE]
  kr_slope <- if (nrow(kr) >= 2)
    kr$abstention_rate[nrow(kr)] - kr$abstention_rate[1] else NA_real_
  list(
    overall_fabricated_rate = mean(data$any_fabricated),
    overall_misgrounded_rate = mean(data$any_misgrounded),
    overall_combined_rate = mean(data$fabricated),
    overall_abstention_rate = mean(data$n_citations == 0),
    mean_citations_per_generation = mean(data$n_citations),
    by_cell_stage = by_cs,
    abstention_by_cell_stage = abst,
    abstention_placebo = list(
      by_stage = kr,
      slope_base_to_max_pp = unname(kr_slope),
      note = paste("Known-real abstention by stage. A decline comparable to the",
                   "obscure-real cell supports the global-abstention-threshold",
                   "alternative and qualifies the T1 reward-hacking reading.")
    )
  )
}

# ---- Directional threshold test (H1.1) ----------------------------------------

#' Compute end-to-end rise (base -> RLHF) in obscure_real cell on probability
#' scale and report pass/fail relative to the 10pp declared threshold.
check_directional_h1_1 <- function(model, data) {
  d <- data[data$cell == "obscure_real", ]
  if (nrow(d) == 0) return(list(observed_rise_pp = NA, passes_threshold = NA))

  emm <- emmeans(model, ~ stage_ordinal,
                 at = list(stage_ordinal = c(0, 3), cell = "obscure_real"),
                 type = "response", lmer.df = "asymptotic")
  emm_df <- as.data.frame(emm)
  # Per-domain rise: the headline 10pp claim is the domain-MARGINAL average over a
  # stage_ordinal*cell*domain three-way; emmeans above marginalizes over domain.
  # The marginal is the pre-registered estimand, but we also report each domain so a
  # reviewer can see whether the average masks heterogeneity (e.g. flat legal, large
  # academic). Statistician panel review 2026-05-25.
  by_domain <- tryCatch({
    emm_d <- as.data.frame(emmeans(model, ~ stage_ordinal | domain,
      at = list(stage_ordinal = c(0, 3), cell = "obscure_real"),
      type = "response", lmer.df = "asymptotic"))
    lapply(split(emm_d, emm_d$domain), function(s) {
      pb <- s$prob[s$stage_ordinal == 0][1]; pr <- s$prob[s$stage_ordinal == 3][1]
      list(domain = as.character(s$domain[1]), base_prob = unname(pb),
           rlhf_prob = unname(pr), rise_pp = unname(pr - pb),
           passes_threshold = unname((pr - pb) >= DIRECTIONAL_THRESHOLD_PP))
    })
  }, error = function(e) NULL)
  if (!is.null(by_domain)) names(by_domain) <- NULL

  if (nrow(emm_df) >= 2) {
    p_base <- emm_df$prob[emm_df$stage_ordinal == 0][1]
    p_rlhf <- emm_df$prob[emm_df$stage_ordinal == 3][1]
    rise_pp <- (p_rlhf - p_base)
    return(list(
      observed_rise_pp = unname(rise_pp),
      base_prob = unname(p_base),
      rlhf_prob = unname(p_rlhf),
      passes_threshold = unname(rise_pp >= DIRECTIONAL_THRESHOLD_PP),
      by_domain = by_domain
    ))
  }
  list(observed_rise_pp = NA, passes_threshold = NA, by_domain = by_domain)
}

# ---- Top-level analysis -------------------------------------------------------

#' Pull a single term's p_value from a coefficient-record list (NA if absent).
.term_pvalue <- function(coef_records, term) {
  for (r in coef_records) if (identical(r$term, term)) return(r$p_value)
  NA_real_
}

#' Pull a single term's Holm significance flag (NA if absent).
.term_sig_holm <- function(coef_records, term) {
  for (r in coef_records) if (identical(r$term, term)) return(r$significant_holm)
  NA
}

#' Did the optimizer converge cleanly (no lme4 convergence messages)?
.converged <- function(model) {
  msgs <- model@optinfo$conv$lme4$messages
  is.null(msgs) || length(msgs) == 0
}

run_e1_analysis <- function(data, bootstrap_nsim = 0L, chunk_job = "e1") {
  primary_terms <- c(
    "stage_ordinal",
    "stage_ordinal:cellobscure_real",
    "stage_ordinal:cellunanswerable",
    "stage_ordinal:cellobscure_real:domainmedical",
    "stage_ordinal:cellobscure_real:domainlegal"
  )

  cat("Fitting confirmatory E1 model (random intercepts, stage_ordinal continuous)...\n")
  model_primary <- fit_e1_model(data)

  coefs_primary <- extract_coefficients(model_primary)
  coefs_primary <- apply_holm(coefs_primary, family_terms = primary_terms)

  cat("Computing directional H1.1 check (obscure_real, base -> RLHF rise)...\n")
  directional <- check_directional_h1_1(model_primary, data)

  if (bootstrap_nsim > 0) {
    cat(sprintf("Computing bootstrap CIs (nsim=%d) for primary interaction terms...\n",
                bootstrap_nsim))
  }
  boot_cis <- compute_bootstrap_cis(
    model_primary,
    terms = c("stage_ordinal:cellobscure_real", "stage_ordinal:cellunanswerable"),
    nsim = bootstrap_nsim,
    chunk_job = chunk_job)

  # Pre-reg §6 profile-CI escalation: when the directional rise sits within 2pp of
  # the 10pp threshold (the ambiguous regime), additionally report a profile CI.
  profile_cis <- NULL
  if (bootstrap_nsim > 0 && .profile_escalation_triggered(directional$observed_rise_pp)) {
    cat("Directional rise within 2pp of threshold; adding profile-likelihood CI...\n")
    profile_cis <- compute_profile_cis(model_primary, PRIMARY_INTERACTION)
  }

  # Co-primary T1 decision (intersection-union, pre-reg §6 sensitivity #5):
  # headline claim holds ONLY if BOTH the continuous stage:obscure interaction
  # (Holm-significant) AND the binary base-vs-post:obscure contrast (p < alpha)
  # reject. The pair is one E1 family member.
  cat("Fitting binary base-vs-post model (co-primary stage coding)...\n")
  model_binary <- tryCatch(fit_e1_binary_stage_model(data), error = function(e) NULL)
  coefs_binary <- if (!is.null(model_binary)) extract_coefficients(model_binary) else NULL
  cont_sig <- isTRUE(.term_sig_holm(coefs_primary, "stage_ordinal:cellobscure_real"))
  binary_p <- if (!is.null(coefs_binary))
    .term_pvalue(coefs_binary, "postpost:cellobscure_real") else NA_real_
  binary_sig <- isTRUE(!is.na(binary_p) && binary_p < ALPHA)

  cat("Fitting sensitivity model (stage_ordinal as 4-level factor)...\n")
  model_factor <- tryCatch(fit_e1_factor_model(data), error = function(e) NULL)
  coefs_factor <- if (!is.null(model_factor)) extract_coefficients(model_factor) else NULL

  cat("Fitting sensitivity model (extended / maximal random effects)...\n")
  model_ext <- tryCatch(fit_e1_extended_model(data), error = function(e) {
    cat("Extended-RE model failed:", conditionMessage(e), "\n"); NULL })

  cat("Fitting sensitivity model (response-length / citation-count covariate)...\n")
  model_cov <- tryCatch(fit_e1_covariate_model(data), error = function(e) {
    cat("Covariate model failed:", conditionMessage(e), "\n"); NULL })

  list(
    primary = list(
      formula = deparse(formula(model_primary)),
      n_observations = nrow(data),
      n_models = nlevels(data$model_id),
      n_questions = nlevels(data$question_id),
      coefficients = coefs_primary,
      bootstrap_cis = boot_cis,
      profile_cis = profile_cis,
      directional_h1_1 = directional
    ),
    co_primary_t1 = list(
      rule = "intersection-union: continuous AND binary base-vs-post must both reject",
      continuous_interaction_significant_holm = cont_sig,
      binary_base_vs_post_interaction_p = unname(binary_p),
      binary_base_vs_post_significant = binary_sig,
      headline_t1_supported = isTRUE(cont_sig && binary_sig)
    ),
    sensitivity_stage_as_factor = list(
      formula = if (!is.null(model_factor)) deparse(formula(model_factor)) else NA,
      coefficients = coefs_factor
    ),
    sensitivity_extended_re = if (!is.null(model_ext)) list(
      formula = deparse(formula(model_ext)),
      is_singular = lme4::isSingular(model_ext),
      converged = .converged(model_ext),
      coefficients = apply_holm(extract_coefficients(model_ext), primary_terms)
    ) else list(formula = NA, fit_failed = TRUE),
    sensitivity_citation_covariate = if (!is.null(model_cov)) list(
      formula = deparse(formula(model_cov)),
      coefficients = extract_coefficients(model_cov)
    ) else list(formula = NA, fit_failed_or_no_covariate = TRUE),
    status_split = summarize_status_split(data),
    metadata = list(
      alpha = ALPHA,
      directional_threshold_pp = DIRECTIONAL_THRESHOLD_PP,
      primary_interaction_term = PRIMARY_INTERACTION,
      bootstrap_nsim = bootstrap_nsim,
      unverifiable_as_fabricated =
        isTRUE(attr(data, "unverifiable_as_fabricated")),
      r_version = R.version.string,
      lme4_version = as.character(packageVersion("lme4")),
      timestamp_utc = format(Sys.time(), tz = "UTC",
                             format = "%Y-%m-%dT%H:%M:%SZ")
    )
  )
}

# ---- Synthetic-data dry-run ---------------------------------------------------

generate_synthetic_e1_inputs <- function(n_per_cell_domain = 30,
                                        seed = RANDOM_SEED) {
  set.seed(seed)

  models_df <- expand.grid(
    family = c("tulu3", "olmo2"),
    size = c("small", "large"),
    stage_ordinal = 0:3,
    stringsAsFactors = FALSE
  )
  models_df$model_id <- with(models_df, paste(family, size, stage_ordinal,
                                              sep = "_"))
  models_df$is_post_trained <- as.integer(models_df$stage_ordinal > 0)

  questions_df <- expand.grid(
    cell = CELL_LEVELS,
    domain = DOMAIN_LEVELS,
    q_within = seq_len(n_per_cell_domain),
    stringsAsFactors = FALSE
  )
  questions_df$id <- with(questions_df, paste(domain, cell, q_within,
                                              sep = "_"))
  questions_df$q_within <- NULL

  generations_df <- merge(
    models_df[, c("model_id")],
    questions_df[, c("id")],
    by = NULL
  )
  names(generations_df) <- c("model_id", "question_id")
  generations_df$condition <- "neutral"
  generations_df$trial_id <- 0L

  cell_map <- setNames(questions_df$cell, questions_df$id)
  stage_map <- setNames(models_df$stage_ordinal, models_df$model_id)
  baseline_logit <- qlogis(0.30)
  cell_offset <- c(known_real = -1.5, obscure_real = 0.0, unanswerable = 1.0)
  per_stage_logit <- (qlogis(0.40) - qlogis(0.30)) / 3

  is_obscure <- cell_map[generations_df$question_id] == "obscure_real"
  stage_for <- stage_map[generations_df$model_id]
  stage_effect <- ifelse(is_obscure, stage_for * per_stage_logit, 0)
  eta <- baseline_logit +
    cell_offset[as.character(cell_map[generations_df$question_id])] +
    stage_effect
  fabricated <- rbinom(nrow(generations_df), 1, plogis(eta))

  verifications_df <- generations_df
  verifications_df$status <- ifelse(fabricated == 1, "FABRICATED", "REAL_CONFIRMED")

  list(
    generations = generations_df,
    verifications = verifications_df,
    questions = questions_df,
    models = models_df
  )
}

# ---- Main ---------------------------------------------------------------------

parse_args <- function(args) {
  out <- list(synthetic = FALSE, output = NA_character_,
              generations = NA_character_, verifications = NA_character_,
              questions = NA_character_, models = NA_character_,
              unverifiable_as_fabricated = FALSE)
  i <- 1
  while (i <= length(args)) {
    a <- args[i]
    if (a == "--synthetic") {
      out$synthetic <- TRUE
      i <- i + 1
    } else if (a == "--unverifiable-as-fabricated") {
      out$unverifiable_as_fabricated <- TRUE
      i <- i + 1
    } else if (a %in% c("--generations", "--verifications", "--questions",
                        "--models", "--output")) {
      out[[sub("^--", "", a)]] <- args[i + 1]
      i <- i + 2
    } else {
      stop(sprintf("Unknown argument: %s", a))
    }
  }
  out
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  if (args$synthetic) {
    cat("E1 synthetic-data dry-run\n")
    syn <- generate_synthetic_e1_inputs()
    data <- build_e1_data(syn$generations, syn$verifications,
                          syn$questions, syn$models)
    results <- run_e1_analysis(data)
    cat("\nPrimary coefficients:\n")
    print(do.call(rbind, lapply(results$primary$coefficients, as.data.frame)))
    cat(sprintf("\nDirectional H1.1: base -> RLHF rise = %.3f, threshold passed: %s\n",
                results$primary$directional_h1_1$observed_rise_pp,
                results$primary$directional_h1_1$passes_threshold))
    return(invisible(results))
  }

  for (p in c("generations", "verifications", "questions", "models")) {
    if (is.na(args[[p]])) stop(sprintf("Missing --%s", p))
  }
  if (is.na(args$output)) {
    args$output <- "data/analysis/e1_results.json"
  }

  cat(sprintf("Loading generations from %s ...\n", args$generations))
  generations <- read_jsonl(args$generations)
  cat(sprintf("Loading verifications from %s ...\n", args$verifications))
  verifications <- read_jsonl(args$verifications)
  cat(sprintf("Loading questions from %s ...\n", args$questions))
  questions <- read_jsonl(args$questions)
  cat(sprintf("Loading models from %s ...\n", args$models))
  models <- read_jsonl(args$models)

  data <- build_e1_data(generations, verifications, questions, models,
                        include_unverifiable_as_fabricated =
                          args$unverifiable_as_fabricated)
  if (args$unverifiable_as_fabricated) {
    cat("Sensitivity #1: UNVERIFIABLE recoded as fabricated (no rows excluded).\n")
  }
  cat(sprintf("Analysis frame: %d rows, %d models, %d questions\n",
              nrow(data), nlevels(data$model_id), nlevels(data$question_id)))

  # Chunks are namespaced by the output artifact, which the launcher already
  # guarantees is unique per job. e1_corrected and e1_registered are the same
  # formula on different inputs, so a shared namespace would let one pool the
  # other's resamples and corrupt the sub-amendment #12 comparison.
  results <- run_e1_analysis(data, bootstrap_nsim = boot_nsim(),
                             chunk_job = boot_job_name(args$output))

  dir.create(dirname(args$output), recursive = TRUE, showWarnings = FALSE)
  # digits = NA writes full precision. jsonlite defaults to 4 DECIMAL places,
  # which silently rendered p = 1.97e-05 as 0 while leaving p = 2.45e-08 intact,
  # because it switches to scientific notation only below its own threshold. The
  # arithmetic was never affected (Holm ordering was computed on the real
  # values), but a p of 0 must never reach a manuscript.
  writeLines(jsonlite::toJSON(results, auto_unbox = TRUE, pretty = TRUE,
                              na = "null", digits = NA),
             args$output)
  cat(sprintf("Wrote results to %s\n", args$output))
  invisible(results)
}

if (sys.nframe() == 0L) main()
