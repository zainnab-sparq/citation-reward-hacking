# Primary analysis for E2 (prompt-condition manipulation, T2 causal).
#
# Pre-reg §6 CONFIRMATORY model (parsimony-primary, random intercepts):
#
#   glmer(fabricated ~ condition * is_post_trained * cell + domain
#                    + (1 | family/model_id)
#                    + (1 | question_id),
#         data = ..., family = binomial(link = "logit"))
#
# This is the structure the §3 power analysis was computed on (E2). The
# design-justified maximal (random condition-slopes at both family and model_id)
# structure is the SENSITIVITY model (fit_e2_extended_model): condition varies
# WITHIN every model so its slope is identifiable at both levels, while
# is_post_trained is BETWEEN-model and gets no random slope. See pre-reg §6.
#
# Conditions (pre-reg §5, 4 levels): neutral (reference), citations_expected,
# abstention_permitted, and length_matched (a length-matched no-citation-cue
# control that isolates the citation request from the verbosity demand;
# citations_expected vs length_matched is the de-confounded contrast).
#
# Outcome coding mirrors E1: abstention (zero citations) -> fabricated = 0;
# cited-only-UNVERIFIABLE -> NA/excluded (default). See e1_checkpoint_cascade.R.
#
# Pre-reg §2 E2 hypothesis structure:
#   H1.2 (open-weights, PRIMARY): condition x is_post_trained interaction is
#     significant in the open-weights subsample where base + post-trained
#     checkpoints exist for the same base model. citations_expected raises
#     fabrication for post-trained checkpoints by >= 10pp more than for base.
#   H1.2b (closed-API, SECONDARY): within post-trained closed-API only,
#     citations_expected raises fabrication >= 10pp relative to neutral.
#     No interaction with is_post_trained because closed APIs have no
#     base-model comparator (acknowledged in pre-reg §6 limitation).
#   H0.2: condition effect uniform across model types in the open-weights set.
#
# Inputs (all JSONL):
#   --generations    one record per (question_id, condition, model_id, trial_id)
#   --verifications  one record per citation (status field is required)
#   --questions      one record per question (id, cell, domain)
#   --models         one record per model_id (family, is_post_trained, is_open_weights)
# Output: --output JSON file with primary-model coefficients, secondary-model
# (closed-API only) coefficients, and pass/fail of the directional H1.2 and
# H1.2b thresholds.
#
# Usage:
#   Rscript src/analysis/e2_prompt_manipulation.R \
#       --generations data/generations/all.jsonl \
#       --verifications data/verifications/all.jsonl \
#       --questions data/questions/frozen_900q.jsonl \
#       --models data/models.jsonl \
#       --output data/analysis/e2_results.json
#
#   # Synthetic-data dry-run (used by src/analysis/test_analysis.R):
#   Rscript src/analysis/e2_prompt_manipulation.R --synthetic
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
DIRECTIONAL_THRESHOLD_PP <- 0.10
PROFILE_ESCALATION_TOL_PP <- 0.02  # pre-reg §6: profile CI when the directional
                                   # interaction delta lands within 2pp of threshold
N_BOOTSTRAP_CI <- 10000  # pre-reg §6: 10k resamples (real-data path; tests use 0)
RANDOM_SEED <- 42

# Worker count for the parametric bootstrap. Same contract as E1; see the
# comment on boot_ncpus in e1_checkpoint_cascade.R for the measured costs and
# why distributing the resamples is not a pre-registration deviation.
boot_ncpus <- function() {
  n <- suppressWarnings(as.integer(Sys.getenv("CITATION_BOOT_NCPUS", "1")))
  if (is.na(n) || n < 1) 1L else n
}

# Resample count, read from CITATION_BOOTSTRAP_NSIM. Same contract as E1; see
# the comment on boot_nsim in e1_checkpoint_cascade.R.
boot_nsim <- function() {
  raw <- Sys.getenv("CITATION_BOOTSTRAP_NSIM", "")
  if (!nzchar(raw)) return(N_BOOTSTRAP_CI)
  n <- suppressWarnings(as.integer(raw))
  if (is.na(n) || n < 0) N_BOOTSTRAP_CI else n
}

CELL_LEVELS <- c("known_real", "obscure_real", "unanswerable")
DOMAIN_LEVELS <- c("academic", "medical", "legal")
# 4-level condition set (pre-reg §5). length_matched is the length-matched
# no-citation-cue control added 2026-05-25 to de-confound citation-ask from
# verbosity-ask. Neutral is the reference for all contrasts.
CONDITION_LEVELS <- c("neutral", "citations_expected", "abstention_permitted",
                      "length_matched")
REFERENCE_CONDITION <- "neutral"

# ---- Data loading (shared shape with e1_checkpoint_cascade.R) -----------------

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

# ---- Outcome construction (shared logic with e1_checkpoint_cascade.R) ---------

GEN_KEYS <- c("question_id", "condition", "model_id", "trial_id")
GOOD_STATUSES <- c("REAL_CONFIRMED", "EXISTENT_METADATA_ERROR", "EXISTENT_WRONG_ID")

#' Per-generation citation tallies across ALL verification rows (every status).
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

#' Outcome mapping: any-bad -> 1; abstention (no citation) -> 0; >=1 good, no bad
#' -> 0; cited-only-UNVERIFIABLE -> NA (excluded). See e1 for full rationale.
classify_fabricated <- function(n_bad, n_good, n_citations) {
  ifelse(n_bad > 0, 1L,
         ifelse(n_citations == 0L, 0L,
                ifelse(n_good > 0, 0L, NA_integer_)))
}

build_e2_data <- function(generations, verifications, questions, models,
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
  data <- merge(data, models[, c("model_id", "family", "is_post_trained",
                                 "is_open_weights")],
                by = "model_id")

  data$cell <- factor(data$cell, levels = CELL_LEVELS)
  data$domain <- factor(data$domain, levels = DOMAIN_LEVELS)
  data$condition <- factor(data$condition, levels = CONDITION_LEVELS)
  data$condition <- relevel(data$condition, ref = REFERENCE_CONDITION)
  data$family <- factor(data$family)
  data$model_id <- factor(data$model_id)
  data$question_id <- factor(data$question_id)
  data$is_post_trained <- as.integer(data$is_post_trained)
  data$is_open_weights <- as.integer(data$is_open_weights)
  data$fabricated <- as.integer(data$fabricated)
  # Tag the frame with its outcome coding so the artifact can record it. Carried
  # on the data rather than passed separately: a sensitivity #1 run and a primary
  # run differ ONLY in this coding, and two JSON files that cannot be told apart
  # is exactly the failure worth engineering out.
  attr(data, "unverifiable_as_fabricated") <- isTRUE(include_unverifiable_as_fabricated)
  data
}

# ---- Primary model (H1.2: open-weights subsample) ----------------------------
# CONFIRMATORY (parsimony-primary, random intercepts). Pre-reg §6. This is the
# structure + roster (23 open-weights model_ids, 6 families) the §3 power
# analysis is computed on, so the registered power describes exactly this test.
fit_e2_primary_model <- function(data) {
  # is_post_trained must be numeric (0/1) so the Holm-family term
  # 'conditioncitations_expected:is_post_trained' is named as the analysis expects;
  # a factor encoding silently renames the coefficient and breaks the Holm lookup.
  stopifnot(is.numeric(data$is_post_trained))
  d <- droplevels(data[data$is_open_weights == 1, ])
  if (nrow(d) == 0) stop("No open-weights observations for primary E2 model")
  glmer(
    fabricated ~ condition * is_post_trained * cell + domain
                + (1 | family/model_id)
                + (1 | question_id),
    data = d,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 1e5))
  )
}

# ---- Sensitivity: extended (maximal) random effects, identifiability-repaired -
# Pre-reg §6. condition varies WITHIN every model, so its random slope is
# identifiable at both the family and model_id levels; is_post_trained is
# BETWEEN-model so it stays fixed-only (no random slope).
fit_e2_extended_model <- function(data) {
  d <- droplevels(data[data$is_open_weights == 1, ])
  if (nrow(d) == 0) stop("No open-weights observations for extended E2 model")
  glmer(
    fabricated ~ condition * is_post_trained * cell + domain
                + (1 + condition + cell | family)
                + (1 + condition + cell | model_id)
                + (1 | question_id),
    data = d,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 1e5))
  )
}

# ---- Secondary model (H1.2b: closed-API post-trained only) -------------------
# Also parsimony-primary intercepts, consistent with the confirmatory choice.
fit_e2_secondary_model <- function(data) {
  d <- droplevels(data[data$is_open_weights == 0 & data$is_post_trained == 1, ])
  if (nrow(d) == 0) stop("No closed-API post-trained observations for secondary E2 model")
  glmer(
    fabricated ~ condition * cell + domain
                + (1 | family/model_id)
                + (1 | question_id),
    data = d,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 1e5))
  )
}

# ---- Sensitivity: response-length / citation-count covariate adjustment ------
# Verbosity confound (pre-reg §6). Open-weights subsample, same collider caveat
# as E1. Returns NULL if no usable covariate.
fit_e2_covariate_model <- function(data) {
  d <- droplevels(data[data$is_open_weights == 1, ])
  if (nrow(d) == 0) return(NULL)
  if (any(!is.na(d$output_chars))) {
    d <- d[!is.na(d$output_chars), , drop = FALSE]
    d$verbosity <- as.numeric(scale(log1p(d$output_chars)))
  } else {
    d$verbosity <- as.numeric(scale(log1p(d$n_citations)))
  }
  if (length(unique(d$verbosity)) < 2) return(NULL)
  glmer(
    fabricated ~ condition * is_post_trained * cell + domain + verbosity
                + (1 | family/model_id)
                + (1 | question_id),
    data = d,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 1e5))
  )
}

# ---- Reporting helpers (same as E1) ------------------------------------------

extract_coefficients <- function(model) {
  cf <- summary(model)$coefficients
  z_crit <- qnorm(1 - ALPHA / 2)
  lapply(seq_len(nrow(cf)), function(i) {
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
}

apply_holm <- function(coefficient_records, family_terms) {
  for (i in seq_along(coefficient_records)) {
    coefficient_records[[i]]$p_holm <- NA_real_
    coefficient_records[[i]]$significant_holm <- NA
  }
  in_family <- vapply(coefficient_records, function(r) r$term %in% family_terms,
                      logical(1))
  if (!any(in_family)) return(coefficient_records)
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

#' Parametric bootstrap CIs (pre-reg §6, 10k resamples on real data). See E1 for
#' why this delegates to the chunked, resumable implementation and why pooling
#' chunks is the same estimator as one undivided run.
compute_bootstrap_cis <- function(model, terms, nsim, chunk_job = "e2") {
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

#' Profile-likelihood CIs (pre-reg §6 escalation), reported alongside the
#' percentile bootstrap when the directional delta is near the threshold.
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

#' Is the directional interaction delta within tolerance of the threshold?
#' Pre-reg §6: profile-CI escalation triggers within 2pp of the 10pp threshold.
.profile_escalation_triggered <- function(delta_pp,
                                          threshold_pp = DIRECTIONAL_THRESHOLD_PP,
                                          tol_pp = PROFILE_ESCALATION_TOL_PP) {
  isTRUE(!is.na(delta_pp) && abs(delta_pp - threshold_pp) <= tol_pp)
}

#' FABRICATED-vs-MISGROUNDED split, reported by condition x is_post_trained.
summarize_status_split <- function(data) {
  by_ci <- aggregate(
    cbind(any_fabricated, any_misgrounded, fabricated) ~ condition + is_post_trained,
    data = data, FUN = mean)
  # Abstention rate by condition x is_post_trained: the abstention_permitted
  # condition is expected to raise zero-citation rates, which mechanically lowers
  # fabrication; reporting it separates "fabricated less" from "abstained more".
  # Panel review 2026-05-25.
  abst <- aggregate(I(n_citations == 0) ~ condition + is_post_trained, data = data, FUN = mean)
  names(abst)[ncol(abst)] <- "abstention_rate"
  list(
    overall_fabricated_rate = mean(data$any_fabricated),
    overall_misgrounded_rate = mean(data$any_misgrounded),
    overall_combined_rate = mean(data$fabricated),
    overall_abstention_rate = mean(data$n_citations == 0),
    mean_citations_per_generation = mean(data$n_citations),
    by_condition_is_post_trained = by_ci,
    abstention_by_condition_is_post_trained = abst
  )
}

#' Did the optimizer converge cleanly (no lme4 convergence messages)?
.converged <- function(model) {
  msgs <- model@optinfo$conv$lme4$messages
  is.null(msgs) || length(msgs) == 0
}

# ---- Directional checks -------------------------------------------------------

#' H1.2 directional: in the open-weights subsample, citations_expected raises
#' fabrication for post-trained checkpoints by >= 10pp more than for base.
check_directional_h1_2 <- function(model, data) {
  d <- droplevels(data[data$is_open_weights == 1, ])
  # is_post_trained is numeric (0/1); pin the reference grid to both values so the
  # base/post picks resolve (emmeans would otherwise evaluate it at its mean).
  emm <- emmeans(model, ~ condition * is_post_trained,
                 at = list(is_post_trained = c(0, 1)),
                 type = "response", lmer.df = "asymptotic")
  emm_df <- as.data.frame(emm)
  pick <- function(cond, ipt) {
    row <- emm_df[emm_df$condition == cond & emm_df$is_post_trained == ipt, ]
    if (nrow(row) == 0) return(NA_real_)
    row$prob[1]
  }
  p_neutral_base <- pick("neutral", 0)
  p_neutral_post <- pick("neutral", 1)
  p_cite_base <- pick("citations_expected", 0)
  p_cite_post <- pick("citations_expected", 1)
  delta_base <- p_cite_base - p_neutral_base
  delta_post <- p_cite_post - p_neutral_post
  interaction_delta <- delta_post - delta_base
  # Per-domain interaction delta on the probability scale. The logit interaction is
  # domain-constant (domain enters additively), but the probability-scale delta
  # shifts with each domain's baseline; reported so a reviewer sees the marginal is
  # not masking domain heterogeneity. Statistician panel review 2026-05-25.
  by_domain <- tryCatch({
    emm_d <- as.data.frame(emmeans(model, ~ condition * is_post_trained | domain,
      at = list(is_post_trained = c(0, 1)),
      type = "response", lmer.df = "asymptotic"))
    pick_d <- function(df, cond, ipt) {
      r <- df[df$condition == cond & df$is_post_trained == ipt, ]
      if (nrow(r) == 0) NA_real_ else r$prob[1]
    }
    lapply(split(emm_d, emm_d$domain), function(s) {
      db <- pick_d(s, "citations_expected", 0) - pick_d(s, "neutral", 0)
      dp <- pick_d(s, "citations_expected", 1) - pick_d(s, "neutral", 1)
      list(domain = as.character(s$domain[1]),
           interaction_delta_pp = unname(dp - db),
           passes_threshold = unname((dp - db) >= DIRECTIONAL_THRESHOLD_PP))
    })
  }, error = function(e) NULL)
  if (!is.null(by_domain)) names(by_domain) <- NULL
  list(
    p_neutral_base = unname(p_neutral_base),
    p_neutral_post = unname(p_neutral_post),
    p_cite_base = unname(p_cite_base),
    p_cite_post = unname(p_cite_post),
    delta_base = unname(delta_base),
    delta_post = unname(delta_post),
    interaction_delta_pp = unname(interaction_delta),
    passes_threshold = unname(interaction_delta >= DIRECTIONAL_THRESHOLD_PP),
    by_domain = by_domain
  )
}

#' H1.2b directional: in closed-API post-trained only, citations_expected
#' raises fabrication >= 10pp relative to neutral.
check_directional_h1_2b <- function(model) {
  emm <- emmeans(model, ~ condition, type = "response",
                 lmer.df = "asymptotic")
  emm_df <- as.data.frame(emm)
  p_neutral <- emm_df$prob[emm_df$condition == "neutral"][1]
  p_cite <- emm_df$prob[emm_df$condition == "citations_expected"][1]
  rise <- p_cite - p_neutral
  list(
    p_neutral = unname(p_neutral),
    p_cite = unname(p_cite),
    rise_pp = unname(rise),
    passes_threshold = unname(rise >= DIRECTIONAL_THRESHOLD_PP)
  )
}

# ---- Amendment #10: the citations_expected vs length_matched de-confounding ----

# Pre-reg §11 amendment #10 fixes this computation exactly, and it is worth
# restating why the details are load-bearing: the amendment shows the SAME Phase
# B data landing in different bins under different Delta computations (-2.0pp as
# a model-based marginal contrast, -7.0pp as a raw rate gap). So the canonical
# computation is registered, not chosen here:
#   Delta = emmeans marginal PROBABILITY-scale contrast citations_expected minus
#   length_matched, weights = "proportional" over cells and domains, evaluated at
#   the is_post_trained = 1 margin of the joint E2 GLMM -- the same fit that
#   yields the registered E2 test, NOT a subset refit.
# is_post_trained = 1 is registered because base checkpoints do not follow the
# prompt conditions at all (every one of the 4,761 empty Phase C generations is
# base), so the citation-cue-vs-length-cue question is only defined where the
# model responds to the cue.
AMENDMENT10_BIN_PP <- 0.05          # the registered +/-5pp bin boundary
AMENDMENT10_BOUNDARY_TOL_PP <- 0.02 # registered "within 2pp of the boundary"

#' Registered three-bin rule plus the CI-based boundary rule. One computed Delta
#' maps to exactly one decision; the thresholds are pre-committed, not tuned.
.amendment10_bin <- function(delta_pp, ci_lo_pp, ci_hi_pp,
                             bin_pp = AMENDMENT10_BIN_PP,
                             tol_pp = AMENDMENT10_BOUNDARY_TOL_PP) {
  if (is.na(delta_pp)) return(NA_character_)
  # The boundary rule is checked first because it overrides a definite bin. It
  # requires BOTH the point estimate within 2pp of a +/-5pp boundary AND the CI
  # crossing that same boundary. Requiring both is the point: a merely wide CI
  # must not turn every imprecise estimate into "ambiguous".
  straddles <- function(b) {
    !is.na(ci_lo_pp) && !is.na(ci_hi_pp) &&
      abs(delta_pp - b) <= tol_pp && ci_lo_pp < b && b < ci_hi_pp
  }
  if (straddles(bin_pp) || straddles(-bin_pp)) return("boundary_ambiguous")
  if (delta_pp >= bin_pp) return("citation_cue_supported")
  if (delta_pp <= -bin_pp) return("length_demand_dominant")
  "not_separable"
}

#' Pull the back-transformed estimate column out of a regridded emmGrid. emmeans
#' names it "prob" for a binomial fit and "response" elsewhere; hard-coding
#' either one makes this silently return NULL if emmeans changes its labelling.
.emm_estimate_column <- function(df) {
  hit <- intersect(c("prob", "response", "emmean", "estimate"), names(df))
  if (length(hit) == 0) stop("no estimate column found on the emmeans grid")
  hit[1]
}

#' Confidence-limit columns from an emmeans confint() frame. emmeans names them
#' asymp.LCL/asymp.UCL when the contrast carries infinite df and lower.CL/upper.CL
#' otherwise, and which one you get depends on the reference grid rather than on
#' anything the caller wrote. Hard-coding one pair silently yields NULL limits
#' for the other, which is how the secondary summary first came back with no CI.
.emm_ci_bounds <- function(df) {
  lo <- intersect(c("lower.CL", "asymp.LCL", "LCL"), names(df))
  hi <- intersect(c("upper.CL", "asymp.UCL", "UCL"), names(df))
  if (length(lo) == 0 || length(hi) == 0) {
    stop(sprintf("no confidence-limit columns on the emmeans frame (saw: %s)",
                 paste(names(df), collapse = ", ")))
  }
  list(lo = unname(df[[lo[1]]][1]), hi = unname(df[[hi[1]]][1]))
}

#' Amendment #10 contrast. Returns the signed probability-scale Delta, its
#' emmeans 95% CI, the registered bin, and the secondary all-open-weights
#' summary. Gated off by default in run_e2_analysis; see the note there.
compute_amendment10_contrast <- function(model, data) {
  # Link-scale grid first. regrid() then moves it to the probability scale
  # BEFORE contrasting, which is what makes the contrast a difference of
  # probabilities. Contrasting a type="response" grid directly would return an
  # ODDS RATIO and silently answer a different question than the one registered.
  emm <- emmeans(model, ~ condition, at = list(is_post_trained = 1),
                 weights = "proportional", lmer.df = "asymptotic")
  emm_p <- regrid(emm)
  pr <- as.data.frame(emm_p)
  est_col <- .emm_estimate_column(pr)
  lv <- as.character(pr$condition)
  if (!all(c("citations_expected", "length_matched", "neutral") %in% lv)) {
    stop("amendment #10 needs citations_expected, length_matched and neutral in the grid")
  }
  w <- as.numeric(lv == "citations_expected") - as.numeric(lv == "length_matched")

  ct <- contrast(emm_p, method = list(cite_minus_length = w))
  ci <- as.data.frame(confint(ct, level = 0.95))
  bounds <- .emm_ci_bounds(ci)
  delta <- ci$estimate[1]
  lo <- bounds$lo
  hi <- bounds$hi

  # Odds-ratio ordering, reported alongside but explicitly NOT bin-determining:
  # the amendment fixes the bin on the pp scale precisely because the OR ordering
  # and the pp gap can disagree.
  or_vs_neutral <- function(cond) {
    wv <- as.numeric(lv == cond) - as.numeric(lv == "neutral")
    unname(exp(as.data.frame(contrast(emm, method = list(x = wv)))$estimate[1]))
  }

  # Secondary summary: the same difference averaged proportionally over the
  # OBSERVED is_post_trained distribution rather than pinned at 1. is_post_trained
  # is numeric, so emmeans would otherwise evaluate it at its mean; the weights
  # are taken from the analysis frame explicitly.
  secondary <- tryCatch({
    d_ow <- droplevels(data[data$is_open_weights == 1, ])
    p_post <- mean(d_ow$is_post_trained == 1)
    emm2 <- emmeans(model, ~ condition * is_post_trained,
                    at = list(is_post_trained = c(0, 1)),
                    weights = "proportional", lmer.df = "asymptotic")
    emm2_p <- regrid(emm2)
    g <- as.data.frame(emm2_p)
    share <- ifelse(g$is_post_trained == 1, p_post, 1 - p_post)
    wv <- share * (as.numeric(g$condition == "citations_expected") -
                     as.numeric(g$condition == "length_matched"))
    c2 <- as.data.frame(confint(contrast(emm2_p, method = list(sec = wv)),
                                level = 0.95))
    b2 <- .emm_ci_bounds(c2)
    list(share_post_trained = unname(p_post),
         delta_pp = unname(c2$estimate[1]),
         ci_lo_pp = b2$lo,
         ci_hi_pp = b2$hi)
  }, error = function(e) NULL)

  list(
    delta_pp = unname(delta),
    ci_lo_pp = unname(lo),
    ci_hi_pp = unname(hi),
    bin = .amendment10_bin(delta, lo, hi),
    p_citations_expected = unname(pr[[est_col]][lv == "citations_expected"][1]),
    p_length_matched = unname(pr[[est_col]][lv == "length_matched"][1]),
    p_neutral = unname(pr[[est_col]][lv == "neutral"][1]),
    odds_ratio_ordering = list(
      citations_expected_vs_neutral = or_vs_neutral("citations_expected"),
      length_matched_vs_neutral = or_vs_neutral("length_matched")
    ),
    secondary_all_open_weights = secondary
  )
}

# ---- Top-level analysis -------------------------------------------------------

run_e2_analysis <- function(data, bootstrap_nsim = 0L, chunk_job = "e2",
                            amendment10 = FALSE) {
  primary_terms <- c(
    "conditioncitations_expected",
    "conditionabstention_permitted",
    "conditioncitations_expected:is_post_trained",
    "conditionabstention_permitted:is_post_trained",
    "conditioncitations_expected:is_post_trained:cellobscure_real"
  )

  cat("Fitting E2 confirmatory model (open-weights subsample, random intercepts)...\n")
  m_primary <- tryCatch(fit_e2_primary_model(data), error = function(e) {
    cat("Primary model failed:", conditionMessage(e), "\n"); NULL
  })

  primary_block <- if (!is.null(m_primary)) {
    coefs <- apply_holm(extract_coefficients(m_primary), primary_terms)
    boot_cis <- compute_bootstrap_cis(
      m_primary, terms = "conditioncitations_expected:is_post_trained",
      nsim = bootstrap_nsim, chunk_job = chunk_job)
    directional <- check_directional_h1_2(m_primary, data)
    # Pre-reg §6 profile-CI escalation: interaction delta within 2pp of threshold.
    profile_cis <- NULL
    if (bootstrap_nsim > 0 && .profile_escalation_triggered(directional$interaction_delta_pp)) {
      cat("Interaction delta within 2pp of threshold; adding profile-likelihood CI...\n")
      profile_cis <- compute_profile_cis(
        m_primary, "conditioncitations_expected:is_post_trained")
    }
    # Amendment #10 is default-OFF and must be asked for. It resolves a
    # registered estimand whose bin the abstract's mechanism parenthetical is
    # pre-committed against, so it should be computed deliberately rather than
    # falling out of any routine re-run of this script.
    amendment10_block <- if (isTRUE(amendment10)) {
      cat("Computing amendment #10 de-confounding contrast (registered estimand)...\n")
      tryCatch(compute_amendment10_contrast(m_primary, data), error = function(e) {
        cat("Amendment #10 contrast failed:", conditionMessage(e), "\n"); NULL
      })
    } else {
      NULL
    }
    list(
      formula = deparse(formula(m_primary)),
      n_observations = nrow(droplevels(data[data$is_open_weights == 1, ])),
      coefficients = coefs,
      bootstrap_cis = boot_cis,
      profile_cis = profile_cis,
      directional_h1_2 = directional,
      amendment10_contrast = amendment10_block
    )
  } else {
    list(formula = NA, coefficients = NULL, bootstrap_cis = NULL,
         directional_h1_2 = NULL)
  }

  cat("Fitting E2 extended-RE sensitivity (open-weights, maximal condition slopes)...\n")
  m_ext <- tryCatch(fit_e2_extended_model(data), error = function(e) {
    cat("Extended-RE model failed:", conditionMessage(e), "\n"); NULL
  })
  extended_block <- if (!is.null(m_ext)) list(
    formula = deparse(formula(m_ext)),
    is_singular = lme4::isSingular(m_ext),
    converged = .converged(m_ext),
    coefficients = apply_holm(extract_coefficients(m_ext), primary_terms)
  ) else list(formula = NA, fit_failed = TRUE)

  cat("Fitting E2 citation-count covariate sensitivity...\n")
  m_cov <- tryCatch(fit_e2_covariate_model(data), error = function(e) {
    cat("Covariate model failed:", conditionMessage(e), "\n"); NULL
  })
  covariate_block <- if (!is.null(m_cov)) list(
    formula = deparse(formula(m_cov)),
    coefficients = extract_coefficients(m_cov)
  ) else list(formula = NA, fit_failed_or_no_covariate = TRUE)

  cat("Fitting E2 secondary model (closed-API post-trained only)...\n")
  m_secondary <- tryCatch(fit_e2_secondary_model(data), error = function(e) {
    cat("Secondary model failed:", conditionMessage(e), "\n"); NULL
  })

  secondary_block <- if (!is.null(m_secondary)) {
    list(
      formula = deparse(formula(m_secondary)),
      n_observations = nrow(droplevels(
        data[data$is_open_weights == 0 & data$is_post_trained == 1, ])),
      coefficients = extract_coefficients(m_secondary),
      directional_h1_2b = check_directional_h1_2b(m_secondary)
    )
  } else {
    list(formula = NA, coefficients = NULL, directional_h1_2b = NULL)
  }

  list(
    primary_open_weights = primary_block,
    sensitivity_extended_re = extended_block,
    sensitivity_citation_covariate = covariate_block,
    secondary_closed_api = secondary_block,
    status_split = summarize_status_split(data),
    metadata = list(
      alpha = ALPHA,
      directional_threshold_pp = DIRECTIONAL_THRESHOLD_PP,
      reference_condition = REFERENCE_CONDITION,
      bootstrap_nsim = bootstrap_nsim,
      unverifiable_as_fabricated =
        isTRUE(attr(data, "unverifiable_as_fabricated")),
      amendment10_computed = isTRUE(amendment10),
      r_version = R.version.string,
      lme4_version = as.character(packageVersion("lme4")),
      timestamp_utc = format(Sys.time(), tz = "UTC",
                             format = "%Y-%m-%dT%H:%M:%SZ")
    )
  )
}

# ---- Synthetic-data dry-run ---------------------------------------------------

generate_synthetic_e2_inputs <- function(n_per_cell_domain = 20,
                                        seed = RANDOM_SEED) {
  set.seed(seed)

  models_df <- data.frame(
    model_id = c(
      # open-weights base
      "tulu3_8b_base", "tulu3_70b_base", "olmo2_13b_base", "llama31_70b_base",
      # open-weights post-trained (same base, different stage)
      "tulu3_8b_rlhf", "tulu3_70b_rlhf", "olmo2_13b_inst", "llama31_70b_inst",
      # closed-API post-trained
      "opus47", "sonnet46", "haiku45", "gpt55", "gpt5", "gemini25pro"
    ),
    family = c(
      "tulu3", "tulu3", "olmo2", "llama",
      "tulu3", "tulu3", "olmo2", "llama",
      "anthropic", "anthropic", "anthropic", "openai", "openai", "google"
    ),
    is_post_trained = c(0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    is_open_weights = c(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
    stringsAsFactors = FALSE
  )

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
    merge(models_df[, "model_id", drop = FALSE],
          questions_df[, "id", drop = FALSE], by = NULL),
    data.frame(condition = CONDITION_LEVELS, stringsAsFactors = FALSE),
    by = NULL
  )
  names(generations_df)[1:2] <- c("model_id", "question_id")
  generations_df$trial_id <- 0L

  baseline_logit <- qlogis(0.25)
  cell_offset <- c(known_real = -1.5, obscure_real = 0.0, unanswerable = 1.0)
  ipt_map <- setNames(models_df$is_post_trained, models_df$model_id)
  cell_map <- setNames(questions_df$cell, questions_df$id)

  cond_post_effect_logit <- qlogis(0.35) - qlogis(0.25)  # ~10pp raise
  cond_abst_effect_logit <- qlogis(0.18) - qlogis(0.25)

  is_post <- ipt_map[generations_df$model_id]
  cond_effect <- ifelse(
    is_post == 1 & generations_df$condition == "citations_expected",
    cond_post_effect_logit,
    ifelse(is_post == 1 & generations_df$condition == "abstention_permitted",
           cond_abst_effect_logit, 0)
  )

  eta <- baseline_logit +
    cell_offset[as.character(cell_map[generations_df$question_id])] +
    cond_effect
  fabricated <- rbinom(nrow(generations_df), 1, plogis(eta))

  verifications_df <- generations_df
  verifications_df$status <- ifelse(fabricated == 1, "FABRICATED",
                                    "REAL_CONFIRMED")

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
              amendment10 = FALSE, unverifiable_as_fabricated = FALSE)
  i <- 1
  while (i <= length(args)) {
    a <- args[i]
    if (a == "--synthetic") {
      out$synthetic <- TRUE
      i <- i + 1
    } else if (a == "--amendment10") {
      out$amendment10 <- TRUE
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
    cat("E2 synthetic-data dry-run\n")
    syn <- generate_synthetic_e2_inputs()
    data <- build_e2_data(syn$generations, syn$verifications,
                          syn$questions, syn$models)
    results <- run_e2_analysis(data)
    cat("\nPrimary (open-weights) directional H1.2:\n")
    print(as.data.frame(results$primary_open_weights$directional_h1_2))
    cat("\nSecondary (closed-API) directional H1.2b:\n")
    print(as.data.frame(results$secondary_closed_api$directional_h1_2b))
    return(invisible(results))
  }

  for (p in c("generations", "verifications", "questions", "models")) {
    if (is.na(args[[p]])) stop(sprintf("Missing --%s", p))
  }
  if (is.na(args$output)) {
    args$output <- "data/analysis/e2_results.json"
  }

  cat(sprintf("Loading generations from %s ...\n", args$generations))
  generations <- read_jsonl(args$generations)
  cat(sprintf("Loading verifications from %s ...\n", args$verifications))
  verifications <- read_jsonl(args$verifications)
  cat(sprintf("Loading questions from %s ...\n", args$questions))
  questions <- read_jsonl(args$questions)
  cat(sprintf("Loading models from %s ...\n", args$models))
  models <- read_jsonl(args$models)

  data <- build_e2_data(generations, verifications, questions, models,
                        include_unverifiable_as_fabricated =
                          args$unverifiable_as_fabricated)
  if (args$unverifiable_as_fabricated) {
    cat("Sensitivity #1: UNVERIFIABLE recoded as fabricated (no rows excluded).\n")
  }
  cat(sprintf("Analysis frame: %d rows, %d models, %d questions\n",
              nrow(data), nlevels(data$model_id), nlevels(data$question_id)))

  # Chunks are namespaced by the output artifact; see the same call in E1.
  results <- run_e2_analysis(data, bootstrap_nsim = boot_nsim(),
                             chunk_job = boot_job_name(args$output),
                             amendment10 = args$amendment10)

  dir.create(dirname(args$output), recursive = TRUE, showWarnings = FALSE)
  # digits = NA writes full precision; see the note at the same call in
  # e1_checkpoint_cascade.R for what jsonlite's 4-decimal default did to small
  # p-values.
  writeLines(jsonlite::toJSON(results, auto_unbox = TRUE, pretty = TRUE,
                              na = "null", digits = NA),
             args$output)
  cat(sprintf("Wrote results to %s\n", args$output))
  invisible(results)
}

if (sys.nframe() == 0L) main()
