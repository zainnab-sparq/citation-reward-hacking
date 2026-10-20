# Phase C judge-instrument drift: differentiality tests.
#
# EXPLORATORY / ROBUSTNESS. Not a member of the registered m=5 confirmatory
# family. Specification (locked before any fit): docs/drift_interaction_prespec.md
#
# The J2a judge's behaviour changed twice mid-collection. That only threatens the
# registered E1/E2 contrasts if the misclassification is DIFFERENTIAL, i.e. if the
# instrument's error rate depends on the treatment variables. Non-differential
# error attenuates a contrast toward the null (conservative); differential error
# can bias it either way.
#
# Balance of era across condition, which is what we have, constrains the MIX of
# conditions within each era. It cannot constrain whether the instrument TREATS
# those conditions differently. These models measure that directly:
#
#   D1  condition:era              likelihood-ratio test, E2-facing
#   D2  stage_ordinal:era          likelihood-ratio test, E1-facing
#   D3  condition:grounding        continuous, no era boundaries drawn
#
# Each takes the corresponding registered model verbatim and adds era. Neither
# e1_checkpoint_cascade.R nor e2_prompt_manipulation.R is modified or run here.
#
# Run from repo root:
#   Rscript src/analysis/phase_c_drift_interaction.R \
#     --generations <dir>/generations.jsonl --verifications <dir>/verifications.jsonl \
#     --questions <dir>/questions.jsonl --models <dir>/models.jsonl \
#     --era data/analysis_inputs/phase_c_drift/judgment_era.jsonl \
#     --output data/analysis/phase_c/drift_interaction.json
#
# Dependencies: lme4, jsonlite (same renv pin as E1/E2).

suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
  library(jsonlite)
})

# ---- Constants ----------------------------------------------------------------

ALPHA <- 0.05
SEPARATION_SE_THRESHOLD <- 100
CELL_LEVELS <- c("known_real", "obscure_real", "unanswerable")
DOMAIN_LEVELS <- c("academic", "medical", "legal")
CONDITION_LEVELS <- c("neutral", "citations_expected", "abstention_permitted",
                      "length_matched")
REFERENCE_CONDITION <- "neutral"
ERA_LEVELS <- c("EARLY", "DEGRADED", "RECOVERED")
REFERENCE_ERA <- "EARLY"
MIXED_ERA <- "MIXED"
RANDOM_SEED <- 20260811

GEN_KEYS <- c("question_id", "condition", "model_id", "trial_id")
# Identical to phase_b_descriptive.R / e2_prompt_manipulation.R. The test asserts
# this script's outcome column matches phase_b_descriptive.R's element-for-element,
# so the drift test cannot silently score a different outcome than the registered
# analysis it is diagnosing.
GOOD_STATUSES <- c("REAL_CONFIRMED", "EXISTENT_METADATA_ERROR", "EXISTENT_WRONG_ID")

RE_TERMS <- "(1 | family/model_id) + (1 | question_id)"
D1_BASE_RHS <- paste("condition * is_post_trained * cell + domain + era +", RE_TERMS)
D2_BASE_RHS <- paste("stage_ordinal * cell * domain + era +", RE_TERMS)
D3_BASE_RHS <- paste("condition * is_post_trained * cell + domain + grounding_c +", RE_TERMS)

# ---- Data loading -------------------------------------------------------------

#' Flat JSONL -> data.frame. The adapter (build_confirmatory_inputs.py) projects
#' every emitted line to a fixed key set, so stream_in yields a rectangular frame.
read_jsonl_frame <- function(path) {
  if (!file.exists(path)) stop(sprintf("File not found: %s", path))
  con <- file(path, open = "r")
  on.exit(close(con), add = TRUE)
  jsonlite::stream_in(con, verbose = FALSE)
}

# ---- Outcome construction -----------------------------------------------------

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

#' NA means "cited, but every citation was UNVERIFIABLE" -> excluded from the
#' model. Pre-reg §5. Its rate by era is reported below precisely because a rate
#' that moves with era is the one drift consequence that reaches past limitations.
classify_fabricated <- function(n_bad, n_good, n_citations) {
  ifelse(n_bad > 0, 1L,
         ifelse(n_citations == 0L, 0L,
                ifelse(n_good > 0, 0L, NA_integer_)))
}

#' Full outcome frame: one row per generation that has at least one JUDGED
#' citation, with era attached. `fabricated` may be NA (excluded) so the
#' exclusion rate can be tabulated before anything is dropped.
#'
#' Generations straddling an era boundary carry era == MIXED and are dropped
#' here (prespec §4); the count is returned as an attribute for reporting.
build_drift_frame <- function(generations, verifications, questions, models, era,
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

  # INNER join on era: a generation with no judged citation has no era, and its
  # outcome is verifier-only, so it is immune to judge drift and carries no
  # information about the era contrast (prespec §4).
  data <- merge(data, era, by = GEN_KEYS)
  n_mixed <- sum(data$era == MIXED_ERA)
  data <- data[data$era != MIXED_ERA, , drop = FALSE]

  data <- merge(data, questions[, c("id", "cell", "domain")],
                by.x = "question_id", by.y = "id")
  data <- merge(data, models[, c("model_id", "family", "stage_ordinal",
                                 "is_post_trained", "is_open_weights")],
                by = "model_id")

  data$cell <- factor(data$cell, levels = CELL_LEVELS)
  data$domain <- factor(data$domain, levels = DOMAIN_LEVELS)
  data$condition <- relevel(factor(data$condition, levels = CONDITION_LEVELS),
                            ref = REFERENCE_CONDITION)
  data$era <- relevel(factor(data$era, levels = ERA_LEVELS), ref = REFERENCE_ERA)
  data$question_id <- factor(data$question_id)
  data$family <- factor(data$family)
  data$model_id <- factor(data$model_id)
  # Centred so the condition main effects in D3 stay interpretable at mean grounding.
  data$grounding_c <- data$mean_search_count - mean(data$mean_search_count, na.rm = TRUE)
  attr(data, "n_mixed_era_dropped") <- n_mixed
  data
}

#' Drop the pre-reg §5 exclusions and refactor. Kept separate from the frame
#' builder so the exclusion rate can be tabulated on the pre-drop frame.
modelable <- function(frame) {
  out <- frame[!is.na(frame$fabricated), , drop = FALSE]
  out$fabricated <- as.integer(out$fabricated)
  droplevels(out)
}

open_weights_only <- function(frame) {
  out <- frame[frame$is_open_weights == 1, , drop = FALSE]
  if (nrow(out) == 0) stop("No open-weights observations")
  out$condition <- relevel(droplevels(out$condition), ref = REFERENCE_CONDITION)
  droplevels(out)
}

# ---- Exclusion reporting ------------------------------------------------------

#' Rate of pre-reg §5 exclusions (cited only UNVERIFIABLE sources), grouped by
#' arbitrary columns. Reported by era IN ADDITION to the two registered
#' stratifications, because era did not exist at registration and the era-level
#' UNVERIFIABLE rate moved substantially.
exclusion_rate_by <- function(frame, by) {
  keys <- lapply(by, function(k) frame[[k]])
  names(keys) <- by
  agg <- aggregate(list(n = rep(1L, nrow(frame)),
                        n_excluded = as.integer(is.na(frame$fabricated))),
                   by = keys, FUN = sum)
  agg$exclusion_rate <- agg$n_excluded / agg$n
  agg[order(agg[[by[1]]]), , drop = FALSE]
}

#' TRUE when the exclusion rate is monotone across the ordered levels of `key`
#' within obscure_real. Pre-reg §5's escalation trigger: if it fires, Sensitivity
#' #1 is promoted from footnote to co-primary.
is_monotone_in <- function(frame, key) {
  sub <- frame[!is.na(frame$cell) & frame$cell == "obscure_real", , drop = FALSE]
  if (nrow(sub) == 0) return(NA)
  tab <- exclusion_rate_by(sub, key)
  rates <- tab$exclusion_rate[order(tab[[key]])]
  if (length(rates) < 3) return(NA)
  d <- diff(rates)
  all(d >= 0) || all(d <= 0)
}

# ---- Model fitting ------------------------------------------------------------

fit_glmm <- function(formula, data) {
  glmer(formula, data = data, family = binomial(link = "logit"),
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
}

model_health <- function(fit) {
  list(is_singular = isSingular(fit),
       converged = length(fit@optinfo$conv$lme4$messages) == 0,
       messages = as.character(fit@optinfo$conv$lme4$messages))
}

#' Fit the null (additive era) and alternative (era interacted with the treatment
#' variable) models and likelihood-ratio test them. Returns NULL on a fit failure
#' rather than aborting the whole run, so one unfittable model does not cost the
#' other two.
run_interaction_test <- function(data, base_rhs, interaction_term, label) {
  null_formula <- as.formula(paste("fabricated ~", base_rhs))
  alt_formula <- as.formula(paste("fabricated ~", base_rhs, "+", interaction_term))
  fits <- tryCatch({
    list(null = fit_glmm(null_formula, data), alt = fit_glmm(alt_formula, data))
  }, error = function(e) {
    message(sprintf("[%s] fit failed: %s", label, conditionMessage(e)))
    NULL
  })
  if (is.null(fits)) {
    return(list(label = label, fitted = FALSE,
                null_formula = deparse1(null_formula),
                alt_formula = deparse1(alt_formula)))
  }
  cmp <- anova(fits$null, fits$alt)
  chisq <- cmp[["Chisq"]][2]
  df <- cmp[["Df"]][2]
  p <- cmp[["Pr(>Chisq)"]][2]
  list(
    label = label,
    fitted = TRUE,
    null_formula = deparse1(null_formula),
    alt_formula = deparse1(alt_formula),
    interaction_term = interaction_term,
    n_observations = nrow(data),
    chisq = chisq, df = df, p_value = p,
    significant = !is.na(p) && p < ALPHA,
    aic_null = cmp[["AIC"]][1], aic_alt = cmp[["AIC"]][2],
    null_health = model_health(fits$null),
    alt_health = model_health(fits$alt),
    interaction_coefficients = interaction_coefficients(fits$alt, interaction_term)
  )
}

#' Fixed-effect estimates for the interaction terms only, on the log-odds scale.
#' Reported so a significant test can be READ (which cell moved), never so the
#' registered estimand can be adjusted.
interaction_coefficients <- function(fit, interaction_term) {
  co <- summary(fit)$coefficients
  parts <- strsplit(interaction_term, ":", fixed = TRUE)[[1]]
  keep <- Reduce(`&`, lapply(parts, function(p) grepl(p, rownames(co), fixed = TRUE)))
  if (!any(keep)) return(list())
  lapply(which(keep), function(i) {
    list(term = rownames(co)[i], estimate = unname(co[i, 1]),
         std_error = unname(co[i, 2]), p_value = unname(co[i, 4]))
  })
}

# ---- Robustness panel ---------------------------------------------------------
#
# Required by the prespec §6 "significant" branch. This re-estimates the
# REGISTERED condition contrasts under restrictions (within era, and dropping
# evidence-free judgments). It is therefore gated behind an explicit
# --robustness-panel flag: running it before the one-shot confirmatory fit would
# be peeking at the registered result. It never fires as a side effect of the
# drift tests, which are instrument diagnostics and reveal no registered
# estimand.

#' Drop generations any of whose judged citations came back with no evidence
#' gathering at all. Those judgments skew toward REAL, which is the direction
#' that would inflate the judge's rescues of verifier-FABRICATED calls.
drop_zero_effort <- function(frame) {
  droplevels(frame[!is.na(frame$n_search_zero) & frame$n_search_zero == 0, , drop = FALSE])
}

.floor_p <- function(p) {
  underflow <- !is.finite(p) || p <= 0 || p < .Machine$double.xmin
  list(p_value = if (underflow) .Machine$double.xmin else p, p_underflow = underflow)
}

#' Condition-vs-neutral marginal contrasts on the OR scale, computed the same way
#' as phase_b_descriptive.R: emmeans treatment-vs-control on the LINK scale, then
#' exponentiated. weights="proportional" averages over cells by observed frequency.
condition_contrasts <- function(model, weights = "proportional") {
  lv <- levels(model@frame$condition)
  ref_idx <- which(lv == REFERENCE_CONDITION)
  emm <- emmeans(model, ~ condition, weights = weights)
  cdf <- as.data.frame(contrast(emm, method = "trt.vs.ctrl", ref = ref_idx))
  z_crit <- qnorm(1 - ALPHA / 2)
  lapply(seq_len(nrow(cdf)), function(i) {
    est <- cdf$estimate[i]
    se <- cdf$SE[i]
    pf <- .floor_p(cdf$p.value[i])
    separated <- !is.finite(se) || se > SEPARATION_SE_THRESHOLD
    list(contrast = as.character(cdf$contrast[i]),
         log_or = unname(est), std_error = unname(se),
         odds_ratio = unname(exp(est)),
         or_ci_lo = unname(exp(est - z_crit * se)),
         or_ci_hi = unname(exp(est + z_crit * se)),
         p_value = unname(pf$p_value), p_underflow = unname(pf$p_underflow),
         separation_flag = unname(separated),
         significant = unname(!separated && pf$p_value < ALPHA))
  })
}

#' The registered E2 primary structure with era REMOVED, for re-estimation
#' within an era stratum.
fit_registered_condition_model <- function(data) {
  fit_glmm(
    fabricated ~ condition * is_post_trained * cell + domain
                 + (1 | family/model_id) + (1 | question_id),
    data)
}

#' Re-estimate the registered condition contrasts within each era. A stratum
#' that fails to fit is reported as fitted=FALSE rather than aborting the panel,
#' since the thin RECOVERED stratum is the most likely to fail.
era_stratified_panel <- function(data, fitter = fit_registered_condition_model) {
  out <- list()
  for (era in ERA_LEVELS) {
    subset <- droplevels(data[data$era == era, , drop = FALSE])
    if (nrow(subset) == 0) {
      out[[era]] <- list(n = 0, fitted = FALSE, reason = "no observations")
      next
    }
    subset$condition <- relevel(subset$condition, ref = REFERENCE_CONDITION)
    fit <- tryCatch(fitter(subset), error = function(e) {
      message(sprintf("[panel %s] fit failed: %s", era, conditionMessage(e)))
      NULL
    })
    if (is.null(fit)) {
      out[[era]] <- list(n = nrow(subset), fitted = FALSE, reason = "fit failed")
      next
    }
    out[[era]] <- list(n = nrow(subset), fitted = TRUE,
                       health = model_health(fit),
                       condition_contrasts = condition_contrasts(fit))
  }
  out
}

# ---- Descriptive tables -------------------------------------------------------

#' n per era x condition cell, so a null interaction can be read against the
#' power that was actually available (prespec §8).
era_cell_counts <- function(data) {
  tab <- as.data.frame(table(era = data$era, condition = data$condition),
                       stringsAsFactors = FALSE)
  tab[tab$Freq > 0, , drop = FALSE]
}

# ---- Synthetic fixtures (tests) -----------------------------------------------

#' Synthetic inputs in the adapter's output shape. `differential_effect` plants a
#' condition-specific era shift, giving the test suite a positive control: the
#' interaction test must detect it when planted and must not when it is 0.
generate_synthetic_inputs <- function(n_questions = 24, n_trials = 2,
                                      differential_effect = 0,
                                      seed = RANDOM_SEED) {
  set.seed(seed)
  questions <- expand.grid(cell = CELL_LEVELS, domain = DOMAIN_LEVELS,
                           q_within = seq_len(n_questions), stringsAsFactors = FALSE)
  questions$id <- with(questions, paste(domain, cell, q_within, sep = "_"))
  questions$q_within <- NULL

  models <- data.frame(
    model_id = c("fam_a_base", "fam_a_post", "fam_b_base", "fam_b_post"),
    family = c("fam_a", "fam_a", "fam_b", "fam_b"),
    stage_ordinal = c(0L, 3L, 0L, 3L),
    is_post_trained = c(0L, 1L, 0L, 1L),
    is_open_weights = 1L, stringsAsFactors = FALSE)

  generations <- expand.grid(question_id = questions$id,
                             condition = CONDITION_LEVELS,
                             model_id = models$model_id,
                             trial_id = seq_len(n_trials) - 1L,
                             stringsAsFactors = FALSE)
  generations$output_text <- "synthetic"

  era <- generations[, GEN_KEYS]
  era$era <- sample(ERA_LEVELS, nrow(era), replace = TRUE)
  era$n_judged <- 1L
  era$n_eras <- 1L
  era$mean_search_count <- ifelse(era$era == "DEGRADED",
                                  rnorm(nrow(era), 1.66, 0.5),
                                  rnorm(nrow(era), 2.05, 0.5))
  era$n_search_zero <- 0L

  cond_logit <- c(neutral = 0, citations_expected = 0.6,
                  abstention_permitted = -0.5, length_matched = 0.4)
  era_logit <- c(EARLY = 0, DEGRADED = 0.5, RECOVERED = 0.6)
  merged <- merge(merge(generations, era, by = GEN_KEYS), models, by = "model_id")
  merged <- merge(merged, questions, by.x = "question_id", by.y = "id")
  eta <- -1.2 + cond_logit[merged$condition] + era_logit[merged$era] +
    0.8 * merged$is_post_trained +
    differential_effect * (merged$condition == "citations_expected") *
      (merged$era != "EARLY")
  is_bad <- rbinom(nrow(merged), 1, plogis(eta))

  verifications <- merged[, GEN_KEYS]
  verifications$citation_index <- 0L
  verifications$status <- ifelse(is_bad == 1L, "FABRICATED", "REAL_CONFIRMED")
  verifications$verifier_status <- verifications$status
  # A slice of genuine UNVERIFIABLE-only generations so the exclusion machinery
  # has something to report on.
  unv <- sample(seq_len(nrow(verifications)), size = floor(nrow(verifications) * 0.05))
  verifications$status[unv] <- "UNVERIFIABLE"

  list(generations = generations, verifications = verifications,
       questions = questions, models = models, era = era)
}

# ---- CLI ----------------------------------------------------------------------

parse_args <- function(args) {
  out <- list(synthetic = FALSE, robustness_panel = FALSE, output = NA_character_,
              generations = NA_character_, verifications = NA_character_,
              questions = NA_character_, models = NA_character_, era = NA_character_)
  i <- 1
  while (i <= length(args)) {
    a <- args[i]
    if (a == "--synthetic") { out$synthetic <- TRUE; i <- i + 1 }
    else if (a == "--robustness-panel") { out$robustness_panel <- TRUE; i <- i + 1 }
    else if (a %in% c("--generations", "--verifications", "--questions",
                      "--models", "--era", "--output")) {
      out[[sub("^--", "", a)]] <- args[i + 1]; i <- i + 2
    } else stop(sprintf("Unknown argument: %s", a))
  }
  out
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  if (is.na(args$output)) stop("--output is required")

  inputs <- if (args$synthetic) {
    generate_synthetic_inputs()
  } else {
    list(generations = read_jsonl_frame(args$generations),
         verifications = read_jsonl_frame(args$verifications),
         questions = read_jsonl_frame(args$questions),
         models = read_jsonl_frame(args$models),
         era = read_jsonl_frame(args$era))
  }

  frame <- build_drift_frame(inputs$generations, inputs$verifications,
                             inputs$questions, inputs$models, inputs$era)
  n_mixed <- attr(frame, "n_mixed_era_dropped")
  cat(sprintf("frame: %d generations with >=1 judged citation (%d MIXED-era dropped)\n",
              nrow(frame), n_mixed))

  out <- list(
    metadata = list(
      analysis = "phase_c_drift_interaction",
      status = "EXPLORATORY / ROBUSTNESS. Not a member of the registered m=5 confirmatory family.",
      specification = "docs/drift_interaction_prespec.md (locked before any fit)",
      era_boundaries = list(early_end_row = 33 * 500, degraded_end_row = 48 * 500,
                            basis = "grounding metrics, set before this test was specified, not tuned to the outcome"),
      unit = "generation with >=1 judged citation; generations with no judged citation carry no era and are drift-immune",
      n_generations_with_era = nrow(frame),
      n_mixed_era_dropped = n_mixed,
      alpha = ALPHA
    )
  )

  # Exclusion reporting BEFORE the primary is interpreted (pre-reg §5), with era
  # added as a third stratification.
  out$exclusions <- list(
    by_era = exclusion_rate_by(frame, "era"),
    by_cell_stage = exclusion_rate_by(frame, c("cell", "stage_ordinal")),
    by_condition_post = exclusion_rate_by(frame, c("condition", "is_post_trained")),
    monotone_in_stage_within_obscure_real = is_monotone_in(frame, "stage_ordinal"),
    monotone_in_condition_within_obscure_real = is_monotone_in(frame, "condition"),
    note = "Pre-reg §5: a rate monotone in stage or condition within obscure_real promotes Sensitivity #1 to co-primary. Sensitivity #1 is reported regardless."
  )

  modeled <- modelable(frame)
  ow <- open_weights_only(modeled)
  out$counts <- list(n_modeled = nrow(modeled), n_open_weights = nrow(ow),
                     era_by_condition = era_cell_counts(ow))

  cat("\nfitting D1 (condition:era) ...\n")
  out$d1_condition_era <- run_interaction_test(ow, D1_BASE_RHS, "condition:era",
                                               "D1 condition:era")
  cat("fitting D2 (stage_ordinal:era) ...\n")
  out$d2_stage_era <- run_interaction_test(modeled, D2_BASE_RHS, "stage_ordinal:era",
                                           "D2 stage_ordinal:era")
  cat("fitting D3 (condition:grounding) ...\n")
  out$d3_condition_grounding <- run_interaction_test(ow, D3_BASE_RHS,
                                                     "condition:grounding_c",
                                                     "D3 condition:grounding_c")

  # Sensitivity #1: UNVERIFIABLE recoded as fabricated. Excludes nothing by
  # construction, so it is immune to era-varying exclusion.
  cat("fitting D1 under Sensitivity #1 (UNVERIFIABLE -> fabricated) ...\n")
  frame_s1 <- build_drift_frame(inputs$generations, inputs$verifications,
                                inputs$questions, inputs$models, inputs$era,
                                include_unverifiable_as_fabricated = TRUE)
  out$d1_sensitivity1 <- run_interaction_test(open_weights_only(modelable(frame_s1)),
                                              D1_BASE_RHS, "condition:era",
                                              "D1 condition:era (Sensitivity #1)")

  for (key in c("d1_condition_era", "d2_stage_era", "d3_condition_grounding",
                "d1_sensitivity1")) {
    r <- out[[key]]
    if (isTRUE(r$fitted)) {
      cat(sprintf("  %-34s chisq=%8.3f df=%d p=%.4g  significant=%s\n",
                  r$label, r$chisq, r$df, r$p_value, r$significant))
    } else {
      cat(sprintf("  %-34s DID NOT FIT\n", r$label))
    }
  }

  if (isTRUE(args$robustness_panel)) {
    cat("\nrobustness panel (re-estimates the REGISTERED contrasts) ...\n")
    out$robustness_panel <- list(
      note = paste("EXPLORATORY. Re-estimates of the registered condition",
                   "contrasts under restrictions. Run only alongside or after",
                   "the one-shot confirmatory fit, never before it."),
      by_era = era_stratified_panel(ow),
      zero_effort_dropped = local({
        kept <- drop_zero_effort(ow)
        list(n_before = nrow(ow), n_after = nrow(kept),
             condition_contrasts = tryCatch(
               condition_contrasts(fit_registered_condition_model(kept)),
               error = function(e) NULL))
      })
    )
  } else {
    out$robustness_panel <- list(
      run = FALSE,
      note = "Not run. Pass --robustness-panel to re-estimate the registered contrasts; withheld by default so the drift diagnostics cannot peek at the confirmatory result."
    )
  }

  out$interpretation <- list(
    if_null = "Non-differential misclassification supported: registered contrasts stand; drift reported in limitations with the three-era table.",
    if_significant = "Registered estimands are NOT adjusted and remain primary. Add an era-stratified robustness panel and state that this test detects differentiality but does not identify the direction of bias, because eras contain different items. Only the paired re-judge sub-study, which holds the item fixed, can attribute the difference to the instrument.",
    power_caveat = "A non-significant interaction bounds the evidence; it does not prove non-differentiality. See counts$era_by_condition for achieved n."
  )

  dir.create(dirname(args$output), recursive = TRUE, showWarnings = FALSE)
  # digits = NA: jsonlite's 4-decimal default wrote D3's p as 0 rather than
  # 2.534e-05. See the note in e1_checkpoint_cascade.R.
  writeLines(jsonlite::toJSON(out, auto_unbox = TRUE, pretty = TRUE, na = "null",
                              digits = NA),
             args$output)
  cat(sprintf("\nWrote results to %s\n", args$output))
  invisible(out)
}

if (sys.nframe() == 0L) main()
