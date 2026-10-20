# Phase A.6 simulation-based power analysis (gate before Phase B).
#
# Plan reference: §8 Verification gates, §13 Statistical analysis specification.
# Gate: >= 80% power for the declared 10pp interaction effect at alpha = 0.05.
#
# Design (E1, checkpoint cascade):
#   Y_ijk ~ Bernoulli(p_ijk)
#   logit(p_ijk) = beta0 + beta_stage * stage_ordinal_i +
#                  beta_cell[cell_j] +
#                  beta_stage_obscure * (stage_ordinal_i * I[cell_j == "obscure_real"]) +
#                  family_re[family(i)] + model_re[model_id(i)] +
#                  question_re[question_id(k)]
#
#   The H1 is beta_stage_obscure != 0 — fabrication rises with post-training stage
#   in the obscure-but-real cell. We power on this single coefficient via a
#   z-test, mirroring the primary E1 test in §13.
#
# Design (E2, prompt manipulation): analogous, with condition x is_post_trained
# as the primary interaction. The H1 is that the citations_expected condition
# raises fabrication more for post-trained models than base models.
#
# Run:
#   Rscript src/analysis/power_simulation.R          # both experiments
#   Rscript src/analysis/power_simulation.R e1
#   Rscript src/analysis/power_simulation.R e2
#
# Dependencies: lme4 1.1.35+, simr 1.0.7+. Frozen via renv at A.6 freeze.

suppressPackageStartupMessages({
  library(lme4)
  library(simr)
})

# ---- Configurable parameters --------------------------------------------------

N_QUESTIONS_PER_CELL_DOMAIN <- as.integer(Sys.getenv("POWER_SIM_N_PER_CELL", "100"))  # 100 -> 900 (the frozen set, raised from 600 on 2026-05-24); env-overridable. A bare run now reproduces the registered 900-question gate, not the superseded 600 design.
DOMAINS <- c("academic", "medical", "legal")
CELLS <- c("known_real", "obscure_real", "unanswerable")
ALPHA <- 0.05
# Holm-Bonferroni within a hypothesis family of ~5 primary tests (pre-reg §13)
# makes the strictest threshold the primary interaction can face = ALPHA / 5.
# We report power at both the nominal ALPHA and this conservative Holm floor so
# the gate is honored even if the primary test is the most significant in its
# family (and therefore tested against the smallest Holm threshold).
HOLM_FAMILY_SIZE <- 5L
ALPHA_HOLM_FLOOR <- ALPHA / HOLM_FAMILY_SIZE
TARGET_POWER <- 0.80
N_SIM <- as.integer(Sys.getenv("POWER_SIM_NSIM", "200"))   # 200 reproduces the registered
                                                           # pre-reg §3 numbers (E1 97.5%, CI
                                                           # 94.26-99.18); env-overridable for a
                                                           # tighter CI, but the recorded gate
                                                           # numbers are the nsim=200 run.
SEED <- 42

# Effect-size assumptions (tunable based on pilot data):
BASELINE_FAB_PROB <- 0.30  # base-stage rate in obscure_real cell
EFFECT_PER_STAGE_PP <- 0.10 / 3  # plan declares 10pp end-to-end; 4 stages = 3 deltas
CONDITION_EFFECT_POST_PP <- 0.10  # citations_expected raises post-trained rate 10pp

# Variance components (informed by pilot; conservative until pilot N grows):
FAMILY_SD <- 0.4
MODEL_SD <- 0.2
QUESTION_SD <- 0.5

# ---- Declared effect sizes (single source of truth) ---------------------------
# Pre-reg §13 powers on the DECLARED 10pp interaction effect, not the coefficient
# that happens to be fitted to one simulated dataset. The same value drives both
# the outcome simulation and the a-priori fixef() pin in powerSim, so the two can
# never silently drift apart. Returned on the logit scale.

declared_e1_interaction_logit <- function() {
  baseline_logit <- qlogis(BASELINE_FAB_PROB)
  end_logit <- qlogis(BASELINE_FAB_PROB + EFFECT_PER_STAGE_PP * 3)
  (end_logit - baseline_logit) / 3
}

declared_e2_interaction_logit <- function() {
  baseline_logit <- qlogis(BASELINE_FAB_PROB)
  raised_logit <- qlogis(BASELINE_FAB_PROB + CONDITION_EFFECT_POST_PP)
  raised_logit - baseline_logit
}

# ---- Power reporting (nominal + Holm floor from one simulation) ---------------
# simr's powerSim stores the per-simulation p-values, so we recompute power at
# both the nominal alpha and the Holm floor from the SAME draws -- no second run.
# Falls back to summary() (nominal only) if the p-value vector is unavailable.

report_power_dual_alpha <- function(power_result, label) {
  pvals <- power_result$pval
  if (is.null(pvals)) {
    p_nominal <- summary(power_result)$mean
    p_holm <- NA_real_
  } else {
    p_nominal <- mean(pvals < ALPHA, na.rm = TRUE)
    p_holm <- mean(pvals < ALPHA_HOLM_FLOOR, na.rm = TRUE)
  }
  cat(sprintf("\n%s power: %.1f%% (gate: %.0f%%) @ alpha=%.2f -- %s\n",
              label, p_nominal * 100, TARGET_POWER * 100, ALPHA,
              if (p_nominal >= TARGET_POWER) "PASS" else "FAIL"))
  if (is.na(p_holm)) {
    cat(sprintf("%s Holm-floor power: UNAVAILABLE ($pval missing); rerun with POWER_SIM alpha=%.3f\n",
                label, ALPHA_HOLM_FLOOR))
  } else {
    cat(sprintf("%s power: %.1f%% (gate: %.0f%%) @ alpha=%.3f (Holm floor, m=%d) -- %s\n",
                label, p_holm * 100, TARGET_POWER * 100, ALPHA_HOLM_FLOOR,
                HOLM_FAMILY_SIZE,
                if (p_holm >= TARGET_POWER) "PASS" else "FAIL"))
  }
  invisible(list(p_nominal = p_nominal, p_holm = p_holm))
}

# ---- E1: checkpoint cascade ---------------------------------------------------

DEFAULT_REALIZED_REGISTRY <- "data/analysis_inputs/phase_c_final/models.jsonl"

#' The roster Phase C ACTUALLY ran, read from the registry the confirmatory fit
#' used rather than retyped here, so the two cannot drift apart.
#'
#' This is deliberately not the registered roster. The registered §3 power
#' figures (E1 97.5%, E2 95.5%) were computed on 23 open-weights model_ids
#' across 6 families; Phase C collected 12 checkpoints across 3 families, having
#' dropped the 405B anchor and the whole cross-family base-to-Instruct breadth
#' arm. Powering the realized design is how a null result is shown to be
#' informative rather than merely quiet.
#'
#' Note this stays an A-PRIORI calculation: the effect is pinned to the
#' pre-registered declared value by the callers below, exactly as the registered
#' run pinned it, so nothing here is a function of the observed result. Reporting
#' observed power for a null would be circular; this is not that.
realized_open_weights_roster <- function(path = DEFAULT_REALIZED_REGISTRY) {
  if (!file.exists(path)) stop(sprintf("model registry not found: %s", path))
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  rows <- lapply(lines, jsonlite::fromJSON)
  # The confirmatory registry spells the grouping column `family`; the Phase C
  # checkpoint registry spells it `model_family`. Same rule as
  # build_confirmatory_inputs.to_confirmatory_models, so both registries read
  # into the same roster shape.
  family_of <- function(r) {
    if (!is.null(r$family)) return(as.character(r$family))
    if (!is.null(r$model_family)) return(as.character(r$model_family))
    stop(sprintf("registry record %s has neither 'family' nor 'model_family'",
                 as.character(r$model_id)))
  }
  reg <- data.frame(
    model_id = vapply(rows, function(r) as.character(r$model_id), character(1)),
    family = vapply(rows, family_of, character(1)),
    stage_ordinal = vapply(rows, function(r) as.integer(r$stage_ordinal), integer(1)),
    is_post_trained = vapply(rows, function(r) as.integer(r$is_post_trained), integer(1)),
    is_open_weights = vapply(rows, function(r) as.integer(r$is_open_weights), integer(1)),
    stringsAsFactors = FALSE
  )
  # Confirmatory E1/E2 both subset to the open-weights subsample. Filtering here
  # means a closed-API row added to the registry later cannot silently inflate
  # this roster, which is precisely the error the 31-to-23 correction fixed.
  reg <- reg[reg$is_open_weights == 1L, , drop = FALSE]
  if (nrow(reg) == 0L) stop("no open-weights models in registry: ", path)
  rownames(reg) <- NULL
  reg
}

DEFAULT_CHECKPOINT_REGISTRY <- "data/models/phase_c_checkpoints.jsonl"

#' The roster after the §11 amendment #13 breadth collection: the 12 realized
#' checkpoints plus the three collected on 2026-09-06 (Llama-3.1-70B-Instruct,
#' Qwen2.5-72B, Qwen2.5-72B-Instruct), read from the checkpoint registry that
#' generation ran from, so the roster powered here is the roster that exists.
#'
#' Grouping rule (amendment #13, fixed before any breadth checkpoint was
#' judged): family = shared base model. Llama-3.1-70B is one checkpoint with one
#' set of generations, already the Tulu-3-70B stage-0 base, so
#' Llama-3.1-70B-Instruct joins `tulu3_70b` as a sibling post-trained
#' checkpoint. The power ladder's `llama` breadth pair duplicated that base as
#' a second family, which the real data cannot do without copying 3,600
#' observations; this roster is what the ladder's 14- and 16-model rungs should
#' have been, and its result supersedes them.
#'
#' Still an A-PRIORI calculation: callers pin the effect to the registered
#' declared value exactly as every other mode does.
breadth_open_weights_roster <- function(path = DEFAULT_CHECKPOINT_REGISTRY) {
  reg <- realized_open_weights_roster(path)
  if (any(duplicated(reg$model_id))) {
    stop("breadth roster has duplicate model_ids in ", path)
  }
  reg
}

#' The roster the REGISTERED §3 power figures were computed on. Frozen: changing
#' it would silently invalidate the numbers printed in the pre-registration.
registered_open_weights_roster <- function() {
  # FAITHFUL E1 roster (plan §4) = the open-weights subset of the E2 roster,
  # 23 model_ids across 6 families. Earlier this used a symmetric 2-family x
  # 2-size x 4-stage = 16-model toy roster; that understated family count and
  # ignored the asymmetric stage coverage, so its power number did not describe
  # the real design (methodology-critic 2026-05-23). The cascade families
  # (tulu3, olmo2) carry all four stages; the cross-family deltas
  # (llama, qwen, mistral, deepseek) carry only base (0) and final (3).
  data.frame(
    model_id = c(
      # base (stage_ordinal = 0)
      "tulu3_base_8b", "tulu3_base_70b", "olmo2_base_13b",
      "llama31_base_70b", "llama31_base_405b", "qwen25_base_72b",
      "mistral_base_large2", "deepseek_base_v3",
      # SFT (stage_ordinal = 1)
      "tulu3_sft_8b", "tulu3_sft_70b", "olmo2_sft_13b",
      # DPO (stage_ordinal = 2)
      "tulu3_dpo_8b", "tulu3_dpo_70b", "olmo2_dpo_13b",
      # RLHF / final instruct (stage_ordinal = 3)
      "tulu3_rlvr_8b", "tulu3_rlvr_70b", "tulu3_rlvr_405b",
      "olmo2_instruct_13b", "llama31_instruct_70b", "llama31_instruct_405b",
      "qwen25_instruct_72b", "mistral_instruct_large2", "deepseek_chat_v3"
    ),
    family = c(
      "tulu3", "tulu3", "olmo2", "llama", "llama", "qwen", "mistral", "deepseek",
      "tulu3", "tulu3", "olmo2",
      "tulu3", "tulu3", "olmo2",
      "tulu3", "tulu3", "tulu3", "olmo2", "llama", "llama", "qwen", "mistral", "deepseek"
    ),
    stage_ordinal = c(
      0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1,
      2, 2, 2,
      3, 3, 3, 3, 3, 3, 3, 3, 3
    ),
    stringsAsFactors = FALSE
  )
}

#' The registered breadth-arm pairs, keyed by what one collection session buys.
#'
#' Each entry is a base/final pair from the registered roster that was never
#' collected. They are keyed separately from `family` because affordability
#' splits inside a family: the Llama breadth arm registers both a 70B pair and a
#' 405B pair, and only the 70B pair is reachable on the GPU budget. Both keep
#' family = "llama" so the registered family grouping is preserved and adding
#' both does not invent a family that the pre-registration never declared.
#'
#' The 405B Tulu-3 anchor is deliberately absent: it is a single final-stage
#' checkpoint with no base comparator, so it cannot be expressed as a pair and
#' contributes nothing to either the stage slope or the is_post_trained contrast.
BREADTH_ADDITIONS <- list(
  llama    = c("llama31_base_70b",    "llama31_instruct_70b"),
  llama405 = c("llama31_base_405b",   "llama31_instruct_405b"),
  qwen     = c("qwen25_base_72b",     "qwen25_instruct_72b"),
  mistral  = c("mistral_base_large2", "mistral_instruct_large2"),
  deepseek = c("deepseek_base_v3",    "deepseek_chat_v3")
)

#' Realized roster plus a chosen subset of the registered breadth pairs.
#'
#' This answers "what would collecting N more checkpoints actually buy?" without
#' spending anything, by powering the intermediate designs between the 12-model
#' roster that ran and the 23-model roster that was registered. Every added
#' model_id is taken from `registered_open_weights_roster()` rather than retyped,
#' so a partial roster is always a genuine subset of what was promised.
partial_open_weights_roster <- function(add = character(0),
                                        registry = DEFAULT_REALIZED_REGISTRY) {
  unknown <- setdiff(add, names(BREADTH_ADDITIONS))
  if (length(unknown) > 0L) {
    stop(sprintf("unknown breadth family: %s (expected one of: %s)",
                 paste(unknown, collapse = ", "),
                 paste(names(BREADTH_ADDITIONS), collapse = ", ")))
  }

  keep <- c("model_id", "family", "stage_ordinal", "is_post_trained",
            "is_open_weights")
  base <- realized_open_weights_roster(registry)[, keep, drop = FALSE]
  add <- unique(add)
  if (length(add) == 0L) {
    rownames(base) <- NULL
    return(base)
  }

  reg <- registered_open_weights_roster()
  extra <- do.call(rbind, lapply(add, function(key) {
    ids <- BREADTH_ADDITIONS[[key]]
    idx <- match(ids, reg$model_id)
    if (anyNA(idx)) {
      stop(sprintf("breadth pair '%s' names a model_id absent from the registered roster: %s",
                   key, paste(ids[is.na(idx)], collapse = ", ")))
    }
    reg[idx, c("model_id", "family", "stage_ordinal"), drop = FALSE]
  }))
  extra$is_post_trained <- as.integer(extra$stage_ordinal > 0L)
  extra$is_open_weights <- 1L

  out <- rbind(base, extra[, keep, drop = FALSE])
  if (any(duplicated(out$model_id))) {
    stop("partial roster has duplicate model_ids; a breadth pair overlaps the realized registry")
  }
  rownames(out) <- NULL
  out
}

build_e1_design <- function(n_per_cell_domain = N_QUESTIONS_PER_CELL_DOMAIN,
                            models = registered_open_weights_roster()) {
  models <- models[, c("model_id", "family", "stage_ordinal"), drop = FALSE]

  questions <- expand.grid(
    cell = CELLS,
    domain = DOMAINS,
    q_within = seq_len(n_per_cell_domain),
    stringsAsFactors = FALSE
  )
  questions$question_id <- with(questions, paste(domain, cell, q_within, sep = "_"))
  questions$q_within <- NULL

  full <- merge(models, questions, by = NULL)
  full$cell <- factor(full$cell, levels = CELLS)
  full$domain <- factor(full$domain, levels = DOMAINS)
  full$family <- factor(full$family)
  full$model_id <- factor(full$model_id)
  full$question_id <- factor(full$question_id)
  full
}

simulate_e1_outcomes <- function(design, family_sd = FAMILY_SD,
                                  model_sd = MODEL_SD, question_sd = QUESTION_SD) {
  set.seed(SEED)

  baseline_logit <- qlogis(BASELINE_FAB_PROB)
  per_stage_logit <- declared_e1_interaction_logit()

  # Cell offsets: known_real near zero, obscure_real at baseline, unanswerable high
  cell_offset <- c(known_real = -1.5, obscure_real = 0.0, unanswerable = 1.0)

  family_re <- rnorm(nlevels(design$family), 0, family_sd)
  names(family_re) <- levels(design$family)
  model_re <- rnorm(nlevels(design$model_id), 0, model_sd)
  names(model_re) <- levels(design$model_id)
  question_re <- rnorm(nlevels(design$question_id), 0, question_sd)
  names(question_re) <- levels(design$question_id)

  is_obscure <- design$cell == "obscure_real"
  stage_effect <- ifelse(is_obscure, design$stage_ordinal * per_stage_logit, 0)

  eta <- baseline_logit +
    cell_offset[as.character(design$cell)] +
    stage_effect +
    family_re[as.character(design$family)] +
    model_re[as.character(design$model_id)] +
    question_re[as.character(design$question_id)]

  rbinom(nrow(design), 1, plogis(eta))
}

run_e1_power_analysis <- function(design = build_e1_design(), label = "E1") {
  cat(sprintf("=== %s power analysis ===\n", label))
  cat(sprintf("Design: %d families x %d models x %d questions = %d observations\n",
              nlevels(design$family), nlevels(design$model_id),
              nlevels(design$question_id), nrow(design)))
  cat(sprintf("Effect: %.1fpp baseline -> %.1fpp at stage 3 in obscure_real\n",
              BASELINE_FAB_PROB * 100,
              (BASELINE_FAB_PROB + EFFECT_PER_STAGE_PP * 3) * 100))

  design$fabricated <- simulate_e1_outcomes(design)
  cat("\nSimulated fabrication rates per cell x stage:\n")
  agg <- aggregate(fabricated ~ cell + stage_ordinal, data = design, FUN = mean)
  print(agg)

  cat("\nFitting mixed-effects model on simulated data...\n")
  # Drop the three-way interaction with domain for tractability; we test on the
  # primary stage:cell interaction. Sensitivity analysis later restores domain.
  model <- glmer(
    fabricated ~ stage_ordinal * cell + domain
                + (1 | family / model_id)
                + (1 | question_id),
    data = design,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  )

  # Pin the interaction to the DECLARED effect so powerSim returns a-priori power,
  # not observed power from the noisy single-dataset fit (simr's default behavior).
  target_logit <- declared_e1_interaction_logit()
  fixef(model)["stage_ordinal:cellobscure_real"] <- target_logit
  cat(sprintf("Pinned a-priori effect: stage_ordinal:cellobscure_real = %.5f logit\n",
              target_logit))

  cat(sprintf("\nRunning powerSim for stage_ordinal:cellobscure_real (n_sim=%d)...\n", N_SIM))
  power_result <- powerSim(
    model,
    test = fixed("stage_ordinal:cellobscure_real", method = "z"),
    nsim = N_SIM,
    seed = SEED,
    progress = interactive()
  )
  print(power_result)

  report_power_dual_alpha(power_result, label)
  invisible(power_result)
}

# ---- E2: prompt manipulation --------------------------------------------------

#' The registered E2 roster. Same 23 models as E1, carrying is_post_trained
#' rather than the stage ordinal. Frozen for the same reason.
registered_e2_roster <- function() {
  # E2 CONFIRMATORY roster = the OPEN-WEIGHTS subsample only (pre-reg §6 / H1.2):
  # 23 model_ids across 6 families, identical to the E1 roster. The closed APIs
  # (Anthropic / OpenAI / Google) are EXCLUDED here because they have no
  # base-model comparator, so is_post_trained is collinear with model identity
  # within them and the confirmatory glmer subsets to is_open_weights == 1. The
  # power gate must describe the estimand the confirmatory test actually runs on,
  # so it is computed on these 23 models, NOT the full 31-model roster (the
  # earlier 31-model design over-counted by including closed APIs that the
  # confirmatory model drops; panel review 2026-05-25). 8 base + 15 post-trained;
  # six families carry both base and post checkpoints, identifying is_post_trained
  # within-family. Closed-API condition effects are the SECONDARY H1.2b test
  # (within-post-trained), powered separately and not gated here.
  data.frame(
    model_id = c(
      # base (is_post_trained = 0)
      "tulu3_base_8b", "tulu3_base_70b", "olmo2_base_13b",
      "llama31_base_70b", "llama31_base_405b", "qwen25_base_72b",
      "mistral_base_large2", "deepseek_base_v3",
      # post-trained open weights
      "tulu3_sft_8b", "tulu3_dpo_8b", "tulu3_rlvr_8b",
      "tulu3_sft_70b", "tulu3_dpo_70b", "tulu3_rlvr_70b", "tulu3_rlvr_405b",
      "olmo2_sft_13b", "olmo2_dpo_13b", "olmo2_instruct_13b",
      "llama31_instruct_70b", "llama31_instruct_405b",
      "qwen25_instruct_72b", "mistral_instruct_large2", "deepseek_chat_v3"
    ),
    family = c(
      "tulu3", "tulu3", "olmo2", "llama", "llama", "qwen", "mistral", "deepseek",
      "tulu3", "tulu3", "tulu3", "tulu3", "tulu3", "tulu3", "tulu3",
      "olmo2", "olmo2", "olmo2", "llama", "llama", "qwen", "mistral", "deepseek"
    ),
    is_post_trained = c(
      0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1
    ),
    stringsAsFactors = FALSE
  )
}

build_e2_design <- function(n_per_cell_domain = N_QUESTIONS_PER_CELL_DOMAIN,
                            models = registered_e2_roster()) {
  # is_post_trained must stay NUMERIC so the interaction term is named
  # 'conditioncitations_expected:is_post_trained', the same constraint
  # fit_e2_primary_model enforces with a hard stopifnot.
  models <- models[, c("model_id", "family", "is_post_trained"), drop = FALSE]
  models$is_post_trained <- as.numeric(models$is_post_trained)

  # 4-level condition set (pre-reg §5; length_matched added 2026-05-25).
  conditions <- c("neutral", "citations_expected", "abstention_permitted",
                  "length_matched")

  questions <- expand.grid(
    cell = CELLS,
    domain = DOMAINS,
    q_within = seq_len(n_per_cell_domain),
    stringsAsFactors = FALSE
  )
  questions$question_id <- with(questions, paste(domain, cell, q_within, sep = "_"))
  questions$q_within <- NULL

  full <- merge(merge(models, questions, by = NULL),
                data.frame(condition = conditions, stringsAsFactors = FALSE),
                by = NULL)
  full$cell <- factor(full$cell, levels = CELLS)
  full$domain <- factor(full$domain, levels = DOMAINS)
  full$condition <- factor(full$condition, levels = conditions)
  full$family <- factor(full$family)
  full$model_id <- factor(full$model_id)
  full$question_id <- factor(full$question_id)
  full
}

simulate_e2_outcomes <- function(design, family_sd = FAMILY_SD,
                                  model_sd = MODEL_SD, question_sd = QUESTION_SD) {
  set.seed(SEED)
  baseline_logit <- qlogis(BASELINE_FAB_PROB)

  cond_post_effect <- declared_e2_interaction_logit()
  abst_logit <- qlogis(max(BASELINE_FAB_PROB - 0.05, 0.01))
  cond_abst_effect <- abst_logit - baseline_logit

  cell_offset <- c(known_real = -1.5, obscure_real = 0.0, unanswerable = 1.0)

  family_re <- rnorm(nlevels(design$family), 0, family_sd)
  names(family_re) <- levels(design$family)
  model_re <- rnorm(nlevels(design$model_id), 0, model_sd)
  names(model_re) <- levels(design$model_id)
  question_re <- rnorm(nlevels(design$question_id), 0, question_sd)
  names(question_re) <- levels(design$question_id)

  cond_effect <- ifelse(
    design$is_post_trained == 1 & design$condition == "citations_expected", cond_post_effect,
    ifelse(design$is_post_trained == 1 & design$condition == "abstention_permitted", cond_abst_effect, 0)
  )

  eta <- baseline_logit +
    cell_offset[as.character(design$cell)] +
    cond_effect +
    family_re[as.character(design$family)] +
    model_re[as.character(design$model_id)] +
    question_re[as.character(design$question_id)]

  rbinom(nrow(design), 1, plogis(eta))
}

run_e2_power_analysis <- function(design = build_e2_design(),
                                  label = "E2 (open-weights confirmatory roster)") {
  cat(sprintf("\n=== %s power analysis ===\n", label))
  cat(sprintf("Design: %d open-weights models x %d questions x %d conditions = %d observations\n",
              nlevels(design$model_id), nlevels(design$question_id),
              nlevels(design$condition), nrow(design)))

  design$fabricated <- simulate_e2_outcomes(design)
  cat("\nSimulated fabrication rates per condition x is_post_trained:\n")
  agg <- aggregate(fabricated ~ condition + is_post_trained, data = design, FUN = mean)
  print(agg)

  cat("\nFitting mixed-effects model on simulated data...\n")
  model <- glmer(
    fabricated ~ condition * is_post_trained * cell + domain
                + (1 | family / model_id)
                + (1 | question_id),
    data = design,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  )

  # Pin the interaction to the DECLARED effect so powerSim returns a-priori power,
  # not observed power from the noisy single-dataset fit (simr's default behavior).
  target_logit <- declared_e2_interaction_logit()
  fixef(model)["conditioncitations_expected:is_post_trained"] <- target_logit
  cat(sprintf("Pinned a-priori effect: conditioncitations_expected:is_post_trained = %.5f logit\n",
              target_logit))

  cat(sprintf("\nRunning powerSim for condition:is_post_trained (n_sim=%d)...\n", N_SIM))
  power_result <- powerSim(
    model,
    test = fixed("conditioncitations_expected:is_post_trained", method = "z"),
    nsim = N_SIM,
    seed = SEED,
    progress = interactive()
  )
  print(power_result)

  report_power_dual_alpha(power_result, label)
  invisible(power_result)
}

# ---- Variance-component sensitivity (pre-reg §6 power sensitivity) -------------
# The headline power numbers assume the pilot-informed variance components
# (FAMILY_SD/MODEL_SD/QUESTION_SD). A reviewer will ask whether the gate survives
# if the true between-group variance is larger. We re-run power at a grid of
# multipliers applied to all three SDs and report nominal power at each, so the
# gate's robustness to the variance assumption is documented rather than assumed.
run_power_variance_sensitivity <- function(experiment = "e1",
                                           multipliers = c(0.5, 1.0, 1.5, 2.0)) {
  cat(sprintf("\n=== %s power: variance-component sensitivity ===\n",
              toupper(experiment)))
  if (experiment == "e1") {
    design <- build_e1_design()
    target_logit <- declared_e1_interaction_logit()
    term <- "stage_ordinal:cellobscure_real"
    fit_one <- function(d) glmer(
      fabricated ~ stage_ordinal * cell + domain
                  + (1 | family/model_id) + (1 | question_id),
      data = d, family = binomial("logit"),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
    sim_one <- simulate_e1_outcomes
  } else {
    design <- build_e2_design()
    target_logit <- declared_e2_interaction_logit()
    term <- "conditioncitations_expected:is_post_trained"
    fit_one <- function(d) glmer(
      fabricated ~ condition * is_post_trained * cell + domain
                  + (1 | family/model_id) + (1 | question_id),
      data = d, family = binomial("logit"),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
    sim_one <- simulate_e2_outcomes
  }

  for (m in multipliers) {
    d <- design
    d$fabricated <- sim_one(d, family_sd = FAMILY_SD * m, model_sd = MODEL_SD * m,
                            question_sd = QUESTION_SD * m)
    model <- fit_one(d)
    fixef(model)[term] <- target_logit
    pr <- powerSim(model, test = fixed(term, method = "z"),
                   nsim = N_SIM, seed = SEED, progress = interactive())
    res <- report_power_dual_alpha(pr, sprintf("%s [SDx%.1f]", toupper(experiment), m))
    cat(sprintf("  -> SD multiplier %.1f: nominal power %.1f%%\n", m,
                res$p_nominal * 100))
  }
}

# ---- main ---------------------------------------------------------------------

# sys.nframe() == 0L, not !interactive(): under Rscript both a direct run and a
# source() from a test harness are non-interactive, so the old guard made this
# file impossible to source without launching a full 200-simulation run. Same
# convention as e1_checkpoint_cascade.R and e2_prompt_manipulation.R.
if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  experiment <- if (length(args) > 0) args[1] else "both"

  valid <- c("e1", "e2", "both", "varsens-e1", "varsens-e2",
             "realized-e1", "realized-e2", "partial-e1", "partial-e2",
             "breadth-e1", "breadth-e2")
  if (!experiment %in% valid) {
    stop(sprintf("Unknown experiment '%s' (expected one of: %s)",
                 experiment, paste(valid, collapse = ", ")))
  }

  if (experiment %in% c("partial-e1", "partial-e2")) {
    # Remaining args name the registered breadth pairs to add to the realized
    # roster, e.g. `partial-e1 llama qwen`. No args means the realized roster
    # itself, which reproduces the already-measured baseline.
    add <- args[-1]
    roster <- partial_open_weights_roster(add)
    label <- sprintf("%s (realized + %s)",
                     if (experiment == "partial-e1") "E1" else "E2",
                     if (length(add) == 0L) "nothing" else paste(add, collapse = "+"))
    cat(sprintf("\nPartial roster: %d model_ids across %d families (added: %s)\n",
                nrow(roster), length(unique(roster$family)),
                if (length(add) == 0L) "none" else paste(add, collapse = ", ")))
    if (experiment == "partial-e1") {
      run_e1_power_analysis(design = build_e1_design(models = roster), label = label)
    } else {
      run_e2_power_analysis(design = build_e2_design(models = roster), label = label)
    }
  } else if (experiment == "varsens-e1") {
    run_power_variance_sensitivity("e1")
  } else if (experiment == "varsens-e2") {
    run_power_variance_sensitivity("e2")
  } else if (experiment %in% c("breadth-e1", "breadth-e2")) {
    roster <- breadth_open_weights_roster()
    cat(sprintf("\nBreadth roster (amendment #13): %d model_ids across %d families\n",
                nrow(roster), length(unique(roster$family))))
    if (experiment == "breadth-e1") {
      run_e1_power_analysis(design = build_e1_design(models = roster),
                            label = "E1 (15-model roster after amendment #13 collection)")
    } else {
      run_e2_power_analysis(design = build_e2_design(models = roster),
                            label = "E2 (15-model roster after amendment #13 collection)")
    }
  } else if (experiment == "realized-e1") {
    run_e1_power_analysis(
      design = build_e1_design(models = realized_open_weights_roster()),
      label = "E1 (realized Phase C roster)")
  } else if (experiment == "realized-e2") {
    run_e2_power_analysis(
      design = build_e2_design(models = realized_open_weights_roster()),
      label = "E2 (realized Phase C roster)")
  } else {
    if (experiment %in% c("e1", "both")) run_e1_power_analysis()
    if (experiment %in% c("e2", "both")) run_e2_power_analysis()
  }
}
