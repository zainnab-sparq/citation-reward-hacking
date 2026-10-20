# Amendment #14 escalation check: Wald CIs for the directional contrasts.
#
# Pre-reg §6 adds a profile-likelihood CI for any primary directional contrast
# whose interval lower bound lands within 2pp of the 10pp probability-scale
# threshold. Amendment #14 replaced the bootstrap percentile interval with the
# Wald interval, so this script refits the registered primary model, takes the
# directional contrast on the probability scale exactly as the confirmatory
# script defines it, and reports its delta-method (Wald) 95% CI with the
# escalation flag. The fitted model and the contrast definitions are the
# registered ones: the confirmatory scripts are sourced, not copied.
#
#   Rscript scripts/_wald_escalation_check.R --exp e1 \
#     --inputs data/analysis_inputs/phase_c15_final \
#     --output data/analysis/phase_c15/e1_corrected_wald_escalation.json

suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
  library(jsonlite)
})

THRESHOLD <- 0.10
TOLERANCE <- 0.02

parse_cli <- function(args) {
  out <- list(exp = NA_character_, inputs = NA_character_, output = NA_character_)
  for (i in seq(1, length(args), by = 2)) out[[sub("^--", "", args[i])]] <- args[i + 1]
  stopifnot(out$exp %in% c("e1", "e2"), !is.na(out$inputs), !is.na(out$output))
  out
}

summarize_contrast <- function(ct, label) {
  ci <- as.data.frame(confint(ct, level = 0.95))
  lo <- ci$asymp.LCL[1]
  list(contrast = label, estimate_pp = ci$estimate[1], ci_lo_pp = lo,
       ci_hi_pp = ci$asymp.UCL[1], se_pp = ci$SE[1],
       escalation_triggered = isTRUE(lo <= THRESHOLD + TOLERANCE))
}

e1_contrasts <- function(model) {
  at <- list(stage_ordinal = c(0, 3), cell = "obscure_real")
  overall <- regrid(emmeans(model, ~ stage_ordinal, at = at, lmer.df = "asymptotic"))
  out <- list(summarize_contrast(contrast(overall, list(rise = c(-1, 1))), "rise_marginal"))
  by_dom <- regrid(emmeans(model, ~ stage_ordinal | domain, at = at, lmer.df = "asymptotic"))
  ct <- contrast(by_dom, list(rise = c(-1, 1)))
  ci <- as.data.frame(confint(ct, level = 0.95))
  for (k in seq_len(nrow(ci))) {
    out[[length(out) + 1]] <- list(
      contrast = paste0("rise_", ci$domain[k]), estimate_pp = ci$estimate[k],
      ci_lo_pp = ci$asymp.LCL[k], ci_hi_pp = ci$asymp.UCL[k], se_pp = ci$SE[k],
      escalation_triggered = isTRUE(ci$asymp.LCL[k] <= THRESHOLD + TOLERANCE))
  }
  out
}

e2_weights <- function(grid) {
  w <- numeric(nrow(grid))
  w[grid$condition == "citations_expected" & grid$is_post_trained == 1] <- 1
  w[grid$condition == "neutral" & grid$is_post_trained == 1] <- -1
  w[grid$condition == "citations_expected" & grid$is_post_trained == 0] <- -1
  w[grid$condition == "neutral" & grid$is_post_trained == 0] <- 1
  stopifnot(sum(abs(w)) == 4)
  w
}

e2_contrasts <- function(model) {
  at <- list(is_post_trained = c(0, 1))
  overall <- regrid(emmeans(model, ~ condition * is_post_trained, at = at, lmer.df = "asymptotic"))
  w <- e2_weights(as.data.frame(overall))
  out <- list(summarize_contrast(contrast(overall, list(interaction = w)), "interaction_marginal"))
  by_dom <- regrid(emmeans(model, ~ condition * is_post_trained | domain, at = at,
                           lmer.df = "asymptotic"))
  grid <- as.data.frame(by_dom)
  for (dom in unique(as.character(grid$domain))) {
    sub_grid <- grid[grid$domain == dom, ]
    ct <- contrast(by_dom, list(interaction = e2_weights(sub_grid)), by = "domain")
    ci <- as.data.frame(confint(ct, level = 0.95))
    row <- ci[ci$domain == dom, ]
    out[[length(out) + 1]] <- list(
      contrast = paste0("interaction_", dom), estimate_pp = row$estimate[1],
      ci_lo_pp = row$asymp.LCL[1], ci_hi_pp = row$asymp.UCL[1], se_pp = row$SE[1],
      escalation_triggered = isTRUE(row$asymp.LCL[1] <= THRESHOLD + TOLERANCE))
  }
  out
}

main <- function() {
  args <- parse_cli(commandArgs(trailingOnly = TRUE))
  env <- new.env()
  script <- if (args$exp == "e1") "src/analysis/e1_checkpoint_cascade.R" else "src/analysis/e2_prompt_manipulation.R"
  sys.source(script, envir = env)
  load <- function(name) env$read_jsonl(file.path(args$inputs, paste0(name, ".jsonl")))
  build <- if (args$exp == "e1") env$build_e1_data else env$build_e2_data
  data <- build(load("generations"), load("verifications"), load("questions"), load("models"))
  cat(sprintf("frame: %d rows\n", nrow(data)))
  model <- if (args$exp == "e1") env$fit_e1_model(data) else env$fit_e2_primary_model(data)
  contrasts <- if (args$exp == "e1") e1_contrasts(model) else e2_contrasts(model)
  for (c in contrasts) {
    cat(sprintf("%-26s %7.4f [%7.4f, %7.4f] escalate=%s\n", c$contrast, c$estimate_pp,
                c$ci_lo_pp, c$ci_hi_pp, c$escalation_triggered))
  }
  result <- list(exp = args$exp, inputs = args$inputs, n_rows = nrow(data),
                 threshold = THRESHOLD, tolerance = TOLERANCE,
                 rule = "profile CI required when the Wald lower bound <= threshold + tolerance (amendment #14)",
                 any_escalation = any(vapply(contrasts, `[[`, logical(1), "escalation_triggered")),
                 contrasts = contrasts)
  writeLines(toJSON(result, auto_unbox = TRUE, digits = NA, pretty = TRUE), args$output)
  cat(sprintf("wrote %s\n", args$output))
}

if (sys.nframe() == 0L) main()
