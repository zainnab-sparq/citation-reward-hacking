# Tests for the power-simulation rosters.
#
# These exist because the registered power figures (pre-reg §3: E1 97.5%, E2
# 95.5%) are computed on a roster that is NOT the roster Phase C actually ran.
# The registered roster must stay reproducible byte-for-byte, and the realized
# roster must be read from the registry the confirmatory fit used rather than
# retyped, or the two can drift apart silently and the comparison becomes
# meaningless.
#
# Run from repo root:
#   Rscript src/analysis/test_power_simulation.R

suppressPackageStartupMessages({ library(lme4) })

PASS_COUNT <- 0L
FAIL_COUNT <- 0L
run_test <- function(name, body) {
  cat(sprintf("  %-64s ... ", name))
  ok <- tryCatch({ body(); TRUE },
                 error = function(e) { cat("FAIL\n      ", conditionMessage(e), "\n"); FALSE })
  if (ok) { cat("ok\n"); PASS_COUNT <<- PASS_COUNT + 1L } else { FAIL_COUNT <<- FAIL_COUNT + 1L }
}

# simr is slow to load and is not needed for roster tests; the script sources
# fine without running any simulation because its main block is interactive-gated.
ps <- new.env()
suppressWarnings(suppressMessages(source("src/analysis/power_simulation.R", local = ps)))

cat("\n[power] registered roster (must not drift) ------------------------------\n")

run_test("registered open-weights roster is 23 model_ids across 6 families", function() {
  r <- ps$registered_open_weights_roster()
  stopifnot(nrow(r) == 23L)
  stopifnot(length(unique(r$family)) == 6L)
})

run_test("registered roster has 8 base and 15 post-trained checkpoints", function() {
  r <- ps$registered_open_weights_roster()
  stopifnot(sum(r$stage_ordinal == 0) == 8L)
  stopifnot(sum(r$stage_ordinal > 0) == 15L)
})

run_test("registered E1 design still builds at 23 models / 6 families", function() {
  d <- ps$build_e1_design(n_per_cell_domain = 1L)
  stopifnot(nlevels(d$model_id) == 23L, nlevels(d$family) == 6L)
})

cat("\n[power] realized roster (read from the confirmatory registry) -----------\n")

REGISTRY <- "data/analysis_inputs/phase_c_final/models.jsonl"

run_test("realized roster loads from the registry the confirmatory fit used", function() {
  if (!file.exists(REGISTRY)) stop("registry missing: ", REGISTRY)
  r <- ps$realized_open_weights_roster(REGISTRY)
  stopifnot(all(c("model_id", "family", "stage_ordinal") %in% names(r)))
  stopifnot(nrow(r) > 0)
})

run_test("realized roster keeps only open-weights rows", function() {
  r <- ps$realized_open_weights_roster(REGISTRY)
  # Every Phase C checkpoint is open-weights, so this is currently a no-op; the
  # filter exists so that adding a closed-API row later cannot silently inflate
  # the confirmatory roster, which is the exact error the 31-vs-23 correction fixed.
  stopifnot(nrow(r) == 12L)
})

run_test("realized roster is materially smaller than the registered one", function() {
  reg <- ps$registered_open_weights_roster()
  rea <- ps$realized_open_weights_roster(REGISTRY)
  stopifnot(nrow(rea) < nrow(reg))
  stopifnot(length(unique(rea$family)) < length(unique(reg$family)))
})

run_test("realized roster stage codes are the registered 0..3 ordinal", function() {
  r <- ps$realized_open_weights_roster(REGISTRY)
  stopifnot(all(r$stage_ordinal %in% 0:3))
  stopifnot(all(0:3 %in% r$stage_ordinal))
})

run_test("every realized family carries a base checkpoint", function() {
  # E1 estimates a stage slope within family. A family with no stage-0 row
  # contributes no within-family base comparison, so this is the structural
  # property the E1 interaction depends on.
  r <- ps$realized_open_weights_roster(REGISTRY)
  by_fam <- tapply(r$stage_ordinal, r$family, function(s) 0 %in% s)
  stopifnot(all(by_fam))
})

run_test("E1 design accepts an injected roster and reflects its size", function() {
  r <- ps$realized_open_weights_roster(REGISTRY)
  d <- ps$build_e1_design(n_per_cell_domain = 1L, models = r)
  stopifnot(nlevels(d$model_id) == nrow(r))
  stopifnot(nlevels(d$family) == length(unique(r$family)))
})

run_test("E2 design accepts an injected roster and derives is_post_trained", function() {
  r <- ps$realized_open_weights_roster(REGISTRY)
  d <- ps$build_e2_design(n_per_cell_domain = 1L, models = r)
  stopifnot("is_post_trained" %in% names(d))
  stopifnot(all(d$is_post_trained %in% c(0, 1)))
  # is_post_trained must be numeric for the interaction term to be named as the
  # analysis expects, the same constraint fit_e2_primary_model enforces.
  stopifnot(is.numeric(d$is_post_trained))
})

run_test("injecting a roster does not mutate the registered default", function() {
  r <- ps$realized_open_weights_roster(REGISTRY)
  invisible(ps$build_e1_design(n_per_cell_domain = 1L, models = r))
  again <- ps$build_e1_design(n_per_cell_domain = 1L)
  stopifnot(nlevels(again$model_id) == 23L)
})

cat("\n[power] partial rosters (realized + a registered breadth family) --------\n")

run_test("partial roster with no additions is exactly the realized roster", function() {
  base <- ps$realized_open_weights_roster(REGISTRY)
  p <- ps$partial_open_weights_roster(character(0), registry = REGISTRY)
  stopifnot(nrow(p) == nrow(base))
  stopifnot(setequal(p$model_id, base$model_id))
  stopifnot(setequal(p$family, base$family))
})

run_test("adding llama adds 2 model_ids and exactly 1 family", function() {
  base <- ps$realized_open_weights_roster(REGISTRY)
  p <- ps$partial_open_weights_roster("llama", registry = REGISTRY)
  stopifnot(nrow(p) == nrow(base) + 2L)
  stopifnot(length(unique(p$family)) == length(unique(base$family)) + 1L)
})

run_test("each added breadth family carries a base and a final checkpoint", function() {
  # Without both, is_post_trained is collinear with family and E2's contrast is
  # not identified within that family, which is the whole point of adding it.
  for (fam in c("llama", "qwen", "mistral")) {
    p <- ps$partial_open_weights_roster(fam, registry = REGISTRY)
    added <- p[p$family == fam, , drop = FALSE]
    stopifnot(nrow(added) == 2L)
    stopifnot(any(added$stage_ordinal == 0L), any(added$stage_ordinal == 3L))
    stopifnot(any(added$is_post_trained == 0L), any(added$is_post_trained == 1L))
  }
})

run_test("adding two families is cumulative and order-independent", function() {
  base <- ps$realized_open_weights_roster(REGISTRY)
  a <- ps$partial_open_weights_roster(c("llama", "qwen"), registry = REGISTRY)
  b <- ps$partial_open_weights_roster(c("qwen", "llama"), registry = REGISTRY)
  stopifnot(nrow(a) == nrow(base) + 4L)
  stopifnot(setequal(a$model_id, b$model_id))
})

run_test("added model_ids never collide with the realized ones", function() {
  base <- ps$realized_open_weights_roster(REGISTRY)
  p <- ps$partial_open_weights_roster(c("llama", "qwen", "mistral"), registry = REGISTRY)
  added <- setdiff(p$model_id, base$model_id)
  stopifnot(length(added) == 6L)
  stopifnot(!any(duplicated(p$model_id)))
})

run_test("an unknown breadth family fails loudly instead of silently no-oping", function() {
  ok <- tryCatch({
    ps$partial_open_weights_roster("gpt5", registry = REGISTRY)
    FALSE
  }, error = function(e) grepl("gpt5", conditionMessage(e), fixed = TRUE))
  stopifnot(ok)
})

run_test("every partial roster addition is drawn from the registered roster", function() {
  # A partial roster is only meaningful as a subset of what was promised. If a
  # model_id appears here that is not in the registered 23, the ladder is
  # measuring a design nobody registered.
  reg <- ps$registered_open_weights_roster()
  base <- ps$realized_open_weights_roster(REGISTRY)
  p <- ps$partial_open_weights_roster(c("llama", "qwen", "mistral"), registry = REGISTRY)
  added <- setdiff(p$model_id, base$model_id)
  stopifnot(all(added %in% reg$model_id))
})

run_test("partial E1 and E2 designs build at the expected model and family counts", function() {
  p <- ps$partial_open_weights_roster(c("llama", "qwen"), registry = REGISTRY)
  d1 <- ps$build_e1_design(n_per_cell_domain = 1L, models = p)
  stopifnot(nlevels(d1$model_id) == 16L, nlevels(d1$family) == 5L)
  d2 <- ps$build_e2_design(n_per_cell_domain = 1L, models = p)
  stopifnot(nlevels(d2$model_id) == 16L, nlevels(d2$family) == 5L)
  stopifnot(is.numeric(d2$is_post_trained))
})

cat("\n[power] breadth roster (the 15 checkpoints actually collected) -----------\n")

CHECKPOINTS <- "data/models/phase_c_checkpoints.jsonl"

run_test("breadth roster reads the checkpoint registry's model_family column", function() {
  if (!file.exists(CHECKPOINTS)) stop("registry missing: ", CHECKPOINTS)
  r <- ps$breadth_open_weights_roster(CHECKPOINTS)
  stopifnot(all(c("model_id", "family", "stage_ordinal", "is_post_trained") %in% names(r)))
  stopifnot(!"model_family" %in% names(r))
})

run_test("breadth roster is 15 model_ids across 4 families with no duplicates", function() {
  # Amendment #13 grouping rule: family = shared base model. Llama-3.1-70B is
  # ONE checkpoint with ONE set of generations, so it appears once, and the
  # ladder's separate two-member `llama` family (which duplicated it) is not
  # what the real data can carry.
  r <- ps$breadth_open_weights_roster(CHECKPOINTS)
  stopifnot(nrow(r) == 15L)
  stopifnot(!any(duplicated(r$model_id)))
  stopifnot(length(unique(r$family)) == 4L)
})

run_test("Llama-3.1-70B-Instruct sits in tulu3_70b as a post-trained stage-3 sibling", function() {
  r <- ps$breadth_open_weights_roster(CHECKPOINTS)
  row <- r[r$model_id == "meta-llama/Llama-3.1-70B-Instruct", , drop = FALSE]
  stopifnot(nrow(row) == 1L)
  stopifnot(row$family == "tulu3_70b", row$stage_ordinal == 3L, row$is_post_trained == 1L)
  stopifnot(sum(r$family == "tulu3_70b") == 5L)
  stopifnot(sum(r$family == "tulu3_70b" & r$stage_ordinal == 0L) == 1L)
})

run_test("the Qwen pair forms its own two-member family with a base and a final", function() {
  r <- ps$breadth_open_weights_roster(CHECKPOINTS)
  q <- r[r$family == "qwen25_72b", , drop = FALSE]
  stopifnot(nrow(q) == 2L)
  stopifnot(setequal(q$stage_ordinal, c(0L, 3L)))
  stopifnot(setequal(q$is_post_trained, c(0L, 1L)))
})

run_test("breadth roster is the realized roster plus exactly the three collected checkpoints", function() {
  base <- ps$realized_open_weights_roster(REGISTRY)
  r <- ps$breadth_open_weights_roster(CHECKPOINTS)
  added <- setdiff(r$model_id, base$model_id)
  stopifnot(setequal(added, c("meta-llama/Llama-3.1-70B-Instruct",
                              "Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct")))
  stopifnot(all(base$model_id %in% r$model_id))
})

run_test("every breadth family carries a base checkpoint", function() {
  r <- ps$breadth_open_weights_roster(CHECKPOINTS)
  by_fam <- tapply(r$stage_ordinal, r$family, function(s) 0 %in% s)
  stopifnot(all(by_fam))
})

run_test("breadth E1 and E2 designs build at 15 models / 4 families", function() {
  r <- ps$breadth_open_weights_roster(CHECKPOINTS)
  d1 <- ps$build_e1_design(n_per_cell_domain = 1L, models = r)
  stopifnot(nlevels(d1$model_id) == 15L, nlevels(d1$family) == 4L)
  d2 <- ps$build_e2_design(n_per_cell_domain = 1L, models = r)
  stopifnot(nlevels(d2$model_id) == 15L, nlevels(d2$family) == 4L)
  stopifnot(is.numeric(d2$is_post_trained))
})

cat(sprintf("\nTests passed: %d, failed: %d\n", PASS_COUNT, FAIL_COUNT))
if (FAIL_COUNT > 0L) quit(status = 1L)
