# §11 amendment #13: the confirmatory open-weights roster is smaller than the one registered, and smaller than the one the power figures describe

**Status: FILED 2026-09-06 as §11 amendment #13 in `PRE_REGISTRATION_OSF.md`, before any of the three new checkpoints was judged.** Written 2026-09-04 as a draft. This one is not a self-report of a deviation the project already knew about: the reduction was found by reading the registered roster back against the registry the confirmatory fit actually used. The filed entry is the authoritative wording; this document keeps the full working rationale.

**Why this is worth reading first.** E1's headline registered result is a **null**. The pre-registration's answer to "is that null informative?" is its §3 power table. That table was computed on a roster the study did not run.

---

## The discrepancy, stated plainly

**Registered** (pre-reg §4 model list, and §3 "Faithful model rosters"):

> E1 = 23 open-weights model_ids across 6 families ... **E2 confirmatory = the same 23-model open-weights roster (8 base + 15 post-trained, 6 families).**

The registered §4 list enumerates: the Tülu-3 8B cascade (4 stages), the Tülu-3 70B cascade (4 stages), a Tülu-3 405B final-checkpoint anchor (1), the OLMo-2-13B cascade (4 stages), and a **cross-family base→Instruct breadth arm** of five base/final pairs (Llama-3.1-70B, Llama-3.1-405B, Qwen2.5-72B, Mistral-Large-2, DeepSeek-V3). 4 + 4 + 1 + 4 + 10 = 23.

**Realized** (`data/analysis_inputs/phase_c_final/models.jsonl`, the registry the confirmatory fit read):

12 checkpoints in 3 declared families, each carrying a complete four-stage cascade:

| family | stage 0 | stage 1 | stage 2 | stage 3 |
|---|---|---|---|---|
| `tulu3_8b` | Llama-3.1-8B | Tulu-3-8B-SFT | Tulu-3-8B-DPO | Tulu-3-8B |
| `tulu3_70b` | Llama-3.1-70B | Tulu-3-70B-SFT | Tulu-3-70B-DPO | Tulu-3-70B |
| `olmo2_13b` | OLMo-2-13B | OLMo-2-13B-SFT | OLMo-2-13B-DPO | OLMo-2-13B-Instruct |

**Not run: 11 of the 23 registered model_ids**, namely the 405B anchor and the entire cross-family breadth arm. The registered roster was 8 base + 15 post-trained; the realized roster is 3 base + 9 post-trained.

## What this does to the registered power figures

The §3 table reports **E1 97.5%** (90.0% at the Holm floor α = 0.01) and **E2 95.5%** (85.0%), and §3 is explicit that these are computed on the 23-model roster, with the E2 note going out of its way to record that an earlier 31-model figure was *withdrawn* precisely because it described a roster the confirmatory model does not run on. That correction was right, and the same objection applies again: **the registered figures describe a 23-model design, and the study ran a 12-model one.** They are not the achieved power of the reported tests.

Three structural consequences, all pushing the same way:

1. **Fewer model-level replicates.** E1's `stage_ordinal` slope is estimated across model_ids; 12 instead of 23 is roughly half the between-model information.
2. **Fewer base comparators.** E2's `is_post_trained` contrast is identified by families carrying both base and post-trained checkpoints. Registered: six do. Realized: three.
3. **Three family levels.** The confirmatory models carry `(1 | family/model_id)`. A random intercept estimated from three levels is at the edge of what is estimable, and both experiments' maximal fits are reported as singular. Under the registered roster's own family scheme (which grouped the 8B and 70B Tülu-3 cascades as one `tulu3` family) these 12 checkpoints would be **two** families, not six.

Partly offsetting: every realized family carries a *complete* four-stage cascade, whereas ten of the registered roster's models were base/final pairs contributing no intermediate stages. So the realized design is worse in breadth but not in stage coverage. Whether that offset matters is an empirical question, which is why it was measured rather than argued.

## What was measured

`src/analysis/power_simulation.R` gained a `realized-e1` / `realized-e2` mode. It reuses the registered machinery unchanged and swaps only the roster, which it **reads from `data/analysis_inputs/phase_c_final/models.jsonl`** rather than retyping, so the power roster cannot drift from the roster that was actually fitted. The registered roster is preserved in `registered_open_weights_roster()` and pinned by tests at 23 model_ids / 6 families, so the registered numbers stay reproducible.

**This is an a-priori calculation, not observed power.** The interaction coefficient is pinned to the same pre-registered declared effect the registered run pinned (E1 0.14728 logit, a 30→40pp rise over three stages in obscure-but-real; E2 0.44183 logit, a 10pp post-trained condition lift) before `powerSim`, so nothing in the result is a function of the observed estimate. Reporting *observed* power for a null would be circular and is not what this is.

**Result (nsim = 200, seed = 42, dual-α, realized 12-model roster; E1 run 2026-09-04 over 10,800 observations from 3 families × 12 models × 900 questions, E2 run 2026-09-05 over 43,200 observations from 12 models × 900 questions × 4 conditions; both 0 errors / 0 warnings / 0 messages, which is what §3 says confirms the a-priori pin took effect rather than silently falling back to observed power):**

| Test | Registered roster (23 models) | Realized roster (12 models) | |
|---|---|---|---|
| E1 `stage_ordinal × cell` @ α = 0.05 | 97.5% [94.26, 99.18] | **73.0% [66.28, 79.02]** | **FAILS the 80% gate** |
| E1 @ α = 0.01 (Holm floor) | 90.0% | **54.0%** | **FAILS the 80% gate** |
| E2 `condition × is_post_trained` @ α = 0.05 | 95.5% [91.63, 97.92] | **73.0% [66.28, 79.02]** | **FAILS the 80% gate** |
| E2 @ α = 0.01 (Holm floor) | 85.0% | **51.0%** | **FAILS the 80% gate** |

The two α = 0.05 figures are identical to the printed precision, which is a coincidence and not a copied result: both are 146 rejections out of 200 simulations, so both carry the same binomial interval. The runs are demonstrably distinct at the Holm floor, where the same stored p-value vectors give 54.0% for E1 and 51.0% for E2. The two logs also differ in design line, pinned effect and runtime (E1 34 minutes, E2 18 h 26 m, the difference being a 26-coefficient GLMM refitted on 42,845 rows against E1's 18).

### 🔴 The E1 figure is the finding, and it is not a small correction

**Registered 97.5%, realized 73.0%.** The gate the pre-registration set for itself is 80%, and §3 is explicit that it reports power at the Holm floor as well because that is "the threshold a hostile reviewer applies." At that floor the realized design has **54.0%** power, against a registered 90.0%. The continuous `stage_ordinal:cellobscure_real` interaction is family member 2 of the E1 Holm family, so the Holm floor is a threshold it can genuinely face.

The consequence for the paper is direct and unavoidable. **E1's registered result is a null, and the realized design would fail to detect the pre-registered effect roughly 27% of the time at nominal α and roughly 46% of the time at the Holm floor.** That is not a confident null. It cannot be reported as "post-training does not raise fabrication selectively"; it can only be reported as "this design, as actually run, was not powered to settle that question at the registered effect size."

Two guardrails on how to say it:

- **Do not convert this into observed or post-hoc power.** The observed interaction (0.084 logit) is smaller than the registered declared effect (0.14728 logit), so power at the observed estimate would be lower still, but that quantity is a deterministic function of the p-value and is circular for a null. Everything above is a-priori power at the registered effect on the realized design, which is a property of the design, not of the result.
- **Do not use this to rescue T1.** Low power explains why a null might occur; it is not evidence the effect exists. The registered conclusion stands as "not supported." What changes is that the null is uninformative rather than informative, which is a weaker statement than the pre-registration anticipated being able to make, and weaker in the direction that costs the paper a claim rather than buying one.

E2 is unaffected in kind: its headline result rejects decisively (Holm p 2.0e-34), and power matters for interpreting failures to reject, not rejections. The E2 figure is still reported for completeness.

## 🔒 Pre-committed collection rule, fixed 2026-09-06 BEFORE any new data exists

**This section was written and committed before a single new generation was produced. Its timestamp is the point.**

The realized-power measurement above showed E1 at 73.0% against a registered 80% gate. A power ladder over intermediate rosters, run on the same a-priori machinery with the effect pinned to the registered declared value, measured what collecting registered-but-uncollected checkpoints would buy:

| roster | model_ids | families | α = 0.05 | α = 0.01 (Holm floor) |
|---|---|---|---|---|
| realized (what Phase C ran) | 12 | 3 | 73.0% | 54.0% |
| + `llama31_instruct_70b` | 14 | 4 | **80.0%** | 66.0% |
| + Qwen2.5-72B pair | 16 | 5 | **91.0%** | 73.5% |
| + Mistral-Large-2 pair | 18 | 6 | 88.0% | 74.5% |
| registered | 23 | 6 | 97.5% | 90.0% |

Logs: `data/analysis/phase_c/logs/power_partial_e1_*.log`. The 88.0% at 18 models is not a decrease from 91.0%; at n_sim = 200 the standard error near 0.9 is about 2.1pp and the two differ by well under two of those.

**The decision and the rule, both fixed now:**

1. **What will be collected:** exactly three checkpoints, `meta-llama/Llama-3.1-70B-Instruct`, `Qwen/Qwen2.5-72B` and `Qwen/Qwen2.5-72B-Instruct`. All three are **already in the registered §4 roster**; none is a new model invented after seeing a result. This moves the realized roster toward the registered one, it does not depart from it.
2. **Why these three and no others:** they are the registered checkpoints reachable within the available GPU budget, and the ladder shows they clear the registered 80% gate. Mistral-Large-2 is excluded because it adds nothing measurable at more than twice the cost; the 405B pair and DeepSeek-V3 are excluded because they exceed the budget and, for DeepSeek-V3, the largest available instance.
3. **The stopping rule is the registered power gate, NOT statistical significance.** Collection stops when the roster clears 80% a-priori power at α = 0.05 for the registered effect. The E1 result will not be examined to decide whether to collect more. There will be no second round of collection contingent on what the interaction does.
4. **The analysis does not change.** Same hypotheses, same estimands, same Holm families, same GLMM specification, same outcome coding, same exclusion rules. Only the roster grows, toward what was registered.
5. **Both outcomes are reportable in advance.** If the interaction remains null at ~91% power, that null is *informative* and will be reported as evidence against selective concentration. If it becomes significant, T1 is supported. Neither outcome licenses re-opening collection.

**The honest sequence, stated plainly because a reviewer will reconstruct it anyway:** E1 returned a null; the roster reduction was then discovered; realized power was measured and failed the gate; the ladder was measured; and only then were the three registered checkpoints collected. Every step after the null is a design repair toward the registered design, with the effect size pinned a priori throughout. What would make this illegitimate is choosing the roster by looking at the E1 estimate, and that is precisely what rule 3 forecloses.

**Interface changes made for collection, in the same class as amendment #11:** three rows appended to `data/models/phase_c_checkpoints.jsonl` (revisions pinned to HF commit SHAs `1605565b…`, `efba10c8…`, `495f3936…`; `tensor_parallel_size` 2; bf16 like every existing row) and three `BatchKey` join codes added to `_MODEL_CODE` in `src/elicit.py` (`l70i`, `q72b`, `q72i`). Neither touches any estimand, coding or test. Collection ran 2026-09-06 on a 2x A100-80GB Thunder Compute instance with vLLM 0.28.0, `dtype=bfloat16`, `max_model_len` 4096, temperature 0, seed 20260614, the same generation code path (`src/run_phase_c.py --single-model`) and the same frozen 900-question file as the twelve Phase C checkpoints.

**Grouping decision, made 2026-09-06 before any of the three was judged (was: deferred to analysis).** The registered roster lists `meta-llama/Llama-3.1-70B` twice, as `tulu3_base_70b` and as `llama31_base_70b`, under two families, and `BREADTH_ADDITIONS$llama` in the ladder mirrored that by adding the base as a second family. In the data it is one checkpoint with one set of 3,600 generations, already `tulu3_70b` stage 0, so the ladder's 14- and 16-model rungs were simulated on a structure the real data cannot carry without duplicating 3,600 observations. **Rule: family = shared base model.** `Llama-3.1-70B-Instruct` joins `tulu3_70b` as a sibling post-trained checkpoint (stage 3, `is_post_trained` 1); the Qwen pair is its own family `qwen25_72b`; 15 checkpoints in 4 families, no duplicated rows. Applied in `data/models/phase_c_checkpoints.jsonl` (commit `347a732`), the column the analysis registry derives `family` from. Rejected: a separate family that duplicates the base (double-counts), and a one-member `llama31_70b` family (hides the shared base, estimates a family intercept from one model). The a-priori power is being rerun on this exact roster (`breadth-e1` / `breadth-e2`, launched 2026-09-06, logs `data/analysis/phase_c/logs/power_breadth_e{1,2}.log`), and **that rerun, not the ladder, is what the 80% gate in the rule above is measured against.**

**Collection outcome (2026-09-06):** all three checkpoints generated (3,600 each, 0 errors, commit `bba6ac9`) on a 2x A100-80GB Thunder instance in 1.68 h (about $3.80), then extracted on the paid Batch API with the Phase C extractor `claude-haiku-4-5-20251001`: batch `msgbatch_01HgUADC6Mw4aTx8CE23rnNb`, 10,800 / 10,800 succeeded, 0 errors, 12,056 citations across 7,378 citing generations, priced $14.30 plus a $0.17 pilot at published Batch rates (the $5.94 pre-pilot estimate had measured the extractor's output as the bare citations JSON and was 3.2x low). Next: judge the 12,056 citations on the registered J2a stack, rebuild the 15-model analysis inputs, refit E1/E2 under both pipelines with Pass B from zero.

**The Holm floor is not repaired and this is not claimed.** No reachable roster reaches 80% at α = 0.01; the best is 74.5%, and even the full registered 23-model roster reached only 90%. The interaction is member 2 of a five-test Holm family, so that floor is a threshold it genuinely faces. Collecting these three checkpoints repairs the nominal-α reading and leaves the strict-threshold reading materially where it was.

## Effect on the registered analysis

No hypothesis, estimand, Holm family, GLMM specification, outcome coding or exclusion rule changes. Every reported test is the registered test on the registered model. What changes is the **description of what the design could detect**, which is exactly what a null result has to be read against.

Two consequences to carry into the text regardless of how the numbers land:

- **E1's null must be reported against the realized power, not the registered 97.5%.** If realized power at the declared effect is comfortably above the 80% gate, the null is informative and should be stated as such. If it is not, the honest reading is "underpowered for the registered effect on the realized roster," and the null cannot carry the weight the pre-registration expected of it.
- **The generalization claim narrows.** The cross-family arm was registered as "E1 breadth," and it is gone. Every reported open-weights result rests on two base models (Llama-3.1 and OLMo-2) and two post-training recipes (Tülu-3 and OLMo-2's). The paper should not describe the finding as spanning model families in a way the realized roster does not support. E4's cross-family transfer split already adapted to this (train Tülu-3, evaluate OLMo-2 only, per `docs/e4_surface_feature_spec.md`); the E1/E2 framing has not.

## The related gap: the closed-API arm, disclosed in the results but not as an amendment

The same reading turns up a second roster gap, which is in better shape but not in good shape. Pre-reg §4 registers a closed-API block (three Anthropic snapshots, three OpenAI, two Google), and §2 registers **H1.2b** against it as a secondary hypothesis. **All three Phase C registries (`phase_c_final`, `phase_c_registered`, `phase_c_drift`) contain 12 models, every one of them open-weights, and zero closed-API rows.**

The *consequence* is disclosed: §4 of the paper states that the secondary block carries no formula, no coefficients and no directional result, and that H1.2b is untested rather than unsupported. That is the right way to report it. What is missing is the same thing missing for the open-weights reduction: a §11 record of the decision not to collect it, and a note that a registered secondary hypothesis was therefore never tested.

Taken together, the registered roster was roughly 31 model_ids (23 open-weights plus the closed-API block) and the realized roster is 12. Amendment #13 should cover both, since they are one decision seen from two sides: Phase C collected the three open-weights cascades and nothing else.

## Why it was not caught earlier

Amendment #11 registered the twelve Phase C checkpoints in the `BatchKey` code table and states "No per-cell, per-test, estimand, outcome-coding, or taxonomy change; the custom_id is a join key only." That is true of the encoding change it describes, and the number 12 is visible in it. But #11 is an interface amendment, and the roster reduction it incidentally reveals was never recorded as a scope deviation, nor propagated to the §3 power figures. The paper says "all 12 checkpoints" and "twelve checkpoints" without anywhere stating that 23 across 6 families were registered.

## When

Commit `347a732` (grouping rule and `breadth-*` power modes); the §11 entry is filed in the commit immediately following it. Realized-power tooling `98f4925`; collection rule `d076c4a`; generations `bba6ac9`. Roster comparison reproducible with `Rscript src/analysis/test_power_simulation.R` (26 tests: pins the registered roster at 23/6, reads the realized one from the confirmatory registry, and pins the breadth roster at 15/4 with no duplicates). Power runs logged at `data/analysis/phase_c/logs/power_realized_e1.log`, `power_realized_e2.log`, `power_breadth_e1.log`, `power_breadth_e2.log`.

## Insertion checklist

1. ✅ Filed as §11 amendment #13 on 2026-09-06, stated in its first line as a disclosure of an already-executed reduction plus the prospective rules for the repair.
2. ✅ Four realized power numbers filled from the logs.
3. ✅ §3 power table annotated with a cross-reference to #13.
4. ✅ Row 13 added to the paper's amendment table in `paper/appendix/A_deviations.tex` with a subsection.
5. ⬜ Narrow the generalization language in §3 methods and §7 discussion. After collection the honest scope is three base models (Llama-3.1, OLMo-2, Qwen2.5) and three post-training recipes; this is written with the 15-model refit, not before it.
6. 🟡 15-model a-priori power: **E1 done 2026-09-06: 83.0% [77.06, 87.93] at α = 0.05 (PASS), 69.0% at the Holm floor (FAIL)**, 200 sims, 0 errors / warnings / messages, 53 min; appended to the #13 entry and the appendix subsection. E2 pending (~18 h); append it when `power_breadth_e2.log` finishes.
