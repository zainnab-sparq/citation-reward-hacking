# Judge-instrument drift: differentiality test specification

**Status: EXPLORATORY / ROBUSTNESS. Not a pre-registration amendment.** These models are diagnostics of the measurement instrument. They are not members of the registered m=5 confirmatory family, they do not alter any registered estimand, and no registered result is conditioned on their outcome.

**Written 2026-08-11, before any model in this document was fitted.** Locked here so the analysis cannot be shaped by its own result.

## 1. Why this test exists

The J2a judge ran over roughly two months, and its behaviour changed twice mid-collection (`scripts/_judge_drift_three_era.py`). Verdict distributions differ substantially between eras while item composition stays balanced, so the shift is a property of the instrument rather than of the citations being judged.

That matters for the registered analysis only if the misclassification is **differential**, meaning the instrument's error rate depends on the treatment variables (`condition` for E2, `stage_ordinal` for E1). Non-differential misclassification attenuates contrasts toward the null, which is conservative. Differential misclassification can bias a contrast in either direction, including away from the null.

The standing argument for non-differentiality has been that era is balanced across condition. Balance is necessary but not sufficient: it constrains the *mix* of conditions within each era, not whether the instrument *treats* those conditions differently. This test measures the thing balance cannot.

## 2. Prior looks at the data (disclosed)

Per-era verdict rates have been monitored throughout collection, and on 2026-08-11 a citation-level `era x condition` verdict table was computed and showed a suggestive but not conclusive pattern. This specification is therefore written with knowledge that such a pattern may exist. It is fixed before any model fit, and the decision rules in §6 are committed in advance, so the disclosure is the safeguard: the reader knows the order of operations.

The registered E1/E2 models have **not** been fitted and their outcome is unseen.

## 3. Era definition

Era is assigned from the judgment's row index in `data/judgments/phase_c/claude-subagent-sonnet.jsonl`, which is append-only in judging order.

| era | rows | basis |
|---|---|---|
| EARLY | `[0, 16500)` | pre-degradation |
| DEGRADED | `[16500, 24000)` | grounding collapse (blocks 33 to 47) |
| RECOVERED | `[24000, end)` | grounding restored (block 48 onward) |

Both boundaries were set from **grounding metrics** (`judge_search_count` distribution, tool-use counts, wall-clock) before this test was conceived, and are **not** re-tuned to the outcome. They stay fixed at these values regardless of what the models show. `scripts/_judge_drift_three_era.py` is the source of the constants.

## 4. Unit of analysis

Generation level, matching the registered `fabricated` outcome, restricted to generations with at least one judged citation.

Justification for the restriction: era exists only where the judge fired. A generation whose citations were all verifier-`REAL_CONFIRMED` was never judged, so its outcome cannot be affected by judge drift and it carries no information about the era contrast. Including such generations would dilute the test without adding signal. The estimand is therefore "among generations with at least one verifier-flagged citation", and that is how it will be reported.

Era is well defined at this level: of 18,714 generations judged as of 2026-08-11, exactly 1 has citations spanning an era boundary. **Generations spanning more than one era are dropped**, and the count dropped is reported.

## 5. Models

Each drift model takes the corresponding registered model verbatim and adds era. Nothing else about the registered structure changes.

**D1, E2-facing (condition).** Restricted to open-weights, as E2 is.
- null: `fabricated ~ condition * is_post_trained * cell + domain + era + (1 | family/model_id) + (1 | question_id)`
- alt: the same plus `condition:era`
- test: likelihood-ratio test of alt vs null, 3 numerator df (4 conditions x 3 eras, additive baseline).

**D2, E1-facing (stage_ordinal).**
- null: `fabricated ~ stage_ordinal * cell * domain + era + (1 | family/model_id) + (1 | question_id)`
- alt: the same plus `stage_ordinal:era`
- test: likelihood-ratio test, 2 numerator df.

**D3, continuous grounding.** Replaces the 3-level era factor with per-generation mean `judge_search_count`, which avoids drawing boundaries at all.
- null: `fabricated ~ condition * is_post_trained * cell + domain + grounding + (1 | family/model_id) + (1 | question_id)`
- alt: the same plus `condition:grounding`
- test: likelihood-ratio test, 3 numerator df.

D3 is the more honest specification if the two agree, because it does not depend on where the era lines were drawn. If D1 and D3 disagree, that disagreement is itself the finding and both are reported.

**Random effects and convergence.** Same convention as `phase_b_descriptive.R`: fit as specified, report singularity and convergence warnings, and if a model is singular report the fallback alongside rather than silently substituting it. `bobyqa`, `maxfun = 1e5`.

## 6. Decision rules, committed in advance

**If the interaction is not significant at alpha = 0.05:** non-differential misclassification is supported. The registered contrasts stand unchanged. The drift is reported in limitations with the three-era table and this test as the evidence that it does not reach the treatment contrasts.

**If the interaction is significant:** the registered estimands are still **not** adjusted, and the registered result remains primary. Added to the report are (i) an era-stratified re-estimate of the same contrasts as a robustness panel, (ii) an explicit statement that this test detects differentiality but does **not** identify the direction of bias, because eras contain different items. Only the paired re-judge sub-study, which holds the item fixed, can attribute the difference to the instrument.

**Ordering constraint on the robustness panel.** That panel re-estimates the registered condition contrasts, so producing it before the one-shot confirmatory fit would amount to reading the registered result early. It is therefore gated behind an explicit `--robustness-panel` flag and never runs as a side effect of the drift diagnostics, which reveal no registered estimand. Run it alongside or after the confirmatory fit, not before. The zero-effort robustness check of §7.3 sits behind the same flag for the same reason.

**Either way, all three tests are reported**, with their p-values, whether or not they agree. There is no outcome-dependent choice about which model gets reported.

## 7. Reported alongside (drift-adjacent reporting additions)

These are reporting additions, not amendments.

1. **Era-stratified UNVERIFIABLE exclusion rate.** Pre-registration §5 requires the generation-level UNVERIFIABLE-only exclusion rate stratified by `cell x stage_ordinal` (E1) and `condition x is_post_trained` (E2), reported before the primary is interpreted, with a pre-committed rule promoting Sensitivity #1 to co-primary if the rate is monotone in stage or condition within `obscure_real`. Era did not exist at registration, and the era-level UNVERIFIABLE rate moved from 16.7% to 7.2%, so **era is added as a third stratification**.
2. **Sensitivity #1 (UNVERIFIABLE recoded as fabricated) is reported regardless** of whether the escalation rule fires. It excludes nothing by construction, so it is immune to era-varying exclusion. If it agrees with the primary, the drift's reach into the registered result is bounded and that can be stated explicitly.
3. **The `search_count == 0` robustness check** (re-estimate dropping generations whose judged citations were all evidence-free), already planned, is reported in the same panel.

## 8. Multiplicity

Three interaction tests, reported together, exploratory. No alpha adjustment is claimed and none is needed for their intended use, which is diagnostic rather than confirmatory. They will not be described as establishing an absence of drift effects; a non-significant interaction bounds the evidence, it does not prove non-differentiality. Power against small interactions is limited and the achieved n per era cell is reported so the reader can judge it.

## 9. What would falsify the reassuring reading

- A significant `condition:era` interaction whose largest component sits in `citations_expected`, the E2 treatment condition, would mean the instrument shift tracks the treatment and the E2 contrast cannot be assumed conservative.
- Disagreement between D1 and D3 would mean the result is an artifact of where the era boundaries fall.
- A paired sub-study transition matrix that varies by condition would confirm differentiality directly, and outranks all three observational tests, which cannot separate instrument effects from item effects.

## 10. Implementation

`src/analysis/phase_c_drift_interaction.R` with `src/analysis/test_phase_c_drift_interaction.R`. Era and grounding columns come from `src/analysis/judge_era.py`, which derives them from judgment row order and joins them to generation keys. Neither `e1_checkpoint_cascade.R` nor `e2_prompt_manipulation.R` is modified.

The paired sub-study of §9 is implemented by `src/analysis/substudy_rejudge.py` (sampling and batch emission) and `src/analysis/substudy_rejudge_score.py` (transition matrices, per-condition flip rates, the differentiality test, and the design-weighted corpus rate). Sampling draws 250 from EARLY and 250 from DEGRADED, balanced across the four conditions rather than proportional, because the comparison of interest is across conditions and its precision is set by the smallest cell. Design weights are retained so a corpus-level rate can still be recovered.
