# Judge-instrument drift: disclosure section (draft)

Draft of the paper's judge-limitations disclosure. **Every figure below is final at the complete 31,980 judgments, refreshed 2026-08-30.** Produced by re-running `scripts/_judge_drift_audit.py`, `scripts/_judge_drift_confound.py`, `scripts/_judge_drift_three_era.py`, `scripts/_judge_drift_unv_channel.py` and `scripts/_unv_exclusion_by_era.py`, then `src/analysis/phase_c_drift_interaction.R` into `data/analysis/phase_c/drift_interaction.json`.

The prose is drafted now, while the reasoning is fresh, precisely because the tempting version of this section is the reassuring one. An earlier reading of these data (2026-08-05/06) claimed the instrument had "recovered" to its original behaviour; a fuller analysis two days later showed it had moved to a third state instead. The wording below is chosen to survive that kind of correction.

## What happened

The J2a judge is an agentic verification step: a Sonnet subagent that reads a fixed prompt, gathers evidence with web search and direct fetches, and returns one of REAL, FABRICATED, or UNVERIFIABLE under the §6 taxonomy. It ran over roughly two months. Its evidence-gathering behaviour changed twice during that period, and its verdict distribution changed with it.

Three regimes, defined by position in the judging order. Boundaries were fixed from the grounding metrics at blocks 33 and 48 before any interaction test was specified, and were never re-tuned to an outcome:

| | EARLY (rows 0 to 16,499) | DEGRADED (16,500 to 23,999) | RECOVERED (24,000+) |
|---|---|---|---|
| n | 16,500 | 7,500 | 7,980 |
| REAL | 46.4% | 37.1% | 43.1% |
| FABRICATED | 36.9% | 49.1% | 49.8% |
| UNVERIFIABLE | 16.7% | 13.8% | 7.1% |
| mean verification effort | 2.03 | 1.66 | 2.09 |

The RECOVERED regime is now 7,980 judgments rather than the 4,266 available when this section was drafted, and the extra data sharpens the reading rather than changing it. Verification effort in RECOVERED matches EARLY (2.09 against 2.03, with zero-effort judgments back under 1%), so the degradation was a bounded episode that fully reversed. The verdict distribution did not reverse with it.

Item composition is matched across regimes to within 1.3 percentage points on domain, question type, model and condition, the largest single deviation being legal at 48.6% in EARLY against 47.3% in RECOVERED. So the shift is a property of the instrument rather than of the citations being judged. Batch ordering was effectively interleaved with respect to the experimental design.

## What the effort metric is, and is not

The per-judgment counter records the agent's self-reported evidence-gathering effort across **all** tools. It is not a count of web searches. Agents that reported search as entirely unavailable still returned high counts by verifying through the Crossref REST API, DOI resolution, PubMed, and CourtListener instead. The honest label is "self-reported verification effort," and the drift must not be described as "search calls dropped." A zero remains meaningful: it marks a judgment where evidence gathering genuinely failed or was skipped.

**The cause of both regime shifts is not established.** A mid-collection change in tool availability or model behaviour would explain both, but that is speculation and the paper should not assert it.

## The stable finding is the UNVERIFIABLE collapse

The most reliable contrast is not the REAL rate. It is UNVERIFIABLE falling from 16.7% to 7.1%, less than half, and it appears in every question type: known_real 10.9 to 5.1, obscure_real 25.7 to 11.8, unanswerable 14.3 to 4.7. The REAL rate moved less and settled later, so it is reported but not leaned on.

At the complete judgment set this stops being a description and becomes a measurement, because effort recovered while UNVERIFIABLE kept falling. Scoring the same judgments under the registered primary coding and under registered Sensitivity #1 shows where the movement lives. Below, each cell is the RECOVERED-minus-EARLY gap in the fabrication rate:

| | registered primary (UNVERIFIABLE NA-excluded) | Sensitivity #1 (UNVERIFIABLE as fabricated) |
|---|---|---|
| all citations | +9.3pp | +3.3pp |
| known_real | +11.2pp | +6.4pp |
| obscure_real | +14.1pp | +3.2pp |
| unanswerable | +4.0pp | +0.6pp |

Closing the UNVERIFIABLE channel absorbs most of the drift in the two question types the hypotheses are about, and leaves the unanswerable cell essentially flat. **known_real is the exception and is not claimed as solved:** there the judge genuinely moved real work into FABRICATED, and only about half the gap closes. That direction inflates the fabrication floor in the cell where fabrication should be rarest, which works against the hypothesis rather than for it, but it is a real residual and is reported as one.

The mechanism is coherent and is arguably the judge working better rather than worse. Under §6, UNVERIFIABLE is the conservative default when investigation is blocked. When evidence gathering works well, fewer items are blocked, so the judge converts "I could not check this" into "I checked, and it is not there."

**This has a consequence that must not be glossed: FABRICATED means different things in different regimes.** In the DEGRADED regime an elevated FABRICATED rate reflects failure to confirm. In the RECOVERED regime it reflects successful positive evidence of absence. Same label, different epistemic content. For this reason the pooled verdict distribution should never be reported without the regime breakdown beside it.

## The degraded regime is not evidence-free

Working against the natural reading that degraded judgments are guesses: through the DEGRADED period the judge continued to make the discriminations that require real evidence. It kept applying the §6 dividing line correctly, separating a real title attached to a different real paper's byline (fabricated) from an ordinary metadata error such as a one-year discrepancy (real). Examples include a real 1994 sequence-alignment paper cited with a fabricated author line, a real 1983 clinical paper cited under a 1987 byline, and a real trial title carrying a different trial's authors. It also produced internal-consistency catches that depend on domain knowledge rather than lookup, such as reporter volumes that could not correspond to the cited year, and it correctly returned REAL for citations carrying genuine metadata errors.

The reasonable conclusion is that degradation hit **coverage**, meaning how many citations got fully checked, rather than **judgment quality per checked citation**.

## Does the drift reach the registered contrasts?

This is the question that matters, and it is not answered by any of the above.

Drift threatens the registered E1/E2 estimands only if the misclassification is **differential**, meaning the instrument's error rate depends on the treatment variables. Non-differential error attenuates a contrast toward the null and is therefore conservative. Differential error can bias a contrast in either direction.

The standing argument for non-differentiality was that regime is balanced across condition. **That argument is insufficient and the paper should not rest on it.** Balance constrains the mix of conditions within each regime; it says nothing about whether the instrument treats those conditions differently.

Three pre-specified tests address it directly (`docs/drift_interaction_prespec.md`, locked before any model was fitted):

- **D1** the `condition x regime` interaction on the registered generation-level outcome, E2-facing
- **D2** the `stage_ordinal x regime` interaction, E1-facing
- **D3** the same question with continuous verification effort in place of the regime factor, which avoids drawing regime boundaries at all

**Final results at the complete 31,980 judgments (20,811 modelled generations). All four models converged and none was singular.**

| test | chi-square | df | p |
|---|---|---|---|
| D1 `condition x regime` | 10.074 | 6 | 0.122 |
| D2 `stage_ordinal x regime` | 0.366 | 2 | 0.833 |
| D3 `condition x verification effort` | 23.970 | 3 | 0.000025 |
| D1 under Sensitivity #1 | 10.418 | 6 | 0.108 |

**D2 is clean.** There is no evidence that the drift interacts with the checkpoint cascade. The E1 contrast is not implicated.

**D1 does not reject at the complete data, and the earlier provisional result did not survive.** At 28,266 judgments D1 stood at chi-square 13.219, p = 0.040. That draft flagged the reason to distrust it: the terms carrying it were the RECOVERED ones, which had the smallest n and were still accumulating. RECOVERED then grew from 4,266 judgments to 7,980 and the test moved to p = 0.122. Adding the interaction now makes the model slightly worse by AIC (16086.1 against 16084.2). The provisional number is recorded here rather than quietly dropped, because it was seen before the final one and the sequence is part of the audit trail.

Individual D1 terms still show movement, and they must not be read as significant when the six-degree-of-freedom test does not reject: `citations_expected` in RECOVERED versus EARLY is +0.32 (p = 0.025) and `length_matched` +0.34 in DEGRADED (p = 0.016) and +0.26 in RECOVERED (p = 0.055), with `abstention_permitted` null in both regimes.

**D3 is significant, is stronger than it was, and its entire signal is in one condition.** `abstention_permitted x verification effort` is +0.42 (SE 0.10, p < 0.0001), while `citations_expected` (+0.03, p = 0.71) and `length_matched` (-0.06, p = 0.48) are indistinguishable from zero. That structure is unchanged from the provisional fit; only its strength grew.

**D1 and D3 now disagree on significance, not merely on location.** The pre-specification anticipated this and named it: disagreement between the two would mean the result depends on where the regime lines fall, and both are then reported with the disagreement as the finding. The asymmetry between them is what carries the interpretation, and at the final data it points the opposite way from the provisional read. Regime is a time slice and is exogenous to the individual citation. Verification effort is measured per judgment and is partly a *consequence* of what was being judged, so a `condition x effort` interaction can reflect genuine differences in what each condition's citations require rather than instrument state. **The only test that rejects is the one whose predictor is endogenous, and it rejects in the one condition where models cite least and abstain most, which is exactly where the surviving citations are most selected.** That is a weak basis for asserting instrument differentiality, and it is not the basis on which the registered contrasts are defended.

**Sensitivity #1 keeps the interaction below detection** (p = 0.108), and the mechanism is visible in the coefficients: the `citations_expected` RECOVERED term falls from +0.32 (p = 0.025) to +0.10 (p = 0.49) once UNVERIFIABLE is recoded as fabricated. Recoding excludes nothing and closes the UNVERIFIABLE channel, so this is the same finding the regime table and the paired sub-study reach independently: the movement runs through the abstention category, not through a changed REAL-versus-FABRICATED boundary. **A pre-registered analysis exists that is substantially drift-robust.** State it as attenuation, not as an effect that vanishes.

**What the pre-specified decision rule requires.** D3 is significant, so its committed branch applies: the registered estimands are **not** adjusted and remain primary, an era-stratified re-estimate of the same contrasts is added as a robustness panel, and the report states explicitly that these tests detect differentiality without identifying the direction of bias, since the regimes contain different items. That panel re-estimates registered contrasts, so it is gated behind an explicit flag and runs alongside or after the confirmatory fit, never before.

**And what it does not license.** Three of the four tests failing to reject is not evidence of no drift effect. No alpha adjustment is claimed, power against small interactions is limited, and a non-significant interaction bounds the evidence rather than proving non-differentiality. Of the three falsifiers named in advance, the decisive one is the paired sub-study, which the pre-specification ranks above all three observational tests because it holds the item fixed. It found no condition-dependent flip pattern. That is the strongest available evidence and it is reassuring; it is not proof.

A limit of all three deserves stating plainly: because the regimes contain **different citations**, an observed interaction cannot by itself separate "the instrument is differential" from "these particular items differ." Only a design that holds the item fixed can do that.

## The paired re-judge sub-study

Every original prompt file was retained, so a sample of already-judged citations can be judged again under the current instrument reading the byte-identical prompt. Any difference in verdict is then attributable to the instrument alone, with prompt-construction variance eliminated by construction rather than assumed away.

500 citations were sampled, 250 from EARLY and 250 from DEGRADED, balanced across the four prompt conditions, with design weights retained so corpus-level rates can be recovered from the balanced sample. Balanced rather than proportional allocation was chosen because the comparison of interest is across conditions, where precision is limited by the smallest cell. This is the same design used for the Phase B EXISTENT_WRONG_ID sub-study.

Both verdicts are kept, giving two paired transition matrices: EARLY to current, and DEGRADED to current. The first is the more important of the two, since EARLY is 52% of the corpus. The flip probability is then regressed on condition, which turns differentiality from an assumption into a measurement.

**Results.** All 500 sampled citations were re-judged, none missing. Corpus-level flip rates, design-reweighted with a finite-population correction, are **16.9% [12.0, 21.9] for EARLY to current** and **16.3% [11.6, 21.1] for DEGRADED to current**. Neither differentiality test rejects: EARLY chi-square 5.08 on 3 df, p = 0.17; DEGRADED 1.38 on 3 df, p = 0.71.

The direction of the flips is the finding, and it agrees with the regime analysis above by a completely different route. Almost none of the movement crosses the REAL to FABRICATED boundary. Out of 250 EARLY pairs, that boundary accounts for **4 flips** (1 FABRICATED to REAL, 3 REAL to FABRICATED); the other 35 all begin at UNVERIFIABLE, and 29 of them land on FABRICATED. Out of 250 DEGRADED pairs the boundary accounts for **5 flips**, against 28 that begin at UNVERIFIABLE and split almost evenly between FABRICATED (14) and REAL (14).

This is the design that holds the item fixed, so it says what the regime comparison could not: the instrument's movement is a re-allocation of the abstention category, not a re-drawing of the real-versus-fabricated line. Two independent routes, one holding the item fixed and one scoring the whole corpus under both codings, reach the same conclusion.

**A null differentiality test bounds the evidence, it does not prove non-differentiality.** With 62 or 63 citations per condition these tests detect only large differences, and the EARLY p of 0.17 is not a clean bill of health. The claim supported is that no large condition-dependent flip pattern is visible, not that none exists.

## What is also reported

1. **Exclusion rates by regime.** Generation-level exclusion rates are **EARLY 1.01%, DEGRADED 4.54%, RECOVERED 0.38%**, an order of magnitude below the citation-level UNVERIFIABLE rates, because a generation is excluded only when *every* citation is UNVERIFIABLE. Neither registered monotonicity trigger fires (not monotone in stage, not monotone in condition, within `obscure_real`), so the escalation rule does not promote Sensitivity #1 to co-primary. It is reported regardless, and given its drift-robustness above, that reporting matters more than the rule firing would have.

   The pre-registration excludes a generation only when every one of its citations is UNVERIFIABLE, and requires that exclusion rate be reported by `cell x stage_ordinal` and `condition x is_post_trained` before the primary is interpreted, with a pre-committed rule promoting Sensitivity #1 to co-primary if the rate is monotone in stage or condition within `obscure_real`. Because the UNVERIFIABLE rate moved substantially across regimes and regime did not exist at registration, **regime is added as a third stratification**. This is a reporting addition, not an amendment.
2. **Sensitivity #1 regardless of the escalation rule.** Recoding UNVERIFIABLE as fabricated excludes nothing by construction, so it is immune to regime-varying exclusion. If it agrees with the primary, the drift's reach into the registered result is bounded, and that can be stated explicitly.
3. **Zero-effort robustness.** Judgments returned with no evidence gathering skew toward REAL (**64.3% against a 43.4% baseline**, over 1,328 such judgments, 4.15% of the corpus), which is the direction that would inflate the judge's rescues of verifier-FABRICATED calls. The affected volume is small. The registered models are re-estimated with those generations dropped, reported as an explicitly labelled robustness check and never as a substitute for the registered estimands.

## A separate limitation, stated separately

Stability and validity are different properties, and the drift analysis speaks only to stability: a judge could be perfectly stable and consistently wrong. The validity question is answered elsewhere and must be stated accurately here, because an earlier draft of this section got it wrong.

**The retracted claim.** An earlier version said J2a is "a single judge whose accuracy has not been validated against human annotation." That is false and must not reach the paper. J2a cleared the pre-registered §8 A.5 gate at **96.0%** three-way agreement with blind human labels against a registered threshold of 90%, where the structured verifier alone scores 46.0%. A second, fabrication-enriched re-validation on the Phase C corpus gives design-weighted accuracy 0.719 against a trivial always-REAL baseline of 0.546, **exact McNemar p = 5.6e-05**, with FABRICATED recall **0.781** [0.673, 0.860].

**The limitation that is real, and narrower.** The human labels in both validations come from **a single annotator, the author**. There is no inter-annotator reliability and no Cohen's kappa. The annotation was blinded to every pipeline label, but blinded is not independent. Concurrent work on a closely related construct used multiple blinded PhD annotators and a three-judge panel, and reported its own error rates as lower bounds; this work has no equivalent, and the paired sub-study does not supply one, since it measures stability rather than truth. Closing this gap needs a second annotator, which was considered and declined as out of budget, and it is disclosed rather than worked around.

**A third limitation, from the corrective re-judge.** Amendment #6(c) accepted verifier-REAL citations without judging them. That trust step failed on this corpus and was corrected by re-judging all 6,852 at-risk citations, of which 29.5% did not survive. The corrected labels still rest on the judge, whose FABRICATED precision the hand-check puts at 80.0% [68.2, 88.2], so roughly a fifth residual error on that stratum remains. The correction is a large improvement on blind trust, not an elimination of error. Details in §11 sub-amendment #12.

Each of these is stated in its own right rather than folded into the drift discussion.
