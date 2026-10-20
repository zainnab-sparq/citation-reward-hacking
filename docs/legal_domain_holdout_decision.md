# Legal-domain holdout decision for Phase C

**Status: ANALYSIS DECISION, recorded before the confirmatory fit.** The registered E1/E2 models have not been fitted and their outcome is unseen. This document exists so the decision cannot be shaped by the result it governs.

**Decision date:** 2026-08-11. **Decision: Phase C does NOT hold the legal domain out of the primary analysis.** The UNVERIFIABLE-as-fabricated sensitivity is elevated to co-primary, and the judge's legal-specific leniency risk is carried as a disclosed limitation.

## 1. The question

Phase B reported its primary on academic + medical only, holding legal out. The question for Phase C is whether to do the same, and it has to be settled and stated before E1/E2 run.

Phase B's holdout rested on two distinct legs, both recorded in the Phase B analysis notes:

1. **Non-ignorable missingness.** Verifier-only legal exclusions were condition-correlated at 23 to 85 percent, so whether a legal generation was observable at all depended on the experimental condition.
2. **An unvalidated judge on legal.** The J2a legal UNVERIFIABLE-to-FABRICATED over-call was never validated, because CourtListener is the authoritative legal source (pre-reg §4 legal-domain verification routing) and it is a **verifier-only** tool. The judge is a Claude subagent with WebSearch and WebFetch and cannot query it, so on legal the judge is structurally worse-equipped than on academic or medical.

## 2. The registered constraint that settles it

Phase B's academic+medical primary was a Phase B **analysis** choice. It was never registered. The Phase C pre-registration points the other way, in two places.

**§6 confirmatory families (pre-reg line 269).** The E1 family of m = 5 primary tests is enumerated explicitly, and member 5 is:

    stage_ordinal:cellobscure_real:domainlegal

Legal sits **inside the registered confirmatory family**. Dropping the legal domain from the Phase C primary would delete a pre-registered confirmatory test, which is a substantially larger deviation than the one it would be trying to avoid, and it is the specific move pre-registration exists to prevent: removing a registered test after seeing that its data is inconvenient.

**§6 sensitivity #1 (pre-reg line 275).** The pre-registration already anticipates informative UNVERIFIABLE exclusion and specifies its own remedy, domain-agnostically:

> Excluding cited-only-UNVERIFIABLE generations from the primary analysis conditions on "the model cited something verifiable," which is post-treatment if UNVERIFIABLE rates differ by stage/condition. **Pre-committed reporting:** before interpreting the primary result we report the UNVERIFIABLE exclusion rate stratified by `cell × stage_ordinal` (E1) and `condition × is_post_trained` (E2). **Escalation rule:** if that rate is monotone in stage/condition within the obscure-but-real cell (a sign the exclusion is informative, not ignorable), the UNVERIFIABLE-as-fabricated sensitivity is elevated from a footnote to a **co-primary** result.

The registered answer to "legal exclusions are condition-correlated" is therefore the sensitivity #1 escalation, not a domain holdout.

## 3. What the Phase C data shows

Computed by `src/analysis/unverifiable_selection_check.py` (21 tests, red-first). This is a missingness diagnostic: it reports who is excluded and never a fabrication rate by condition, so it reveals no registered estimand and is safe to run before the one-shot fit.

### Leg 1 is largely cured by J2a

Verifier-only exclusion (cited-only-UNVERIFIABLE / cited at least once), all 43,200 generations, verification complete:

| domain | range across the four conditions | spread |
|---|---|---|
| academic | 0.5 to 1.6 % | 1.1 pp |
| medical | 0.1 to 1.6 % | 1.5 pp |
| legal | **63.9 to 90.7 %** | **26.8 pp** |

Phase B's leg 1 replicates in Phase C at full force under verifier-only labels. Under the J2a-combined label that the primary actually uses, on the 23,076 generations whose fired citations are all judged:

| domain | range across the four conditions | spread |
|---|---|---|
| academic | 0.0 to 0.1 % | 0.1 pp |
| medical | 0.0 to 0.2 % | 0.2 pp |
| legal | **2.3 to 5.6 %** | **3.3 pp** |

Legal exclusion falls by roughly an order of magnitude, from 64-91 percent to 2-6 percent, and the condition-correlation falls from 26.8 pp to 3.3 pp. **Phase B's leg 1 was a property of the verifier-only pipeline. Phase C has a judge layer, and the judge resolves most of it.** The residual is real but no longer disqualifying on its own.

### The registered trigger does not fire, but only just

Post-J2a exclusion within `obscure_real` by `stage_ordinal`, which is the registered trigger stratification:

| grouping | stage 0 | stage 1 | stage 2 | stage 3 | monotone? |
|---|---|---|---|---|---|
| **pooled (registered)** | 2.53 % | 3.09 % | 2.85 % | 2.43 % | **NO** |
| academic | 0.00 % | 0.00 % | 0.26 % | 0.13 % | NO |
| legal | 8.89 % | 8.45 % | 7.07 % | 6.23 % | **DOWN** |
| medical | 1.05 % | 0.18 % | 0.14 % | 0.00 % | DOWN |

The registered trigger is written at `cell × stage_ordinal` with no domain term, and the pooled sequence is non-monotone, so **the literal trigger returns False** (`registered_trigger_fires: false` in the artifact).

That is a thin reed to stand on. Legal alone is monotone decreasing across all four stages at a level roughly 30 to 60 times the academic rate, and medical is technically monotone but over a 1.05-to-0.00 percent range where monotonicity is noise.

### Leg 2 is not cured, and there is new evidence it is a live risk

The judge's own behaviour differs sharply on legal (pooled over conditions, so this is an instrument-coverage diagnostic, not an estimand):

| domain | n judged | judge REAL | judge FAB | judge UNV | mean searches |
|---|---|---|---|---|---|
| academic | 7,044 | 33.8 % | 53.3 % | 12.9 % | 2.02 |
| **legal** | 13,661 | **57.2 %** | 30.9 % | 11.9 % | **1.71** |
| medical | 7,561 | 27.2 % | 52.0 % | 20.7 % | 2.29 |

The judge rescues legal citations to REAL at roughly double the rate of the other domains while performing fewer searches. Broken out by search depth, **48.8 percent of all legal judgments resolve at exactly one search, and 89.2 percent of those return REAL** (academic: 2,306 at one search, 76.1 percent REAL; medical: 1,780, 73.1 percent REAL).

A "few searches implies REAL" gradient appears in all three domains and is the rational signature of an easy confirmation, so low search count is not leniency by itself. What is legal-specific is the concentration, and the reason it matters is structural: a US case is identified by its **reporter cite** (for example `386 F. Supp. 3d 760`), and a general web search confirms a case *name* far more easily than it confirms the reporter, court, and year together. A one-search REAL on a legal citation is therefore consistent with confirming only the caption, which is exactly the EXISTENT_WRONG_ID failure mode (a real case name attached to the wrong reporter or court).

The judge does sometimes get this right. The first record in the judgments file is a legal citation where it spent its full three-search budget, found the case existed in a different district than cited, declined to confirm the reporter, and returned UNVERIFIABLE. The concern is the rate, not the capability.

## 4. Decision and consequences

**Legal stays in the Phase C primary.** Grounds, in order of weight:

1. `stage_ordinal:cellobscure_real:domainlegal` is registered E1 family member 5. Removing legal deletes a registered confirmatory test.
2. Phase B's leg 1 rationale is obsolete under J2a: legal exclusion drops from 63.9-90.7 percent to 2.3-5.6 percent, and its condition-correlation from 26.8 pp to 3.3 pp.
3. The pre-registration's own remedy for informative exclusion is the sensitivity #1 escalation, which is domain-agnostic and does not contemplate a holdout.

**The UNVERIFIABLE-as-fabricated sensitivity is elevated to co-primary anyway**, despite the literal trigger returning False. Reasons:

- Legal is monotone in stage within `obscure_real` at 8.9 to 6.2 percent, an order of magnitude above the other domains. The pooled test masks it only because academic and medical are near zero and dominate the pooled n.
- The escalation costs nothing. It is a re-coding of existing labels, not new data, and under it **no generation is excluded at all**, so the entire non-ignorable-missingness question is dissolved rather than argued.
- The asymmetry favours running it. Reporting a sensitivity that turns out to be unnecessary costs a paragraph; omitting one that turns out to be necessary is not recoverable after the one-shot fit.

This is a conservative deviation (running more than the trigger demands), it is recorded before the fit, and it does not alter any registered estimand, per-test, or family membership.

**Carried as a disclosed limitation, not resolved:** the legal-specific judge leniency risk in §3. It cannot be settled from the judgments file alone, because there is no legal ground truth the judge did not already see. Two existing artifacts bear on it partially: the 2026-08-12 paired re-judge sub-study measures the misclassification matrix directly on degraded-era rows, and Phase B's EXISTENT_WRONG_ID sub-study validated the same construct at 4.3 percent, though on Phase B and on verifier-REAL citations. Neither is a legal-specific validation. **The honest statement is that the legal arm's judge verdicts rest on a search channel that cannot reach the authoritative source for the domain, and legal results should be read with that caveat attached.**

## 5. Reproduction

    python -m src.analysis.unverifiable_selection_check \
      --verifications data/verifications/phase_c/verifications.jsonl \
      --judgments data/judgments/phase_c/claude-subagent-sonnet.jsonl \
      --registry data/models/phase_c_checkpoints.jsonl \
      --output data/analysis/phase_c/unverifiable_selection_check.json

Artifact as of this decision: `data/analysis/phase_c/unverifiable_selection_check_PROVISIONAL.json`, computed at 28,266 of 31,980 judgments (88.4 percent), with 2,454 generations skipped as not-yet-fully-judged and excluded from the tables above. Because judging proceeds in file order, scoring them partially would have biased the rate by judging order rather than by condition.

**Re-run at 100 percent on 2026-08-30: `data/analysis/phase_c/unverifiable_selection_check.json`, 25,530 generations scored and none skipped.** The decision above stands unchanged on complete data. The registered escalation trigger still does not fire, since the pooled obscure-real rate by stage is 2.25 / 2.85 / 2.58 / 2.21 percent and is not monotone. The per-domain directions are also unchanged: academic flat, legal monotone down 7.84 to 5.55 percent, medical monotone down 0.94 to 0.00. Sensitivity #1 remains elevated to co-primary as the conservative deviation described above, not because the trigger fired.
