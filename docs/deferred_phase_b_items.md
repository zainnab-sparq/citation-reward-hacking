# The three deferred Phase B items, resolved

Item 4 of the remaining-to-publish checklist. All three were deferred out of Phase B into the Phase C §5 protocol. Each is settled below: one was already computed and only needed stating, one is moot on inspection, and one was run here and returned the opposite of what the deferral assumed.

## 4.1 The J2a-versus-verifier CIs overlap. State it.

**Status: already computed, needed stating.** `data/analysis/phase_b/descriptive_results.json` carries a `ci_overlap_caveat` field under `divergence_academic_medical`. The numbers behind it, on the `citations_expected − neutral` marginal contrast, academic + medical only:

| scoring | OR | 95% CI |
|---|---|---|
| J2a | 2.79 | [1.93, 4.04] |
| verifier-only | 1.78 | [1.26, 2.50] |

The intervals overlap on [1.93, 2.50]. **The J2a > verifier gap is directional only and is not a formal difference.** Both scorings independently place the effect in `obscure_real` (J2a interaction OR 2.49, verifier 4.56). The judge adds resolution; it does not supply the signal.

This matters because it is the retraction of a stronger claim. The all-domain `citations_expected` OR of 23.85 collapses to 2.79 once legal is removed, confirming that figure was legal-UNVERIFIABLE-inflated, and it must not be the headline. The honest sentence for the paper is that the two scorings agree on where the effect lives and differ only in magnitude, within overlapping intervals.

## 4.2 The judge-leniency MISGROUNDED spot-check on legal and rescued citations

**Status: MOOT as worded. The underlying question is answered from a better source.**

The check cannot be run because **MISGROUNDED is never emitted by this pipeline.** Verified three ways:

- Phase C judgments (28,266 rows) contain only `UNVERIFIABLE` 4,106 / `REAL` 12,252 / `FABRICATED` 11,908. Zero MISGROUNDED.
- Phase B judgments: zero occurrences.
- `src/judge_via_workflow.py` line 50: `_VALID_STATUSES = frozenset({"REAL", "FABRICATED", "UNVERIFIABLE"})`. The status is not in the schema enum, so the judge cannot return it.

MISGROUNDED survives only as an inert "bad status" string in the analysis code. (This is the same finding that makes the pre-reg's κ ≥ 0.7 gate non-binding: that gate is MISGROUNDED-specific and conditional on reporting a human-plus-judge MISGROUNDED table, which will never exist.)

**The real question underneath, "is the judge lenient when it rescues a citation", is answered directly by the fabrication-enriched Phase C hand-check** (150 hand-labelled citations, `data/analysis/phase_c/handcheck_result.json`):

| stratum | accuracy | 95% CI |
|---|---|---|
| `judge_REAL_rescue` | **43/45 = 95.6 %** | [85.2, 98.8] |
| `judge_FABRICATED` | 48/60 = 80.0 % | [68.2, 88.2] |

The rescue mechanism, which is the exact behaviour the deferred spot-check was aimed at, is the **best-validated stratum in the whole instrument**. The judge is not systematically lenient when overriding a verifier FAB/UNV to REAL. The error is on the other side: roughly 20 % of `judge_FABRICATED` calls are false alarms.

**One legal-specific caveat does survive, and it is carried in `docs/legal_domain_holdout_decision.md` §3 rather than here.** The judge rescues legal to REAL at 57.2 % against 33.8 % academic and 27.2 % medical, at lower search depth (1.71 against 2.02 and 2.29), with 48.8 % of legal judgments resolving at one search and 89.2 % of those returning REAL. The hand-check's 45-row rescue stratum is not powered to break that out by domain. So: the rescue mechanism validates overall, and the legal arm keeps a disclosed caveat.

## 4.3 rePCA(maximal) on the abstention slope

**Status: RUN. Result contradicts the assumption behind the deferral.**

Code: `src/analysis/phase_b_repca.R`. Output: `data/analysis/phase_b/repca_maximal.json`.

**Why it was deferred.** `.summarize_re_robustness` reports `isSingular()` and the condition-SE comparison, and the maximal model `fabricated ~ condition * cell + domain + (1 + condition | question_id)` reports non-singular and converged for all three scorings. But the `abstention_permitted` fixed-effect SE inflates about 6x from intercept-only to maximal RE (0.393 → 2.350 on the J2a academic+medical fit). That pattern is consistent with a variance component that is technically estimable but carries no information, which `isSingular()` would not flag. `rePCA` is the diagnostic that separates the two readings: it reports how many dimensions of the RE covariance matrix the data actually support.

**Result: all four dimensions are supported, in all three scorings.**

| scoring | rotated SDs | relative SDs | unsupported dims |
|---|---|---|---|
| J2a academic+medical | 7.696, 3.488, 2.055, 1.040 | 1, 0.453, 0.267, **0.135** | **0 of 4** |
| J2a all domains | 6.261, 3.546, 2.357, 0.958 | 1, 0.566, 0.377, **0.153** | **0 of 4** |
| verifier-only | 4.688, 3.001, 1.916, 1.359 | 1, 0.640, 0.409, **0.290** | **0 of 4** |

The smallest relative standard deviation is 0.135, three orders of magnitude above the 1e-3 negligibility threshold fixed in advance in the script. No dimension collapses.

**And the abstention slope is the opposite of degenerate. It has the LARGEST random-effects SD in every fit:**

| scoring | intercept | citations_expected | **abstention_permitted** | length_matched |
|---|---|---|---|---|
| J2a academic+medical | 2.763 | 2.113 | **7.643** | 2.486 |
| J2a all domains | 3.312 | 3.236 | **5.543** | 2.467 |
| verifier-only | 3.800 | 2.641 | **3.011** | 2.452 |

**Reading.** The wide abstention CI is not a numerical artifact of an unidentified variance component. It reflects genuine and large between-question heterogeneity in the abstention effect, which is what you would expect given that `abstention_permitted` drove fabrication to near zero overall and questions differ sharply in whether the model took the abstention option at all. The quasi-separation noted in the Phase B write-up is real, and the maximal RE structure is correctly specified rather than over-parameterised.

**Consequence for the paper:** the registered parsimony-primary-with-extended-RE-sensitivity rule (pre-reg §6) is safe as written, the maximal model is legitimately the primary where it fits cleanly, and the abstention term's wide interval should be reported as heterogeneity plus boundary behaviour, not as a fitting problem. The intercept-only SEs remain anti-conservative for the condition terms and should not be quoted.

## Reproduction

    Rscript src/analysis/phase_b_repca.R \
      --generations data/analysis/phase_b/generations.csv \
      --verifications data/analysis/phase_b/verifications.csv \
      --questions data/analysis/phase_b/questions.csv \
      --output data/analysis/phase_b/repca_maximal.json
