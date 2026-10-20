# E4 pre-analysis specification: features, pairing, and scoring (locked 2026-08-13, before any model is fit)

Pre-registration §5 (E4) names the naive judge's four inputs but does not define them operationally: "citation count, format conformity, plausible-author-name density, plausible-venue-name density -- NOT existence." It also leaves the pairing and the tie rule open. This document fixes all three. It is written and committed **before any preference rate has been computed**, for the same reason `docs/drift_interaction_prespec.md` was written before the drift models were fitted.

Implementation: `src/analysis/e4_surface_features.py` (47 tests) and `src/analysis/e4_pairs.py` (24 tests).

## Why this document exists

"Plausible" is the loose word in the pre-registration, and it is load-bearing. H1.4 is a threshold claim (preference rate ≥ 60%, bootstrap CI lower bound > 50%). If the plausibility heuristics can be adjusted after a preference rate is visible, E4 becomes a search over feature definitions with a pre-registered-looking bar attached. Freezing the definitions in advance is the only thing that makes the 60% bar mean what it is supposed to mean.

**Any change to these definitions after a preference rate has been computed is a deviation and goes in the §11 amendment log, with the pre-change and post-change rates both reported.**

## Unit of analysis

One feature vector per generation, computed from that generation's extraction record (`data/extractions/phase_c/extractions.jsonl`). The registered comparison pairs a base generation against a post-trained generation on the same question and condition, so the features are per-generation, not per-citation.

## Blindness to existence

The features read the extracted citation fields only: `title`, `year`, `venue`, `reporter`, `authors`, `case_name`. They never read a verification status, a judge verdict, or any external source. A test asserts that a real citation and a fabricated citation of identical surface form receive identical vectors, and a second test asserts that verification fields merged into a record by mistake are ignored rather than silently used.

This is the assumption H1.4 rests on. A judge that could see existence would not be a naive proxy.

## The four features

**1. `citation_count`.** Number of extracted citations. Kept on the raw scale, not normalised, because the pre-registered count-matched control subsets on raw counts within ±1.

**2. `format_conformity`.** Mean over citations of the fraction of four format slots filled:

| slot | fields, first non-empty wins |
|---|---|
| title | `title` |
| year | `year` |
| venue | `venue`, `reporter` |
| attribution | `authors`, `case_name` |

The alternates exist because legal citations carry `reporter` and `case_name` where academic and medical citations carry `venue` and `authors`. Without them every legal generation would score as malformed for being legal, in both arms. Empty string, empty list, and missing are all treated as unfilled. A generation with no citations scores 0.

**3. `plausible_author_density`.** Fraction of citations carrying at least one name-shaped string in `authors`. Any one element qualifying carries the citation, because the extractor sometimes splits a single person across list entries (`["Smith", "J"]`).

**4. `plausible_venue_density`.** Fraction of citations whose `venue` or `reporter` is venue-shaped.

Legal citations rarely carry `authors` (20.5% of legal citations, against 92.6% academic), so legal generations score low on feature 3 in both arms. This is harmless: pairing is within-question, so the domain is identical on both sides of every comparison and a domain-constant offset cancels.

## The shape test

Features 3 and 4 use **one** rule, deliberately, so that there is little to tune:

> A string qualifies when every word is either capitalised, an allowed lowercase particle, or a numeric fragment, and at least one word is capitalised.

The two features differ in exactly one parameter and two word lists:

| | author | venue |
|---|---|---|
| digits | disqualify the string | skipped |
| lowercase words allowed | name particles (`van`, `der`, `de`, `al`, ...) | function words (`of`, `the`, `and`, `for`, ...) |
| length cap | 60 characters | 120 characters |

Digits disqualify a name because a person's name does not contain a year, which is what separates `Cronbach, L. J.` from `Smith 2019`. Digits are skipped in a venue because reporter cites routinely contain them (`F. Supp. 3d`). The length caps reject prose that the extractor dropped into a field ("the authors of the study") without rejecting long real venue names.

An earlier draft also carried a keyword list (`Journal`, `Proceedings`, `Transactions`, ...). It was removed before any fitting: every venue it would have accepted is already accepted by the capitalisation rule, so it added tunable surface without adding coverage. Recording the removal here because a reviewer comparing this file to the implementation should not have to wonder whether a list was dropped after seeing results.

## Instrument check (2026-08-13, pooled)

Run over all 43,200 extraction records, deliberately **pooled** rather than broken out by stage or family, because a by-stage breakdown is E4's registered comparison in descriptive form and the confirmatory model is unfit.

| feature | mean | sd | median | max | zero |
|---|---|---|---|---|---|
| citation_count | 0.945 | 1.048 | 1.00 | 17 | 40.9% |
| format_conformity | 0.510 | 0.451 | 0.67 | 1.00 | 40.9% |
| plausible_author_density | 0.325 | 0.455 | 0.00 | 1.00 | 65.3% |
| plausible_venue_density | 0.428 | 0.475 | 0.00 | 1.00 | 53.2% |

0 extractor errors, no density outside [0, 1], no constant feature. The 40.9% zero rate on both count and conformity is the same 17,670 generations that cited nothing, which independently agrees with the extraction stage's 25,530 generations carrying at least one citation.

## Pairing (decided 2026-08-13)

A pair holds question, prompt condition, and model family fixed, so the only thing varying across it is post-training. Domain and question difficulty are identical on both sides by construction and need no modelling.

**Primary pairs are base against the final checkpoint only** (stage_ordinal 0 against 3): base vs RLVR for Tülu-3, base vs Instruct for OLMo-2. §5 names "cross-family base/Instruct deltas", which points at those two checkpoints specifically, and it keeps pairs independent. The alternative, pairing base against each of SFT/DPO/RLVR, reuses one base generation in three pairs, which makes a plain bootstrap CI too narrow; E4 registers no clustered bootstrap. A per-stage version may be reported as a labelled exploratory addendum, never as the primary.

**Placebo pairs are SFT against DPO**, both post-trained, oriented by stage ordinal. Using the two intermediate stages keeps the placebo from sharing generations with the primary, so a quirk in the final checkpoint cannot move both at once. Orientation by stage is arbitrary with respect to quality, which is what makes 50% the null.

**Split:** train on `tulu3_8b` + `tulu3_70b`, evaluate on `olmo2_13b`. Cross-family transfer is the publishable claim, so the splits are disjoint and a test asserts it.

**A pair is dropped, not imputed, when either side is missing or its extraction errored.** An extraction failure is not evidence that the model cited nothing, and scoring it as zero would credit the other side with a difference that was never observed.

Yield on the frozen Phase C extractions, structural counts only:

| | primary | placebo |
|---|---|---|
| pairs | 10,800 | 10,800 |
| evaluation split (OLMo-2) | 3,600 | 3,600 |
| per family / per condition | 3,600 / 2,700 | 3,600 / 2,700 |
| count-matched (±1), eval | 2,283 (63.4%) | 2,914 (80.9%) |

10,800 is exactly 900 questions × 4 conditions × 3 families, so no generation is missing anywhere.

## Ties (decided 2026-08-13, and it is load-bearing)

**A tie contributes 0.5 to the preference rate.** Not a win, not a loss.

This is not a detail. On the evaluation split, **856 of 3,600 primary pairs (23.8%) have identical feature vectors on both sides**, and 808 of those are both sides citing nothing at all. On the count-matched subsample it is **856 of 2,283 (37.5%)**, because count-matching removes non-tied pairs and keeps every tie. A judge of any capacity must be indifferent on an identical input, so the tie rate is a property of the data, not of the model.

Scoring ties as losses would cap the count-matched preference rate at **62.5% before any modelling**, against a registered bar of 60% with a CI lower bound above 50%. The pre-registered strongest form of H1.4 could then fail for arithmetic reasons that have nothing to do with the hypothesis. Scoring ties at 0.5 puts the count-matched ceiling at 81.3% and keeps the bar meaningful.

**The tie rate is reported alongside every preference rate**, so a reader can recompute under either convention. The base-model arm is the reason it is so high: base models frequently cite nothing, which is itself a finding rather than a nuisance.

## What the DeBERTa arm reads (decided 2026-08-13)

§5 says the DeBERTa-v3-base model is trained "using only surface features", which cannot be read literally: a transformer over four scalars is strictly worse than a logistic regression on the same four scalars, which would make the pre-registered 5pp ablation vacuous.

**DeBERTa reads the concatenated raw citation spans and nothing else.** Not the prose answer, not any verification field. It stays blind to existence and to the answer's semantics, so a preference it expresses is still a preference over citation surface, while it is genuinely the higher-capacity arm the ablation is supposed to test. This is consistent with §5 describing the hand-coded logistic as the model that "cannot be accused of learning a hidden semantic signal", which implies the other arm could.

The rejected alternative worth recording: giving DeBERTa the full generation text would let it key on fluency or helpfulness, so a preference could come from something other than citation quality, and E4 would no longer be a citation-quality proxy.

## Still open

Nothing in the feature, pairing, or scoring definitions. The remaining E4 work is the two model fits and the bootstrap, all of which consume the artifacts specified above.
