# Related work: "Solution Hacking" (arXiv 2608.02442)

Source material for the paper's related-work and discussion sections. Not a pre-registration amendment: this paper is a different domain and a different failure mode, and it touches nothing in the registered m = 5 confirmatory family.

## The citation

Ren, Zhai, Pu, Zhu, Wei, Zhao. "Right Answer, Wrong Method: Shortcut Hacking Misleads the Evaluation of LLM Reasoning on Frontier Science Benchmarks." arXiv:2608.02442v1 [cs.AI, cs.CL], submitted 3 August 2026. Alibaba Group and Alibaba DAMO Academy.

**Cite as a preprint.** Its own comments field says "working in progress"; it is not peer-reviewed. It is concurrent work, not prior work, and should be described that way.

## 1. Construct boundary: a clean three-way separation, and no scoop

Their construct, *solution hacking*, is an LLM reaching the **correct answer** through an **invalid shortcut** (numerical search, enumeration, pattern or answer guessing, answer-first verification) that bypasses the reasoning capability the benchmark meant to test. Their Definition 1 rests on three operational criteria: targeted crux, capability bypass, derivational support.

Their own §2 draws the line we need: their judge asks *how* a correct answer was obtained, not *whether* it is correct. Ours asks whether the cited thing **exists at all**. That yields a three-way separation worth stating explicitly in the paper:

| | question asked | failure it detects |
|---|---|---|
| standard benchmark scoring | is the answer right? | wrong answers |
| Solution Hacking | was the right answer reached legitimately? | correct outcome, invalid process |
| this work | does the cited artifact have a referent? | artifact that looks right and refers to nothing |

There is no overlap claim to defend and no priority dispute.

## 2. Their role-based boundary is precedent for our §6 dividing-line test

Their Table 1 boundary is **role-based, not technique-based**: the same enumeration is legitimate or hacking depending on the role it plays in the derivation.

That is structurally the same move as our §6 dividing-line test, where an identical metadata discrepancy is a REAL typo or a FABRICATED `EXISTENT_WRONG_ID` depending on whether the discrepancy identifies a *different real work*. Worth citing as precedent that a construct boundary of this shape is coherent and publishable, since it is the part of our taxonomy most exposed to the objection that it is arbitrary.

## 3. Dose-response convergence: the strongest empirical parallel (Discussion)

Their hack ratio rises monotonically with difficulty, and monotonically within every subject: **2.2 % easy/textbook, 28.3 % competition/olympiad, 37.4 % frontier/HLE**.

That mirrors our Phase B shape across cell (`length_matched` odds ratios 2.84 known_real, 7.90 obscure_real, 10.90 unanswerable). Different domain, different task, same finding: **hacking scales with how hard legitimate success is.**

They also report that weaker models hack more (GPT-4.1 hack ratio 60.4 % at accuracy 10.1, against GPT-5.5 at 22.0 % and 48.1). That is directly relevant to how we frame the E1 checkpoint cascade, and it is a useful external anchor for the claim that the gradient is about difficulty rather than about a particular model generation.

## 4. Abstention parallel, and a divergence not to over-read

All four of their anti-hack prompt variants let the model abstain via "CANNOT SOLVE RIGOROUSLY", justified exactly as our `abstention_permitted` condition is: without an abstention option, the prompt may push the model to **hide** shortcuts rather than admit it cannot solve.

Their abstention went from 0 % to 27.3 % (reported accuracy 50.6 to 37.3; correct-and-non-hacked 41.2 to 35.2). Ours was roughly 0.4 % and quasi-separated.

**Do not over-read the divergence.** The populations differ enormously: frontier 2026 models under an elaborate multi-variant intervention, against our 8B to 70B open-weights checkpoints under a single condition line. The honest use of this is to note that the abstention lever is real and that our near-zero rate is a property of the model tier we studied, not evidence the lever does not work.

## 5. The validation comparison, stated correctly

**An earlier reading of this paper concluded that our J2a is "a single unvalidated judge." That is wrong and must not go in the paper.** The audit that corrected it is recorded in the judge-validation status memory. The accurate comparison is narrower and still worth making.

**What they did.** Validated against blinded PhD expert annotation: 300 solutions (101 math, 100 physics, 99 chemistry), judge-stratified pool, 8 + 7 + 6 PhD annotators, released written guidelines, annotator self-review pass, split 180 dev and 120 held-out test used exactly once. The deployed detector is a **majority vote of three peer-level judges** (Gemini-3.1-Pro-Preview, Claude Opus 4.7, GPT-5.2) under a no-self-audit rule, blinded from both the reference answer and the correctness verdict. Agreement 75.4 % pooled, against 71.6 % for a single auditor.

**What we have.** Two independent validations, both passing:

1. **Step 0 blind hand-check (June 2026), the pre-registered §8 A.5 gate.** Production forced-Sonnet J2a agreed with blind human labels on 48/50 = **96.0 %** [86.5, 98.9], clearing the registered ≥ 90 % three-way gate. The structured verifier alone scores 23/50 = 46.0 %, so the judge layer takes the pipeline from **46 % to 96 %**.
2. **Fabrication-enriched re-validation on the Phase C corpus (August 2026), 150 hand-labelled citations.** Weighted pipeline accuracy 0.719 against a trivial always-REAL baseline of 0.546, **McNemar p = 5.6e-05**. **FABRICATED recall 0.781** [0.673, 0.860] on 73 human-labelled fabrications. The judge's core mechanism, overriding a verifier FAB/UNV to REAL, validates at 43/45 = 95.6 %.

**The real gap, and it is the one to concede.** Our annotation is **single-annotator (the author)**, so there is no inter-annotator reliability and no Cohen's κ. Theirs used multiple blinded PhD annotators. Ours is blinded-to-labels but not independent-of-author. That is a genuine limitation of a different kind than "unvalidated", and it is the honest thing to write. Closing it requires recruiting a second annotator, which is a paper-stage decision.

Their asymmetric error reporting is also worth borrowing as a presentation standard: they report falsely flagging 17 % of expert-clean cases and **missing 39 % of expert-confirmed hacks (recall 61.2 %)**, and therefore frame every reported hack ratio as a conservative **lower bound**. We should report our error profile with the same candour rather than a single accuracy number.

## 6. Borrow their framing, sign-flipped

Their detector under-flags, so their rates are **floors**.

Our degraded-era judge fails in the opposite direction: it fails to *confirm* real works, pushing REAL toward FAB/UNV, so **our degraded-era fabrication rates are ceilings, i.e. upper bounds.** Same rhetorical structure, opposite sign. This is a considerably cleaner way to present the instrument drift than "verdicts shifted", because it tells the reader which direction to discount in.

## 7. What they do not have, which is ours to claim

They run **no temporal-stability or instrument-drift analysis at all**, and their judge is a static API call with no web search, so our tool-availability failure mode cannot arise for them. Our drift documentation (`docs/drift_interaction_prespec.md`, `docs/judge_drift_disclosure_draft.md`, the paired re-judge sub-study) is therefore a contribution rather than a liability, and should be presented as one.

Keep the two axes distinct, because conflating them is easy and wrong: **the drift work measures instrument STABILITY** (does the judge agree with itself across conditions and time), **the validation work measures instrument VALIDITY** (does the judge agree with ground truth). They are orthogonal. A perfectly stable judge can be consistently wrong, and drift analysis is not a substitute for validation.

## 8. Minor note worth one sentence

They built their audit rubric with Claude Code (Opus 4.7), "iteratively refined by a computer scientist against manually reviewed examples". That is useful precedent that LLM-assisted rubric construction is publishable, which is relevant to how we describe our own prompt and taxonomy development.

Their corpus is 3,528 frontier solutions across 11 models; easy tier from SciBench and MATH-500, medium from IMO-Bench, PHYBench and SciOlympiad, hard from HLE.
