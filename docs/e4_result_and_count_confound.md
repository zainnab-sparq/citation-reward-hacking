# E4 result, and why the citation-count confound is the whole story (2026-08-13)

Registered artifact: `data/analysis/phase_c/e4_naive_judge.json`, produced once by `src/analysis/run_e4.py --run-confirmatory`. Pre-analysis spec: `docs/e4_surface_feature_spec.md`, locked before the fit. Post-hoc diagnostic: `data/analysis/phase_c/e4_count_diagnostic.json`.

## The registered result

Trained on 14,400 Tülu-3 generations (7,200 post-trained), evaluated on 3,600 held-out OLMo-2 base-vs-Instruct pairs.

| arm | preference rate | 95% bootstrap CI | ties |
|---|---|---|---|
| 4-feature logistic | 0.8578 | [0.8500, 0.8661] | 23.8% |
| DeBERTa-v3-base | 0.8724 | [0.8619, 0.8829] | 2.4% |

DeBERTa leads by 1.46pp, inside the pre-registered 5pp band, so **the hand-coded logistic is the primary reported model** exactly as §5's choice rule specifies.

**H1.4 is supported on the registered test.** The rate clears the 60% threshold and the CI lower bound of 0.8500 is far above 50%. Both legs are met.

Count-matched (±1), the registered control: logistic 0.7876 [0.7766, 0.7981]. Under §5's rule this is **"count-sensitive plausible-proxy preference"**, because the matched lower bound falls between 50% and the unconstrained lower bound. The confound is weakened, not excluded.

## Both pre-registered controls turned out to be weaker than their names suggest

This is the part that matters for how E4 is written up, and neither problem is visible from the registered numbers alone.

### The ±1 window is permissive on this data

Base checkpoints average **0.09** citations per generation; the final checkpoints average **1.31**. A ±1 window therefore still admits the 0-against-1 contrast, which is the dominant comparison in the corpus. The control removes the large count gaps and leaves the one that carries the effect.

### The placebo is not direction-free as specified

§5 assumes two post-trained outputs carry no expected direction. But the placebo pairs SFT against DPO, and those differ in **stage**, with citation surface still rising across the cascade:

| feature | SFT | DPO |
|---|---|---|
| citation count | 0.758 | 1.329 |
| format conformity | 0.454 | 0.636 |
| plausible-author density | 0.291 | 0.407 |
| plausible-venue density | 0.385 | 0.528 |

The observed placebo rate of 0.6471 [0.6347, 0.6589] is therefore ambiguous between two opposite conclusions: the judge has a spurious directional bias, or the judge is correctly reading a real increase. As specified, the control cannot separate them.

## The diagnostic that resolves both, and it is decisive

Restricting to pairs whose citation counts are **exactly equal** removes the count channel entirely, so any residual preference must come from the other three features.

| tier | primary (base vs Instruct) | placebo (SFT vs DPO) |
|---|---|---|
| all pairs | 0.8578 [0.8500, 0.8661], n=3600 | 0.6471 [0.6347, 0.6589], n=3600 |
| within ±1 (registered) | 0.7876 [0.7766, 0.7981], n=2283 | 0.5808 [0.5683, 0.5932], n=2914 |
| **exactly equal counts** | **0.5155 [0.5083, 0.5238], n=904** | **0.5035 [0.4923, 0.5146], n=1882** |

Two readings, and they point the same way.

**The placebo is clean once counts are equal.** It lands on 0.5035 with a CI straddling 0.50. So the judge carries no directional bias of its own; its entire placebo signal was the real SFT-to-DPO citation increase. The control passes, once it is asked a question it can answer.

**The primary preference nearly vanishes once counts are equal.** 0.5155, barely above chance, on 904 pairs of which 856 are ties. Only 48 pairs are decided at all: 38 for the post-trained side, 10 for base.

**So the naive judge's preference for post-trained output is almost entirely "cites more", not "cites in a more plausible-looking way."** Format conformity, author-name plausibility and venue plausibility add close to nothing once quantity is held fixed.

## What this does and does not do to the argument

It does not weaken E4's role in the thesis. E4 exists to show that a plausible reward proxy would reward the post-trained behaviour, and citation quantity is a perfectly ordinary thing for a reward model to reward. A proxy that keys on "did it cite sources" prefers post-trained output at 86%, and that is the mechanism the paper is pointing at.

It does change what may be claimed. The stronger reading, that a naive judge is fooled by *fabricated-looking scholarly apparatus*, is not supported: the plausibility features carry almost no independent weight. E4 supports "quantity is rewarded", not "surface plausibility is rewarded".

**Write it as the registered verdict plus the diagnostic, not as either alone.** H1.4 is supported; the count-matched result is count-sensitive per the registered rule; and the exact-match diagnostic shows the count channel accounts for nearly all of it.

## Two limitations to disclose

**Exact-count matching is post-hoc.** The registered control is ±1 and its result stands as the registered one. The exact-match tier is a diagnostic added after seeing that ±1 was permissive on this data, and is labelled as such in the artifact.

**The DeBERTa arm has no exact-match number.** `run_e4` did not persist the fine-tuned weights, and retraining would produce a second version of a one-shot registered number. Since the logistic is the primary model, the gap is in the secondary check only. Persist the weights before any future run.
