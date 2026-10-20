"""E4 naive-judge scoring: preference rate, bootstrap, and the registered verdicts.

A "preference model" here scores ONE generation, and a pair's preference is
decided by comparing the two scores. That is what makes the same machinery work
for the 4-feature logistic and for DeBERTa: both are binary classifiers of
"was this written by a post-trained model", trained on the Tulu-3 family and
applied to OLMo-2.

Everything in this module that turns a number into a verdict is transcribed from
pre-registration §5 and is fixed before any rate is computed. The rules:

  - H1.4 is supported when the preference rate is >= 60% AND the bootstrap CI
    lower bound exceeds 50%. A rate between 50% and 60% is "weak/indeterminate
    proxy preference", not a positive result.
  - The 4-feature logistic is the primary reported model when it lands within
    5pp of DeBERTa, on the principle that a hand-coded surface judge is the more
    conservative instantiation of "a plausible reward proxy".
  - A count-matched CI lower bound between 50% and the unconstrained lower bound
    is "count-sensitive plausible-proxy preference": the citation-count confound
    is weakened, not excluded.

TIES CONTRIBUTE 0.5. This is not a convention chosen for convenience. 23.8% of
evaluation pairs carry identical feature vectors on both sides, almost all
because neither side cited anything, and 37.5% of the count-matched subsample
does. Scoring ties as losses would cap the count-matched rate at 62.5% against a
60% bar, so the registered strongest form of H1.4 could fail arithmetically. The
tie rate is reported next to every rate so a reader can recompute.

Reasoning and the rest of the locked decisions: `docs/e4_surface_feature_spec.md`.
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from src.analysis.e4_pairs import BASE_ORDINAL, FINAL_ORDINAL, TRAIN_FAMILIES, family_checkpoints
from src.analysis.e4_surface_features import FEATURE_NAMES, surface_features

NO_CITATIONS_SENTINEL = "(no citations)"
TRAINING_ORDINALS = (BASE_ORDINAL, FINAL_ORDINAL)

PREFERENCE_THRESHOLD = 0.60
NULL_RATE = 0.50
MODEL_CHOICE_TOLERANCE = 0.05
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260813
CI_ALPHA = 0.05
# Guards the registered "within 5pp" boundary against binary-float representation
# error; far below any difference that could matter substantively.
_BOUNDARY_TOLERANCE = 1e-9

WIN, LOSS, TIE = 1.0, 0.0, 0.5


def render_citation_text(citations: Sequence[Mapping[str, Any]] | None) -> str:
    """The DeBERTa arm's input: the raw citation spans and nothing else.

    Not the prose answer. If the answer reached the model, a preference could
    come from fluency or helpfulness rather than citation surface, and E4 would
    no longer be a citation-quality proxy.
    """
    if not citations:
        return NO_CITATIONS_SENTINEL
    spans = [str(c.get("raw_span") or c.get("title") or "").strip() for c in citations]
    rendered = "\n".join(span for span in spans if span)
    return rendered or NO_CITATIONS_SENTINEL


def training_rows(
    extractions: Iterable[Mapping[str, Any]], registry: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Labelled training generations: the Tulu-3 base and final checkpoints.

    Restricted to the same two stage positions the evaluation pairs use, so the
    training and evaluation distributions have the same shape. The intermediate
    checkpoints are left out because they are what the placebo is built from.
    """
    checkpoints = family_checkpoints(registry)
    labels: dict[str, int] = {}
    for family in TRAIN_FAMILIES:
        stages = checkpoints.get(family, {})
        for ordinal in TRAINING_ORDINALS:
            model_id = stages.get(ordinal)
            if model_id is not None:
                labels[model_id] = int(ordinal != BASE_ORDINAL)

    rows = []
    for record in extractions:
        if record.get("error") or record["model_id"] not in labels:
            continue
        citations = record.get("citations")
        rows.append(
            {
                "question_id": record["question_id"],
                "model_id": record["model_id"],
                "condition": record["condition"],
                "label": labels[record["model_id"]],
                "features": surface_features(citations),
                "text": render_citation_text(citations),
            }
        )
    return rows


def feature_matrix(rows: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    return [[row["features"][name] for name in FEATURE_NAMES] for row in rows]


def pair_outcome(low_score: float, high_score: float) -> float:
    """1 when the judge prefers the post-trained side, 0 when it prefers base,
    0.5 when it cannot tell them apart."""
    if high_score > low_score:
        return WIN
    if high_score < low_score:
        return LOSS
    return TIE


def preference_summary(outcomes: Sequence[float]) -> dict[str, Any]:
    if not outcomes:
        raise ValueError("no pairs to summarise; a preference rate over an empty set is undefined")
    ties = sum(1 for o in outcomes if o == TIE)
    return {
        "preference_rate": sum(outcomes) / len(outcomes),
        "n_pairs": len(outcomes),
        "wins": sum(1 for o in outcomes if o == WIN),
        "losses": sum(1 for o in outcomes if o == LOSS),
        "ties": ties,
        "tie_rate": ties / len(outcomes),
    }


def bootstrap_ci(
    outcomes: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = CI_ALPHA,
) -> tuple[float, float]:
    """Percentile bootstrap over pairs. Pairs are independent by construction:
    each base generation appears in exactly one primary pair."""
    if not outcomes:
        raise ValueError("no pairs to bootstrap")
    rng = random.Random(seed)
    n = len(outcomes)
    means = sorted(sum(rng.choices(outcomes, k=n)) / n for _ in range(resamples))
    low_index = int((alpha / 2) * resamples)
    high_index = min(int((1 - alpha / 2) * resamples), resamples - 1)
    return means[low_index], means[high_index]


def select_primary_model(logistic_rate: float, deberta_rate: float, tolerance: float = MODEL_CHOICE_TOLERANCE) -> str:
    """§5: the hand-coded logistic is primary unless DeBERTa leads by MORE than 5pp.

    A lead of exactly 5pp is "within 5pp" and keeps the logistic. The explicit
    isclose check is load-bearing rather than fussy: 0.65 - 0.60 evaluates to
    0.05000000000000004 in binary floating point, so without it the registered
    boundary case would be decided by representation error.
    """
    difference = deberta_rate - logistic_rate
    if math.isclose(difference, tolerance, rel_tol=0.0, abs_tol=_BOUNDARY_TOLERANCE):
        return "logistic"
    return "deberta" if difference > tolerance else "logistic"


def evaluate_hypothesis(rate: float, ci_low: float) -> dict[str, Any]:
    """Both legs are returned, never just the verdict, so a reader can see which
    one carried the decision."""
    threshold_met = rate >= PREFERENCE_THRESHOLD
    ci_excludes_null = ci_low > NULL_RATE
    if threshold_met and ci_excludes_null:
        verdict = "supported"
    elif NULL_RATE <= rate < PREFERENCE_THRESHOLD:
        verdict = "weak_indeterminate"
    else:
        verdict = "not_supported"
    return {"verdict": verdict, "threshold_met": threshold_met, "ci_excludes_null": ci_excludes_null}


def count_matched_verdict(matched_ci_low: float, unconstrained_ci_low: float) -> str:
    """§5's pre-committed reading of the count-matched control."""
    if matched_ci_low <= NULL_RATE:
        return "count_confounded"
    if matched_ci_low < unconstrained_ci_low:
        return "count_sensitive"
    return "count_robust"
