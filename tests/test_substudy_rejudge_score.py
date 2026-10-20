"""Contract tests for src/analysis/substudy_rejudge_score.py.

Scores the paired re-judge sub-study. The design holds the ITEM fixed, so a
verdict difference is attributable to the instrument rather than to which
citations happened to land in which era. That buys two things the observational
era comparison cannot give:

  * a transition MATRIX (original verdict x re-judged verdict), which shows the
    DIRECTION of the movement, not just the net rate;
  * a differentiality test: does the flip probability depend on the prompt
    condition? If it does not, the misclassification is non-differential and the
    registered contrasts are attenuated (conservative) rather than biased.

One subtlety pinned here: the ACROSS-CONDITION comparison uses UNWEIGHTED
per-stratum rates, because it compares strata to each other. Design weights are
for recovering a CORPUS-level rate, which is a different estimand.
"""
from __future__ import annotations

import pytest

from src.analysis.substudy_rejudge_score import (
    differentiality_test,
    flip_rate_by_condition,
    reweighted_flip_rate,
    score_rejudge,
    transition_matrix,
)


def _pair(era="EARLY", condition="neutral", original="FABRICATED",
          rejudged="FABRICATED", weight=10.0, population=100):
    return {
        "era": era,
        "condition": condition,
        "original_status": original,
        "rejudged_status": rejudged,
        "design_weight": weight,
        "stratum_population": population,
    }


def _pairs(n, **kw):
    return [_pair(**kw) for _ in range(n)]


# ---------- transition matrix ----------


def test_transition_matrix_counts_original_by_rejudged() -> None:
    pairs = (
        _pairs(3, original="FABRICATED", rejudged="FABRICATED")
        + _pairs(2, original="FABRICATED", rejudged="REAL")
        + _pairs(1, original="UNVERIFIABLE", rejudged="FABRICATED")
    )
    m = transition_matrix(pairs)
    assert m[("FABRICATED", "FABRICATED")] == 3
    assert m[("FABRICATED", "REAL")] == 2
    assert m[("UNVERIFIABLE", "FABRICATED")] == 1


def test_transition_matrix_can_be_restricted_to_one_era() -> None:
    pairs = _pairs(2, era="EARLY") + _pairs(5, era="DEGRADED")
    assert sum(transition_matrix(pairs, era="EARLY").values()) == 2
    assert sum(transition_matrix(pairs, era="DEGRADED").values()) == 5


# ---------- flip rates ----------


def test_flip_rate_counts_any_change_of_verdict() -> None:
    pairs = (
        _pairs(6, original="FABRICATED", rejudged="FABRICATED")
        + _pairs(4, original="FABRICATED", rejudged="REAL")
    )
    row = flip_rate_by_condition(pairs)["neutral"]
    assert row["n"] == 10
    assert row["n_flipped"] == 4
    assert row["flip_rate"] == pytest.approx(0.4)


def test_flip_rate_reports_a_confidence_interval() -> None:
    pairs = _pairs(50, rejudged="REAL")
    row = flip_rate_by_condition(pairs)["neutral"]
    lo, hi = row["ci"]
    assert 0.0 <= lo <= row["flip_rate"] <= hi <= 1.0


def test_flip_rate_splits_by_condition() -> None:
    pairs = (
        _pairs(10, condition="neutral", rejudged="FABRICATED")
        + _pairs(10, condition="citations_expected", rejudged="REAL")
    )
    rates = flip_rate_by_condition(pairs)
    assert rates["neutral"]["flip_rate"] == 0.0
    assert rates["citations_expected"]["flip_rate"] == 1.0


# ---------- differentiality ----------


def test_differentiality_test_detects_condition_dependent_flipping() -> None:
    """The positive control: flipping concentrated in one condition."""
    pairs = (
        _pairs(40, condition="neutral", rejudged="FABRICATED")
        + _pairs(40, condition="citations_expected", rejudged="REAL")
    )
    result = differentiality_test(pairs)
    assert result["p_value"] < 0.05
    assert result["differential"] is True


def test_differentiality_test_is_null_when_flipping_is_uniform() -> None:
    """The negative control: identical flip rates across conditions."""
    pairs = []
    for condition in ("neutral", "citations_expected", "length_matched"):
        pairs += _pairs(20, condition=condition, rejudged="FABRICATED")
        pairs += _pairs(20, condition=condition, rejudged="REAL")
    result = differentiality_test(pairs)
    assert result["p_value"] > 0.05
    assert result["differential"] is False


def test_differentiality_test_reports_degrees_of_freedom() -> None:
    pairs = []
    for condition in ("neutral", "citations_expected", "length_matched"):
        pairs += _pairs(10, condition=condition, rejudged="REAL")
        pairs += _pairs(10, condition=condition, rejudged="FABRICATED")
    assert differentiality_test(pairs)["df"] == 2


def test_differentiality_test_needs_two_conditions() -> None:
    result = differentiality_test(_pairs(10))
    assert result["p_value"] is None
    assert result["differential"] is None


# ---------- reweighting ----------


def test_reweighted_flip_rate_weights_by_stratum_population() -> None:
    """Balanced sampling over unbalanced strata: the corpus rate is the
    population-weighted average, not the pooled one."""
    pairs = (
        _pairs(10, condition="neutral", rejudged="REAL", weight=90.0, population=900)
        + _pairs(10, condition="citations_expected", rejudged="FABRICATED",
                 weight=10.0, population=100)
    )
    # neutral flips 100%, citations_expected 0%; neutral is 9x the population.
    assert reweighted_flip_rate(pairs)["rate"] == pytest.approx(0.9)


def test_reweighted_rate_differs_from_the_pooled_rate_when_strata_differ() -> None:
    pairs = (
        _pairs(10, condition="neutral", rejudged="REAL", weight=90.0, population=900)
        + _pairs(10, condition="citations_expected", rejudged="FABRICATED",
                 weight=10.0, population=100)
    )
    pooled = flip_rate_by_condition(pairs)
    n_flipped = sum(r["n_flipped"] for r in pooled.values())
    n = sum(r["n"] for r in pooled.values())
    assert reweighted_flip_rate(pairs)["rate"] != pytest.approx(n_flipped / n)


def test_reweighted_rate_is_unbiased_by_differential_missingness() -> None:
    """The stratum population comes from the frozen manifest, not from
    design_weight x n_observed, so strata that lose items to API failures keep
    their correct weight instead of silently shrinking."""
    full = (
        _pairs(10, condition="neutral", rejudged="REAL", weight=90.0, population=900)
        + _pairs(10, condition="citations_expected", rejudged="FABRICATED",
                 weight=10.0, population=100)
    )
    # citations_expected loses half its items to failures; its weight must not move.
    lossy = full[:10] + full[10:15]
    assert reweighted_flip_rate(lossy)["rate"] == pytest.approx(
        reweighted_flip_rate(full)["rate"]
    )


def test_reweighted_rate_reports_a_design_based_interval() -> None:
    pairs = (
        _pairs(30, condition="neutral", rejudged="REAL", weight=20.0, population=600)
        + _pairs(30, condition="citations_expected", rejudged="FABRICATED",
                 weight=20.0, population=600)
    )
    result = reweighted_flip_rate(pairs)
    lo, hi = result["ci"]
    assert lo <= result["rate"] <= hi


# ---------- end to end ----------


def test_score_rejudge_pairs_manifest_with_rejudgments() -> None:
    manifest = [
        {"custom_id": "q1__m__neutral__t0__c0", "era": "EARLY", "condition": "neutral",
         "original_status": "FABRICATED", "design_weight": 10.0, "stratum_population": 100},
        {"custom_id": "q2__m__neutral__t0__c0", "era": "EARLY", "condition": "neutral",
         "original_status": "FABRICATED", "design_weight": 10.0, "stratum_population": 100},
    ]
    rejudgments = [
        {"custom_id": "q1__m__neutral__t0__c0", "judge_status": "REAL", "error": None},
    ]
    result = score_rejudge(manifest, rejudgments)
    assert result["n_sampled"] == 2
    assert result["n_rejudged"] == 1
    assert result["n_missing"] == 1
    assert result["by_era"]["EARLY"]["flip_rate_by_condition"]["neutral"]["n"] == 1


def test_score_rejudge_reports_each_era_separately() -> None:
    """EARLY -> current and DEGRADED -> current are different questions."""
    manifest = [
        {"custom_id": f"q{i}__m__neutral__t0__c0", "era": era, "condition": "neutral",
         "original_status": "FABRICATED", "design_weight": 10.0, "stratum_population": 100}
        for i, era in enumerate(["EARLY", "DEGRADED"])
    ]
    rejudgments = [
        {"custom_id": "q0__m__neutral__t0__c0", "judge_status": "REAL", "error": None},
        {"custom_id": "q1__m__neutral__t0__c0", "judge_status": "FABRICATED", "error": None},
    ]
    result = score_rejudge(manifest, rejudgments)
    assert result["by_era"]["EARLY"]["flip_rate_by_condition"]["neutral"]["flip_rate"] == 1.0
    assert result["by_era"]["DEGRADED"]["flip_rate_by_condition"]["neutral"]["flip_rate"] == 0.0


def test_score_rejudge_ignores_errored_rejudgments(tmp_path) -> None:
    manifest = [
        {"custom_id": "q1__m__neutral__t0__c0", "era": "EARLY", "condition": "neutral",
         "original_status": "FABRICATED", "design_weight": 10.0, "stratum_population": 100},
    ]
    rejudgments = [
        {"custom_id": "q1__m__neutral__t0__c0", "judge_status": "REAL", "error": "429"},
    ]
    result = score_rejudge(manifest, rejudgments)
    assert result["n_rejudged"] == 0
