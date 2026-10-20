"""Tests for E4's preference scoring, bootstrap, and pre-registered decision rules.

The decision rules are the point of this file. H1.4 is a threshold claim, and
every rule that turns a number into a verdict is written from the
pre-registration text rather than from anything observed, so these tests are
what stop the verdict from drifting once a rate is visible.
"""
from __future__ import annotations

import pytest

from src.analysis.e4_naive_judge import (
    BOOTSTRAP_RESAMPLES,
    MODEL_CHOICE_TOLERANCE,
    NO_CITATIONS_SENTINEL,
    NULL_RATE,
    PREFERENCE_THRESHOLD,
    TRAINING_ORDINALS,
    bootstrap_ci,
    count_matched_verdict,
    evaluate_hypothesis,
    pair_outcome,
    preference_summary,
    render_citation_text,
    select_primary_model,
    training_rows,
)
from src.analysis.e4_pairs import build_pairs

REGISTRY = [
    {"model_id": "tulu-base", "model_family": "tulu3_8b", "stage_ordinal": 0, "is_post_trained": 0},
    {"model_id": "tulu-sft", "model_family": "tulu3_8b", "stage_ordinal": 1, "is_post_trained": 1},
    {"model_id": "tulu-dpo", "model_family": "tulu3_8b", "stage_ordinal": 2, "is_post_trained": 1},
    {"model_id": "tulu-rlvr", "model_family": "tulu3_8b", "stage_ordinal": 3, "is_post_trained": 1},
    {"model_id": "olmo-base", "model_family": "olmo2_13b", "stage_ordinal": 0, "is_post_trained": 0},
    {"model_id": "olmo-instruct", "model_family": "olmo2_13b", "stage_ordinal": 3, "is_post_trained": 1},
]


def extraction(question_id, model_id, condition, spans):
    return {
        "question_id": question_id,
        "model_id": model_id,
        "condition": condition,
        "trial_id": 0,
        "citations": [{"raw_span": s, "title": s} for s in spans],
        "error": None,
    }


class TestCitationTextRendering:
    def test_renders_the_raw_spans_only(self):
        citations = [{"raw_span": "Smith (2001). A paper.", "title": "A paper"}]
        assert render_citation_text(citations) == "Smith (2001). A paper."

    def test_joins_multiple_citations_on_separate_lines(self):
        citations = [{"raw_span": "One"}, {"raw_span": "Two"}]
        assert render_citation_text(citations) == "One\nTwo"

    def test_does_not_include_the_prose_answer(self):
        # DeBERTa reads citation spans only. If the answer text ever reached it,
        # a preference could come from fluency rather than citation surface and
        # E4 would stop being a citation-quality proxy.
        citations = [{"raw_span": "One", "output_text": "a long helpful answer"}]
        assert "helpful" not in render_citation_text(citations)

    def test_no_citations_renders_as_an_explicit_sentinel(self):
        assert render_citation_text([]) == NO_CITATIONS_SENTINEL
        assert render_citation_text(None) == NO_CITATIONS_SENTINEL

    def test_falls_back_to_the_title_when_a_span_is_missing(self):
        assert render_citation_text([{"title": "Only a title"}]) == "Only a title"


class TestTrainingRows:
    def test_uses_only_the_training_families(self):
        records = [
            extraction("q1", "tulu-base", "neutral", []),
            extraction("q1", "tulu-rlvr", "neutral", ["A"]),
            extraction("q1", "olmo-base", "neutral", []),
            extraction("q1", "olmo-instruct", "neutral", ["A"]),
        ]
        rows = training_rows(records, REGISTRY)
        assert {r["model_id"] for r in rows} == {"tulu-base", "tulu-rlvr"}

    def test_uses_only_the_paired_checkpoints(self):
        # Training on the same checkpoints the evaluation pairs use keeps the
        # training and evaluation distributions the same shape.
        records = [extraction("q1", m, "neutral", ["A"]) for m in ("tulu-base", "tulu-sft", "tulu-dpo", "tulu-rlvr")]
        rows = training_rows(records, REGISTRY)
        assert {r["model_id"] for r in rows} == {"tulu-base", "tulu-rlvr"}
        assert TRAINING_ORDINALS == (0, 3)

    def test_labels_post_trained_as_one_and_base_as_zero(self):
        records = [
            extraction("q1", "tulu-base", "neutral", []),
            extraction("q1", "tulu-rlvr", "neutral", ["A"]),
        ]
        labels = {r["model_id"]: r["label"] for r in training_rows(records, REGISTRY)}
        assert labels == {"tulu-base": 0, "tulu-rlvr": 1}

    def test_carries_both_the_feature_vector_and_the_text(self):
        rows = training_rows([extraction("q1", "tulu-rlvr", "neutral", ["A"])], REGISTRY)
        assert rows[0]["features"]["citation_count"] == 1.0
        assert rows[0]["text"] == "A"

    def test_skips_errored_extractions(self):
        record = extraction("q1", "tulu-rlvr", "neutral", ["A"])
        record["error"] = "timeout"
        assert training_rows([record], REGISTRY) == []


class TestPairOutcome:
    def test_higher_score_on_the_post_trained_side_is_a_win(self):
        assert pair_outcome(low_score=0.2, high_score=0.8) == 1.0

    def test_higher_score_on_the_base_side_is_a_loss(self):
        assert pair_outcome(low_score=0.8, high_score=0.2) == 0.0

    def test_equal_scores_are_half(self):
        # 23.8% of evaluation pairs are ties, almost all of them because neither
        # side cited anything. Scoring them as losses would cap the count-matched
        # rate at 62.5% against a registered 60% bar.
        assert pair_outcome(low_score=0.5, high_score=0.5) == 0.5


class TestPreferenceSummary:
    def test_rate_counts_a_tie_as_half(self):
        summary = preference_summary([1.0, 0.0, 0.5, 0.5])
        assert summary["preference_rate"] == pytest.approx(0.5)

    def test_reports_the_tie_rate_alongside_the_preference_rate(self):
        summary = preference_summary([1.0, 1.0, 0.5, 0.0])
        assert summary["n_pairs"] == 4
        assert summary["wins"] == 2
        assert summary["losses"] == 1
        assert summary["ties"] == 1
        assert summary["tie_rate"] == pytest.approx(0.25)

    def test_no_pairs_raises_rather_than_reporting_a_rate(self):
        with pytest.raises(ValueError):
            preference_summary([])


class TestBootstrap:
    def test_is_deterministic_under_a_fixed_seed(self):
        outcomes = [1.0] * 60 + [0.0] * 40
        assert bootstrap_ci(outcomes, resamples=200, seed=1) == bootstrap_ci(outcomes, resamples=200, seed=1)

    def test_a_different_seed_gives_a_different_interval(self):
        outcomes = [1.0] * 60 + [0.0] * 40
        assert bootstrap_ci(outcomes, resamples=200, seed=1) != bootstrap_ci(outcomes, resamples=200, seed=2)

    def test_interval_brackets_the_point_estimate(self):
        outcomes = [1.0] * 60 + [0.0] * 40
        low, high = bootstrap_ci(outcomes, resamples=500, seed=7)
        assert low < 0.6 < high

    def test_an_all_wins_sample_has_a_degenerate_interval_rather_than_an_error(self):
        assert bootstrap_ci([1.0] * 50, resamples=100, seed=3) == (1.0, 1.0)

    def test_default_resample_count_is_pre_registered(self):
        assert BOOTSTRAP_RESAMPLES == 2000


class TestModelChoiceRule:
    """§5: the 4-feature logistic is primary if within 5pp of DeBERTa."""

    def test_logistic_is_primary_when_within_tolerance(self):
        assert select_primary_model(logistic_rate=0.62, deberta_rate=0.66) == "logistic"

    def test_logistic_is_primary_when_it_beats_deberta(self):
        assert select_primary_model(logistic_rate=0.70, deberta_rate=0.61) == "logistic"

    def test_deberta_is_primary_when_it_leads_by_more_than_tolerance(self):
        assert select_primary_model(logistic_rate=0.60, deberta_rate=0.70) == "deberta"

    def test_exactly_five_points_still_favours_the_logistic(self):
        assert select_primary_model(logistic_rate=0.60, deberta_rate=0.65) == "logistic"
        assert MODEL_CHOICE_TOLERANCE == pytest.approx(0.05)


class TestHypothesisRule:
    def test_supported_needs_both_the_threshold_and_the_interval(self):
        assert evaluate_hypothesis(rate=0.64, ci_low=0.61)["verdict"] == "supported"

    def test_a_rate_between_the_null_and_the_threshold_is_weak(self):
        assert evaluate_hypothesis(rate=0.55, ci_low=0.52)["verdict"] == "weak_indeterminate"

    def test_a_high_rate_whose_interval_touches_the_null_is_not_supported(self):
        result = evaluate_hypothesis(rate=0.63, ci_low=0.49)
        assert result["verdict"] == "not_supported"
        assert result["threshold_met"] is True
        assert result["ci_excludes_null"] is False

    def test_a_rate_at_or_below_the_null_is_not_supported(self):
        assert evaluate_hypothesis(rate=0.48, ci_low=0.44)["verdict"] == "not_supported"

    def test_thresholds_are_the_pre_registered_ones(self):
        assert PREFERENCE_THRESHOLD == pytest.approx(0.60)
        assert NULL_RATE == pytest.approx(0.50)

    def test_both_legs_are_always_reported(self):
        result = evaluate_hypothesis(rate=0.64, ci_low=0.61)
        assert set(result) == {"verdict", "threshold_met", "ci_excludes_null"}


class TestCountMatchedRule:
    """§5's pre-committed reading of an intermediate count-matched result."""

    def test_a_count_matched_bound_at_or_above_the_unconstrained_one_is_robust(self):
        assert count_matched_verdict(matched_ci_low=0.60, unconstrained_ci_low=0.58) == "count_robust"

    def test_a_bound_between_the_null_and_the_unconstrained_one_is_count_sensitive(self):
        # The confound is weakened but not excluded, which §5 says must not be
        # reported as full support.
        assert count_matched_verdict(matched_ci_low=0.54, unconstrained_ci_low=0.60) == "count_sensitive"

    def test_a_bound_at_or_below_the_null_leaves_the_confound_standing(self):
        assert count_matched_verdict(matched_ci_low=0.49, unconstrained_ci_low=0.60) == "count_confounded"
        assert count_matched_verdict(matched_ci_low=0.50, unconstrained_ci_low=0.60) == "count_confounded"


class TestPairsIntegration:
    def test_evaluation_pairs_come_from_the_held_out_family(self):
        records = [
            extraction("q1", "olmo-base", "neutral", []),
            extraction("q1", "olmo-instruct", "neutral", ["A", "B"]),
        ]
        pairs = [p for p in build_pairs(records, REGISTRY) if p.split == "eval"]
        assert len(pairs) == 1
        assert pairs[0].low_family == "olmo2_13b"
