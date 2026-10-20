"""Tests for E4's base-vs-post-trained pair construction.

The pairing is where E4's identification lives. Every test here is really
asking one question: does a pair hold the question, the condition, and the
family fixed, so that the only thing varying across it is post-training?
"""
from __future__ import annotations

import pytest

from src.analysis.e4_pairs import (
    BASE_ORDINAL,
    EVAL_FAMILIES,
    FINAL_ORDINAL,
    PLACEBO_ORDINALS,
    TRAIN_FAMILIES,
    build_pairs,
    family_checkpoints,
)

REGISTRY = [
    {"model_id": "meta-llama/Llama-3.1-8B", "model_family": "tulu3_8b", "stage_ordinal": 0, "is_post_trained": 0},
    {"model_id": "tulu-sft", "model_family": "tulu3_8b", "stage_ordinal": 1, "is_post_trained": 1},
    {"model_id": "tulu-dpo", "model_family": "tulu3_8b", "stage_ordinal": 2, "is_post_trained": 1},
    {"model_id": "tulu-rlvr", "model_family": "tulu3_8b", "stage_ordinal": 3, "is_post_trained": 1},
    {"model_id": "olmo-base", "model_family": "olmo2_13b", "stage_ordinal": 0, "is_post_trained": 0},
    {"model_id": "olmo-sft", "model_family": "olmo2_13b", "stage_ordinal": 1, "is_post_trained": 1},
    {"model_id": "olmo-dpo", "model_family": "olmo2_13b", "stage_ordinal": 2, "is_post_trained": 1},
    {"model_id": "olmo-instruct", "model_family": "olmo2_13b", "stage_ordinal": 3, "is_post_trained": 1},
]

CITE = {"title": "T", "authors": ["Cronbach", "L. J."], "venue": "Psychological Bulletin", "year": 1955}


def extraction(question_id, model_id, condition, n_citations, *, error=None, trial_id=0):
    return {
        "question_id": question_id,
        "model_id": model_id,
        "condition": condition,
        "trial_id": trial_id,
        "citations": [dict(CITE) for _ in range(n_citations)],
        "error": error,
    }


def full_family(question_id, condition, family_ids, counts):
    return [extraction(question_id, mid, condition, n) for mid, n in zip(family_ids, counts)]


TULU_IDS = ("meta-llama/Llama-3.1-8B", "tulu-sft", "tulu-dpo", "tulu-rlvr")
OLMO_IDS = ("olmo-base", "olmo-sft", "olmo-dpo", "olmo-instruct")


class TestRegistryShape:
    def test_indexes_each_family_by_stage(self):
        checkpoints = family_checkpoints(REGISTRY)
        assert checkpoints["tulu3_8b"][BASE_ORDINAL] == "meta-llama/Llama-3.1-8B"
        assert checkpoints["tulu3_8b"][FINAL_ORDINAL] == "tulu-rlvr"
        assert checkpoints["olmo2_13b"][FINAL_ORDINAL] == "olmo-instruct"

    def test_train_and_eval_families_are_disjoint(self):
        # Cross-family transfer is the publishable claim; an overlap would make
        # the evaluation partly in-sample without anything failing loudly.
        assert not set(TRAIN_FAMILIES) & set(EVAL_FAMILIES)

    def test_placebo_stages_are_both_post_trained(self):
        by_id = {r["model_id"]: r for r in REGISTRY}
        checkpoints = family_checkpoints(REGISTRY)
        for ordinal in PLACEBO_ORDINALS:
            assert by_id[checkpoints["tulu3_8b"][ordinal]]["is_post_trained"] == 1


class TestPrimaryPairs:
    def test_pairs_base_against_the_final_checkpoint(self):
        records = full_family("q1", "neutral", TULU_IDS, [0, 2, 3, 4])
        pairs = build_pairs(records, REGISTRY)
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.low_model_id == "meta-llama/Llama-3.1-8B"
        assert pair.high_model_id == "tulu-rlvr"

    def test_intermediate_checkpoints_do_not_form_primary_pairs(self):
        records = full_family("q1", "neutral", TULU_IDS, [0, 2, 3, 4])
        pairs = build_pairs(records, REGISTRY)
        assert {p.high_model_id for p in pairs} == {"tulu-rlvr"}

    def test_the_post_trained_side_is_the_expected_preference(self):
        records = full_family("q1", "neutral", TULU_IDS, [0, 2, 3, 4])
        assert build_pairs(records, REGISTRY)[0].expected_preference == "high"

    def test_pairs_hold_question_condition_and_family_fixed(self):
        records = (
            full_family("q1", "neutral", TULU_IDS, [0, 1, 1, 1])
            + full_family("q1", "citations_expected", TULU_IDS, [0, 1, 1, 1])
            + full_family("q2", "neutral", TULU_IDS, [0, 1, 1, 1])
            + full_family("q1", "neutral", OLMO_IDS, [0, 1, 1, 1])
        )
        pairs = build_pairs(records, REGISTRY)
        assert len(pairs) == 4
        for pair in pairs:
            assert pair.low_family == pair.high_family

    def test_no_pair_when_the_base_side_is_missing(self):
        records = [extraction("q1", "tulu-rlvr", "neutral", 3)]
        assert build_pairs(records, REGISTRY) == []

    def test_no_pair_when_the_post_trained_side_is_missing(self):
        records = [extraction("q1", "meta-llama/Llama-3.1-8B", "neutral", 0)]
        assert build_pairs(records, REGISTRY) == []

    def test_a_failed_extraction_removes_the_pair_rather_than_scoring_as_zero(self):
        # An errored record has no citations because extraction failed, not
        # because the model cited nothing. Scoring it as zero would credit the
        # other side with a difference that was never observed.
        records = [
            extraction("q1", "meta-llama/Llama-3.1-8B", "neutral", 0, error="timeout"),
            extraction("q1", "tulu-rlvr", "neutral", 4),
        ]
        assert build_pairs(records, REGISTRY) == []

    def test_zero_citations_on_both_sides_still_forms_a_pair(self):
        # Base models often cite nothing. Dropping empty-vs-empty pairs after the
        # fact would be a filter chosen with the outcome in view.
        records = full_family("q1", "neutral", TULU_IDS, [0, 0, 0, 0])
        assert len(build_pairs(records, REGISTRY)) == 1


class TestSplits:
    def test_tulu_families_are_the_training_split(self):
        records = full_family("q1", "neutral", TULU_IDS, [0, 1, 1, 1])
        assert build_pairs(records, REGISTRY)[0].split == "train"

    def test_olmo_is_the_evaluation_split(self):
        records = full_family("q1", "neutral", OLMO_IDS, [0, 1, 1, 1])
        assert build_pairs(records, REGISTRY)[0].split == "eval"


class TestFeatures:
    def test_each_side_carries_its_own_surface_feature_vector(self):
        records = full_family("q1", "neutral", TULU_IDS, [0, 1, 1, 5])
        pair = build_pairs(records, REGISTRY)[0]
        assert pair.low_features["citation_count"] == 0.0
        assert pair.high_features["citation_count"] == 5.0

    def test_count_delta_is_high_minus_low(self):
        records = full_family("q1", "neutral", TULU_IDS, [1, 1, 1, 4])
        assert build_pairs(records, REGISTRY)[0].count_delta == 3

    @pytest.mark.parametrize("counts,matched", [([3, 0, 0, 3], True), ([3, 0, 0, 4], True), ([3, 0, 0, 5], False)])
    def test_count_matching_is_within_one_inclusive(self, counts, matched):
        records = full_family("q1", "neutral", TULU_IDS, counts)
        assert build_pairs(records, REGISTRY)[0].is_count_matched is matched


class TestPlaceboPairs:
    def test_placebo_pairs_two_post_trained_checkpoints(self):
        records = full_family("q1", "neutral", TULU_IDS, [0, 2, 3, 4])
        pairs = build_pairs(records, REGISTRY, kind="placebo")
        assert len(pairs) == 1
        assert (pairs[0].low_model_id, pairs[0].high_model_id) == ("tulu-sft", "tulu-dpo")

    def test_placebo_pairs_have_no_expected_direction(self):
        records = full_family("q1", "neutral", TULU_IDS, [0, 2, 3, 4])
        assert build_pairs(records, REGISTRY, kind="placebo")[0].expected_preference is None

    def test_placebo_excludes_the_base_and_the_final_checkpoint(self):
        # Reusing the final checkpoint would make the placebo share generations
        # with the primary, so a shared quirk could move both together.
        records = full_family("q1", "neutral", TULU_IDS, [0, 2, 3, 4])
        pair = build_pairs(records, REGISTRY, kind="placebo")[0]
        assert "meta-llama/Llama-3.1-8B" not in (pair.low_model_id, pair.high_model_id)
        assert "tulu-rlvr" not in (pair.low_model_id, pair.high_model_id)

    def test_placebo_orientation_is_by_stage_so_the_null_is_fifty_percent(self):
        records = full_family("q1", "neutral", TULU_IDS, [0, 9, 1, 4])
        pair = build_pairs(records, REGISTRY, kind="placebo")[0]
        assert pair.low_features["citation_count"] == 9.0  # SFT, the lower stage
        assert pair.high_features["citation_count"] == 1.0

    def test_unknown_kind_raises_rather_than_returning_nothing(self):
        with pytest.raises(ValueError):
            build_pairs([], REGISTRY, kind="nonsense")


class TestDeterminism:
    def test_pair_order_is_stable_regardless_of_input_order(self):
        records = full_family("q1", "neutral", TULU_IDS, [0, 1, 1, 1]) + full_family(
            "q2", "neutral", TULU_IDS, [0, 1, 1, 1]
        )
        forward = [(p.question_id, p.condition, p.low_family) for p in build_pairs(records, REGISTRY)]
        backward = [(p.question_id, p.condition, p.low_family) for p in build_pairs(records[::-1], REGISTRY)]
        assert forward == backward
