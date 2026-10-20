"""Tests for the pre-registered UNVERIFIABLE-exclusion selection check (§6 sensitivity #1)."""
from __future__ import annotations

import json

import pytest

from src.analysis.unverifiable_selection_check import (
    cell_of_question_id,
    domain_of_question_id,
    generation_final_statuses,
    is_excluded,
    monotone_direction,
    selection_check,
)


class TestQuestionIdParsing:
    def test_reads_cell_from_question_id(self):
        assert cell_of_question_id("legal_obscure_real_0445") == "obscure_real"
        assert cell_of_question_id("academic_known_real_0001") == "known_real"
        assert cell_of_question_id("medical_unanswerable_0100") == "unanswerable"

    def test_known_real_is_not_swallowed_by_obscure_real(self):
        # 'obscure_real' and 'known_real' both end in '_real'; the matcher must not
        # collapse them, or every obscure row would be counted as known.
        assert cell_of_question_id("legal_obscure_real_0445") != "known_real"

    def test_reads_domain(self):
        assert domain_of_question_id("legal_obscure_real_0445") == "legal"
        assert domain_of_question_id("academic_known_real_0001") == "academic"

    def test_unrecognised_cell_raises_rather_than_silently_bucketing(self):
        with pytest.raises(ValueError):
            cell_of_question_id("legal_nonsense_0445")


class TestJ2aFallbackSemantics:
    """The load-bearing rule: a judge verdict of UNVERIFIABLE does NOT rescue."""

    def _index(self, statuses):
        return {("q", "m", "c", 0, i): s for i, s in enumerate(statuses)}

    def test_judge_real_rescues_to_real(self):
        final, complete = generation_final_statuses(
            "q", "m", "c", 0, ["FABRICATED"], self._index(["REAL"])
        )
        assert complete is True
        assert final == ["REAL"]

    def test_judge_fabricated_overrides_verifier(self):
        final, _ = generation_final_statuses(
            "q", "m", "c", 0, ["UNVERIFIABLE"], self._index(["FABRICATED"])
        )
        assert final == ["FABRICATED"]

    def test_judge_unverifiable_falls_back_to_verifier_label(self):
        final, _ = generation_final_statuses(
            "q", "m", "c", 0, ["UNVERIFIABLE"], self._index(["UNVERIFIABLE"])
        )
        assert final == ["UNVERIFIABLE"]

    def test_verifier_real_is_never_sent_to_the_judge_and_stays_real(self):
        final, complete = generation_final_statuses("q", "m", "c", 0, ["REAL"], {})
        assert complete is True
        assert final == ["REAL"]

    def test_missing_judgment_marks_generation_incomplete(self):
        # Judging is still in flight; a partially judged generation must be
        # skipped, not scored, or the exclusion rate is biased by judging order.
        final, complete = generation_final_statuses("q", "m", "c", 0, ["FABRICATED"], {})
        assert complete is False
        assert final is None


class TestExclusionRule:
    def test_all_unverifiable_is_excluded(self):
        assert is_excluded(["UNVERIFIABLE", "UNVERIFIABLE"]) is True

    def test_any_resolved_citation_keeps_the_generation(self):
        assert is_excluded(["UNVERIFIABLE", "FABRICATED"]) is False
        assert is_excluded(["UNVERIFIABLE", "REAL"]) is False

    def test_empty_citation_list_is_not_an_exclusion(self):
        # A generation that cited nothing is a separate outcome (abstention),
        # not a cited-only-UNVERIFIABLE exclusion.
        assert is_excluded([]) is False


class TestMonotoneDirection:
    def test_detects_decreasing(self):
        assert monotone_direction([8.9, 8.5, 7.1, 6.2]) == "DOWN"

    def test_detects_increasing(self):
        assert monotone_direction([0.4, 0.7, 1.1, 1.4]) == "UP"

    def test_non_monotone_returns_no(self):
        assert monotone_direction([2.5, 3.1, 2.9, 2.4]) == "NO"

    def test_flat_sequence_is_not_reported_as_a_trend(self):
        assert monotone_direction([1.0, 1.0, 1.0, 1.0]) == "FLAT"

    def test_ignores_missing_strata(self):
        assert monotone_direction([8.9, None, 7.1, 6.2]) == "DOWN"


class TestSelectionCheck:
    def _fixture(self, tmp_path):
        registry = [
            {"model_id": "base", "stage_ordinal": 0, "is_post_trained": 0, "stage": "base"},
            {"model_id": "post", "stage_ordinal": 3, "is_post_trained": 1, "stage": "RLVR"},
        ]
        verifications = [
            # legal obscure, base model, excluded (all UNV, judge also UNV)
            {
                "question_id": "legal_obscure_real_0001",
                "condition": "neutral",
                "model_id": "base",
                "trial_id": 0,
                "citation_statuses": ["UNVERIFIABLE"],
            },
            # legal obscure, post model, rescued to REAL -> not excluded
            {
                "question_id": "legal_obscure_real_0002",
                "condition": "neutral",
                "model_id": "post",
                "trial_id": 0,
                "citation_statuses": ["UNVERIFIABLE"],
            },
        ]
        judgments = [
            {
                "custom_id": "legal_obscure_real_0001__base__neutral__t0__c0",
                "judge_status": "UNVERIFIABLE",
                "error": None,
            },
            {
                "custom_id": "legal_obscure_real_0002__post__neutral__t0__c0",
                "judge_status": "REAL",
                "error": None,
            },
        ]
        return registry, verifications, judgments

    def test_counts_exclusions_per_stratum(self, tmp_path):
        registry, verifications, judgments = self._fixture(tmp_path)
        result = selection_check(verifications, judgments, registry)
        e1 = result["e1_cell_by_stage"]
        assert e1["obscure_real"]["0"] == {"n": 1, "excluded": 1, "rate": 100.0}
        assert e1["obscure_real"]["3"] == {"n": 1, "excluded": 0, "rate": 0.0}

    def test_errored_judgment_is_treated_as_no_verdict(self):
        registry, verifications, judgments = self._fixture(None)
        judgments[1] = {**judgments[1], "error": "boom", "judge_status": "REAL"}
        result = selection_check(verifications, judgments, registry)
        # the rescued row is now unjudged -> skipped, not counted as resolved
        assert result["n_skipped_incomplete"] == 1

    def test_reports_per_domain_obscure_real_monotonicity(self):
        registry, verifications, judgments = self._fixture(None)
        result = selection_check(verifications, judgments, registry)
        assert "legal" in result["escalation"]["per_domain"]

    def test_escalation_verdict_is_present_and_boolean(self):
        registry, verifications, judgments = self._fixture(None)
        result = selection_check(verifications, judgments, registry)
        assert isinstance(result["escalation"]["registered_trigger_fires"], bool)
