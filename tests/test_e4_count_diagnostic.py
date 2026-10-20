"""Tests for the post-hoc exact-count-match diagnostic."""
from __future__ import annotations

import pytest

from src.analysis.e4_count_diagnostic import EXACT, TIERS, run_diagnostic, select_tier
from tests.test_run_e4 import REGISTRY, synthetic_extractions


class FakePair:
    def __init__(self, delta):
        self.count_delta = delta
        self.is_count_matched = abs(delta) <= 1


PAIRS = [FakePair(0), FakePair(1), FakePair(-1), FakePair(3), FakePair(0)]


class TestTierSelection:
    def test_all_keeps_everything(self):
        assert len(select_tier(PAIRS, "all")) == 5

    def test_within_one_keeps_the_registered_window(self):
        assert len(select_tier(PAIRS, "within_one")) == 4

    def test_exact_keeps_only_equal_counts(self):
        # The point of the diagnostic: with counts identical, a residual
        # preference cannot be a citation-count artifact.
        assert len(select_tier(PAIRS, EXACT)) == 2

    def test_exact_is_a_strict_subset_of_the_registered_window(self):
        assert set(map(id, select_tier(PAIRS, EXACT))) <= set(map(id, select_tier(PAIRS, "within_one")))

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError):
            select_tier(PAIRS, "nonsense")


class TestDiagnostic:
    def test_reports_all_three_tiers_for_both_pair_kinds(self):
        result = run_diagnostic(synthetic_extractions(), REGISTRY)
        assert set(result["tiers"]) == {"primary", "placebo"}
        for kind in result["tiers"]:
            assert set(result["tiers"][kind]) == set(TIERS)

    def test_marks_itself_post_hoc(self):
        # This artifact sits next to a registered one; it has to say which it is.
        assert "POST-HOC" in run_diagnostic(synthetic_extractions(), REGISTRY)["note"]

    def test_an_empty_tier_is_reported_as_not_computable(self):
        result = run_diagnostic(synthetic_extractions(base_cites_on_even=False), REGISTRY)
        assert result["tiers"]["primary"]["within_one"]["computable"] is False
