"""Contract tests for src/analysis/judge_drift_descriptives.py.

The drift subsection makes four numeric claims, and each one is a different way
to get an aggregation wrong:

  1. the verdict mix per era, which must be a share of citations judged in that
     era and not of every citation in the corrected verification file (those
     include verifier-trusted citations the judge never saw);
  2. grounding per era, where the zero-effort share is a ratio of sums and a
     citation carrying no effort record must leave both the numerator and the
     denominator rather than count as a zero;
  3. composition balance, which is a gap in PERCENTAGE POINTS between an era's
     share of a design level and that level's overall share, so a factor with
     many rare levels cannot look balanced by averaging;
  4. the per-cell REAL share, which is the diagnostic that separates a failure
     to confirm from a rise in fabrication and must be computed within cell.
"""
from __future__ import annotations

import pytest

from src.analysis.judge_drift_descriptives import (
    ERAS,
    composition_gaps,
    fab_share_by_effort_era,
    grounding_by_era,
    real_share_by_cell_era,
    verdict_mix_by_era,
)


def _rec(era: str, status: str = "REAL", search_count: int | None = 2,
         question_id: str = "q1", model_id: str = "m1",
         condition: str = "neutral") -> dict:
    return {
        "era": era,
        "judge_status": status,
        "judge_search_count": search_count,
        "question_id": question_id,
        "model_id": model_id,
        "condition": condition,
        "trial_id": 0,
    }


class TestVerdictMix:
    def test_shares_sum_to_one_within_each_era(self):
        meta = [_rec("EARLY", "REAL"), _rec("EARLY", "FABRICATED"),
                _rec("EARLY", "UNVERIFIABLE"), _rec("DEGRADED", "FABRICATED")]
        rows = verdict_mix_by_era(meta)
        for era in ("EARLY", "DEGRADED"):
            total = sum(r["share"] for r in rows if r["era"] == era)
            assert total == pytest.approx(1.0)

    def test_share_is_within_era_not_within_corpus(self):
        # EARLY is 1 of 4 citations overall but REAL is 1 of 1 within EARLY.
        meta = [_rec("EARLY", "REAL")] + [_rec("DEGRADED", "FABRICATED")] * 3
        rows = verdict_mix_by_era(meta)
        early_real = next(r for r in rows if r["era"] == "EARLY" and r["status"] == "REAL")
        assert early_real["share"] == pytest.approx(1.0)
        assert early_real["n"] == 1

    def test_absent_era_is_omitted_not_emitted_as_zero(self):
        rows = verdict_mix_by_era([_rec("EARLY", "REAL")])
        assert {r["era"] for r in rows} == {"EARLY"}

    def test_counts_conserve_the_input(self):
        meta = [_rec("EARLY", "REAL")] * 5 + [_rec("RECOVERED", "FABRICATED")] * 7
        assert sum(r["n"] for r in verdict_mix_by_era(meta)) == 12


class TestGrounding:
    def test_zero_effort_share_is_ratio_of_sums(self):
        meta = [_rec("EARLY", search_count=0)] * 3 + [_rec("EARLY", search_count=4)] * 7
        row = next(r for r in grounding_by_era(meta) if r["era"] == "EARLY")
        assert row["zero_effort_share"] == pytest.approx(0.3)

    def test_mean_effort_is_over_citations(self):
        meta = [_rec("EARLY", search_count=1)] * 2 + [_rec("EARLY", search_count=4)] * 2
        row = next(r for r in grounding_by_era(meta) if r["era"] == "EARLY")
        assert row["mean_effort"] == pytest.approx(2.5)

    def test_missing_effort_leaves_numerator_and_denominator(self):
        # A citation with no effort record is not a zero-effort citation.
        meta = [_rec("EARLY", search_count=None), _rec("EARLY", search_count=0),
                _rec("EARLY", search_count=2)]
        row = next(r for r in grounding_by_era(meta) if r["era"] == "EARLY")
        assert row["n_effort_recorded"] == 2
        assert row["n_missing_effort"] == 1
        assert row["zero_effort_share"] == pytest.approx(0.5)
        assert row["mean_effort"] == pytest.approx(1.0)

    def test_eras_come_back_in_chronological_order(self):
        meta = [_rec("RECOVERED"), _rec("EARLY"), _rec("DEGRADED")]
        assert [r["era"] for r in grounding_by_era(meta)] == list(ERAS)


class TestCompositionGaps:
    def test_perfectly_balanced_factor_has_zero_gap(self):
        meta = []
        for era in ERAS:
            meta += [_rec(era, condition="neutral"), _rec(era, condition="length_matched")]
        gaps = composition_gaps(meta, {"condition": lambda r: r["condition"]})
        assert gaps[0]["max_gap_pp"] == pytest.approx(0.0)

    def test_gap_is_in_percentage_points_against_the_overall_share(self):
        # EARLY is all neutral; overall neutral share is 2/3. Gap = 33.33 points.
        meta = [_rec("EARLY", condition="neutral"),
                _rec("DEGRADED", condition="neutral"),
                _rec("DEGRADED", condition="length_matched")]
        gaps = composition_gaps(meta, {"condition": lambda r: r["condition"]})
        assert gaps[0]["max_gap_pp"] == pytest.approx(100 / 3, abs=1e-9)

    def test_reports_which_era_and_level_was_worst(self):
        meta = [_rec("EARLY", condition="neutral"),
                _rec("DEGRADED", condition="neutral"),
                _rec("DEGRADED", condition="length_matched")]
        gaps = composition_gaps(meta, {"condition": lambda r: r["condition"]})
        assert gaps[0]["worst_era"] in ERAS
        assert gaps[0]["worst_level"] in {"neutral", "length_matched"}


class TestFabShareByEffort:
    """The diagnostic that separates 'checked fewer things' from 'scored them
    differently'. If the era shift were only a coverage failure it would live in
    the zero-effort rows and vanish once effort is held fixed."""

    def test_share_is_computed_within_effort_and_era(self):
        meta = ([_rec("EARLY", "FABRICATED", search_count=3)] * 3
                + [_rec("EARLY", "REAL", search_count=3)]
                + [_rec("EARLY", "REAL", search_count=0)] * 4)
        rows = fab_share_by_effort_era(meta)
        worked = next(r for r in rows if r["effort"] == 3 and r["era"] == "EARLY")
        assert worked["fab_share"] == pytest.approx(0.75)
        assert worked["n"] == 4
        idle = next(r for r in rows if r["effort"] == 0 and r["era"] == "EARLY")
        assert idle["fab_share"] == pytest.approx(0.0)

    def test_citations_with_no_effort_record_are_excluded(self):
        meta = [_rec("EARLY", "FABRICATED", search_count=None),
                _rec("EARLY", "FABRICATED", search_count=1)]
        rows = fab_share_by_effort_era(meta)
        assert sum(r["n"] for r in rows) == 1
        assert all(r["effort"] is not None for r in rows)

    def test_rows_are_ordered_by_effort_then_era(self):
        meta = [_rec("RECOVERED", search_count=1), _rec("EARLY", search_count=1),
                _rec("EARLY", search_count=0)]
        rows = fab_share_by_effort_era(meta)
        assert [(r["effort"], r["era"]) for r in rows] == [
            (0, "EARLY"), (1, "EARLY"), (1, "RECOVERED")]


class TestRealShareByCell:
    def test_share_is_computed_within_cell_and_era(self):
        cells = {"q_known": "known_real", "q_obscure": "obscure_real"}
        meta = [
            _rec("EARLY", "REAL", question_id="q_known"),
            _rec("EARLY", "FABRICATED", question_id="q_known"),
            _rec("EARLY", "REAL", question_id="q_obscure"),
        ]
        rows = real_share_by_cell_era(meta, lambda qid: cells[qid])
        known = next(r for r in rows if r["cell"] == "known_real" and r["era"] == "EARLY")
        obscure = next(r for r in rows if r["cell"] == "obscure_real" and r["era"] == "EARLY")
        assert known["real_share"] == pytest.approx(0.5)
        assert obscure["real_share"] == pytest.approx(1.0)

    def test_reports_the_early_to_degraded_drop_in_points(self):
        cells = {"q": "known_real"}
        meta = ([_rec("EARLY", "REAL", question_id="q")] * 8
                + [_rec("EARLY", "FABRICATED", question_id="q")] * 2
                + [_rec("DEGRADED", "REAL", question_id="q")] * 5
                + [_rec("DEGRADED", "FABRICATED", question_id="q")] * 5)
        rows = real_share_by_cell_era(meta, lambda qid: cells[qid])
        early = next(r for r in rows if r["era"] == "EARLY")
        assert early["real_share"] == pytest.approx(0.8)
        degraded = next(r for r in rows if r["era"] == "DEGRADED")
        assert degraded["real_share"] == pytest.approx(0.5)
        assert early["real_share"] - degraded["real_share"] == pytest.approx(0.3)
