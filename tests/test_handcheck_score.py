"""Contract tests for src/analysis/handcheck_score.py.

Scores the enriched blind hand-check. Because the sample deliberately
oversamples fabricated citations, every POPULATION-level number must be design
weighted, while per-stratum and per-class diagnostics stay unweighted. Mixing
those up would either bias the headline accuracy or destroy the precision the
enrichment was bought for.

The comparison the paper needs is the one Step 0 could not make: is the pipeline
distinguishable from a judge that answers REAL to everything? That needs the
weighted baseline, since the trivial baseline's score depends entirely on the
population class balance, not the sample's.
"""
from __future__ import annotations

import pytest

from src.analysis.handcheck_score import (
    baseline_comparison,
    by_era,
    by_stratum,
    class_performance,
    score_handcheck,
    weighted_accuracy,
)


def _row(human, combined, stratum="judge_FABRICATED", weight=1.0, era="EARLY"):
    return {
        "human_status": human,
        "combined_status": combined,
        "stratum": stratum,
        "design_weight": weight,
        "era": era,
    }


# ---------- weighted accuracy ----------


def test_weighted_accuracy_respects_design_weights() -> None:
    rows = [
        _row("REAL", "REAL", stratum="a", weight=9.0),
        _row("REAL", "FABRICATED", stratum="b", weight=1.0),
    ]
    assert weighted_accuracy(rows) == pytest.approx(0.9)


def test_weighted_accuracy_differs_from_unweighted_when_strata_differ() -> None:
    rows = [
        _row("REAL", "REAL", stratum="a", weight=9.0),
        _row("REAL", "FABRICATED", stratum="b", weight=1.0),
    ]
    unweighted = sum(1 for r in rows if r["human_status"] == r["combined_status"]) / len(rows)
    assert weighted_accuracy(rows) != pytest.approx(unweighted)


# ---------- per-class ----------


def test_class_performance_reports_recall_and_precision() -> None:
    rows = (
        [_row("FABRICATED", "FABRICATED") for _ in range(8)]
        + [_row("FABRICATED", "REAL") for _ in range(2)]
        + [_row("REAL", "FABRICATED") for _ in range(1)]
    )
    fab = class_performance(rows, "FABRICATED")
    assert fab["recall"] == pytest.approx(0.8)
    assert fab["precision"] == pytest.approx(8 / 9)
    assert fab["support"] == 10


def test_class_performance_reports_a_confidence_interval() -> None:
    rows = [_row("FABRICATED", "FABRICATED") for _ in range(50)]
    fab = class_performance(rows, "FABRICATED")
    lo, hi = fab["recall_ci"]
    assert lo <= fab["recall"] <= hi


def test_class_performance_weighted_recall_uses_weights() -> None:
    rows = [
        _row("FABRICATED", "FABRICATED", stratum="a", weight=1.0),
        _row("FABRICATED", "REAL", stratum="b", weight=9.0),
    ]
    fab = class_performance(rows, "FABRICATED")
    assert fab["recall"] == pytest.approx(0.5)
    assert fab["weighted_recall"] == pytest.approx(0.1)


def test_class_performance_handles_absent_class() -> None:
    rows = [_row("REAL", "REAL")]
    fab = class_performance(rows, "FABRICATED")
    assert fab["support"] == 0
    assert fab["recall"] is None


# ---------- baseline ----------


def test_baseline_comparison_uses_the_weighted_population_balance() -> None:
    """The trivial baseline's score depends on the POPULATION class balance,
    not the enriched sample's, so it must be weighted."""
    rows = (
        [_row("REAL", "REAL", stratum="trusted", weight=100.0) for _ in range(10)]
        + [_row("FABRICATED", "FABRICATED", stratum="fab", weight=1.0) for _ in range(10)]
    )
    result = baseline_comparison(rows)
    assert result["baseline_label"] == "REAL"
    assert result["baseline_weighted_accuracy"] == pytest.approx(100 / 101, abs=1e-3)
    assert result["pipeline_weighted_accuracy"] == pytest.approx(1.0)


def test_baseline_comparison_runs_mcnemar_on_discordant_pairs() -> None:
    rows = (
        [_row("FABRICATED", "FABRICATED") for _ in range(20)]
        + [_row("REAL", "REAL") for _ in range(20)]
    )
    result = baseline_comparison(rows)
    # Pipeline right on all 40; baseline wrong on the 20 fabricated ones.
    assert result["judge_right_baseline_wrong"] == 20
    assert result["judge_wrong_baseline_right"] == 0
    assert result["mcnemar_p"] < 0.05


def test_baseline_comparison_is_null_when_the_pipeline_adds_nothing() -> None:
    rows = [_row("REAL", "REAL") for _ in range(30)]
    result = baseline_comparison(rows)
    assert result["mcnemar_p"] is None


# ---------- breakdowns ----------


def test_by_stratum_reports_unweighted_accuracy_per_stratum() -> None:
    rows = (
        [_row("FABRICATED", "FABRICATED", stratum="judge_FABRICATED") for _ in range(3)]
        + [_row("FABRICATED", "REAL", stratum="judge_FABRICATED")]
        + [_row("REAL", "REAL", stratum="judge_REAL_rescue") for _ in range(2)]
    )
    table = by_stratum(rows)
    assert table["judge_FABRICATED"]["n"] == 4
    assert table["judge_FABRICATED"]["accuracy"] == pytest.approx(0.75)
    assert table["judge_REAL_rescue"]["accuracy"] == pytest.approx(1.0)


def test_by_era_exposes_whether_accuracy_moved_across_instrument_regimes() -> None:
    """The drift analysis measures the judge's STABILITY. This is the only
    direct evidence on its ACCURACY per regime, which is a different property."""
    rows = (
        [_row("FABRICATED", "FABRICATED", era="EARLY") for _ in range(4)]
        + [_row("FABRICATED", "REAL", era="DEGRADED") for _ in range(2)]
    )
    table = by_era(rows)
    assert table["EARLY"]["accuracy"] == pytest.approx(1.0)
    assert table["DEGRADED"]["accuracy"] == pytest.approx(0.0)


# ---------- end to end ----------


def test_score_handcheck_joins_and_skips_unlabelled_rows() -> None:
    blind = [
        {"row_id": "1", "your_status": "REAL"},
        {"row_id": "2", "your_status": ""},
        {"row_id": "3", "your_status": "FABRICATED"},
    ]
    reveal = [
        {"row_id": "1", "combined_status": "REAL", "stratum": "s",
         "design_weight": "2.0", "era": "EARLY"},
        {"row_id": "2", "combined_status": "REAL", "stratum": "s",
         "design_weight": "2.0", "era": "EARLY"},
        {"row_id": "3", "combined_status": "FABRICATED", "stratum": "s",
         "design_weight": "2.0", "era": "EARLY"},
    ]
    result = score_handcheck(blind, reveal)
    assert result["n_sampled"] == 3
    assert result["n_labelled"] == 2
    assert result["n_unlabelled"] == 1
    assert result["overall"]["unweighted_accuracy"] == pytest.approx(1.0)


def test_score_handcheck_rejects_an_unrecognised_label() -> None:
    blind = [{"row_id": "1", "your_status": "probably real?"}]
    reveal = [{"row_id": "1", "combined_status": "REAL", "stratum": "s",
               "design_weight": "1.0", "era": "EARLY"}]
    with pytest.raises(ValueError, match="probably real"):
        score_handcheck(blind, reveal)


def test_score_handcheck_accepts_shorthand_labels() -> None:
    blind = [{"row_id": "1", "your_status": " fab "}]
    reveal = [{"row_id": "1", "combined_status": "FABRICATED", "stratum": "s",
               "design_weight": "1.0", "era": "EARLY"}]
    result = score_handcheck(blind, reveal)
    assert result["overall"]["unweighted_accuracy"] == pytest.approx(1.0)
