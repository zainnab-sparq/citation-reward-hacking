"""Tests for src/analysis/batch_equivalence.py (pre-reg §11 deviation #9).

The J2a judge layer ran on two execution paths: a paid `claude -p` subprocess
(n=3,821, halted by the billing leak) and a free Workflow-tool path (n=6,872).
This module checks whether the two paths produced an equivalent distribution of
judge verdicts, controlling for the cell composition each path happened to
cover. Provenance is recovered from the pre-pivot backup: rows the subprocess
left as `error: subprocess_failed_or_unparseable` placeholders are exactly the
ones the Workflow path later judged.

NOT a randomized equivalence trial: path assignment is "whatever cleared before
the halt," so within-stratum residual confounding is disclosed, not eliminated.
"""
from __future__ import annotations

import pytest

from src.analysis import batch_equivalence as be


PLACEHOLDER = "subprocess_failed_or_unparseable"


# ---------- partition_provenance ----------


def test_partition_provenance_splits_on_placeholder_marker():
    backup = [
        {"custom_id": "q_known_real_0001__m__neutral__t0__c0",
         "judge_status": "REAL", "error": None},  # subprocess completed it
        {"custom_id": "q_known_real_0001__m__neutral__t0__c1",
         "judge_status": "None", "error": PLACEHOLDER},  # workflow judged it later
    ]
    prov = be.partition_provenance(backup)
    assert prov["q_known_real_0001__m__neutral__t0__c0"] == "subprocess"
    assert prov["q_known_real_0001__m__neutral__t0__c1"] == "workflow"


def test_partition_provenance_counts():
    backup = (
        [{"custom_id": f"q_a_0001__m__neutral__t0__c{i}", "error": None} for i in range(3)]
        + [{"custom_id": f"q_a_0001__m__neutral__t1__c{i}", "error": PLACEHOLDER}
           for i in range(5)]
    )
    prov = be.partition_provenance(backup)
    n_sub = sum(1 for v in prov.values() if v == "subprocess")
    n_wf = sum(1 for v in prov.values() if v == "workflow")
    assert (n_sub, n_wf) == (3, 5)


# ---------- build_equivalence_rows ----------


def test_build_equivalence_rows_attaches_strata_and_provenance():
    judgments = [
        {"custom_id": "academic_known_real_0000__m__citations_expected__t0__c0",
         "judge_status": "REAL", "error": None, "judge_search_count": 3},
    ]
    provenance = {"academic_known_real_0000__m__citations_expected__t0__c0": "workflow"}
    questions = [{"id": "academic_known_real_0000", "cell": "known_real", "domain": "academic"}]
    rows = be.build_equivalence_rows(judgments, provenance, questions)
    assert rows == [
        {
            "custom_id": "academic_known_real_0000__m__citations_expected__t0__c0",
            "provenance": "workflow",
            "condition": "citations_expected",
            "cell": "known_real",
            "domain": "academic",
            "judge_status": "REAL",
            "judge_search_count": 3,
        }
    ]


def test_build_equivalence_rows_skips_unknown_provenance():
    """A judgment with no backup provenance entry cannot be attributed and is
    dropped (with the count recoverable by the caller)."""
    judgments = [
        {"custom_id": "academic_known_real_0000__m__neutral__t0__c0",
         "judge_status": "REAL", "error": None},
    ]
    rows = be.build_equivalence_rows(judgments, {}, [
        {"id": "academic_known_real_0000", "cell": "known_real", "domain": "academic"}])
    assert rows == []


# ---------- status distribution + chi-square ----------


def _rows(spec):
    """spec: list of (provenance, judge_status, n) -> expanded rows."""
    out = []
    for prov, status, n in spec:
        for _ in range(n):
            out.append({"provenance": prov, "judge_status": status,
                        "condition": "neutral", "cell": "known_real",
                        "domain": "academic"})
    return out


def test_overall_chi_square_identical_distributions_high_p():
    rows = _rows([
        ("subprocess", "REAL", 50), ("subprocess", "FABRICATED", 30),
        ("subprocess", "UNVERIFIABLE", 20),
        ("workflow", "REAL", 50), ("workflow", "FABRICATED", 30),
        ("workflow", "UNVERIFIABLE", 20),
    ])
    res = be.overall_chi_square(rows)
    assert res["dof"] == 2
    assert res["n"] == 200
    assert res["p"] > 0.99  # distributions are identical


def test_overall_chi_square_divergent_distributions_low_p():
    rows = _rows([
        ("subprocess", "REAL", 90), ("subprocess", "FABRICATED", 5),
        ("subprocess", "UNVERIFIABLE", 5),
        ("workflow", "REAL", 10), ("workflow", "FABRICATED", 45),
        ("workflow", "UNVERIFIABLE", 45),
    ])
    res = be.overall_chi_square(rows)
    assert res["p"] < 0.001


def test_overall_chi_square_excludes_none_status():
    rows = _rows([
        ("subprocess", "REAL", 10), ("workflow", "REAL", 10),
    ]) + [{"provenance": "workflow", "judge_status": None,
           "condition": "neutral", "cell": "known_real", "domain": "academic"}]
    res = be.overall_chi_square(rows)
    assert res["n"] == 20  # the None-status row is excluded
    assert res["n_excluded"] == 1


# ---------- per_stratum_deltas ----------


def test_per_stratum_deltas_computes_proportion_gap():
    rows = [
        {"provenance": "subprocess", "judge_status": "REAL",
         "condition": "neutral", "cell": "known_real", "domain": "academic"},
        {"provenance": "subprocess", "judge_status": "REAL",
         "condition": "neutral", "cell": "known_real", "domain": "academic"},
        {"provenance": "workflow", "judge_status": "REAL",
         "condition": "neutral", "cell": "known_real", "domain": "academic"},
        {"provenance": "workflow", "judge_status": "FABRICATED",
         "condition": "neutral", "cell": "known_real", "domain": "academic"},
    ]
    deltas = be.per_stratum_deltas(rows)
    assert len(deltas) == 1
    d = deltas[0]
    assert (d["condition"], d["cell"], d["domain"]) == ("neutral", "known_real", "academic")
    assert d["n_subprocess"] == 2
    assert d["n_workflow"] == 2
    # REAL proportion: subprocess 2/2 = 1.0, workflow 1/2 = 0.5 -> abs delta 0.5
    assert d["prop_subprocess"]["REAL"] == pytest.approx(1.0)
    assert d["prop_workflow"]["REAL"] == pytest.approx(0.5)
    assert d["max_abs_delta"] == pytest.approx(0.5)


def test_per_stratum_deltas_flags_single_arm_strata_insufficient():
    """A stratum present in only one arm cannot be compared; it is reported
    with insufficient=True rather than a misleading delta."""
    rows = [
        {"provenance": "subprocess", "judge_status": "REAL",
         "condition": "neutral", "cell": "obscure_real", "domain": "medical"},
    ]
    deltas = be.per_stratum_deltas(rows)
    assert len(deltas) == 1
    assert deltas[0]["insufficient"] is True
    assert deltas[0]["n_workflow"] == 0


# ---------- search_count_summary (difficulty-selection diagnostic) ----------
#
# If the workflow arm shows systematically higher judge_search_count, the
# leading explanation for the verdict-distribution gap is difficulty
# selection (transient-skip re-queue pushed harder citations to workflow),
# not a dispatch-mechanism effect.


def _sc_row(prov, sc, condition="neutral", cell="known_real", domain="academic"):
    return {"provenance": prov, "judge_status": "REAL", "judge_search_count": sc,
            "condition": condition, "cell": cell, "domain": domain}


def test_search_count_summary_overall_means_by_arm():
    rows = [_sc_row("subprocess", 2), _sc_row("subprocess", 4),
            _sc_row("workflow", 5), _sc_row("workflow", 7)]
    res = be.search_count_summary(rows)
    assert res["overall"]["subprocess"]["n"] == 2
    assert res["overall"]["subprocess"]["mean"] == pytest.approx(3.0)
    assert res["overall"]["workflow"]["mean"] == pytest.approx(6.0)
    assert res["overall"]["delta_mean"] == pytest.approx(3.0)  # workflow - subprocess


def test_search_count_summary_per_stratum_delta():
    rows = (
        [_sc_row("subprocess", 2, domain="legal"), _sc_row("workflow", 6, domain="legal")]
        + [_sc_row("subprocess", 3, domain="medical"), _sc_row("workflow", 3, domain="medical")]
    )
    res = be.search_count_summary(rows)
    by_key = {(s["condition"], s["cell"], s["domain"]): s for s in res["per_stratum"]}
    legal = by_key[("neutral", "known_real", "legal")]
    assert legal["mean_subprocess"] == pytest.approx(2.0)
    assert legal["mean_workflow"] == pytest.approx(6.0)
    assert legal["delta_mean"] == pytest.approx(4.0)
    medical = by_key[("neutral", "known_real", "medical")]
    assert medical["delta_mean"] == pytest.approx(0.0)


def test_search_count_summary_skips_non_integer_counts():
    rows = [_sc_row("subprocess", 2), _sc_row("subprocess", None),
            _sc_row("workflow", 4)]
    res = be.search_count_summary(rows)
    assert res["overall"]["subprocess"]["n"] == 1
    assert res["overall"]["subprocess"]["mean"] == pytest.approx(2.0)
    assert res["n_skipped_no_count"] == 1


# ---------- gap_summary (powered vs sparse strata) ----------
#
# The headline max_abs_delta must not be driven by tiny strata; the summary
# separates n-weighted and well-powered (min-arm n) views.


def _stratum(n_sub, n_wf, max_delta, condition="neutral", cell="known_real",
             domain="academic", insufficient=False):
    return {"condition": condition, "cell": cell, "domain": domain,
            "n_subprocess": n_sub, "n_workflow": n_wf,
            "max_abs_delta": max_delta, "insufficient": insufficient}


def test_gap_summary_weighted_mean_and_well_powered_subset():
    strata = [
        _stratum(50, 50, 0.10),
        _stratum(2, 100, 0.30, domain="legal"),  # sparse arm: excluded from well-powered
    ]
    res = be.gap_summary(strata, min_arm_n=30)
    # n-weighted mean over comparable strata: (100*0.10 + 102*0.30) / 202
    assert res["weighted_mean_max_gap"] == pytest.approx((100 * 0.10 + 102 * 0.30) / 202)
    assert res["min_arm_n"] == 30
    assert res["well_powered"]["n_strata"] == 1
    assert res["well_powered"]["mean_max_gap"] == pytest.approx(0.10)
    assert res["well_powered"]["max_gap"] == pytest.approx(0.10)
    assert res["well_powered"]["max_gap_stratum"] == {
        "condition": "neutral", "cell": "known_real", "domain": "academic"}


def test_gap_summary_excludes_insufficient_strata():
    strata = [
        _stratum(40, 40, 0.05),
        _stratum(10, 0, None, insufficient=True),
    ]
    res = be.gap_summary(strata, min_arm_n=30)
    assert res["weighted_mean_max_gap"] == pytest.approx(0.05)
    assert res["well_powered"]["n_strata"] == 1
