"""Tests for the EXISTENT_WRONG_ID sub-study scorer (amendment #6 #3).

Registered metric (PRE_REGISTRATION_OSF.md §11 #6): per-citation disagreement
rate vs J2a-as-reported under the strict-hybrid combine. Because strict-hybrid
falls back to the verifier label on a judge-UNVERIFIABLE, only a judge
FABRICATED verdict flips a verifier-REAL citation — so "disagreement" is the
judge-FABRICATED rate. Threshold 8%.
"""
from __future__ import annotations

import math

from src.analysis.substudy_score import (
    DISAGREEMENT_STATUS,
    score_substudy,
    stratified_se,
    wilson_ci,
)


def test_disagreement_status_is_fabricated():
    assert DISAGREEMENT_STATUS == "FABRICATED"


def test_wilson_ci_zero_events():
    lo, hi = wilson_ci(0, 200)
    assert lo == 0.0
    assert 0.0 < hi < 0.05


def test_wilson_ci_brackets_point_estimate():
    lo, hi = wilson_ci(10, 200)
    assert lo < 0.05 < hi
    assert 0.0 <= lo < hi <= 1.0


def test_wilson_ci_symmetric_half_width_sane():
    lo, hi = wilson_ci(20, 200)  # 10%
    assert math.isclose((lo + hi) / 2, 0.10, abs_tol=0.02)


def test_stratified_se_matches_hand_calc():
    # Two strata, both r=0.1; design-based SE with FPC, hand-computed.
    # A: N=900,n=100,fab=10 -> term = .9^2 * .1*.9/100 * (1-100/900) = 0.000648
    # B: N=100,n=50, fab=5  -> term = .1^2 * .1*.9/50  * (1-50/100)  = 0.000009
    # Var = 0.000657 -> SE = 0.025631
    per_stratum = [
        {"population": 900, "n_judged": 100, "n_fabricated": 10},
        {"population": 100, "n_judged": 50, "n_fabricated": 5},
    ]
    se = stratified_se(per_stratum, pop_total=1000)
    assert math.isclose(se, 0.025631, abs_tol=1e-5)


def test_stratified_se_zero_when_no_events():
    per_stratum = [{"population": 900, "n_judged": 100, "n_fabricated": 0}]
    assert stratified_se(per_stratum, pop_total=900) == 0.0


def _manifest(strata, selected):
    return {"strata": strata, "selected_custom_ids": selected}


def _qmap(strata_of):
    return {qid: {"id": qid, "domain": d, "cell": c} for qid, (d, c) in strata_of.items()}


def test_score_raw_rate_counts_only_fabricated():
    # 4 citations, 1 FABRICATED, 1 UNVERIFIABLE, 2 REAL -> raw disagreement 1/4.
    strata = [{"domain": "academic", "cell": "known_real", "population": 100, "allocation": 4}]
    sel = ["qa__m__nu__t0__c0", "qb__m__nu__t0__c0", "qc__m__nu__t0__c0", "qd__m__nu__t0__c0"]
    manifest = _manifest(strata, sel)
    qmap = _qmap({"qa": ("academic", "known_real"), "qb": ("academic", "known_real"),
                  "qc": ("academic", "known_real"), "qd": ("academic", "known_real")})
    judgments = [
        {"custom_id": "qa__m__nu__t0__c0", "judge_status": "FABRICATED"},
        {"custom_id": "qb__m__nu__t0__c0", "judge_status": "UNVERIFIABLE"},
        {"custom_id": "qc__m__nu__t0__c0", "judge_status": "REAL"},
        {"custom_id": "qd__m__nu__t0__c0", "judge_status": "REAL"},
    ]
    res = score_substudy(manifest, judgments, qmap, threshold=0.08)
    assert res.n == 4
    assert res.n_judged == 4
    assert res.raw_fab_rate == 0.25
    assert math.isclose(res.raw_unv_rate, 0.25)


def test_score_reweights_by_population():
    # Stratum A: 1/2 fab in sample, pop 900. Stratum B: 1/2 fab, pop 100.
    # Raw = 2/4 = .5; reweighted = .9*.5 + .1*.5 = .5 (equal rates -> same).
    strata = [
        {"domain": "academic", "cell": "known_real", "population": 900, "allocation": 2},
        {"domain": "legal", "cell": "obscure_real", "population": 100, "allocation": 2},
    ]
    sel = ["a0__m__nu__t0__c0", "a1__m__nu__t0__c0", "b0__m__nu__t0__c0", "b1__m__nu__t0__c0"]
    qmap = _qmap({"a0": ("academic", "known_real"), "a1": ("academic", "known_real"),
                  "b0": ("legal", "obscure_real"), "b1": ("legal", "obscure_real")})
    judgments = [
        {"custom_id": "a0__m__nu__t0__c0", "judge_status": "FABRICATED"},
        {"custom_id": "a1__m__nu__t0__c0", "judge_status": "REAL"},
        {"custom_id": "b0__m__nu__t0__c0", "judge_status": "FABRICATED"},
        {"custom_id": "b1__m__nu__t0__c0", "judge_status": "REAL"},
    ]
    res = score_substudy(_manifest(strata, sel), judgments, qmap, threshold=0.08)
    assert res.raw_fab_rate == 0.5
    assert math.isclose(res.reweighted_fab_rate, 0.5)


def test_score_reweight_differs_when_rates_differ_by_stratum():
    # Big safe stratum 0/2 fab (pop 900); small risky 2/2 fab (pop 100).
    # Raw = 2/4 = .5; reweighted = .9*0 + .1*1.0 = .10.
    strata = [
        {"domain": "academic", "cell": "known_real", "population": 900, "allocation": 2},
        {"domain": "legal", "cell": "obscure_real", "population": 100, "allocation": 2},
    ]
    sel = ["a0__m__nu__t0__c0", "a1__m__nu__t0__c0", "b0__m__nu__t0__c0", "b1__m__nu__t0__c0"]
    qmap = _qmap({"a0": ("academic", "known_real"), "a1": ("academic", "known_real"),
                  "b0": ("legal", "obscure_real"), "b1": ("legal", "obscure_real")})
    judgments = [
        {"custom_id": "a0__m__nu__t0__c0", "judge_status": "REAL"},
        {"custom_id": "a1__m__nu__t0__c0", "judge_status": "REAL"},
        {"custom_id": "b0__m__nu__t0__c0", "judge_status": "FABRICATED"},
        {"custom_id": "b1__m__nu__t0__c0", "judge_status": "FABRICATED"},
    ]
    res = score_substudy(_manifest(strata, sel), judgments, qmap, threshold=0.08)
    assert res.raw_fab_rate == 0.5
    assert math.isclose(res.reweighted_fab_rate, 0.10)


def test_score_counts_missing_judgments():
    strata = [{"domain": "academic", "cell": "known_real", "population": 100, "allocation": 2}]
    sel = ["a0__m__nu__t0__c0", "a1__m__nu__t0__c0"]
    qmap = _qmap({"a0": ("academic", "known_real"), "a1": ("academic", "known_real")})
    judgments = [{"custom_id": "a0__m__nu__t0__c0", "judge_status": "REAL"}]  # a1 missing
    res = score_substudy(_manifest(strata, sel), judgments, qmap, threshold=0.08)
    assert res.n == 2
    assert res.n_judged == 1
    assert res.n_missing == 1


def test_raw_rate_denominator_excludes_missing():
    # 3 sampled, 1 FABRICATED, 1 REAL, 1 missing -> raw rate = 1/2 (judged), not 1/3.
    strata = [{"domain": "academic", "cell": "known_real", "population": 100, "allocation": 3}]
    sel = ["a0__m__nu__t0__c0", "a1__m__nu__t0__c0", "a2__m__nu__t0__c0"]
    qmap = _qmap({k: ("academic", "known_real") for k in ("a0", "a1", "a2")})
    judgments = [
        {"custom_id": "a0__m__nu__t0__c0", "judge_status": "FABRICATED"},
        {"custom_id": "a1__m__nu__t0__c0", "judge_status": "REAL"},
        # a2 missing
    ]
    res = score_substudy(_manifest(strata, sel), judgments, qmap, threshold=0.08)
    assert res.n_judged == 2
    assert res.raw_fab_rate == 0.5  # 1/2, missing excluded from denominator


def test_result_carries_reweighted_ci():
    strata = [{"domain": "academic", "cell": "known_real", "population": 100, "allocation": 10}]
    sel = [f"a{i}__m__nu__t0__c0" for i in range(10)]
    qmap = _qmap({f"a{i}": ("academic", "known_real") for i in range(10)})
    judgments = [
        {"custom_id": f"a{i}__m__nu__t0__c0",
         "judge_status": "FABRICATED" if i < 1 else "REAL"}
        for i in range(10)
    ]
    res = score_substudy(_manifest(strata, sel), judgments, qmap, threshold=0.08)
    lo, hi = res.reweighted_ci
    assert 0.0 <= lo <= res.reweighted_fab_rate <= hi <= 1.0


def test_score_gate_triggers_above_threshold():
    strata = [{"domain": "academic", "cell": "known_real", "population": 100, "allocation": 10}]
    sel = [f"a{i}__m__nu__t0__c0" for i in range(10)]
    qmap = _qmap({f"a{i}": ("academic", "known_real") for i in range(10)})
    # 2/10 = 20% fabricated -> exceeds 8%.
    judgments = [
        {"custom_id": f"a{i}__m__nu__t0__c0",
         "judge_status": "FABRICATED" if i < 2 else "REAL"}
        for i in range(10)
    ]
    res = score_substudy(_manifest(strata, sel), judgments, qmap, threshold=0.08)
    assert res.raw_fab_rate == 0.2
    assert res.exceeds_threshold is True


def test_score_gate_clears_below_threshold():
    strata = [{"domain": "academic", "cell": "known_real", "population": 100, "allocation": 100}]
    sel = [f"a{i}__m__nu__t0__c0" for i in range(100)]
    qmap = _qmap({f"a{i}": ("academic", "known_real") for i in range(100)})
    judgments = [
        {"custom_id": f"a{i}__m__nu__t0__c0",
         "judge_status": "FABRICATED" if i < 4 else "REAL"}
        for i in range(100)
    ]  # 4% < 8%
    res = score_substudy(_manifest(strata, sel), judgments, qmap, threshold=0.08)
    assert res.raw_fab_rate == 0.04
    assert res.exceeds_threshold is False


def test_per_stratum_breakdown_present():
    strata = [
        {"domain": "academic", "cell": "known_real", "population": 900, "allocation": 2},
        {"domain": "legal", "cell": "obscure_real", "population": 100, "allocation": 2},
    ]
    sel = ["a0__m__nu__t0__c0", "a1__m__nu__t0__c0", "b0__m__nu__t0__c0", "b1__m__nu__t0__c0"]
    qmap = _qmap({"a0": ("academic", "known_real"), "a1": ("academic", "known_real"),
                  "b0": ("legal", "obscure_real"), "b1": ("legal", "obscure_real")})
    judgments = [
        {"custom_id": "a0__m__nu__t0__c0", "judge_status": "REAL"},
        {"custom_id": "a1__m__nu__t0__c0", "judge_status": "REAL"},
        {"custom_id": "b0__m__nu__t0__c0", "judge_status": "FABRICATED"},
        {"custom_id": "b1__m__nu__t0__c0", "judge_status": "FABRICATED"},
    ]
    res = score_substudy(_manifest(strata, sel), judgments, qmap, threshold=0.08)
    by = {(s["domain"], s["cell"]): s for s in res.per_stratum}
    assert by[("legal", "obscure_real")]["n_fabricated"] == 2
    assert by[("academic", "known_real")]["n_fabricated"] == 0
    assert by[("legal", "obscure_real")]["fab_rate"] == 1.0
