"""Contract tests for src/analysis/handcheck_sample.py.

Builds a fabrication-ENRICHED blind hand-check sample from the Phase C corpus,
to re-validate the J2a judge where the Step 0 check could not.

Why enriched: the Step 0 set was drawn to be representative, which for a rare
class means starving it of positives. It held 6 fabricated citations out of 50,
so FABRICATED recall came out 4/6 with a 95% CI of [30%, 90%], and the whole
pipeline was not separable from a trivial always-REAL baseline (McNemar
p = 0.125). Oversampling the positive class and carrying design weights fixes
the precision problem without biasing the population estimate.

The properties pinned here:

  * BLINDNESS. The file the annotator opens must contain no verifier label, no
    judge label, no combined label, no stratum and no weight. Anything that
    leaks the pipeline's opinion turns the validation into a measure of
    anchoring. This is the single most important test in the module.
  * The reveal file carries those columns and joins on row_id.
  * Presentation order is shuffled and independent of stratum, so the annotator
    cannot infer the stratum from position.
  * Deterministic given a seed, and design weights are population over sample.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.analysis.handcheck_sample import (
    BLIND_COLUMNS,
    DEFAULT_ALLOCATION,
    LEAKY_FIELDS,
    build_citation_records,
    sample_stratified,
    write_blind_csv,
    write_reveal_csv,
)


def _record(row_id, stratum, **kw):
    base = {
        "row_id": row_id,
        "stratum": stratum,
        "custom_id": f"q{row_id}__m__neutral__t0__c0",
        "question_id": f"academic_obscure_real_{row_id:04d}",
        "domain": "academic",
        "cell": "obscure_real",
        "condition": "neutral",
        "model_id": "allenai/Llama-3.1-Tulu-3-8B",
        "citation_index": 0,
        "raw_span": "Some Author (2019). A Title. Venue.",
        "title": "A Title",
        "case_name": "",
        "year": "2019",
        "venue": "Venue",
        "doi": "",
        "output_excerpt": "...text...",
        "verifier_status": "FABRICATED",
        "judge_status": "FABRICATED",
        "combined_status": "FABRICATED",
        "era": "EARLY",
    }
    base.update(kw)
    return base


def _pool(counts):
    records, row_id = [], 0
    for stratum, n in counts.items():
        for _ in range(n):
            row_id += 1
            records.append(_record(row_id, stratum))
    return records


# ---------- stratification ----------


def test_sample_respects_the_requested_allocation() -> None:
    pool = _pool({"judge_FABRICATED": 500, "judge_REAL_rescue": 500})
    sample = sample_stratified(pool, {"judge_FABRICATED": 60, "judge_REAL_rescue": 45})
    counts = {}
    for r in sample:
        counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1
    assert counts == {"judge_FABRICATED": 60, "judge_REAL_rescue": 45}


def test_sample_caps_at_available_population() -> None:
    pool = _pool({"judge_FABRICATED": 10})
    sample = sample_stratified(pool, {"judge_FABRICATED": 60})
    assert len(sample) == 10


def test_sample_is_deterministic_given_a_seed() -> None:
    pool = _pool({"judge_FABRICATED": 200})
    a = sample_stratified(pool, {"judge_FABRICATED": 20}, seed=7)
    b = sample_stratified(pool, {"judge_FABRICATED": 20}, seed=7)
    assert [r["row_id"] for r in a] == [r["row_id"] for r in b]


def test_design_weight_is_population_over_sample() -> None:
    pool = _pool({"judge_FABRICATED": 300})
    sample = sample_stratified(pool, {"judge_FABRICATED": 60})
    assert sample[0]["design_weight"] == pytest.approx(5.0)
    assert sample[0]["stratum_population"] == 300


def test_default_allocation_targets_about_fifty_fabricated_events() -> None:
    """The whole point of enriching: Step 0 had 6 fabricated events."""
    assert DEFAULT_ALLOCATION["judge_FABRICATED"] >= 50
    assert sum(DEFAULT_ALLOCATION.values()) <= 200


# ---------- presentation order ----------


def test_presentation_order_is_a_permutation() -> None:
    pool = _pool({"judge_FABRICATED": 50, "judge_REAL_rescue": 50})
    sample = sample_stratified(pool, {"judge_FABRICATED": 20, "judge_REAL_rescue": 20})
    orders = sorted(r["presentation_order"] for r in sample)
    assert orders == list(range(1, 41))


def test_presentation_order_is_independent_of_stratum() -> None:
    """If order tracked stratum, the annotator could infer the pipeline's view
    from position alone, which would defeat the blinding."""
    pool = _pool({"judge_FABRICATED": 100, "judge_REAL_rescue": 100})
    sample = sample_stratified(pool, {"judge_FABRICATED": 30, "judge_REAL_rescue": 30},
                               seed=11)
    ordered = sorted(sample, key=lambda r: r["presentation_order"])
    first_half = [r["stratum"] for r in ordered[:30]]
    # A stratum-sorted order would put all 30 of one stratum first.
    assert len(set(first_half)) > 1


# ---------- blindness ----------


def test_blind_columns_contain_no_pipeline_opinion() -> None:
    for leaky in LEAKY_FIELDS:
        assert leaky not in BLIND_COLUMNS


def test_blind_csv_never_writes_a_leaky_field(tmp_path: Path) -> None:
    pool = _pool({"judge_FABRICATED": 20})
    sample = sample_stratified(pool, {"judge_FABRICATED": 10})
    out = tmp_path / "blind.csv"
    write_blind_csv(sample, out)
    text = out.read_text(encoding="utf-8")
    with out.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for leaky in LEAKY_FIELDS:
        assert leaky not in rows[0]
    # And the label values themselves must not appear anywhere in the file.
    assert "FABRICATED" not in text


def test_blind_csv_has_empty_answer_columns(tmp_path: Path) -> None:
    pool = _pool({"judge_FABRICATED": 5})
    sample = sample_stratified(pool, {"judge_FABRICATED": 5})
    out = tmp_path / "blind.csv"
    write_blind_csv(sample, out)
    with out.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["your_status"] == "" for r in rows)
    assert all(r["your_evidence_url"] == "" for r in rows)


def test_blind_csv_is_written_in_presentation_order(tmp_path: Path) -> None:
    pool = _pool({"judge_FABRICATED": 40})
    sample = sample_stratified(pool, {"judge_FABRICATED": 20}, seed=3)
    out = tmp_path / "blind.csv"
    write_blind_csv(sample, out)
    with out.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["presentation_order"]) for r in rows] == list(range(1, 21))


# ---------- reveal ----------


def test_reveal_csv_carries_the_labels_and_joins_on_row_id(tmp_path: Path) -> None:
    pool = _pool({"judge_FABRICATED": 20})
    sample = sample_stratified(pool, {"judge_FABRICATED": 10})
    blind_path, reveal_path = tmp_path / "b.csv", tmp_path / "r.csv"
    write_blind_csv(sample, blind_path)
    write_reveal_csv(sample, reveal_path)
    with blind_path.open(encoding="utf-8-sig", newline="") as fh:
        blind = {r["row_id"] for r in csv.DictReader(fh)}
    with reveal_path.open(encoding="utf-8-sig", newline="") as fh:
        reveal = list(csv.DictReader(fh))
    assert {r["row_id"] for r in reveal} == blind
    for field in ("verifier_status", "judge_status", "combined_status",
                  "stratum", "design_weight", "era"):
        assert field in reveal[0]


# ---------- record construction ----------


def _verification(statuses):
    return {
        "question_id": "academic_obscure_real_0100",
        "condition": "neutral",
        "model_id": "allenai/Llama-3.1-Tulu-3-8B",
        "trial_id": 0,
        "citation_statuses": list(statuses),
    }


def _extraction(n):
    return {
        "question_id": "academic_obscure_real_0100",
        "condition": "neutral",
        "model_id": "allenai/Llama-3.1-Tulu-3-8B",
        "trial_id": 0,
        "citations": [
            {"raw_span": f"span {i}", "title": f"Title {i}", "year": 2019,
             "venue": "V", "doi": ""} for i in range(n)
        ],
    }


def _judgment(idx, status):
    return {
        "custom_id": ("academic_obscure_real_0100__allenai/Llama-3.1-Tulu-3-8B"
                      f"__neutral__t0__c{idx}"),
        "judge_status": status,
        "judge_search_count": 2,
        "error": None,
    }


QUESTIONS = {"academic_obscure_real_0100": {"domain": "academic", "cell": "obscure_real"}}
GENERATIONS = [{
    "question_id": "academic_obscure_real_0100", "condition": "neutral",
    "model_id": "allenai/Llama-3.1-Tulu-3-8B", "trial_id": 0,
    "output_text": "text span 0 more text",
}]


def test_records_classify_the_four_strata() -> None:
    records = build_citation_records(
        generations=GENERATIONS,
        extractions=[_extraction(4)],
        verifications=[_verification(["REAL", "FABRICATED", "FABRICATED", "UNVERIFIABLE"])],
        judgments=[_judgment(1, "FABRICATED"), _judgment(2, "REAL"),
                   _judgment(3, "UNVERIFIABLE")],
        questions=QUESTIONS,
    )
    strata = {r["citation_index"]: r["stratum"] for r in records}
    assert strata[0] == "verifier_REAL_trusted"
    assert strata[1] == "judge_FABRICATED"
    assert strata[2] == "judge_REAL_rescue"
    assert strata[3] == "judge_UNVERIFIABLE"


def test_unjudged_citations_are_their_own_stratum() -> None:
    """Judging runs for weeks, so at any point some verifier-FAB/UNV citations
    have no verdict yet. They fall back to the verifier label exactly like a
    judge-UNVERIFIABLE row, but they are NOT the same thing: their combined
    status will change once the judge reaches them, so hand-labelling them is
    wasted effort and pooling them corrupts the judge_UNVERIFIABLE stratum."""
    records = build_citation_records(
        generations=GENERATIONS,
        extractions=[_extraction(2)],
        verifications=[_verification(["FABRICATED", "FABRICATED"])],
        judgments=[_judgment(0, "UNVERIFIABLE")],   # index 1 never judged
        questions=QUESTIONS,
    )
    by_idx = {r["citation_index"]: r for r in records}
    assert by_idx[0]["stratum"] == "judge_UNVERIFIABLE"
    assert by_idx[1]["stratum"] == "never_judged"


def test_never_judged_is_not_drawn_by_the_default_allocation() -> None:
    """The default allocation must not spend the annotator's time on citations
    whose label is not final."""
    assert "never_judged" not in DEFAULT_ALLOCATION


def test_records_apply_the_registered_combine_rule() -> None:
    records = build_citation_records(
        generations=GENERATIONS,
        extractions=[_extraction(2)],
        verifications=[_verification(["FABRICATED", "FABRICATED"])],
        judgments=[_judgment(0, "REAL"), _judgment(1, "UNVERIFIABLE")],
        questions=QUESTIONS,
    )
    by_idx = {r["citation_index"]: r for r in records}
    assert by_idx[0]["combined_status"] == "REAL"        # judge rescues
    assert by_idx[1]["combined_status"] == "FABRICATED"  # judge UNV -> verifier stands


def test_records_carry_citation_metadata_for_the_annotator() -> None:
    records = build_citation_records(
        generations=GENERATIONS, extractions=[_extraction(1)],
        verifications=[_verification(["FABRICATED"])],
        judgments=[_judgment(0, "FABRICATED")], questions=QUESTIONS,
    )
    r = records[0]
    assert r["title"] == "Title 0"
    assert r["domain"] == "academic"
    assert r["output_excerpt"]
