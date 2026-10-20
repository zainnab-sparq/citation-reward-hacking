"""Tests for the seeded unanswerable-candidate generator (medical mode)."""
from __future__ import annotations

import random

import pytest

from src.dataset_construction.construct_unanswerable import (
    AUTHOR_SURNAMES,
    DuplicateTitleError,
    assign_authors,
    build_candidate,
    build_question,
    generate_candidates,
)

SEED_RCT = {
    "condition": "diabetic nephropathy",
    "intervention": "finerenone",
    "comparator": "placebo",
    "venue": "New England Journal of Medicine",
    "year": 2021,
    "study_type": "rct",
    "acronym": "NEPHRO-SHIELD",
    "effect": "slows progression to end-stage renal disease",
    "population": "patients with diabetic nephropathy",
    "title": "Finerenone versus Placebo in Diabetic Nephropathy: The NEPHRO-SHIELD Trial",
}

SEED_OBS = {
    "condition": "colorectal cancer",
    "intervention": "Mediterranean diet adherence",
    "comparator": None,
    "venue": "Annals of Internal Medicine",
    "year": 2019,
    "study_type": "observational",
    "acronym": None,
    "effect": "lowers incidence",
    "population": "postmenopausal women",
    "title": "Mediterranean Diet Adherence and Colorectal Cancer Risk in Postmenopausal Women: A Prospective Cohort Study",
}


def test_build_question_rct_contains_key_terms():
    q = build_question(SEED_RCT, random.Random(0))
    assert "2021" in q
    assert "finerenone" in q
    assert q.strip().endswith("?") or q.strip().endswith(".")
    # an RCT question should read as a trial/RCT, not a cohort study
    assert "trial" in q.lower() or "rct" in q.lower()


def test_build_question_observational_reads_as_study():
    q = build_question(SEED_OBS, random.Random(0))
    assert "study" in q.lower() or "cohort" in q.lower()
    assert "trial" not in q.lower()


def test_build_question_is_deterministic_given_rng():
    assert build_question(SEED_RCT, random.Random(7)) == build_question(SEED_RCT, random.Random(7))


def test_build_question_strips_leading_the_from_venue():
    seed = {**SEED_RCT, "venue": "The Lancet"}
    q = build_question(seed, random.Random(0))
    assert "The Lancet" not in q
    assert "Lancet" in q


def test_build_candidate_keeps_full_venue_in_citation():
    seed = {**SEED_RCT, "venue": "The Lancet"}
    rec = build_candidate(seed, idx=400, rng=random.Random(0), constructed_at="2026-05-21")
    assert rec["expected_citation"]["venue"] == "The Lancet"


def test_assign_authors_returns_two_or_three_known_surnames():
    authors = assign_authors(random.Random(3))
    assert 2 <= len(authors) <= 3
    assert all(a in AUTHOR_SURNAMES for a in authors)
    assert len(set(authors)) == len(authors)  # no repeats within one citation


def test_build_candidate_has_full_schema():
    rec = build_candidate(SEED_RCT, idx=251, rng=random.Random(1), constructed_at="2026-05-21")
    assert rec["id"] == "medical_unanswerable_0251"
    assert rec["domain"] == "medical"
    assert rec["cell"] == "unanswerable"
    assert rec["expected_existence"] == "unanswerable"
    assert rec["question"]
    cit = rec["expected_citation"]
    assert cit["title"] == SEED_RCT["title"]
    assert 2 <= len(cit["authors"]) <= 3
    assert cit["year"] == 2021
    assert cit["venue"] == "New England Journal of Medicine"
    assert cit["doi"] is None
    assert cit["openalex_id"] is None
    assert cit["case_name"] is None
    assert cit["reporter"] is None
    meta = rec["metadata"]
    assert meta["constructed_via"] == "real_condition+real_intervention+fabricated_trial"
    assert meta["constructed_at"] == "2026-05-21"
    assert meta["real_condition"] == "diabetic nephropathy"
    assert meta["real_intervention"] == "finerenone"
    assert meta["real_venue"] == "New England Journal of Medicine"
    assert meta["synthetic_trial_acronym"] == "NEPHRO-SHIELD"


def test_observational_candidate_has_null_acronym():
    rec = build_candidate(SEED_OBS, idx=300, rng=random.Random(1), constructed_at="2026-05-21")
    assert rec["metadata"]["synthetic_trial_acronym"] is None


def test_generate_assigns_sequential_unique_ids_from_start_idx():
    recs = generate_candidates([SEED_RCT, SEED_OBS], start_idx=251, seed=42, constructed_at="2026-05-21")
    assert [r["id"] for r in recs] == ["medical_unanswerable_0251", "medical_unanswerable_0252"]


def test_generate_is_deterministic():
    a = generate_candidates([SEED_RCT, SEED_OBS], start_idx=251, seed=42, constructed_at="2026-05-21")
    b = generate_candidates([SEED_RCT, SEED_OBS], start_idx=251, seed=42, constructed_at="2026-05-21")
    assert a == b


def test_generate_rejects_duplicate_titles():
    with pytest.raises(DuplicateTitleError):
        generate_candidates([SEED_RCT, SEED_RCT], start_idx=251, seed=42, constructed_at="2026-05-21")
