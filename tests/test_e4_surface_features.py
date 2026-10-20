"""Tests for E4's four hand-coded surface features.

The features are the pre-registered ones (§5 E4): citation count, format
conformity, plausible-author-name density, plausible-venue-name density. The
tests that carry the most weight are the ones asserting what the features must
NOT be able to see, because E4's claim depends on the judge being blind to
whether a citation exists.
"""
from __future__ import annotations

import pytest

from src.analysis.e4_surface_features import (
    FEATURE_NAMES,
    citation_count,
    format_conformity,
    is_plausible_author_name,
    is_plausible_venue_name,
    plausible_author_density,
    plausible_venue_density,
    surface_features,
)

REAL_ACADEMIC = {
    "raw_span": "Cronbach, L. J., & Meehl, P. E. (1955). Construct validity in psychological tests. Psychological Bulletin, 52(4), 281-302.",
    "title": "Construct validity in psychological tests",
    "authors": ["Cronbach", "L. J.", "Meehl", "P. E."],
    "venue": "Psychological Bulletin",
    "year": 1955,
}
FABRICATED_ACADEMIC = {
    "raw_span": "Kaminski, R. T. (2019). Adaptive resonance in distributed sensor arrays. Journal of Applied Sensing, 14(2), 77-95.",
    "title": "Adaptive resonance in distributed sensor arrays",
    "authors": ["Kaminski", "R. T."],
    "venue": "Journal of Applied Sensing",
    "year": 2019,
}
BARE_CITATION = {"raw_span": "the 2020 Lancet trial", "title": "the 2020 Lancet trial"}
LEGAL_CITATION = {
    "raw_span": "Polaris Powerled Technologies, LLC v. Samsung Electronics America, Inc., 386 F. Supp. 3d 760 (N.D. Tex. 2019)",
    "case_name": "Polaris Powerled Technologies, LLC v. Samsung Electronics America, Inc.",
    "reporter": "386 F. Supp. 3d 760",
    "title": "Polaris Powerled Technologies, LLC v. Samsung Electronics America, Inc.",
    "venue": "N.D. Tex.",
    "year": 2019,
}


class TestBlindnessToExistence:
    """E4's whole claim is that a judge with no access to existence still prefers
    post-trained output. A feature that can separate a real citation from a
    fabricated one of identical surface form would void that claim."""

    def test_real_and_fabricated_citations_of_equal_form_score_identically(self):
        assert surface_features([REAL_ACADEMIC]) == surface_features([FABRICATED_ACADEMIC])

    def test_features_ignore_verification_fields_if_present(self):
        # Nothing downstream should ever pass these in, but if a caller merges a
        # verified record by mistake the features must not start reading them.
        contaminated = dict(REAL_ACADEMIC, status="FABRICATED", judge_status="FABRICATED")
        assert surface_features([contaminated]) == surface_features([REAL_ACADEMIC])


class TestCitationCount:
    def test_counts_citations(self):
        assert citation_count([REAL_ACADEMIC, BARE_CITATION]) == 2

    def test_no_citations_is_zero(self):
        assert citation_count([]) == 0


class TestFormatConformity:
    def test_fully_slotted_citation_scores_one(self):
        assert format_conformity([REAL_ACADEMIC]) == pytest.approx(1.0)

    def test_title_only_citation_scores_one_quarter(self):
        assert format_conformity([BARE_CITATION]) == pytest.approx(0.25)

    def test_legal_reporter_and_case_name_fill_the_venue_and_attribution_slots(self):
        # Legal citations carry no `authors`; the slot is filled by `case_name`,
        # and `reporter` stands in for `venue`. Without the alternates every
        # legal generation would look malformed regardless of model.
        assert format_conformity([LEGAL_CITATION]) == pytest.approx(1.0)

    def test_averages_over_citations(self):
        assert format_conformity([REAL_ACADEMIC, BARE_CITATION]) == pytest.approx(0.625)

    def test_no_citations_scores_zero(self):
        assert format_conformity([]) == 0.0

    def test_empty_and_missing_fields_are_treated_alike(self):
        blank = {"title": "T", "authors": [], "venue": "", "year": None}
        assert format_conformity([blank]) == pytest.approx(0.25)


class TestPlausibleAuthorName:
    @pytest.mark.parametrize(
        "name", ["Cronbach", "L. J.", "Meehl, P. E.", "van der Waals", "O'Brien", "Muller-Lyer"]
    )
    def test_accepts_name_shaped_strings(self, name):
        assert is_plausible_author_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "   ",
            "et al.",
            "2019",
            "Smith 2019",
            "the authors of the study",
            "A" * 80,
        ],
    )
    def test_rejects_non_name_strings(self, name):
        assert not is_plausible_author_name(name)

    def test_density_is_the_fraction_of_citations_with_a_usable_author(self):
        assert plausible_author_density([REAL_ACADEMIC, BARE_CITATION]) == pytest.approx(0.5)

    def test_one_usable_element_carries_the_citation(self):
        # The extractor sometimes splits a name into surname and initials, so a
        # citation counts as attributed if ANY element is name-shaped.
        split = {"title": "T", "authors": ["Smith", "J"]}
        assert plausible_author_density([split]) == pytest.approx(1.0)

    def test_authors_given_as_a_bare_string_are_accepted(self):
        assert plausible_author_density([{"title": "T", "authors": "Cronbach, L. J."}]) == pytest.approx(1.0)

    def test_no_citations_scores_zero(self):
        assert plausible_author_density([]) == 0.0


class TestPlausibleVenueName:
    @pytest.mark.parametrize(
        "venue",
        [
            "Psychological Bulletin",
            "Journal of Applied Sensing",
            "Proceedings of the National Academy of Sciences",
            "Nature",
            "JAMA",
            "N.D. Tex.",
            "F. Supp. 3d",
        ],
    )
    def test_accepts_venue_shaped_strings(self, venue):
        assert is_plausible_venue_name(venue)

    @pytest.mark.parametrize("venue", ["", "   ", "52", "a journal somewhere", "?"])
    def test_rejects_non_venue_strings(self, venue):
        assert not is_plausible_venue_name(venue)

    def test_reporter_stands_in_when_venue_is_absent(self):
        legal_no_venue = dict(LEGAL_CITATION)
        del legal_no_venue["venue"]
        assert plausible_venue_density([legal_no_venue]) == pytest.approx(1.0)

    def test_density_is_the_fraction_of_citations_with_a_usable_venue(self):
        assert plausible_venue_density([REAL_ACADEMIC, BARE_CITATION]) == pytest.approx(0.5)

    def test_no_citations_scores_zero(self):
        assert plausible_venue_density([]) == 0.0


class TestSurfaceFeatureVector:
    def test_returns_exactly_the_four_registered_features(self):
        assert tuple(surface_features([REAL_ACADEMIC])) == FEATURE_NAMES
        assert len(FEATURE_NAMES) == 4

    def test_all_values_are_floats(self):
        # The logistic baseline and the DeBERTa rendering both consume this
        # vector; an int count leaking through changes sklearn's dtype handling.
        assert all(isinstance(v, float) for v in surface_features([REAL_ACADEMIC]).values())

    def test_a_generation_with_no_citations_is_all_zeros(self):
        assert set(surface_features([]).values()) == {0.0}

    def test_none_is_accepted_as_no_citations(self):
        # Extraction records carry `citations: null` when the model cited nothing.
        assert surface_features(None) == surface_features([])

    def test_count_is_not_normalised(self):
        # The count-matched control subsets on raw counts, so the feature must
        # stay on the raw scale rather than being squashed to a rate.
        assert surface_features([REAL_ACADEMIC] * 7)["citation_count"] == 7.0
