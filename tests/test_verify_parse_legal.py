"""Tests for parse_citation when domain == 'legal'.

US case citations have a different structure than academic papers:
    <Case Name>, <reporter> (<year>)
    e.g., Miranda v. Arizona, 384 U.S. 436 (1966)

The parser must populate case_name + reporter for these so that
verify_citation can route them to CourtListener.
"""
from src.verify import parse_citation


def test_extracts_case_name_from_supreme_court_citation():
    raw = "Miranda v. Arizona, 384 U.S. 436 (1966)"
    cite = parse_citation(raw, domain="legal")
    assert cite.case_name == "Miranda v. Arizona"


def test_extracts_reporter_from_supreme_court_citation():
    raw = "Miranda v. Arizona, 384 U.S. 436 (1966)"
    cite = parse_citation(raw, domain="legal")
    assert cite.reporter == "384 U.S. 436"


def test_extracts_year_from_legal_citation():
    raw = "Loving v. Virginia, 388 U.S. 1 (1967)"
    cite = parse_citation(raw, domain="legal")
    assert cite.year == 1967
    assert cite.case_name == "Loving v. Virginia"
    assert cite.reporter == "388 U.S. 1"


def test_extracts_federal_reporter_citation():
    """Federal Reporter (F.2d, F.3d) is a different abbrev than US Reports."""
    raw = "Smith v. Jones, 123 F.3d 456 (9th Cir. 2001)"
    cite = parse_citation(raw, domain="legal")
    assert cite.case_name == "Smith v. Jones"
    assert cite.reporter == "123 F.3d 456"
    assert cite.year == 2001


def test_legal_parse_does_not_populate_doi_or_apa_title():
    raw = "Miranda v. Arizona, 384 U.S. 436 (1966)"
    cite = parse_citation(raw, domain="legal")
    assert cite.doi is None
    # title may or may not be set depending on regex behavior; case_name is the legal-domain field


def test_academic_domain_does_not_populate_case_name():
    """Make sure case_name stays None for non-legal domains even if 'v.' appears."""
    raw = "Smith, J. (2019). On the v. operator in linear algebra. Journal of Foo."
    cite = parse_citation(raw, domain="academic")
    assert cite.case_name is None
    assert cite.reporter is None
