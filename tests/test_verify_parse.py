"""Tests for citation parsing — Citation extraction from raw text."""
from src.verify import parse_citation


def test_extracts_doi_from_apa_citation():
    raw = (
        "Sharma, M., Tong, M., Korbak, T., et al. (2023). "
        "Towards understanding sycophancy in language models. "
        "arXiv preprint. https://doi.org/10.48550/arXiv.2310.13548"
    )
    cite = parse_citation(raw, domain="academic")
    assert cite.doi == "10.48550/arXiv.2310.13548"


def test_extracts_year_from_apa_citation():
    raw = "Smith, J. (2019). Some paper title. Journal of Foo."
    cite = parse_citation(raw, domain="academic")
    assert cite.year == 2019


def test_extracts_title_from_apa_after_year():
    raw = "Smith, J. (2019). Neural attention in epidemiology. Journal of Foo, 12(3), 45-67."
    cite = parse_citation(raw, domain="academic")
    assert cite.title == "Neural attention in epidemiology"


def test_extracts_first_author_surname():
    raw = "Sharma, M., Tong, M., Korbak, T. (2023). Sycophancy. arXiv."
    cite = parse_citation(raw, domain="academic")
    assert cite.authors[0] == "Sharma"


def test_year_none_when_no_year_present():
    cite = parse_citation("Some unstructured text with no year.", domain="academic")
    assert cite.year is None
    assert cite.title is None
