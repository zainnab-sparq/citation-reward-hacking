"""Tests for pilot.py — citation extraction from free-text responses."""
from src.pilot import extract_citations


def test_extracts_apa_citation_from_paragraph():
    text = (
        "The Transformer architecture was introduced in:\n"
        "Vaswani, A., Shazeer, N. (2017). Attention is all you need. "
        "Advances in Neural Information Processing Systems, 30."
    )
    cites = extract_citations(text)
    assert len(cites) == 1
    assert "Vaswani" in cites[0]
    assert "(2017)" in cites[0]


def test_extracts_two_citations_from_two_paragraphs():
    text = (
        "Vaswani et al. (2017). Attention is all you need. NeurIPS.\n\n"
        "Devlin et al. (2018). BERT. https://doi.org/10.48550/arXiv.1810.04805"
    )
    cites = extract_citations(text)
    assert len(cites) == 2


def test_extracts_numbered_list_items():
    text = (
        "Two relevant papers:\n\n"
        "1. Vaswani et al. (2017). Attention is all you need. NeurIPS.\n"
        "2. Devlin et al. (2018). BERT. https://doi.org/10.48550/arXiv.1810.04805"
    )
    cites = extract_citations(text)
    assert len(cites) == 2
    assert "Vaswani" in cites[0]
    assert "Devlin" in cites[1]


def test_extracts_bracketed_legal_citations():
    text = (
        "[1] Miranda v. Arizona, 384 U.S. 436 (1966) "
        "[2] Loving v. Virginia, 388 U.S. 1 (1967)"
    )
    cites = extract_citations(text)
    assert len(cites) == 2
    assert "Miranda v. Arizona" in cites[0]
    assert "Loving v. Virginia" in cites[1]


def test_no_citations_in_plain_prose():
    text = "The Transformer is a neural network architecture. It uses self-attention."
    assert extract_citations(text) == []
