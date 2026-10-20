"""Tests for src/extract_citations.py — LLM-assisted structured citation extraction.

The production extractor uses Sonnet 4.6 with tool-use to return structured
{authors, year, title, venue, doi, case_name, reporter} per citation. These
tests use a FakeExtractor that returns canned structured responses so the
suite stays hermetic and fast — no network, no API keys.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from src.extract_citations import (
    ExtractedCitation,
    FakeExtractorClient,
    extract_citations_llm,
)


@pytest.mark.asyncio
async def test_extracts_apa_citation_from_paragraph():
    text = (
        "The Transformer architecture was introduced in:\n"
        "Vaswani, A., Shazeer, N. (2017). Attention is all you need. "
        "Advances in Neural Information Processing Systems, 30."
    )
    fake = FakeExtractorClient(canned=[
        {
            "raw_span": "Vaswani, A., Shazeer, N. (2017). Attention is all you need. "
                        "Advances in Neural Information Processing Systems, 30.",
            "authors": ["Vaswani", "Shazeer"],
            "year": 2017,
            "title": "Attention is all you need",
            "venue": "Advances in Neural Information Processing Systems",
            "doi": None,
            "case_name": None,
            "reporter": None,
        }
    ])
    cites = await extract_citations_llm(text, domain="academic", client=fake)
    assert len(cites) == 1
    assert cites[0].authors == ("Vaswani", "Shazeer")
    assert cites[0].year == 2017
    assert "Attention is all you need" in cites[0].title


@pytest.mark.asyncio
async def test_extracts_mid_sentence_quoted_citation():
    """The bug from the Phase A pilot: regex missed citations embedded
    mid-sentence with quoted titles. The LLM extractor must catch them.
    """
    text = (
        "The Transformer architecture was introduced in 'Attention Is All You Need' "
        "by Vaswani et al. (2017), published in NeurIPS."
    )
    fake = FakeExtractorClient(canned=[
        {
            "raw_span": "'Attention Is All You Need' by Vaswani et al. (2017), published in NeurIPS",
            "authors": ["Vaswani"],
            "year": 2017,
            "title": "Attention Is All You Need",
            "venue": "NeurIPS",
            "doi": None,
            "case_name": None,
            "reporter": None,
        }
    ])
    cites = await extract_citations_llm(text, domain="academic", client=fake)
    assert len(cites) == 1
    assert cites[0].title == "Attention Is All You Need"
    assert cites[0].authors == ("Vaswani",)


@pytest.mark.asyncio
async def test_extracts_multiple_citations_in_one_response():
    text = (
        "Two relevant papers: Vaswani 2017 on attention, Devlin 2018 on BERT."
    )
    fake = FakeExtractorClient(canned=[
        {
            "raw_span": "Vaswani 2017 on attention",
            "authors": ["Vaswani"],
            "year": 2017,
            "title": "Attention is all you need",
            "venue": None,
            "doi": None,
            "case_name": None,
            "reporter": None,
        },
        {
            "raw_span": "Devlin 2018 on BERT",
            "authors": ["Devlin"],
            "year": 2018,
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "venue": None,
            "doi": "10.48550/arXiv.1810.04805",
            "case_name": None,
            "reporter": None,
        },
    ])
    cites = await extract_citations_llm(text, domain="academic", client=fake)
    assert len(cites) == 2
    assert cites[0].authors == ("Vaswani",)
    assert cites[1].authors == ("Devlin",)
    assert cites[1].doi == "10.48550/arXiv.1810.04805"


@pytest.mark.asyncio
async def test_extracts_legal_case_with_reporter():
    text = (
        "The landmark case is Miranda v. Arizona, 384 U.S. 436 (1966), which "
        "established the requirement to inform suspects of their rights."
    )
    fake = FakeExtractorClient(canned=[
        {
            "raw_span": "Miranda v. Arizona, 384 U.S. 436 (1966)",
            "authors": [],
            "year": 1966,
            "title": "Miranda v. Arizona",
            "venue": "U.S. Reports",
            "doi": None,
            "case_name": "Miranda v. Arizona",
            "reporter": "384 U.S. 436",
        }
    ])
    cites = await extract_citations_llm(text, domain="legal", client=fake)
    assert len(cites) == 1
    assert cites[0].case_name == "Miranda v. Arizona"
    assert cites[0].reporter == "384 U.S. 436"
    assert cites[0].year == 1966


@pytest.mark.asyncio
async def test_extracts_legal_case_without_title_field():
    """The Haiku extractor in production omits the `title` field on legal
    citations (it relies on case_name + reporter). The parser must accept
    this rather than KeyError on the missing key. Repro for the Phase B
    Step 1 verify-stage failure 2026-06-07: 34/13912 citations had no
    `title` key and all were legal cases.
    """
    fake = FakeExtractorClient(canned=[
        {
            "raw_span": "Denver Homeless Out Loud v. Denver, Colorado, 32 F.4th 1259 (2022)",
            "case_name": "Denver Homeless Out Loud v. Denver, Colorado",
            "reporter": "32 F.4th 1259",
            "year": 2022,
            "authors": [],
        }
    ])
    cites = await extract_citations_llm(text="...", domain="legal", client=fake)
    assert len(cites) == 1
    assert cites[0].case_name == "Denver Homeless Out Loud v. Denver, Colorado"
    assert cites[0].title is None
    assert cites[0].reporter == "32 F.4th 1259"


@pytest.mark.asyncio
async def test_returns_empty_list_when_no_citations():
    text = "I am not certain a specific source exists for this claim."
    fake = FakeExtractorClient(canned=[])  # LLM correctly identifies zero citations
    cites = await extract_citations_llm(text, domain="academic", client=fake)
    assert cites == []


@pytest.mark.asyncio
async def test_extracted_citation_converts_to_verify_format():
    """ExtractedCitation must map cleanly to the verify.Citation type so
    the existing verification pipeline can consume extractor output.
    """
    extracted = ExtractedCitation(
        raw_span="Vaswani et al. (2017). Attention. NeurIPS.",
        authors=("Vaswani",),
        year=2017,
        title="Attention",
        venue="NeurIPS",
        doi=None,
        case_name=None,
        reporter=None,
    )
    verify_citation = extracted.to_verify_citation(domain="academic")
    assert verify_citation.title == "Attention"
    assert verify_citation.year == 2017
    assert verify_citation.authors == ("Vaswani",)
    assert verify_citation.domain == "academic"


def test_arxiv_id_in_raw_span_populates_doi_if_missing():
    """Models often write 'arXiv (1810.04805)' or 'arXiv:1810.04805' rather than
    a full DOI. Without recognizing the arXiv ID, verification falls back to
    flaky title search and false-fabricates real arXiv papers."""
    extracted = ExtractedCitation.from_dict({
        "raw_span": "BERT was introduced by Devlin et al. (2018). arXiv (1810.04805).",
        "title": "BERT",
        "authors": ["Devlin"],
        "year": 2018,
        "venue": None,
        "doi": None,
        "case_name": None,
        "reporter": None,
    })
    assert extracted.doi == "10.48550/arXiv.1810.04805"


def test_arxiv_id_does_not_overwrite_existing_doi():
    extracted = ExtractedCitation.from_dict({
        "raw_span": "Some text 1810.04805 more text",
        "title": "X",
        "authors": [],
        "year": None,
        "venue": None,
        "doi": "10.1234/already.set",
        "case_name": None,
        "reporter": None,
    })
    assert extracted.doi == "10.1234/already.set"


def test_arxiv_colon_form_also_recognized():
    extracted = ExtractedCitation.from_dict({
        "raw_span": "Paper at arXiv:2310.13548 published 2023",
        "title": "X",
        "authors": [],
        "year": 2023,
        "venue": None,
        "doi": None,
        "case_name": None,
        "reporter": None,
    })
    assert extracted.doi == "10.48550/arXiv.2310.13548"


@pytest.mark.asyncio
async def test_handles_malformed_llm_json_gracefully():
    """If the LLM returns junk (or the tool-use call fails), we should
    return [] rather than crash the pipeline.
    """
    fake = FakeExtractorClient(canned=None, raise_on_call=ValueError("malformed JSON"))
    cites = await extract_citations_llm(
        "Some text.", domain="academic", client=fake
    )
    assert cites == []


# ---------- Abstention guard (Step 0 anomaly fix C addon) ----------


def test_extraction_prompt_instructs_to_skip_refusal_text():
    """The extractor prompt must tell Sonnet not to extract citations that
    appear inside refusal/abstention/uncertainty language. Step 0 showed
    Haiku refusing 'United States v. Chen' but naming the case in its refusal;
    the extractor then pulled it as a citation and CourtListener (pre-fix)
    matched it to an unrelated real Chen case. The prompt-level guard removes
    the ghost citation upstream. Pre-reg §11 amendment #3."""
    from src.extract_citations import _EXTRACTION_PROMPT

    lowered = _EXTRACTION_PROMPT.lower()
    # Must instruct skipping citations from refusal contexts.
    assert "do not extract" in lowered or "skip" in lowered
    # Must reference the refusal / abstention / uncertainty signal class.
    assert any(
        kw in lowered
        for kw in (
            "refusal",
            "refuse",
            "abstention",
            "abstain",
            "uncertain",
            "i cannot",
            "i don't have",
            "i do not have",
            "no such",
            "not aware",
        )
    )
    # Must permit extraction when the citation IS asserted (so we don't lose
    # real signal): the prompt should clarify the rule applies to refusal
    # text only, not to confidently asserted citations.
    assert "asserted" in lowered or "claim" in lowered or "answers" in lowered or "answering" in lowered
