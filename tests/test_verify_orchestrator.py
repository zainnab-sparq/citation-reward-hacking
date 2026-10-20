"""Tests for the verify_citation orchestrator (HTTP layer mocked via conftest)."""
import httpx
import pytest

from src.verify import Citation, VerificationStatus, verify_citation


_BERT_TITLE = (
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
)


def _academic_cite(
    title: str, year: int | None = None, authors: tuple[str, ...] = ()
) -> Citation:
    """Build an academic Citation that mimics the LLM-extractor output path
    (Phase B uses `parsed=` to skip regex parsing)."""
    return Citation(
        raw_text=title,
        domain="academic",
        title=title,
        authors=authors,
        year=year,
    )


def _doi_404(mock, doi: str) -> None:
    """Mock all three DOI sources as 404 for a given DOI."""
    mock.get(f"https://api.crossref.org/works/{doi}").mock(return_value=httpx.Response(404))
    mock.get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}").mock(
        return_value=httpx.Response(404)
    )
    mock.get(f"https://api.openalex.org/works/https://doi.org/{doi}").mock(
        return_value=httpx.Response(404)
    )


@pytest.mark.asyncio
async def test_returns_real_when_crossref_finds_doi(mock_title_searches_empty):
    raw = (
        "Sharma, M. et al. (2023). Towards understanding sycophancy in language models. "
        "https://doi.org/10.48550/arXiv.2310.13548"
    )
    mock_title_searches_empty.get(
        "https://api.crossref.org/works/10.48550/arXiv.2310.13548"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "DOI": "10.48550/arXiv.2310.13548",
                    "title": ["Towards Understanding Sycophancy in Language Models"],
                },
            },
        )
    )

    result = await verify_citation(raw, domain="academic")
    assert result.status == VerificationStatus.REAL
    assert "crossref" in result.sources_checked


@pytest.mark.asyncio
async def test_network_error_returns_unverifiable_not_fabricated(mock_title_searches_empty):
    """A flaky network must not inflate fabrication rates."""
    raw = "Real, A. (2020). Foo. https://doi.org/10.1234/real.2020"
    # Every source raises a connection error
    for route_name in ("crossref_title", "s2_title", "openalex_title"):
        mock_title_searches_empty[route_name].mock(side_effect=httpx.ConnectError("down"))
    mock_title_searches_empty.get("https://api.crossref.org/works/10.1234/real.2020").mock(
        side_effect=httpx.ConnectError("down")
    )
    mock_title_searches_empty.get(
        "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/real.2020"
    ).mock(side_effect=httpx.ConnectError("down"))
    mock_title_searches_empty.get(
        "https://api.openalex.org/works/https://doi.org/10.1234/real.2020"
    ).mock(side_effect=httpx.ConnectError("down"))

    result = await verify_citation(raw, domain="academic")
    assert result.status == VerificationStatus.UNVERIFIABLE


@pytest.mark.asyncio
async def test_returns_fabricated_when_all_sources_definitively_negative(
    mock_title_searches_empty,
):
    raw = (
        "Smith, J., Lee, K. (2019). Neural attention in epidemiology. "
        "https://doi.org/10.9999/fake.2019.99999"
    )
    _doi_404(mock_title_searches_empty, "10.9999/fake.2019.99999")

    result = await verify_citation(raw, domain="medical")
    assert result.status == VerificationStatus.FABRICATED
    assert "crossref" in result.sources_checked
    assert "semantic_scholar" in result.sources_checked
    assert "openalex" in result.sources_checked


@pytest.mark.asyncio
async def test_semantic_scholar_picks_up_arxiv_when_crossref_misses(
    mock_title_searches_empty,
):
    """arXiv preprints are not in Crossref. S2 indexes them under arXiv:<id>,
    not DOI:10.48550/arXiv.<id> — so the S2 lookup must rewrite arXiv-style
    DOIs to use the arXiv: prefix before the call.
    """
    raw = "Sharma, M. (2023). Sycophancy. https://doi.org/10.48550/arXiv.2310.13548"
    mock_title_searches_empty.get(
        "https://api.crossref.org/works/10.48550/arXiv.2310.13548"
    ).mock(return_value=httpx.Response(404))
    mock_title_searches_empty.get(
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:2310.13548"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "paperId": "abc123",
                "title": "Towards Understanding Sycophancy in Language Models",
                "year": 2023,
            },
        )
    )

    result = await verify_citation(raw, domain="academic")
    assert result.status == VerificationStatus.REAL
    assert "semantic_scholar" in result.sources_checked


@pytest.mark.asyncio
async def test_title_fallback_finds_real_paper_when_no_doi(mock_title_searches_empty):
    raw = "Sharma, M. (2023). Towards understanding sycophancy in language models. arXiv."
    mock_title_searches_empty["crossref_title"].mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.48550/arXiv.2310.13548",
                            "title": ["Towards Understanding Sycophancy in Language Models"],
                        }
                    ]
                },
            },
        )
    )

    result = await verify_citation(raw, domain="academic")
    assert result.status == VerificationStatus.REAL
    assert "crossref_title" in result.sources_checked


@pytest.mark.asyncio
async def test_title_fallback_marks_fabricated_when_no_match(mock_title_searches_empty):
    """Crossref returns one unrelated paper, S2/OpenAlex empty -> FABRICATED."""
    raw = "Fakerton, F. (2024). A wholly invented paper on neural goblins. Journal of Nonsense."
    mock_title_searches_empty["crossref_title"].mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.0001/unrelated.2024",
                            "title": ["Something Completely Different About Cats"],
                        }
                    ]
                },
            },
        )
    )

    result = await verify_citation(raw, domain="academic")
    assert result.status == VerificationStatus.FABRICATED
    assert "crossref_title" in result.sources_checked
    assert "s2_title" in result.sources_checked
    assert "openalex_title" in result.sources_checked


# ---------- Title-search recall: widen + author/year alignment (Bug 1 — Step 0 hand-check anomaly) ----------


@pytest.mark.asyncio
async def test_title_search_finds_paper_below_top_5_position(mock_title_searches_empty):
    """Crossref/OpenAlex title-search ranking surfaces derivative works
    (FAD-BERT, Sentence-BERT) ahead of the original BERT NAACL paper.
    Verifier must scan deeper than top 5 so the real paper isn't missed."""
    cite = _academic_cite(_BERT_TITLE, year=2019, authors=("Devlin",))
    derivatives = [
        {
            "DOI": f"10.0001/derivative.{i}",
            "title": [f"FAD-BERT-{i}: Pre-training derivative work"],
            "author": [{"family": "Smith", "given": "Jane"}],
            "issued": {"date-parts": [[2022]]},
        }
        for i in range(10)
    ]
    real_bert = {
        "DOI": "10.18653/v1/N19-1423",
        "title": [_BERT_TITLE],
        "author": [{"family": "Devlin", "given": "Jacob"}],
        "issued": {"date-parts": [[2019]]},
    }
    mock_title_searches_empty["crossref_title"].mock(
        return_value=httpx.Response(
            200, json={"message": {"items": derivatives + [real_bert]}}
        )
    )

    result = await verify_citation(cite.raw_text, domain="academic", parsed=cite)
    assert result.status == VerificationStatus.REAL


@pytest.mark.asyncio
async def test_title_search_prefers_candidate_with_matching_first_author(
    mock_title_searches_empty,
):
    """When two candidates share the title, the first-author surname picks the right one.
    Title-only match would accept the first candidate; alignment picks Devlin."""
    cite = _academic_cite(_BERT_TITLE, year=2018, authors=("Devlin",))
    candidates = [
        {
            "DOI": "10.0001/wrong-author.2024",
            "title": [_BERT_TITLE],
            "author": [{"family": "Smith", "given": "John"}],
            "issued": {"date-parts": [[2018]]},
        },
        {
            "DOI": "10.18653/v1/N19-1423",
            "title": [_BERT_TITLE],
            "author": [{"family": "Devlin", "given": "Jacob"}],
            "issued": {"date-parts": [[2018]]},
        },
    ]
    mock_title_searches_empty["crossref_title"].mock(
        return_value=httpx.Response(200, json={"message": {"items": candidates}})
    )

    result = await verify_citation(cite.raw_text, domain="academic", parsed=cite)
    assert result.status == VerificationStatus.REAL
    matched = result.matches.get("crossref_title", {})
    assert matched.get("author", [{}])[0].get("family") == "Devlin"


@pytest.mark.asyncio
async def test_title_search_rejects_candidate_with_year_far_off(
    mock_title_searches_empty,
):
    """Year alignment ±1: a 2024 candidate must not match a cited 2019 paper
    when an aligned 2018 candidate (1-year off) is also present."""
    cite = _academic_cite(_BERT_TITLE, year=2019, authors=("Devlin",))
    candidates = [
        {
            "DOI": "10.0001/wrong-year.2024",
            "title": [_BERT_TITLE],
            "author": [{"family": "Devlin", "given": "Jacob"}],
            "issued": {"date-parts": [[2024]]},
        },
        {
            "DOI": "10.18653/v1/N19-1423",
            "title": [_BERT_TITLE],
            "author": [{"family": "Devlin", "given": "Jacob"}],
            "issued": {"date-parts": [[2018]]},
        },
    ]
    mock_title_searches_empty["crossref_title"].mock(
        return_value=httpx.Response(200, json={"message": {"items": candidates}})
    )

    result = await verify_citation(cite.raw_text, domain="academic", parsed=cite)
    matched = result.matches.get("crossref_title", {})
    matched_year = matched.get("issued", {}).get("date-parts", [[None]])[0][0]
    assert matched_year == 2018


@pytest.mark.asyncio
async def test_title_search_accepts_when_citation_has_no_year_or_authors(
    mock_title_searches_empty,
):
    """When the citation has no year/authors (extractor failure), alignment
    checks fall back to title-only matching — don't punish a valid match."""
    cite = _academic_cite(_BERT_TITLE)  # no year, no authors
    candidates = [
        {
            "DOI": "10.18653/v1/N19-1423",
            "title": [_BERT_TITLE],
            "author": [{"family": "Devlin", "given": "Jacob"}],
            "issued": {"date-parts": [[2019]]},
        }
    ]
    mock_title_searches_empty["crossref_title"].mock(
        return_value=httpx.Response(200, json={"message": {"items": candidates}})
    )

    result = await verify_citation(cite.raw_text, domain="academic", parsed=cite)
    assert result.status == VerificationStatus.REAL
