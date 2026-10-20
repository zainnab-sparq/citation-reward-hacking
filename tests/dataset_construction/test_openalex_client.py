"""Tests for the OpenAlex client used in dataset construction.

OpenAlex is the no-API-key alternative to Semantic Scholar. We use it for:
 - obscure-but-real recruitment via citation-count filter [2, 50)
 - known-real moderately-known recruitment via [1000, 5000]
 - field-of-study sampling via concepts.id filter
 - title search for the negative-existence proof in the unanswerable cell

All tests use respx to mock HTTP. No live calls.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from src.dataset_construction.openalex_client import (
    OpenAlexClient,
    Work,
)


@pytest.mark.asyncio
async def test_search_by_concept_filters_citation_band():
    """A query for obscure-but-real papers must apply citation_count [2,50)
    AND publication_year cutoff AND concept filter — all in one request."""
    expected_url = "https://api.openalex.org/works"
    captured_params: dict = {}

    async def handler(request):
        captured_params.update(request.url.params)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "doi": "https://doi.org/10.1234/example",
                        "title": "An obscure NLP paper",
                        "publication_year": 2020,
                        "cited_by_count": 4,
                        "authorships": [
                            {"author": {"display_name": "Jane Doe"}}
                        ],
                        "primary_location": {
                            "source": {"display_name": "Workshop on Tiny Things"}
                        },
                    }
                ]
            },
        )

    with respx.mock as mock:
        mock.get(expected_url).mock(side_effect=handler)
        client = OpenAlexClient(mailto="researcher@example.com")
        works = await client.search_by_concept(
            concept_id="C41008148",  # Computer science
            citation_min=2,
            citation_max=49,
            year_max=2022,
            per_page=25,
        )

    assert "filter" in captured_params
    fil = captured_params["filter"]
    assert "cited_by_count:2-49" in fil
    assert "publication_year:<2023" in fil
    assert "concepts.id:C41008148" in fil
    assert captured_params.get("mailto") == "researcher@example.com"
    assert len(works) == 1
    assert works[0].title == "An obscure NLP paper"
    assert works[0].cited_by_count == 4
    assert works[0].first_author == "Jane Doe"
    assert works[0].venue == "Workshop on Tiny Things"


@pytest.mark.asyncio
async def test_search_by_title_returns_top_n_for_negative_existence():
    """Search by title fetches top-N candidate papers for the cosine-similarity
    arm of the unanswerable-cell negative-existence proof."""
    with respx.mock as mock:
        mock.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "https://openalex.org/W1",
                            "doi": None,
                            "title": "Some paper title",
                            "publication_year": 2019,
                            "cited_by_count": 12,
                            "authorships": [],
                            "primary_location": None,
                        },
                    ]
                },
            )
        )
        client = OpenAlexClient(mailto="r@example.com")
        works = await client.search_by_title("Imaginary attention paper", per_page=5)

    assert len(works) == 1
    assert works[0].id == "https://openalex.org/W1"
    assert works[0].venue is None
    assert works[0].first_author is None


@pytest.mark.asyncio
async def test_get_work_by_doi_returns_metadata():
    with respx.mock as mock:
        mock.get(
            "https://api.openalex.org/works/https://doi.org/10.48550/arXiv.1810.04805"
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "https://openalex.org/W2",
                    "doi": "https://doi.org/10.48550/arXiv.1810.04805",
                    "title": "BERT",
                    "publication_year": 2018,
                    "cited_by_count": 100000,
                    "authorships": [
                        {"author": {"display_name": "Jacob Devlin"}}
                    ],
                    "primary_location": {"source": {"display_name": "arXiv"}},
                },
            )
        )
        client = OpenAlexClient(mailto="r@example.com")
        work = await client.get_work_by_doi("10.48550/arXiv.1810.04805")

    assert work is not None
    assert work.cited_by_count == 100000
    assert work.first_author == "Jacob Devlin"


@pytest.mark.asyncio
async def test_get_work_by_doi_returns_none_on_404():
    with respx.mock as mock:
        mock.get(
            "https://api.openalex.org/works/https://doi.org/10.9999/fake"
        ).mock(return_value=httpx.Response(404))
        client = OpenAlexClient(mailto="r@example.com")
        work = await client.get_work_by_doi("10.9999/fake")
    assert work is None


@pytest.mark.asyncio
async def test_polite_pool_user_agent_includes_mailto():
    """OpenAlex 'polite pool' is requested via the User-Agent header containing
    a mailto address. Without it, requests share an anonymous queue with much
    lower throughput. We must send it on every request."""
    captured_headers: dict = {}

    async def handler(request):
        captured_headers.update(request.headers)
        return httpx.Response(200, json={"results": []})

    with respx.mock as mock:
        mock.get("https://api.openalex.org/works").mock(side_effect=handler)
        client = OpenAlexClient(mailto="researcher@example.com")
        await client.search_by_title("anything", per_page=1)

    ua = captured_headers.get("user-agent", "")
    assert "researcher@example.com" in ua


def test_work_filters_out_papers_missing_required_fields():
    """A search hit with no title is unusable for our purposes — drop it
    rather than carrying a None through downstream similarity checks."""
    parsed = Work.from_openalex({
        "id": "W3",
        "doi": None,
        "title": None,
        "publication_year": 2020,
        "cited_by_count": 5,
        "authorships": [],
        "primary_location": None,
    })
    assert parsed is None
