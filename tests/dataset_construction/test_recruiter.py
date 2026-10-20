"""Tests for the OpenAlex-based candidate recruiter."""
from __future__ import annotations

import httpx
import pytest
import respx

from src.dataset_construction.recruiter import (
    CandidateRecruiter,
    UnsupportedCellDomainError,
)
from src.dataset_construction.openalex_client import OpenAlexClient, Work


@pytest.fixture
def fake_oa_response():
    return {
        "results": [
            {
                "id": f"https://openalex.org/W{i}",
                "doi": None,
                "title": f"Paper {i}",
                "publication_year": 2020,
                "cited_by_count": 25,
                "authorships": [{"author": {"display_name": f"Author {i}"}}],
                "primary_location": {"source": {"display_name": "Some Venue"}},
            }
            for i in range(10)
        ]
    }


@pytest.mark.asyncio
async def test_recruits_academic_obscure_real_with_correct_filters(fake_oa_response):
    captured_params: dict = {}

    async def handler(request):
        captured_params.update(request.url.params)
        return httpx.Response(200, json=fake_oa_response)

    with respx.mock as mock:
        mock.get("https://api.openalex.org/works").mock(side_effect=handler)
        oa = OpenAlexClient(mailto="r@example.com")
        recruiter = CandidateRecruiter(oa)
        works = await recruiter.recruit(
            cell="obscure_real",
            domain="academic",
            n=10,
            seed=42,
        )

    fil = captured_params["filter"]
    assert "cited_by_count:2-49" in fil
    assert "publication_year:<2023" in fil
    assert "concepts.id:C41008148" in fil  # CS concept
    assert captured_params.get("sample") == "10"
    assert captured_params.get("seed") == "42"
    assert len(works) == 10


@pytest.mark.asyncio
async def test_recruits_academic_known_real_canonical_with_high_cite_filter(fake_oa_response):
    captured_params: dict = {}

    async def handler(request):
        captured_params.update(request.url.params)
        return httpx.Response(200, json=fake_oa_response)

    with respx.mock as mock:
        mock.get("https://api.openalex.org/works").mock(side_effect=handler)
        oa = OpenAlexClient(mailto="r@example.com")
        recruiter = CandidateRecruiter(oa)
        await recruiter.recruit(
            cell="known_real_canonical",
            domain="academic",
            n=10,
            seed=1,
        )

    fil = captured_params["filter"]
    assert "cited_by_count:>9999" in fil
    assert "publication_year:<2023" in fil
    assert "concepts.id:C41008148" in fil


@pytest.mark.asyncio
async def test_recruits_medical_cochrane_with_source_id_filter(fake_oa_response):
    captured_params: dict = {}

    async def handler(request):
        captured_params.update(request.url.params)
        return httpx.Response(200, json=fake_oa_response)

    with respx.mock as mock:
        mock.get("https://api.openalex.org/works").mock(side_effect=handler)
        oa = OpenAlexClient(mailto="r@example.com")
        recruiter = CandidateRecruiter(oa)
        await recruiter.recruit(
            cell="known_real_cochrane",
            domain="medical",
            n=10,
        )

    fil = captured_params["filter"]
    assert "primary_location.source.id:S4210172715" in fil
    assert "publication_year:<2023" in fil
    assert "cited_by_count:>49" in fil


@pytest.mark.asyncio
async def test_unsupported_cell_domain_combo_raises():
    oa = OpenAlexClient(mailto="r@example.com")
    recruiter = CandidateRecruiter(oa)
    with pytest.raises(UnsupportedCellDomainError):
        await recruiter.recruit(cell="obscure_real", domain="legal", n=5)


@pytest.mark.asyncio
async def test_unknown_cell_raises():
    oa = OpenAlexClient(mailto="r@example.com")
    recruiter = CandidateRecruiter(oa)
    with pytest.raises(UnsupportedCellDomainError):
        await recruiter.recruit(cell="bogus_cell", domain="academic", n=5)
