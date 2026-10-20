"""Candidate recruiter for known-real and obscure-but-real cells via OpenAlex.

Maps (cell, domain) → OpenAlex filter list, issues a sample request, returns
Works. Random sampling via OpenAlex's `sample=N&seed=<int>` mode for
reproducibility.

Legal-domain recruitment is NOT handled here — US case law is not in OpenAlex.
See src/dataset_construction/courtlistener_recruiter.py for legal.

Unanswerable-cell construction is NOT handled here either — that's
src/dataset_construction/construct_unanswerable.py.
"""
from __future__ import annotations

from typing import Any

from src.dataset_construction.openalex_client import OpenAlexClient, Work

# OpenAlex concept IDs (verified via /concepts endpoint 2026-05-03):
#   C41008148  Computer science
#   C71924100  Medicine
_CONCEPT_ACADEMIC = "C41008148"
_CONCEPT_MEDICINE = "C71924100"
_COCHRANE_SOURCE_ID = "S4210172715"  # Cochrane Database of Systematic Reviews

# Single source of truth for cell × domain → OpenAlex filter spec.
# Mirror of src/dataset_construction/feasibility.py — keep in sync.
_CELL_FILTERS: dict[tuple[str, str], list[str]] = {
    # academic
    ("known_real_canonical", "academic"): [
        "cited_by_count:>9999",
        "publication_year:<2023",
        f"concepts.id:{_CONCEPT_ACADEMIC}",
    ],
    ("known_real_moderate", "academic"): [
        "cited_by_count:1000-5000",
        "publication_year:<2023",
        f"concepts.id:{_CONCEPT_ACADEMIC}",
    ],
    ("obscure_real", "academic"): [
        "cited_by_count:2-49",
        "publication_year:<2023",
        f"concepts.id:{_CONCEPT_ACADEMIC}",
    ],
    # medical
    ("known_real_cochrane", "medical"): [
        f"primary_location.source.id:{_COCHRANE_SOURCE_ID}",
        "publication_year:<2023",
        "cited_by_count:>49",  # exclude protocol-only / unread reviews
    ],
    ("known_real_non_cochrane", "medical"): [
        "cited_by_count:>499",
        "publication_year:<2023",
        f"concepts.id:{_CONCEPT_MEDICINE}",
        # Exclude Cochrane source from the non-Cochrane bucket so the 60/40
        # split doesn't double-count.
        f"primary_location.source.id:!{_COCHRANE_SOURCE_ID}",
    ],
    ("obscure_real", "medical"): [
        "cited_by_count:2-49",
        "publication_year:<2023",
        f"concepts.id:{_CONCEPT_MEDICINE}",
    ],
}


class UnsupportedCellDomainError(ValueError):
    pass


class CandidateRecruiter:
    def __init__(self, openalex: OpenAlexClient):
        self._oa = openalex

    async def recruit(
        self,
        cell: str,
        domain: str,
        n: int,
        seed: int = 42,
    ) -> list[Work]:
        key = (cell, domain)
        if key not in _CELL_FILTERS:
            raise UnsupportedCellDomainError(
                f"Cell {cell!r} for domain {domain!r} is not handled by OpenAlex "
                "recruiter. Legal goes through CourtListener; unanswerable goes "
                "through construct_unanswerable.py."
            )
        filters = _CELL_FILTERS[key]
        # OpenAlex caps per-page at 200; sample > 200 requires pagination which
        # we don't need for our 200-question targets.
        per_page = min(n, 200)
        params: dict[str, Any] = {
            "filter": ",".join(filters),
            "per-page": per_page,
            "sample": n,
            "seed": seed,
        }
        response = await self._oa._get("/works", params)
        response.raise_for_status()
        return self._oa._parse_results(response.json())
