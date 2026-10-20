"""Async OpenAlex client for dataset construction.

Replaces Semantic Scholar (which has an API-key waiting period). OpenAlex
covers the same metadata field with no waiting period and a 100K/day
anonymous polite-pool limit when a `mailto:` is supplied via User-Agent.

Used for:
  - obscure-but-real recruitment via cited_by_count band [2, 50)
  - moderately-known recruitment via [1000, 5000]
  - field-of-study sampling via concepts.id filter
  - title search for the unanswerable-cell negative-existence proof
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

_OPENALEX_BASE = "https://api.openalex.org"
_DEFAULT_TIMEOUT = httpx.Timeout(15.0)


@dataclass(frozen=True)
class Work:
    """OpenAlex Work (paper / report / book) with the fields we use downstream."""
    id: str
    title: str
    publication_year: int | None
    cited_by_count: int
    doi: str | None
    first_author: str | None
    venue: str | None

    @classmethod
    def from_openalex(cls, payload: dict[str, Any]) -> Work | None:
        title = payload.get("title")
        if not title:
            return None  # untitled records are unusable downstream
        authorships = payload.get("authorships") or []
        first_author = None
        if authorships:
            author = authorships[0].get("author") or {}
            first_author = author.get("display_name")
        primary_location = payload.get("primary_location") or {}
        source = primary_location.get("source") or {} if primary_location else {}
        venue = source.get("display_name") if source else None
        doi = payload.get("doi")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]
        return cls(
            id=payload["id"],
            title=title,
            publication_year=payload.get("publication_year"),
            cited_by_count=int(payload.get("cited_by_count") or 0),
            doi=doi,
            first_author=first_author,
            venue=venue,
        )


class OpenAlexClient:
    """Polite-pool OpenAlex client. Pass the maintainer's email via `mailto`
    on every request to land in the higher-throughput queue."""

    def __init__(
        self,
        mailto: str,
        client: httpx.AsyncClient | None = None,
    ):
        self._mailto = mailto
        self._client = client
        self._user_agent = f"citation-reward-hacking/0.1 (mailto:{mailto})"

    async def _get(self, path: str, params: dict[str, Any]) -> httpx.Response:
        params = {**params, "mailto": self._mailto}
        headers = {"User-Agent": self._user_agent}
        url = f"{_OPENALEX_BASE}{path}"
        if self._client is not None:
            return await self._client.get(url, params=params, headers=headers)
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            return await client.get(url, params=params, headers=headers)

    async def search_by_concept(
        self,
        concept_id: str,
        citation_min: int,
        citation_max: int,
        year_max: int,
        per_page: int = 25,
        seed: int | None = None,
    ) -> list[Work]:
        """Search for works in a concept (field-of-study) within a citation band.

        Used to recruit obscure-but-real (citation_min=2, citation_max=49) and
        moderately-known (citation_min=1000, citation_max=5000) works.

        `year_max` is inclusive — we cap publication year at the intersection
        of all model training cutoffs (pre-2023 per plan §7.5).
        """
        filters = [
            f"cited_by_count:{citation_min}-{citation_max}",
            f"publication_year:<{year_max + 1}",
            f"concepts.id:{concept_id}",
        ]
        params: dict[str, Any] = {
            "filter": ",".join(filters),
            "per-page": per_page,
        }
        if seed is not None:
            params["sample"] = per_page
            params["seed"] = seed
        response = await self._get("/works", params)
        response.raise_for_status()
        return self._parse_results(response.json())

    async def search_by_title(self, title: str, per_page: int = 5) -> list[Work]:
        """Top-N title-search hits for the negative-existence proof."""
        params: dict[str, Any] = {
            "search": title,
            "per-page": per_page,
        }
        response = await self._get("/works", params)
        response.raise_for_status()
        return self._parse_results(response.json())

    async def get_work_by_doi(self, doi: str) -> Work | None:
        path = f"/works/https://doi.org/{doi}"
        response = await self._get(path, params={})
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return Work.from_openalex(response.json())

    @staticmethod
    def _parse_results(payload: dict[str, Any]) -> list[Work]:
        out: list[Work] = []
        for item in payload.get("results", []) or []:
            work = Work.from_openalex(item)
            if work is not None:
                out.append(work)
        return out
