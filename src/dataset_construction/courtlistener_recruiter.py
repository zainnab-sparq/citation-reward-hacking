"""CourtListener-based recruiter for the legal domain.

US case law isn't in OpenAlex, so legal recruitment goes through CourtListener
v4 search API. Federal-court-only (per plan §4 hardening 2026-05-03 — state
courts have CourtListener coverage gaps that risk accidental real-case
collisions in the unanswerable cell).

Cells:
  - known_real_scotus: SCOTUS opinions, citeCount > 100 (canonical authorities)
  - known_real_circuit: federal Circuit Courts of Appeals, citeCount > 50
  - obscure_real_district: federal District Court reported opinions, citeCount [2, 49]

CourtListener uses Solr-style filter syntax: q=citeCount:[N TO M] AND court_id:X.
Random sampling is approximated by cursor-paginating with a deterministic seed.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import httpx

_CL_BASE = "https://www.courtlistener.com/api/rest/v4/search/"
_TIMEOUT = httpx.Timeout(30.0)

# Federal Circuit Courts of Appeals court_ids
_CIRCUIT_COURTS = ["ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10", "ca11", "cadc", "cafc"]
# A subset of high-volume federal District Courts (representative sample;
# not exhaustive). Restricted to courts with full PACER + CourtListener coverage.
_DISTRICT_COURTS = ["dcd", "nysd", "cand", "ilnd", "txnd", "txed", "njd", "vad", "vaed", "wand"]


@dataclass(frozen=True)
class LegalCandidate:
    cluster_id: int
    case_name: str
    citation: list[str]
    court: str
    court_id: str
    date_filed: str
    cite_count: int
    docket_number: str
    snippet: str
    absolute_url: str


def _parse_result(item: dict[str, Any]) -> LegalCandidate | None:
    name = item.get("caseName")
    if not name:
        return None
    return LegalCandidate(
        cluster_id=int(item.get("cluster_id") or 0),
        case_name=name,
        citation=list(item.get("citation") or []),
        court=item.get("court") or "",
        court_id=item.get("court_id") or "",
        date_filed=item.get("dateFiled") or "",
        cite_count=int(item.get("citeCount") or 0),
        docket_number=item.get("docketNumber") or "",
        snippet=item.get("snippet") or "",
        absolute_url=item.get("absolute_url") or "",
    )


def _headers() -> dict[str, str]:
    token = os.environ.get("COURTLISTENER_API_TOKEN")
    return {"Authorization": f"Token {token}"} if token else {}


async def _fetch_one_page(
    client: httpx.AsyncClient,
    q: str,
    court_id: str,
    cursor: str | None,
    page_size: int,
) -> tuple[list[LegalCandidate], str | None]:
    # CourtListener v4: court_id as a separate query param does not consistently
    # restrict to that court when q= is also provided. Embed court_id in the
    # Solr query (q=) instead — that's reliably AND-combined.
    combined_q = f"({q}) AND court_id:{court_id}"
    params: dict[str, Any] = {
        "type": "o",
        "q": combined_q,
        "page_size": page_size,
    }
    if cursor:
        params["cursor"] = cursor
    response = await client.get(_CL_BASE, params=params, headers=_headers())
    response.raise_for_status()
    data = response.json()
    candidates = []
    for item in data.get("results", []):
        cand = _parse_result(item)
        if cand is not None:
            candidates.append(cand)
    next_url = data.get("next") or ""
    next_cursor = None
    if next_url:
        # Extract cursor= param from the next URL
        from urllib.parse import urlparse, parse_qs
        next_cursor = parse_qs(urlparse(next_url).query).get("cursor", [None])[0]
    return candidates, next_cursor


async def recruit_scotus(n: int = 200, citation_min: int = 100) -> list[LegalCandidate]:
    """Sample N SCOTUS opinions with citeCount >= citation_min.

    Sampled via cursor-paginated walk. Not perfectly random but covers the
    full date range; the bias (toward more-cited opinions) is acceptable for
    "canonical authorities" which is what known-real-scotus targets.
    """
    q = f"citeCount:[{citation_min} TO 100000]"
    candidates: list[LegalCandidate] = []
    cursor = None
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        while len(candidates) < n:
            page, cursor = await _fetch_one_page(client, q, "scotus", cursor, page_size=100)
            if not page:
                break
            candidates.extend(page)
            if cursor is None:
                break
    return candidates[:n]


async def recruit_circuit(n: int = 200, citation_min: int = 50) -> list[LegalCandidate]:
    """Sample N federal Circuit Courts of Appeals opinions, citeCount >= min.

    CourtListener court_id filter accepts a single id at a time, so we round-
    robin across the 13 circuits proportionally.
    """
    per_circuit = max(1, n // len(_CIRCUIT_COURTS) + 1)
    q = f"citeCount:[{citation_min} TO 100000]"
    candidates: list[LegalCandidate] = []
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for court_id in _CIRCUIT_COURTS:
            page, _ = await _fetch_one_page(client, q, court_id, None, page_size=per_circuit)
            candidates.extend(page)
    rng = random.Random(42)
    rng.shuffle(candidates)
    return candidates[:n]


async def recruit_district_obscure(n: int = 200) -> list[LegalCandidate]:
    """Sample N federal District Court opinions with citeCount in [2, 49].

    Same round-robin-across-courts strategy as recruit_circuit. The District
    courts here are the ones with the fullest CourtListener / PACER coverage.
    """
    per_court = max(1, n // len(_DISTRICT_COURTS) + 1)
    q = "citeCount:[2 TO 49]"
    candidates: list[LegalCandidate] = []
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for court_id in _DISTRICT_COURTS:
            page, _ = await _fetch_one_page(client, q, court_id, None, page_size=per_court)
            candidates.extend(page)
    rng = random.Random(42)
    rng.shuffle(candidates)
    return candidates[:n]
