"""Feasibility enumeration for the 600-question dataset (plan §4 / pre-reg §4).

For each (domain × cell), count the candidate pool that meets the eligibility
filters. Per-cell target is 200 questions; we want ≥ 300 candidates to absorb
≥ 33% rejection-pipeline attrition (filter conflicts, near-duplicate rejection,
question-template generation failures).

If a cell is below 300, the pre-declared fallback widens the citation band
([2, 50) → [2, 100)) or year cutoff (2022 → 2023), recorded in the pre-reg.

Run:
    .venv/Scripts/python.exe -m src.dataset_construction.feasibility
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import httpx

# OpenAlex concept IDs (verified via https://api.openalex.org/concepts):
# C41008148  Computer science
# C71924100  Medicine
# C17744445  Political science (law is a subdiscipline; legal queries go via
#            CourtListener — see _enumerate_legal below)
_CONCEPT_IDS = {
    "academic": "C41008148",  # CS as the operational scope
    "medical": "C71924100",
    "legal": "C17744445",
}

_OPENALEX_BASE = "https://api.openalex.org"
_PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v4/search/"
_TARGET_PER_CELL = 200
_MIN_CANDIDATE_POOL = 300
_FALLBACK_CANDIDATE_POOL = 250  # below this and the cell is infeasible

_TIMEOUT = httpx.Timeout(30.0)


@dataclass
class CellCount:
    domain: str
    cell: str
    label: str  # human-readable filter description
    count: int
    source: str
    pass_threshold: int = _MIN_CANDIDATE_POOL

    @property
    def passes(self) -> bool:
        return self.count >= self.pass_threshold


async def _openalex_count(client: httpx.AsyncClient, mailto: str, filters: list[str]) -> int:
    params = {
        "filter": ",".join(filters),
        "per-page": 1,
        "mailto": mailto,
    }
    headers = {"User-Agent": f"citation-reward-hacking/0.1 (mailto:{mailto})"}
    r = await client.get(f"{_OPENALEX_BASE}/works", params=params, headers=headers)
    r.raise_for_status()
    return int(r.json().get("meta", {}).get("count", 0))


async def _pubmed_count(client: httpx.AsyncClient, query: str) -> int:
    params = {"db": "pubmed", "term": query, "retmode": "json", "rettype": "count"}
    r = await client.get(f"{_PUBMED_BASE}/esearch.fcgi", params=params)
    r.raise_for_status()
    return int(r.json().get("esearchresult", {}).get("count", 0))


async def _courtlistener_count(client: httpx.AsyncClient, q: str) -> int:
    token = os.environ.get("COURTLISTENER_API_TOKEN")
    headers = {"Authorization": f"Token {token}"} if token else {}
    params = {"q": q, "type": "o", "page_size": 1}
    r = await client.get(_COURTLISTENER_BASE, params=params, headers=headers)
    r.raise_for_status()
    return int(r.json().get("count", 0))


async def _enumerate_academic(client: httpx.AsyncClient, mailto: str) -> list[CellCount]:
    out = []
    cs = _CONCEPT_IDS["academic"]
    # known-real, hyper-canonical: cited_by_count >= 10000, pre-2023
    out.append(CellCount(
        "academic", "known_real_canonical",
        "OpenAlex CS, cited_by_count>=10000, year<=2022",
        await _openalex_count(client, mailto, [
            "cited_by_count:>9999", "publication_year:<2023", f"concepts.id:{cs}",
        ]),
        "openalex",
    ))
    # known-real, moderately-known: [1000, 5000]
    out.append(CellCount(
        "academic", "known_real_moderate",
        "OpenAlex CS, [1000,5000] cites, year<=2022",
        await _openalex_count(client, mailto, [
            "cited_by_count:1000-5000", "publication_year:<2023", f"concepts.id:{cs}",
        ]),
        "openalex",
    ))
    # obscure-but-real: [2, 49]
    out.append(CellCount(
        "academic", "obscure_real",
        "OpenAlex CS, [2,49] cites, year<=2022",
        await _openalex_count(client, mailto, [
            "cited_by_count:2-49", "publication_year:<2023", f"concepts.id:{cs}",
        ]),
        "openalex",
    ))
    return out


async def _enumerate_medical(client: httpx.AsyncClient, mailto: str) -> list[CellCount]:
    out = []
    med = _CONCEPT_IDS["medical"]

    # known-real, Cochrane: source ID S4210172715 (Cochrane Database of Systematic Reviews)
    cochrane_count = await _openalex_count(client, mailto, [
        "primary_location.source.id:S4210172715", "publication_year:<2023",
        "cited_by_count:>49",  # exclude protocol-only / unread reviews
    ])
    out.append(CellCount(
        "medical", "known_real_cochrane",
        "OpenAlex Cochrane Database of Systematic Reviews (S4210172715), >=50 cites, year<=2022",
        cochrane_count, "openalex",
        pass_threshold=120,  # 60% of 200 target = 120
    ))

    # known-real, non-Cochrane systematic review (PubMed pubtype filter)
    # Query: systematic review[Publication Type] AND citation count >=500 (proxy via
    # OpenAlex cited_by_count for Crossref-DOI-resolved reviews); we approximate
    # via OpenAlex search-tag combination since a strict 500-cite filter on PubMed
    # alone is not supported.
    out.append(CellCount(
        "medical", "known_real_non_cochrane",
        "OpenAlex Medicine, cited_by_count>=500, year<=2022 (proxy for high-cite reviews)",
        await _openalex_count(client, mailto, [
            "cited_by_count:>499", "publication_year:<2023", f"concepts.id:{med}",
        ]),
        "openalex",
        pass_threshold=80,  # 40% of 200 target = 80
    ))

    # obscure-but-real: [2, 49] medicine, pre-2023, NOT yet retraction-filtered
    # (retraction filter is applied at recruitment time; here we just count the
    # raw pool).
    out.append(CellCount(
        "medical", "obscure_real",
        "OpenAlex Medicine, [2,49] cites, year<=2022 (pre Retraction-Watch filter)",
        await _openalex_count(client, mailto, [
            "cited_by_count:2-49", "publication_year:<2023", f"concepts.id:{med}",
        ]),
        "openalex",
    ))
    return out


async def _enumerate_legal(client: httpx.AsyncClient) -> list[CellCount]:
    """Legal counts via CourtListener since OpenAlex doesn't index US case law.

    CourtListener's federal-court coverage is the operational scope (per plan §4
    post-2026-05-03 hardening). For known-real we want a sample of high-citation
    federal opinions; for obscure-but-real we want low-access federal opinions.
    The /search/?type=o endpoint with court= filter gives us federal-only.
    """
    out = []
    # known-real federal: SCOTUS opinions are the canonical anchor
    out.append(CellCount(
        "legal", "known_real_scotus",
        "CourtListener: SCOTUS opinions (court=scotus)",
        await _courtlistener_count(client, "court_id:scotus"),
        "courtlistener",
        pass_threshold=200,  # need 200 known-real legal; SCOTUS is the canonical pool
    ))
    out.append(CellCount(
        "legal", "known_real_circuit",
        "CourtListener: Circuit Courts of Appeals (court_id:ca*)",
        await _courtlistener_count(
            client, "court_id:(ca1 OR ca2 OR ca3 OR ca4 OR ca5 OR ca6 OR ca7 OR ca8 OR ca9 OR ca10 OR ca11 OR cadc OR cafc)"
        ),
        "courtlistener",
        pass_threshold=100,
    ))
    out.append(CellCount(
        "legal", "obscure_real_district",
        "CourtListener: federal District Court reported opinions",
        await _courtlistener_count(client, "court_id:dcd OR court_id:nysd OR court_id:cand"),
        "courtlistener",
    ))
    return out


async def main() -> None:
    mailto = os.environ.get("OPENALEX_MAILTO", "researcher@example.com")
    print(f"Running feasibility enumeration with mailto={mailto}\n")
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results: list[CellCount] = []
        results.extend(await _enumerate_academic(client, mailto))
        results.extend(await _enumerate_medical(client, mailto))
        results.extend(await _enumerate_legal(client))

    print(f"{'domain':<10} {'cell':<28} {'count':>10} {'min':>8} {'OK':>6}  source")
    print("-" * 90)
    all_pass = True
    for r in results:
        flag = "PASS" if r.passes else "FAIL"
        if not r.passes:
            all_pass = False
        print(f"{r.domain:<10} {r.cell:<28} {r.count:>10} {r.pass_threshold:>8} {flag:>6}  {r.source}")
        print(f"           {r.label}")

    print()
    print("OVERALL:", "ALL CELLS PASS" if all_pass else "ONE OR MORE FAILURES — apply fallback rules per pre-reg §4")


if __name__ == "__main__":
    asyncio.run(main())
