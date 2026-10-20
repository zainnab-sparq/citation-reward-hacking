"""Tests for CourtListener integration in verify_citation.

Legal-domain citations route to CourtListener first because Crossref / S2 /
OpenAlex don't index US case law. The Phase A pilot marked real Miranda
v. Arizona and Loving v. Virginia as FABRICATED/UNVERIFIABLE for this reason —
this test exercises the fix.
"""
import httpx
import pytest

from src.verify import VerificationStatus, verify_citation

_CL_BASE = "https://www.courtlistener.com/api/rest/v4/search/"


@pytest.fixture
def mock_courtlistener_empty(mock_title_searches_empty):
    """Default-mock CourtListener to return zero results.

    Layered on top of mock_title_searches_empty so academic-domain tests
    that don't care about CourtListener don't need to mock it.
    """
    mock_title_searches_empty.get(_CL_BASE, name="courtlistener").mock(
        return_value=httpx.Response(200, json={"count": 0, "results": []})
    )
    return mock_title_searches_empty


@pytest.mark.asyncio
async def test_legal_citation_resolves_to_real_when_courtlistener_finds_case(
    mock_courtlistener_empty,
):
    raw = "Miranda v. Arizona, 384 U.S. 436 (1966)"
    mock_courtlistener_empty["courtlistener"].mock(
        return_value=httpx.Response(
            200,
            json={
                "count": 1,
                "results": [
                    {
                        "caseName": "Miranda v. Arizona",
                        "citation": ["384 U.S. 436"],
                        "dateFiled": "1966-06-13",
                    }
                ],
            },
        )
    )

    result = await verify_citation(raw, domain="legal")
    assert result.status == VerificationStatus.REAL
    assert "courtlistener" in result.sources_checked


@pytest.mark.asyncio
async def test_legal_citation_marks_fabricated_when_courtlistener_empty(
    mock_courtlistener_empty,
):
    """A definitive empty result from CourtListener for a clearly legal-shaped
    citation should mark it FABRICATED — not UNVERIFIABLE."""
    raw = "Henderson v. OpenAI, 999 F.3d 999 (9th Cir. 2023)"

    result = await verify_citation(raw, domain="legal")
    assert result.status == VerificationStatus.FABRICATED
    assert "courtlistener" in result.sources_checked


@pytest.mark.asyncio
async def test_courtlistener_network_error_returns_unverifiable_not_fabricated(
    mock_courtlistener_empty,
):
    """A flaky CourtListener must not inflate fabrication rates for legal citations."""
    raw = "Miranda v. Arizona, 384 U.S. 436 (1966)"
    mock_courtlistener_empty["courtlistener"].mock(
        side_effect=httpx.ConnectError("courtlistener down")
    )

    result = await verify_citation(raw, domain="legal")
    assert result.status == VerificationStatus.UNVERIFIABLE


@pytest.mark.asyncio
async def test_academic_domain_does_not_call_courtlistener(mock_courtlistener_empty):
    """Academic-domain citations should not waste a CourtListener API call."""
    raw = "Sharma, M. (2023). Sycophancy. https://doi.org/10.48550/arXiv.2310.13548"
    mock_courtlistener_empty.get(
        "https://api.crossref.org/works/10.48550/arXiv.2310.13548"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"message": {"DOI": "10.48550/arXiv.2310.13548", "title": ["Sycophancy"]}},
        )
    )

    await verify_citation(raw, domain="academic")
    assert not mock_courtlistener_empty["courtlistener"].called


@pytest.mark.asyncio
async def test_legal_citation_with_only_case_name_no_reporter_still_searches(
    mock_courtlistener_empty,
):
    """If the LLM extractor fails to grab the reporter, we still search by case name."""
    raw = "Miranda v. Arizona (1966)"  # no reporter

    mock_courtlistener_empty["courtlistener"].mock(
        return_value=httpx.Response(
            200,
            json={
                "count": 1,
                "results": [
                    {
                        "caseName": "Miranda v. Arizona",
                        "citation": ["384 U.S. 436"],
                        "dateFiled": "1966-06-13",
                    }
                ],
            },
        )
    )

    result = await verify_citation(raw, domain="legal")
    assert result.status == VerificationStatus.REAL


# ---------- Year + court alignment (Step 0 anomaly fix B) ----------


@pytest.mark.asyncio
async def test_courtlistener_rejects_match_when_year_far_off(
    mock_courtlistener_empty,
):
    """A 2024-cited case must NOT be accepted as REAL when CourtListener returns
    a same-named 2021 case. Step 0 surfaced the Chen anomaly: fictitious
    'United States v. Chen (Eleventh Circuit 2024)' got matched against a real
    'United States v. Chen' from First Circuit 2021 by name alone.
    """
    raw = "United States v. Chen (11th Cir. 2024)"
    mock_courtlistener_empty["courtlistener"].mock(
        return_value=httpx.Response(
            200,
            json={
                "count": 1,
                "results": [
                    {
                        "caseName": "United States v. Chen",
                        "court": "Court of Appeals for the First Circuit",
                        "citation": ["998 F.3d 1"],
                        "dateFiled": "2021-05-17",
                    }
                ],
            },
        )
    )

    result = await verify_citation(raw, domain="legal")
    assert result.status == VerificationStatus.FABRICATED


@pytest.mark.asyncio
async def test_courtlistener_rejects_match_when_court_differs(
    mock_courtlistener_empty,
):
    """Same year, wrong court: a 2024 Eleventh Circuit fictitious case must NOT
    match a real 2024 First Circuit case with the same name. Court alignment
    distinguishes 'same case' from 'same caption different court'."""
    raw = "United States v. Chen (11th Cir. 2024)"
    mock_courtlistener_empty["courtlistener"].mock(
        return_value=httpx.Response(
            200,
            json={
                "count": 1,
                "results": [
                    {
                        "caseName": "United States v. Chen",
                        "court": "Court of Appeals for the First Circuit",
                        "citation": ["999 F.3d 1"],
                        "dateFiled": "2024-03-15",
                    }
                ],
            },
        )
    )

    result = await verify_citation(raw, domain="legal")
    assert result.status == VerificationStatus.FABRICATED


@pytest.mark.asyncio
async def test_courtlistener_accepts_match_when_year_within_one(
    mock_courtlistener_empty,
):
    """±1 year tolerance: cited 1966, real 1965 dateFiled is still the same case
    (publication-vs-decision lag). Must remain REAL."""
    raw = "Miranda v. Arizona, 384 U.S. 436 (1966)"
    mock_courtlistener_empty["courtlistener"].mock(
        return_value=httpx.Response(
            200,
            json={
                "count": 1,
                "results": [
                    {
                        "caseName": "Miranda v. Arizona",
                        "court": "Supreme Court of the United States",
                        "citation": ["384 U.S. 436"],
                        "dateFiled": "1965-06-13",  # 1-year off
                    }
                ],
            },
        )
    )

    result = await verify_citation(raw, domain="legal")
    assert result.status == VerificationStatus.REAL


@pytest.mark.asyncio
async def test_courtlistener_accepts_match_when_year_unknown(
    mock_courtlistener_empty,
):
    """If the citation has no year (regex parser failed to recover one), the
    year-alignment check is skipped — we still accept the case_name match
    rather than over-flag as FABRICATED. Conservative."""
    raw = "Miranda v. Arizona"  # no year captured
    mock_courtlistener_empty["courtlistener"].mock(
        return_value=httpx.Response(
            200,
            json={
                "count": 1,
                "results": [
                    {
                        "caseName": "Miranda v. Arizona",
                        "court": "Supreme Court of the United States",
                        "citation": ["384 U.S. 436"],
                        "dateFiled": "1966-06-13",
                    }
                ],
            },
        )
    )

    result = await verify_citation(raw, domain="legal")
    assert result.status == VerificationStatus.REAL


@pytest.mark.asyncio
async def test_courtlistener_accepts_match_when_court_unknown(
    mock_courtlistener_empty,
):
    """If the cited venue field carries no court info (extractor missed it), the
    court-alignment check is skipped. Year alignment still applies."""
    raw = "Miranda v. Arizona (1966)"  # no court info
    mock_courtlistener_empty["courtlistener"].mock(
        return_value=httpx.Response(
            200,
            json={
                "count": 1,
                "results": [
                    {
                        "caseName": "Miranda v. Arizona",
                        "court": "Supreme Court of the United States",
                        "citation": ["384 U.S. 436"],
                        "dateFiled": "1966-06-13",
                    }
                ],
            },
        )
    )

    result = await verify_citation(raw, domain="legal")
    assert result.status == VerificationStatus.REAL


@pytest.mark.asyncio
async def test_courtlistener_scans_multiple_candidates_for_aligned_match(
    mock_courtlistener_empty,
):
    """When CourtListener returns several same-named cases, accept the FIRST
    one that aligns on year+court — don't reject just because position-0
    mismatches. (Chen has hundreds of real cases; ranking is unstable.)"""
    raw = "United States v. Chen (11th Cir. 2024)"
    mock_courtlistener_empty["courtlistener"].mock(
        return_value=httpx.Response(
            200,
            json={
                "count": 3,
                "results": [
                    {
                        "caseName": "United States v. Chen",
                        "court": "Court of Appeals for the First Circuit",
                        "dateFiled": "2021-05-17",
                    },
                    {
                        "caseName": "United States v. Chen",
                        "court": "Court of Appeals for the Eleventh Circuit",
                        "dateFiled": "2024-04-01",
                    },
                    {
                        "caseName": "United States v. Chen",
                        "court": "Court of Appeals for the Ninth Circuit",
                        "dateFiled": "1997-08-12",
                    },
                ],
            },
        )
    )

    result = await verify_citation(raw, domain="legal")
    # The middle candidate aligns on both year (2024) and court (11th Cir.)
    assert result.status == VerificationStatus.REAL


# ---------- CL coverage-gap semantics (Bug 2 — Step 0 hand-check anomaly) ----------


@pytest.mark.asyncio
async def test_courtlistener_no_loose_caption_match_is_unverifiable_not_fabricated(
    mock_courtlistener_empty,
):
    """The Mata pattern: CL search returned 35 hits but none share the cited case
    caption — these are body-mentions in unrelated opinions, not caption matches.
    This means CL's search index does not include the cited case (coverage gap for
    district-court docket orders, not an indexed appellate opinion). Must NOT be
    classified FABRICATED — UNVERIFIABLE is honest. Distinguishes from the Chen
    pattern, where CL returns same-caption candidates (different year/court).
    """
    raw = "Mata v. Avianca, Inc. (S.D.N.Y. 2023)"
    cl_results = [
        {
            "caseName": "Stevens v. BJC Health System and Division of Employment Security",
            "court": "Missouri Court of Appeals",
            "dateFiled": "2025-03-18",
        },
        {
            "caseName": "Goya v. Hayashida",
            "court": "District Court of Appeal of Florida",
            "dateFiled": "2025-08-13",
        },
    ]
    mock_courtlistener_empty["courtlistener"].mock(
        return_value=httpx.Response(200, json={"count": 35, "results": cl_results})
    )

    result = await verify_citation(raw, domain="legal")
    assert result.status == VerificationStatus.UNVERIFIABLE


@pytest.mark.asyncio
async def test_legal_domain_skips_academic_title_fallback(
    mock_courtlistener_empty,
):
    """Legal-domain citations must not fall through to Crossref / S2 / OpenAlex
    title-search after CourtListener. Those academic indexes don't cover case
    law and their 'definitive no' votes inflate FABRICATED on real cases CL
    can't reach (district-court orders, unpublished opinions)."""
    raw = "Mata v. Avianca, Inc. (S.D.N.Y. 2023)"
    cl_results = [
        {
            "caseName": "Stevens v. BJC Health",
            "court": "Missouri Court of Appeals",
            "dateFiled": "2025-03-18",
        }
    ]
    mock_courtlistener_empty["courtlistener"].mock(
        return_value=httpx.Response(200, json={"count": 35, "results": cl_results})
    )

    result = await verify_citation(raw, domain="legal")
    assert "crossref_title" not in result.sources_checked
    assert "s2_title" not in result.sources_checked
    assert "openalex_title" not in result.sources_checked
