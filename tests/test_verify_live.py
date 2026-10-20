"""Live API tests — actually hit Crossref. Skipped unless RUN_LIVE_TESTS=1.

Run with:
    RUN_LIVE_TESTS=1 pytest tests/test_verify_live.py -v
"""
import os

import pytest

from src.verify import VerificationStatus, verify_citation

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to enable live API tests",
)


@pytest.mark.asyncio
async def test_real_arxiv_citation_resolves_to_real():
    raw = (
        "Sharma, M., Tong, M., Korbak, T., et al. (2023). "
        "Towards understanding sycophancy in language models. "
        "https://doi.org/10.48550/arXiv.2310.13548"
    )
    result = await verify_citation(raw, domain="academic")
    assert result.status == VerificationStatus.REAL


@pytest.mark.asyncio
async def test_obviously_fake_doi_resolves_to_fabricated():
    raw = "Fakeson, F. (2024). Imaginary paper. https://doi.org/10.99999/this.is.not.real.zzz"
    result = await verify_citation(raw, domain="academic")
    assert result.status == VerificationStatus.FABRICATED


@pytest.mark.asyncio
async def test_real_paper_no_doi_resolves_via_title():
    raw = (
        "Vaswani, A. et al. (2017). Attention is all you need. "
        "Advances in Neural Information Processing Systems."
    )
    result = await verify_citation(raw, domain="academic")
    assert result.status == VerificationStatus.REAL


@pytest.mark.asyncio
async def test_fabricated_title_no_doi_resolves_to_fabricated():
    raw = (
        "Fakerton, F. (2024). "
        "Quantum entanglement of squirrel-borne neural goblins in suburban North Dakota. "
        "Journal of Imaginary Nonsense, 99(99), 1-99."
    )
    result = await verify_citation(raw, domain="academic")
    assert result.status == VerificationStatus.FABRICATED


# ---------------- CourtListener live tests (legal domain) ----------------

courtlistener_required = pytest.mark.skipif(
    not os.environ.get("COURTLISTENER_API_TOKEN"),
    reason="set COURTLISTENER_API_TOKEN to enable CourtListener live tests",
)


@courtlistener_required
@pytest.mark.asyncio
async def test_real_miranda_case_resolves_to_real():
    raw = "Miranda v. Arizona, 384 U.S. 436 (1966)"
    result = await verify_citation(raw, domain="legal")
    assert result.status == VerificationStatus.REAL
    assert "courtlistener" in result.sources_checked


@courtlistener_required
@pytest.mark.asyncio
async def test_real_loving_case_resolves_to_real():
    raw = "Loving v. Virginia, 388 U.S. 1 (1967)"
    result = await verify_citation(raw, domain="legal")
    assert result.status == VerificationStatus.REAL


@courtlistener_required
@pytest.mark.asyncio
async def test_fictitious_case_resolves_to_fabricated():
    raw = (
        "Henderson v. OpenAI Corp., 999 F.3d 12345 (9th Cir. 2099)"
    )
    result = await verify_citation(raw, domain="legal")
    assert result.status == VerificationStatus.FABRICATED
