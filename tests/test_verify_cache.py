"""Tests for the disk cache layer of verify.py."""
import json
from pathlib import Path

import httpx
import pytest

from src.verify import VerificationCache, VerificationStatus, verify_citation


@pytest.mark.asyncio
async def test_second_call_does_not_hit_network(
    tmp_path: Path, mock_title_searches_empty
):
    raw = "Sharma, M. (2023). Sycophancy. https://doi.org/10.48550/arXiv.2310.13548"
    cache_file = tmp_path / "cache.jsonl"
    cache = VerificationCache(cache_file)

    crossref_route = mock_title_searches_empty.get(
        "https://api.crossref.org/works/10.48550/arXiv.2310.13548"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"message": {"DOI": "10.48550/arXiv.2310.13548"}},
        )
    )

    r1 = await verify_citation(raw, domain="academic", cache=cache)
    r2 = await verify_citation(raw, domain="academic", cache=cache)

    assert r1.status == VerificationStatus.REAL
    assert r2.status == VerificationStatus.REAL
    assert crossref_route.call_count == 1


def test_cache_persists_to_disk(tmp_path: Path):
    cache_file = tmp_path / "cache.jsonl"
    cache = VerificationCache(cache_file)
    cache.set("some-key", {"status": "REAL", "matches": {"crossref": {"DOI": "x"}}})
    cache.flush()

    reloaded = VerificationCache(cache_file)
    assert reloaded.get("some-key") == {"status": "REAL", "matches": {"crossref": {"DOI": "x"}}}


def test_cache_file_is_valid_jsonl(tmp_path: Path):
    cache_file = tmp_path / "cache.jsonl"
    cache = VerificationCache(cache_file)
    cache.set("a", {"status": "REAL"})
    cache.set("b", {"status": "FABRICATED"})
    cache.flush()

    lines = cache_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert "key" in record and "value" in record
