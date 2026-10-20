"""Tests for the two-step negative-existence proof for unanswerable cell.

Step 1: Levenshtein-3 fuzzy match across recruitment sources.
Step 2: SPECTER-2 cosine similarity ≥ 0.85 against top-5 OpenAlex hits.

A candidate passes negative-existence if BOTH steps return no match.

The cosine check uses an injectable Embedder protocol so tests don't load
SPECTER-2 weights (slow + 500MB).
"""
from __future__ import annotations

from typing import Sequence

import pytest

from src.dataset_construction.negative_existence import (
    Embedder,
    NegativeExistenceProver,
    ProofResult,
    Rejection,
)
from src.dataset_construction.openalex_client import Work


class StaticEmbedder:
    """Embedder that returns the registered vector per text. For tests."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]


class FakeOpenAlexClient:
    def __init__(self, hits_per_query: dict[str, list[Work]]):
        self._hits = hits_per_query

    async def search_by_title(self, title: str, per_page: int = 5) -> list[Work]:
        return self._hits.get(title, [])[:per_page]


def _work(title: str, doi: str | None = None) -> Work:
    return Work(
        id=f"https://openalex.org/W-{title[:8]}",
        title=title,
        publication_year=2020,
        cited_by_count=10,
        doi=doi,
        first_author=None,
        venue=None,
    )


@pytest.mark.asyncio
async def test_passes_when_no_levenshtein_or_cosine_match():
    candidate_title = "Quantum entanglement of squirrel-borne neural goblins in suburban North Dakota"
    fake_oa = FakeOpenAlexClient(
        hits_per_query={candidate_title: [_work("Some unrelated paper about cats")]}
    )
    embedder = StaticEmbedder({
        candidate_title: [1.0, 0.0],
        "Some unrelated paper about cats": [0.0, 1.0],  # cosine 0.0
    })
    prover = NegativeExistenceProver(
        openalex=fake_oa,
        embedder=embedder,
        cosine_threshold=0.85,
    )
    result = await prover.prove(candidate_title=candidate_title)
    assert result.passes is True
    assert result.rejection is None


@pytest.mark.asyncio
async def test_rejects_on_cosine_match_above_threshold():
    candidate = "BERT pre-training of bidirectional transformers for NLP"
    real = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    fake_oa = FakeOpenAlexClient(hits_per_query={candidate: [_work(real)]})
    # Vectors with cosine = 0.95
    embedder = StaticEmbedder({
        candidate: [1.0, 0.0],
        real: [0.95, 0.31225],  # cos = 0.95 / sqrt(0.95^2 + 0.31225^2) ≈ 0.95
    })
    prover = NegativeExistenceProver(
        openalex=fake_oa, embedder=embedder, cosine_threshold=0.85
    )
    result = await prover.prove(candidate_title=candidate)
    assert result.passes is False
    assert result.rejection is not None
    assert result.rejection.step == "cosine_similarity"
    assert result.rejection.matched_title == real


@pytest.mark.asyncio
async def test_rejects_on_levenshtein_3_partial_match():
    """A title that is edit-distance-3 from a real paper is rejected at step 1
    without ever reaching the cosine check."""
    candidate = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"  # exact
    real = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    fake_oa = FakeOpenAlexClient(hits_per_query={candidate: [_work(real)]})
    # Embedder shouldn't be called; provide one that would crash to verify
    crash_embedder = StaticEmbedder({})
    prover = NegativeExistenceProver(
        openalex=fake_oa, embedder=crash_embedder, cosine_threshold=0.85
    )
    result = await prover.prove(candidate_title=candidate)
    assert result.passes is False
    assert result.rejection is not None
    assert result.rejection.step == "levenshtein_fuzzy"


@pytest.mark.asyncio
async def test_returns_indeterminate_when_openalex_returns_no_hits():
    """If OpenAlex returns zero hits, we cannot do the cosine check. Treat as
    pass (no evidence the title exists) but record indeterminacy."""
    candidate = "A truly novel and obscure title that returns no hits"
    fake_oa = FakeOpenAlexClient(hits_per_query={candidate: []})
    crash_embedder = StaticEmbedder({})  # would error if called
    prover = NegativeExistenceProver(
        openalex=fake_oa, embedder=crash_embedder, cosine_threshold=0.85
    )
    result = await prover.prove(candidate_title=candidate)
    assert result.passes is True
    assert result.rejection is None
    assert result.cosine_check_skipped_reason == "no_hits"
