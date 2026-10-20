"""Two-step negative-existence proof for unanswerable-cell candidates.

Step 1: Levenshtein-3 fuzzy title match against the top-N OpenAlex hits.
Step 2: SPECTER v1 cosine similarity ≥ 0.85 against the same hits.

Either match → reject. Both clear → candidate is provisionally unanswerable.

The plan §7.4 also requires Crossref / PubMed / CourtListener checks; those are
delegated to the existing src/verify.py pipeline (callable via
verify_citation(text, domain) and inspecting result.status). This module only
covers the OpenAlex semantic-similarity arm, which is the novel addition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from rapidfuzz import fuzz

# Plan §7.4 fuzzy thresholds — must mirror src/verify.py:_title_matches.
# A candidate is rejected at step 1 if either ratio exceeds threshold against
# any returned hit.
_LEVENSHTEIN_TOKEN_SET_THRESHOLD = 95
_LEVENSHTEIN_TOKEN_SORT_THRESHOLD = 85


class Embedder(Protocol):
    """Embed a list of strings; return a list of vectors of equal length."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class _OpenAlexLike(Protocol):
    async def search_by_title(self, title: str, per_page: int = 5): ...


@dataclass(frozen=True)
class Rejection:
    step: str  # "levenshtein_fuzzy" or "cosine_similarity"
    matched_title: str
    matched_score: float


@dataclass(frozen=True)
class ProofResult:
    passes: bool
    rejection: Rejection | None
    cosine_check_skipped_reason: str | None = None
    cosine_max: float | None = None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector dim mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class NegativeExistenceProver:
    def __init__(
        self,
        openalex: _OpenAlexLike,
        embedder: Embedder,
        cosine_threshold: float = 0.85,
        hits_per_query: int = 5,
    ):
        self._openalex = openalex
        self._embedder = embedder
        self._cosine_threshold = cosine_threshold
        self._hits_per_query = hits_per_query

    async def prove(self, candidate_title: str) -> ProofResult:
        hits = await self._openalex.search_by_title(
            candidate_title, per_page=self._hits_per_query
        )
        if not hits:
            return ProofResult(passes=True, rejection=None, cosine_check_skipped_reason="no_hits")

        # Step 1: Levenshtein-3 fuzzy match (cheap, runs first)
        for hit in hits:
            set_ratio = fuzz.token_set_ratio(candidate_title, hit.title)
            sort_ratio = fuzz.token_sort_ratio(candidate_title, hit.title)
            if (
                set_ratio >= _LEVENSHTEIN_TOKEN_SET_THRESHOLD
                and sort_ratio >= _LEVENSHTEIN_TOKEN_SORT_THRESHOLD
            ):
                return ProofResult(
                    passes=False,
                    rejection=Rejection(
                        step="levenshtein_fuzzy",
                        matched_title=hit.title,
                        matched_score=float(set_ratio),
                    ),
                )

        # Step 2: SPECTER v1 cosine similarity
        all_texts = [candidate_title] + [h.title for h in hits]
        vectors = self._embedder.embed(all_texts)
        candidate_vec = vectors[0]
        max_cos = 0.0
        max_match = ""
        for hit, hit_vec in zip(hits, vectors[1:]):
            cos = _cosine(candidate_vec, hit_vec)
            if cos > max_cos:
                max_cos = cos
                max_match = hit.title
            if cos >= self._cosine_threshold:
                return ProofResult(
                    passes=False,
                    rejection=Rejection(
                        step="cosine_similarity",
                        matched_title=hit.title,
                        matched_score=cos,
                    ),
                    cosine_max=cos,
                )
        return ProofResult(passes=True, rejection=None, cosine_max=max_cos)
