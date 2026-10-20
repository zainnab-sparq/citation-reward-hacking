"""SPECTER v1 embedding wrapper for the negative-existence cosine check.

SPECTER (Cohan et al. 2020) is trained to make scientific paper titles +
abstracts of the same work close in cosine. We use it to detect cases where
a constructed "fabricated" candidate is semantically equivalent to a real
paper despite being edit-distance-far at the title level.

Threshold: cosine >= 0.85 → reject candidate as "actually exists." Calibrated
against known duplicate pairs in src/dataset_construction/calibrate_specter.py.
"""
from __future__ import annotations

from functools import cache
from typing import Sequence


_MODEL_NAME = "sentence-transformers/allenai-specter"


@cache
def _load_model():
    """Load once per process — model is ~440MB, init takes a few seconds."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME)


class SpecterEmbedder:
    """Implements the Embedder Protocol from negative_existence.py."""

    def __init__(self):
        # Force lazy load via property so __init__ stays cheap and tests can
        # construct without paying the model-load cost.
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model()
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # SPECTER expects "title[SEP]abstract" format; for unanswerable-cell use
        # we typically have only a title, which the model handles fine — empty
        # abstract is just a slightly less precise embedding.
        encoded = self.model.encode(list(texts), show_progress_bar=False)
        return [vec.tolist() for vec in encoded]
