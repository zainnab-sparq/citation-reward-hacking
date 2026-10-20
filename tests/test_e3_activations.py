"""Tests for E3 activation extraction.

H1.3 is specifically about hidden states "at the answer-token positions". Two
things can go wrong there without raising anything, and both would quietly
invalidate the result:

  - pooling over the prompt as well as the answer, so the probe reads the
    question rather than the answer
  - an off-by-one at the boundary, which for a one-token answer means pooling
    the wrong token entirely

Neither is visible in an AUROC. They are visible here.
"""

import numpy as np
import pytest

from src.e3.activations import (
    answer_token_span,
    mean_pool_span,
    pool_all_layers,
)


def test_span_starts_where_the_prompt_ends():
    prompt = [1, 2, 3, 4]
    full = [1, 2, 3, 4, 10, 11, 12]
    assert answer_token_span(prompt, full) == (4, 7)


def test_span_of_a_single_token_answer_is_length_one():
    assert answer_token_span([1, 2], [1, 2, 9]) == (2, 3)


def test_span_rejects_a_full_sequence_that_does_not_extend_the_prompt():
    with pytest.raises(ValueError, match="prefix"):
        answer_token_span([1, 2, 3], [9, 9, 9, 4])


def test_span_rejects_an_empty_answer():
    with pytest.raises(ValueError, match="no answer tokens"):
        answer_token_span([1, 2, 3], [1, 2, 3])


def test_mean_pool_averages_only_the_span():
    hidden = np.array([[0.0, 0.0], [100.0, 100.0], [1.0, 3.0], [3.0, 5.0]])
    # Span covers rows 2 and 3 only; row 1 is a decoy that would dominate.
    assert mean_pool_span(hidden, (2, 4)).tolist() == [2.0, 4.0]


def test_mean_pool_of_one_token_is_that_token():
    hidden = np.array([[1.0, 2.0], [7.0, 8.0]])
    assert mean_pool_span(hidden, (1, 2)).tolist() == [7.0, 8.0]


def test_mean_pool_rejects_an_out_of_range_span():
    hidden = np.zeros((3, 2))
    with pytest.raises(ValueError):
        mean_pool_span(hidden, (1, 9))


def test_pool_all_layers_returns_one_vector_per_layer():
    n_layers, n_tokens, dim = 5, 6, 4
    layers = [np.random.default_rng(i).normal(size=(n_tokens, dim))
              for i in range(n_layers)]
    pooled = pool_all_layers(layers, (2, 5))
    assert pooled.shape == (n_layers, dim)
    assert np.allclose(pooled[3], layers[3][2:5].mean(axis=0))


def test_pool_all_layers_preserves_layer_order():
    # Layer index is the unit of the registered per-layer ablation, so a
    # reordering here would silently relabel every layer in the sweep.
    layers = [np.full((3, 2), float(i)) for i in range(4)]
    pooled = pool_all_layers(layers, (0, 3))
    assert [row[0] for row in pooled] == [0.0, 1.0, 2.0, 3.0]
