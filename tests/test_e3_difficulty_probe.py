"""Tests for the E3 difficulty probe (pre-registration section 6, guardrail (a)).

The deconfound asks whether a probe that can only see what the hidden states
say about item difficulty separates fabricated from real citations as well as
the fabrication probe does. If it does, the fabrication probe was reading
difficulty. The difficulty probe is a linear regression from hidden states to
the continuous proxy; its predicted proxy is then scored as a fabrication
classifier on the held-out rows.
"""

import numpy as np
import pytest

from src.e3.probe import ProbeResult, difficulty_probe, proxy_alone_auroc


def _difficulty_encoded(n=400, dim=16, seed=0, label_tracks_difficulty=True):
    """Hidden states whose first axis is item difficulty; labels optionally track it."""
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(size=n)
    X = rng.normal(size=(n, dim))
    X[:, 0] += 3.0 * difficulty
    if label_tracks_difficulty:
        y = (difficulty + 0.3 * rng.normal(size=n) > 0).astype(int)
    else:
        y = rng.integers(0, 2, size=n)
    return X, difficulty, y


def test_difficulty_probe_separates_labels_that_track_difficulty():
    X, d, y = _difficulty_encoded()
    Xt, dt, yt = _difficulty_encoded(seed=1)
    res = difficulty_probe(X, d, Xt, yt)
    assert isinstance(res, ProbeResult)
    assert res.auroc > 0.85
    assert res.n_train == len(d) and res.n_test == len(yt)


def test_difficulty_probe_is_orientation_agnostic():
    """The proxy's sign convention must not decide the verdict.

    A probe that predicts the proxy perfectly but in the direction that ranks
    real citations first would score below 0.5 as a fabrication classifier.
    That is the same evidence of difficulty-decodability, so the larger
    orientation is taken. This is conservative for the fabrication claim.
    """
    X, d, y = _difficulty_encoded()
    Xt, dt, yt = _difficulty_encoded(seed=1)
    flipped = difficulty_probe(X, -d, Xt, yt)
    straight = difficulty_probe(X, d, Xt, yt)
    assert flipped.auroc == pytest.approx(straight.auroc, abs=1e-9)
    assert flipped.auroc >= 0.5


def test_difficulty_probe_is_near_chance_when_labels_ignore_difficulty():
    X, d, y = _difficulty_encoded(label_tracks_difficulty=False)
    Xt, dt, yt = _difficulty_encoded(seed=1, label_tracks_difficulty=False)
    res = difficulty_probe(X, d, Xt, yt)
    assert 0.5 <= res.auroc < 0.62


def test_difficulty_probe_rejects_mismatched_rows():
    X, d, y = _difficulty_encoded()
    with pytest.raises(ValueError):
        difficulty_probe(X, d[:-1], X, y)


def test_proxy_alone_auroc_is_orientation_agnostic_and_at_least_half():
    _, d, y = _difficulty_encoded()
    assert proxy_alone_auroc(d, y) == pytest.approx(proxy_alone_auroc(-d, y), abs=1e-9)
    assert proxy_alone_auroc(d, y) > 0.85
