"""Tests for the E3 difficulty-deconfound block that run_e3 reports.

The block joins per-citation activations to the per-question proxy, splits by
question exactly as the within-domain probe does, and scores the fabrication
probe against the difficulty probe on the same held-out rows, once on all of
them and once on the high-difficulty cells the registration names.
"""

import numpy as np
import pytest

from src.e3.run_e3 import HIGH_DIFFICULTY_CELLS, difficulty_deconfound_block


def _world(n_questions=120, per_question=4, dim=12, seed=0, mode="fabrication"):
    """Citation rows over questions; mode decides what the hidden states carry."""
    rng = np.random.default_rng(seed)
    questions, X, y = [], [], []
    proxy, cells = {}, {}
    cell_names = ["known_real", "obscure_real", "unanswerable"]
    for q in range(n_questions):
        qid = f"q{q:03d}"
        proxy[qid] = float(rng.normal())
        cells[qid] = cell_names[q % 3]
        for _ in range(per_question):
            vec = rng.normal(size=dim)
            if mode == "fabrication":
                label = int(rng.integers(0, 2))
                vec[0] += 3.0 * label
            else:  # hidden states carry difficulty only, labels track difficulty
                label = int(proxy[qid] + 0.3 * rng.normal() > 0)
                vec[0] += 3.0 * proxy[qid]
            questions.append(qid)
            X.append(vec)
            y.append(label)
    return np.stack(X), np.asarray(y), questions, proxy, cells


def test_registered_high_difficulty_cells():
    assert set(HIGH_DIFFICULTY_CELLS) == {"obscure_real", "unanswerable"}


def test_fabrication_signal_beats_the_difficulty_probe():
    X, y, questions, proxy, cells = _world(mode="fabrication")
    block = difficulty_deconfound_block(X, y, questions, proxy, cells)
    assert block["fabrication_auroc"] > 0.9
    assert block["difficulty_probe_auroc"] < 0.65
    assert block["verdict"]["reads_difficulty"] is False
    assert block["high_difficulty_subset"]["verdict"]["reads_difficulty"] is False


def test_difficulty_only_states_are_reported_as_difficulty():
    X, y, questions, proxy, cells = _world(mode="difficulty")
    block = difficulty_deconfound_block(X, y, questions, proxy, cells)
    assert block["difficulty_probe_auroc"] > 0.85
    assert block["verdict"]["reads_difficulty"] is True
    assert block["proxy_alone_auroc"] > 0.85


def test_high_difficulty_subset_counts_only_the_registered_cells():
    X, y, questions, proxy, cells = _world(mode="fabrication")
    block = difficulty_deconfound_block(X, y, questions, proxy, cells)
    sub = block["high_difficulty_subset"]
    assert sub["cells"] == sorted(HIGH_DIFFICULTY_CELLS)
    assert 0 < sub["n_test"] < block["n_test"]
    assert block["n_test"] + block["n_train"] == len(y)


def test_a_question_without_a_proxy_is_an_error_not_a_default():
    X, y, questions, proxy, cells = _world(mode="fabrication")
    del proxy["q000"]
    with pytest.raises(ValueError, match="q000"):
        difficulty_deconfound_block(X, y, questions, proxy, cells)
