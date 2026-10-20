"""Tests for the DeBERTa arm's helpers.

Only the parts that do not need torch. The fine-tuning loop itself is exercised
by the E4 run, which is a one-shot and cannot live in the suite.
"""
from __future__ import annotations

import pytest

from src.analysis.e4_deberta import (
    BATCH_SIZE,
    EPOCHS,
    MAX_LENGTH,
    TrainingDivergedError,
    batched,
    require_finite_loss,
)


class TestBatching:
    def test_splits_into_full_batches(self):
        assert list(batched([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    def test_keeps_the_short_final_batch(self):
        assert list(batched([1, 2, 3], 2)) == [[1, 2], [3]]

    def test_empty_input_yields_nothing(self):
        assert list(batched([], 4)) == []

    def test_zero_size_raises_rather_than_looping_forever(self):
        with pytest.raises(ValueError):
            list(batched([1, 2], 0))


class TestDivergenceGuard:
    def test_finite_loss_passes(self):
        require_finite_loss(0.6931, step=1)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_loss_raises_immediately(self, value):
        # The first run trained 200 steps at mean_loss nan. A silent NaN is worse
        # than a crash, because the run still writes out a preference rate and it
        # looks like a result.
        with pytest.raises(TrainingDivergedError):
            require_finite_loss(value, step=200)

    def test_the_message_names_the_dtype_cause(self):
        with pytest.raises(TrainingDivergedError, match="float32"):
            require_finite_loss(float("nan"), step=5)


class TestFixedHyperparameters:
    def test_are_the_documented_ones(self):
        # Fixed rather than tuned: E4 is one-shot, so a tuning loop over a
        # held-out slice of the training family would be a search for a
        # preference rate.
        assert (MAX_LENGTH, BATCH_SIZE, EPOCHS) == (256, 16, 3)
