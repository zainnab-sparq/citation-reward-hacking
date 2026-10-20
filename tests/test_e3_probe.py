"""Tests for the E3 linear probe and its pre-registered decision rules.

The rules encoded here are not conveniences, they are the pre-registration's own
commitments, and each one is the kind that gets quietly forgotten between
running an analysis and writing it up:

  - H1.3 declares AUROC >= 0.70 on the fabrication probe.
  - The agreement rule (section 6 guardrails): if the within-domain probe beats
    the cross-task probe by more than 10pp, the cross-task result is downgraded
    and the within-domain probe becomes the headline.
  - The difficulty deconfound: if the fabrication probe is no better than a
    probe trained on the item-difficulty proxy, the signal is reported as
    difficulty-decodability, not fabrication-decodability.
  - The shuffled-label control must land near chance, or the pipeline is
    leaking the label somewhere.
"""

import numpy as np
import pytest

from src.e3.probe import (
    AGREEMENT_TOLERANCE,
    H1_3_AUROC_THRESHOLD,
    ProbeResult,
    apply_agreement_rule,
    difficulty_deconfound_verdict,
    fit_probe,
    shuffled_label_control,
    sweep_layers,
)


def _separable(n=200, dim=16, seed=0, signal=2.0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(size=(n, dim))
    X[y == 1, 0] += signal
    return X, y


def _noise(n=200, dim=16, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)), np.repeat([0, 1], n // 2)


def test_registered_constants_match_the_preregistration():
    assert H1_3_AUROC_THRESHOLD == 0.70
    assert AGREEMENT_TOLERANCE == 0.10


def test_probe_recovers_a_separable_signal():
    X, y = _separable()
    Xt, yt = _separable(seed=99)
    res = fit_probe(X, y, Xt, yt)
    assert isinstance(res, ProbeResult)
    assert res.auroc > 0.9
    assert res.n_train == len(y) and res.n_test == len(yt)


def test_probe_on_pure_noise_lands_near_chance():
    X, y = _noise()
    Xt, yt = _noise(seed=2)
    assert fit_probe(X, y, Xt, yt).auroc == pytest.approx(0.5, abs=0.15)


def test_shuffled_label_control_destroys_a_real_signal():
    # If this does NOT drop to chance, the pipeline is leaking the label.
    X, y = _separable()
    Xt, yt = _separable(seed=99)
    assert fit_probe(X, y, Xt, yt).auroc > 0.9
    assert shuffled_label_control(X, y, Xt, yt, seed=0).auroc == pytest.approx(0.5, abs=0.15)


def test_layer_sweep_reports_every_layer_and_finds_the_informative_one():
    layers = {0: _noise(seed=3), 1: _noise(seed=4), 2: _separable(seed=5)}
    test_layers = {0: _noise(seed=6), 1: _noise(seed=7), 2: _separable(seed=8)}
    sweep = sweep_layers(layers, test_layers)
    assert set(sweep) == {0, 1, 2}
    assert max(sweep, key=lambda k: sweep[k].auroc) == 2


def test_agreement_rule_keeps_cross_task_headline_when_probes_agree():
    verdict = apply_agreement_rule(cross_task_auroc=0.74, within_domain_auroc=0.79)
    assert verdict.headline == "cross_task"
    assert not verdict.downgraded
    assert verdict.headline_auroc == 0.74


def test_agreement_rule_downgrades_when_within_domain_wins_by_more_than_10pp():
    verdict = apply_agreement_rule(cross_task_auroc=0.62, within_domain_auroc=0.80)
    assert verdict.headline == "within_domain"
    assert verdict.downgraded
    assert verdict.headline_auroc == 0.80


def test_agreement_rule_boundary_of_exactly_10pp_is_not_a_downgrade():
    # "exceeds ... by more than 10pp" is a strict inequality in the registration.
    verdict = apply_agreement_rule(cross_task_auroc=0.70, within_domain_auroc=0.80)
    assert not verdict.downgraded
    assert verdict.headline == "cross_task"


def test_agreement_rule_does_not_downgrade_when_cross_task_is_the_higher_one():
    verdict = apply_agreement_rule(cross_task_auroc=0.85, within_domain_auroc=0.60)
    assert not verdict.downgraded
    assert verdict.headline == "cross_task"


def test_difficulty_deconfound_flags_a_probe_that_is_only_reading_difficulty():
    verdict = difficulty_deconfound_verdict(fabrication_auroc=0.72, difficulty_auroc=0.74)
    assert verdict.reads_difficulty
    assert "difficulty" in verdict.reported_as


def test_difficulty_deconfound_passes_when_fabrication_probe_wins():
    verdict = difficulty_deconfound_verdict(fabrication_auroc=0.80, difficulty_auroc=0.58)
    assert not verdict.reads_difficulty
    assert "fabrication" in verdict.reported_as


def test_probe_refuses_a_single_class_training_set():
    X = np.random.default_rng(0).normal(size=(20, 4))
    with pytest.raises(ValueError, match="one class"):
        fit_probe(X, np.zeros(20, dtype=int), X, np.zeros(20, dtype=int))


def test_probe_refuses_mismatched_shapes():
    X, y = _separable()
    with pytest.raises(ValueError):
        fit_probe(X, y[:-1], X, y)
