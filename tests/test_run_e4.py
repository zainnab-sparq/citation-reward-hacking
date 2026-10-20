"""Tests for the gated E4 runner.

The gate test is the important one. E4 is a one-shot registered test, so the
runner must refuse to produce a number unless the caller says so explicitly.
"""
from __future__ import annotations

import json

import pytest

from src.analysis.run_e4 import fit_logistic, main, run_e4
from src.analysis.e4_naive_judge import training_rows

REGISTRY = [
    {"model_id": "tulu-base", "model_family": "tulu3_8b", "stage_ordinal": 0, "is_post_trained": 0},
    {"model_id": "tulu-sft", "model_family": "tulu3_8b", "stage_ordinal": 1, "is_post_trained": 1},
    {"model_id": "tulu-dpo", "model_family": "tulu3_8b", "stage_ordinal": 2, "is_post_trained": 1},
    {"model_id": "tulu-rlvr", "model_family": "tulu3_8b", "stage_ordinal": 3, "is_post_trained": 1},
    {"model_id": "olmo-base", "model_family": "olmo2_13b", "stage_ordinal": 0, "is_post_trained": 0},
    {"model_id": "olmo-sft", "model_family": "olmo2_13b", "stage_ordinal": 1, "is_post_trained": 1},
    {"model_id": "olmo-dpo", "model_family": "olmo2_13b", "stage_ordinal": 2, "is_post_trained": 1},
    {"model_id": "olmo-instruct", "model_family": "olmo2_13b", "stage_ordinal": 3, "is_post_trained": 1},
]
STAGE_CITATIONS = {0: 0, 1: 2, 2: 2, 3: 3}
IDS_BY_STAGE = {
    "tulu3_8b": {0: "tulu-base", 1: "tulu-sft", 2: "tulu-dpo", 3: "tulu-rlvr"},
    "olmo2_13b": {0: "olmo-base", 1: "olmo-sft", 2: "olmo-dpo", 3: "olmo-instruct"},
}


def synthetic_extractions(n_questions=40, *, base_cites_on_even=True):
    """Post-trained checkpoints cite more and cite better-formed things, which is
    the pattern E4 asks whether a naive judge can pick up.

    On even-numbered questions the base model also cites, so that some pairs land
    within the ±1 count-matching window; otherwise the count-matched control
    would be empty and the fixture would never exercise it.
    """
    records = []
    for q in range(n_questions):
        for condition in ("neutral", "citations_expected"):
            for family, stages in IDS_BY_STAGE.items():
                for ordinal, model_id in stages.items():
                    n = STAGE_CITATIONS[ordinal]
                    if ordinal == 0 and base_cites_on_even and q % 2 == 0:
                        n = 2
                    records.append(
                        {
                            "question_id": f"academic_known_real_{q:04d}",
                            "model_id": model_id,
                            "condition": condition,
                            "trial_id": 0,
                            "citations": [
                                {
                                    "raw_span": f"Cronbach, L. J. ({1950 + i}). A paper. Psychological Bulletin.",
                                    "title": "A paper",
                                    "authors": ["Cronbach", "L. J."],
                                    "venue": "Psychological Bulletin",
                                    "year": 1950 + i,
                                }
                                for i in range(n)
                            ],
                            "error": None,
                        }
                    )
    return records


class TestGate:
    def test_refuses_to_run_without_the_explicit_flag(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            main(["--extractions", "x.jsonl", "--registry", "r.jsonl", "--output", str(tmp_path / "o.json")])
        assert excinfo.value.code != 0

    def test_the_gate_is_off_by_default_in_the_parser(self, tmp_path, capsys):
        # A default of True would make every accidental invocation burn the
        # one-shot test, which is the failure this guard exists to prevent.
        with pytest.raises(SystemExit):
            main(["--extractions", "x.jsonl", "--registry", "r.jsonl", "--output", str(tmp_path / "o.json")])
        assert "--run-confirmatory" in capsys.readouterr().err


class TestLogisticArm:
    def test_learns_the_post_trained_direction_from_surface_features(self):
        rows = training_rows(synthetic_extractions(), REGISTRY)
        model = fit_logistic(rows)
        cited = [[3.0, 1.0, 1.0, 1.0]]
        empty = [[0.0, 0.0, 0.0, 0.0]]
        assert model.predict_proba(cited)[0][1] > model.predict_proba(empty)[0][1]

    def test_trains_only_on_the_training_family(self):
        rows = training_rows(synthetic_extractions(), REGISTRY)
        assert {r["model_id"] for r in rows} == {"tulu-base", "tulu-rlvr"}


class TestRunner:
    def test_produces_both_blocks_and_a_placebo_for_the_logistic_arm(self):
        result = run_e4(synthetic_extractions(), REGISTRY, log=lambda *_: None)
        arm = result["arms"]["logistic"]
        assert set(arm) == {"model", "unconstrained", "count_matched", "count_matched_verdict", "hypothesis"}
        assert "logistic" in result["placebo"]

    def test_evaluates_on_the_held_out_family_only(self):
        result = run_e4(synthetic_extractions(n_questions=10), REGISTRY, log=lambda *_: None)
        assert result["n_eval_pairs"] == 10 * 2  # questions x conditions, OLMo-2 only

    def test_no_model_is_selected_when_the_deberta_arm_is_absent(self):
        # Selecting a "primary" with only one arm fitted would report a
        # pre-registered ablation that was never actually run.
        result = run_e4(synthetic_extractions(), REGISTRY, log=lambda *_: None)
        assert "primary_model" not in result
        assert "h1_4" not in result

    def test_selects_a_primary_model_when_both_arms_are_present(self):
        def fake_fitter(texts, labels):
            return lambda batch: [0.9 if t != "(no citations)" else 0.1 for t in batch]

        result = run_e4(synthetic_extractions(), REGISTRY, deberta_fitter=fake_fitter, log=lambda *_: None)
        assert result["primary_model"] in {"logistic", "deberta"}
        assert result["h1_4"]["verdict"] in {"supported", "weak_indeterminate", "not_supported"}

    def test_an_indifferent_judge_lands_at_the_null(self):
        def flat_fitter(texts, labels):
            return lambda batch: [0.5] * len(batch)

        result = run_e4(synthetic_extractions(), REGISTRY, deberta_fitter=flat_fitter, log=lambda *_: None)
        deberta = result["arms"]["deberta"]["unconstrained"]
        assert deberta["preference_rate"] == pytest.approx(0.5)
        assert deberta["tie_rate"] == pytest.approx(1.0)

    def test_result_is_json_serialisable(self):
        result = run_e4(synthetic_extractions(n_questions=10), REGISTRY, log=lambda *_: None)
        assert json.loads(json.dumps(result))["n_eval_pairs"] == 20

    def test_an_empty_count_matched_subsample_is_reported_rather_than_raised(self):
        # The control failing to exist is a result about the data. Raising here
        # would take the unconstrained estimate down with it.
        records = synthetic_extractions(n_questions=10, base_cites_on_even=False)
        arm = run_e4(records, REGISTRY, log=lambda *_: None)["arms"]["logistic"]
        assert arm["count_matched"]["computable"] is False
        assert arm["count_matched_verdict"] == "not_computable"
        assert arm["unconstrained"]["n_pairs"] == 20
