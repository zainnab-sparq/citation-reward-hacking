"""Tests for the E3 item-level difficulty proxy.

Pre-registration section 6, E3 guardrail (a): the difficulty proxy is "the base
model's log-probability of producing any citation for that item". It is
estimated here from the base model's own Phase C generations, one draw per
prompt condition, so the estimate is a smoothed log-rate over a handful of
draws and the number of draws is recorded next to it.
"""

import json
import math

import pytest

from src.e3.difficulty_proxy import (
    BASE_OF_PROBE_MODEL,
    build_difficulty_proxy,
    load_difficulty_proxy,
    write_difficulty_proxy,
)


def _row(question_id, model_id=BASE_OF_PROBE_MODEL, n_citations=0, condition="neutral"):
    return {
        "question_id": question_id,
        "model_id": model_id,
        "condition": condition,
        "trial_id": 0,
        "citations": [{"raw_span": f"c{i}"} for i in range(n_citations)],
    }


def test_registered_base_model_is_the_llama_base_of_tulu_3_8b():
    assert BASE_OF_PROBE_MODEL == "meta-llama/Llama-3.1-8B"


def test_proxy_is_the_smoothed_log_rate_of_any_citation():
    rows = [
        _row("q1", n_citations=2, condition="neutral"),
        _row("q1", n_citations=0, condition="citations_expected"),
        _row("q1", n_citations=1, condition="length_matched"),
        _row("q1", n_citations=0, condition="abstention_permitted"),
    ]
    proxy = build_difficulty_proxy(rows)
    assert proxy["q1"].n_draws == 4
    assert proxy["q1"].n_with_citation == 2
    # Laplace-style smoothing (k + 0.5) / (n + 1) keeps 0-of-n finite.
    assert proxy["q1"].log_prob_any_citation == pytest.approx(math.log(2.5 / 5))


def test_zero_citation_items_get_a_finite_low_value():
    rows = [_row("q1", n_citations=0, condition=c) for c in ("a", "b", "c", "d")]
    proxy = build_difficulty_proxy(rows)
    value = proxy["q1"].log_prob_any_citation
    assert math.isfinite(value)
    assert value == pytest.approx(math.log(0.5 / 5))


def test_only_the_base_model_rows_count():
    rows = [
        _row("q1", n_citations=1, condition="neutral"),
        _row("q1", model_id="allenai/Llama-3.1-Tulu-3-8B", n_citations=5, condition="neutral"),
    ]
    proxy = build_difficulty_proxy(rows)
    assert proxy["q1"].n_draws == 1
    assert proxy["q1"].n_with_citation == 1


def test_a_different_model_set_can_be_requested_explicitly():
    rows = [
        _row("q1", model_id="base-a", n_citations=1),
        _row("q1", model_id="base-b", n_citations=0),
    ]
    proxy = build_difficulty_proxy(rows, model_ids=("base-a", "base-b"))
    assert proxy["q1"].n_draws == 2


def test_questions_without_base_draws_are_absent_not_defaulted():
    rows = [_row("q1", n_citations=1)]
    proxy = build_difficulty_proxy(rows)
    assert "q2" not in proxy


def test_round_trip_through_jsonl(tmp_path):
    rows = [_row("q1", n_citations=1), _row("q2", n_citations=0)]
    proxy = build_difficulty_proxy(rows)
    path = tmp_path / "proxy.jsonl"
    write_difficulty_proxy(proxy, path)
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert {l["question_id"] for l in lines} == {"q1", "q2"}
    assert all("n_draws" in l and "log_prob_any_citation" in l for l in lines)
    loaded = load_difficulty_proxy(path)
    assert loaded == {q: p.log_prob_any_citation for q, p in proxy.items()}
