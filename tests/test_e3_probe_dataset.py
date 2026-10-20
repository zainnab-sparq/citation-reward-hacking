"""Tests for the E3 probe training set.

The pre-registration (section 3, E3) requires a TruthfulQA + SciQ true/false set
of roughly 5k examples on an 80/10/10 split, and section 9 requires it to be
fully disjoint from the citation-elicitation questions. Both are properties a
test can pin, and both are the kind of thing that silently rots: a resplit with
a different seed, or a dataset swap that quietly lets a citation question in,
would invalidate the cross-task transfer claim that E3 rests on.
"""

import pytest

from src.e3.probe_dataset import (
    SPLIT_FRACTIONS,
    ProbeExample,
    build_probe_dataset,
    split_probe_dataset,
)


def _fake_truthfulqa():
    return [
        {
            "question": f"TQA question {i}?",
            "mc1_targets": {
                "choices": [f"true answer {i}", f"false answer {i}a", f"false answer {i}b"],
                "labels": [1, 0, 0],
            },
        }
        for i in range(40)
    ]


def _fake_sciq():
    return [
        {
            "question": f"SciQ question {i}?",
            "correct_answer": f"correct {i}",
            "distractor1": f"wrong {i}a",
            "distractor2": f"wrong {i}b",
            "distractor3": f"wrong {i}c",
        }
        for i in range(40)
    ]


def test_builds_labeled_true_false_pairs():
    rows = build_probe_dataset(_fake_truthfulqa(), _fake_sciq())
    assert rows, "expected a non-empty probe dataset"
    assert all(isinstance(r, ProbeExample) for r in rows)
    assert {r.label for r in rows} == {0, 1}


def test_every_source_question_contributes_both_a_true_and_a_false_answer():
    # A probe trained on questions that only ever appear with one label learns
    # the question, not truthfulness.
    rows = build_probe_dataset(_fake_truthfulqa(), _fake_sciq())
    by_q = {}
    for r in rows:
        by_q.setdefault(r.question, set()).add(r.label)
    assert all(labels == {0, 1} for labels in by_q.values())


def test_both_sources_are_represented_and_tagged():
    rows = build_probe_dataset(_fake_truthfulqa(), _fake_sciq())
    assert {r.source for r in rows} == {"truthfulqa", "sciq"}


def test_labels_are_balanced_within_each_source():
    # Class imbalance would make AUROC hard to read against the registered 0.70
    # threshold, and the sources have different natural distractor counts.
    rows = build_probe_dataset(_fake_truthfulqa(), _fake_sciq())
    for source in ("truthfulqa", "sciq"):
        subset = [r for r in rows if r.source == source]
        assert sum(r.label for r in subset) * 2 == len(subset)


def test_split_fractions_are_the_registered_80_10_10():
    assert SPLIT_FRACTIONS == (0.8, 0.1, 0.1)


def test_split_is_disjoint_and_covers_everything():
    rows = build_probe_dataset(_fake_truthfulqa(), _fake_sciq())
    train, val, test = split_probe_dataset(rows, seed=42)
    ids = [{id(r) for r in part} for part in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(train) + len(val) + len(test) == len(rows)


def test_split_keeps_a_question_entirely_within_one_split():
    # The true and false answers to one question are near-duplicates. Splitting
    # them across train and test leaks the answer and inflates AUROC.
    rows = build_probe_dataset(_fake_truthfulqa(), _fake_sciq())
    train, val, test = split_probe_dataset(rows, seed=42)
    seen = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        for r in part:
            assert seen.setdefault(r.question, name) == name


def test_split_is_deterministic_for_a_seed_and_changes_with_it():
    rows = build_probe_dataset(_fake_truthfulqa(), _fake_sciq())
    a = [r.answer for r in split_probe_dataset(rows, seed=42)[0]]
    b = [r.answer for r in split_probe_dataset(rows, seed=42)[0]]
    c = [r.answer for r in split_probe_dataset(rows, seed=7)[0]]
    assert a == b
    assert a != c


def test_rejects_a_citation_question_leaking_into_the_probe_set():
    # Section 9 contamination rule: the probe set must be disjoint from the
    # citation-elicitation questions. A silent overlap would turn a cross-task
    # transfer claim into a within-task one.
    tqa = _fake_truthfulqa()
    tqa[0]["question"] = "Which paper established construct validity?"
    with pytest.raises(ValueError, match="overlap"):
        build_probe_dataset(
            tqa, _fake_sciq(),
            citation_questions=["Which paper established construct validity?"],
        )


def test_overlap_check_ignores_case_and_surrounding_whitespace():
    tqa = _fake_truthfulqa()
    tqa[0]["question"] = "  WHICH Paper Established Construct Validity?  "
    with pytest.raises(ValueError, match="overlap"):
        build_probe_dataset(
            tqa, _fake_sciq(),
            citation_questions=["Which paper established construct validity?"],
        )


def test_empty_sources_fail_loudly_rather_than_yielding_an_empty_probe_set():
    with pytest.raises(ValueError):
        build_probe_dataset([], [])
