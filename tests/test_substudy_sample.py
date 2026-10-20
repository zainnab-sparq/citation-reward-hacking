"""Tests for the EXISTENT_WRONG_ID sub-study sampler (amendment #6 #3).

The sub-study draws a stratified subsample of the verifier-REAL citations that
J2a accepted untouched, re-judges every one, and measures the disagreement rate
(judge != REAL) against the 8% sub-amendment threshold. These tests pin the
allocation math and the deterministic stratified draw.
"""
from __future__ import annotations

from src.analysis.substudy_sample import (
    SUBSTUDY_FLOOR,
    SUBSTUDY_N,
    SUBSTUDY_SEED,
    select_substudy_sample,
    stratified_allocation,
)
from src.judge_via_subagents import JudgeInput


def _make_input(qid: str, cidx: int, condition: str = "neutral") -> JudgeInput:
    cid = f"{qid}__h45d__{condition}__t0__c{cidx}"
    return JudgeInput(
        custom_id=cid,
        question_id=qid,
        model_snapshot="h45d",
        condition=condition,
        trial_id=0,
        citation_index=cidx,
        verifier_status="REAL",
        prompt=f"prompt for {cid}",
    )


# ---- stratified_allocation ----------------------------------------------------


def test_allocation_sums_to_total_when_capacity_allows():
    sizes = {"a": 100, "b": 50, "c": 30}
    alloc = stratified_allocation(sizes, 60, floor=5)
    assert sum(alloc.values()) == 60


def test_allocation_respects_floor_on_small_strata():
    sizes = {"big": 1000, "tiny": 8}
    alloc = stratified_allocation(sizes, 100, floor=5)
    assert alloc["tiny"] >= 5
    assert sum(alloc.values()) == 100


def test_allocation_never_exceeds_stratum_size():
    # 'small' has only 3 available; even a generous budget cannot draw more.
    sizes = {"big": 1000, "small": 3}
    alloc = stratified_allocation(sizes, 100, floor=5)
    assert alloc["small"] <= 3
    assert all(alloc[k] <= sizes[k] for k in sizes)


def test_allocation_caps_to_capacity_when_total_exceeds_population():
    sizes = {"a": 10, "b": 7}
    alloc = stratified_allocation(sizes, 100, floor=5)
    assert sum(alloc.values()) == 17  # cannot sample more than exist
    assert alloc == {"a": 10, "b": 7}


def test_allocation_skips_empty_strata():
    sizes = {"a": 50, "empty": 0, "b": 50}
    alloc = stratified_allocation(sizes, 40, floor=5)
    assert "empty" not in alloc or alloc["empty"] == 0
    assert sum(alloc.values()) == 40


def test_allocation_is_proportional_monotone():
    sizes = {"a": 900, "b": 90, "c": 10}
    alloc = stratified_allocation(sizes, 100, floor=2)
    assert alloc["a"] >= alloc["b"] >= alloc["c"]


def test_allocation_is_deterministic():
    sizes = {"a": 137, "b": 211, "c": 19, "d": 5}
    a1 = stratified_allocation(sizes, 80, floor=3)
    a2 = stratified_allocation(sizes, 80, floor=3)
    assert a1 == a2


# ---- select_substudy_sample ---------------------------------------------------


def _population(per_stratum: dict[tuple[str, str], int]):
    """Build a synthetic verifier-REAL population + qmap covering given strata."""
    inputs = []
    qmap = {}
    for (domain, cell), n in per_stratum.items():
        for i in range(n):
            qid = f"{domain[:3]}_{cell[:3]}_{i:04d}"
            qmap[qid] = {"id": qid, "domain": domain, "cell": cell}
            inputs.append(_make_input(qid, 0))
    return inputs, qmap


def test_select_returns_exactly_n_when_population_allows():
    per = {("academic", "known_real"): 500, ("medical", "obscure_real"): 300,
           ("legal", "unanswerable"): 50}
    inputs, qmap = _population(per)
    sample = select_substudy_sample(inputs, qmap, n=100, floor=5, seed=1)
    assert len(sample.selected) == 100


def test_select_composition_matches_allocation():
    per = {("academic", "known_real"): 500, ("medical", "obscure_real"): 300,
           ("legal", "unanswerable"): 50}
    inputs, qmap = _population(per)
    sample = select_substudy_sample(inputs, qmap, n=100, floor=5, seed=1)
    from collections import Counter
    got = Counter(
        (qmap[i.question_id]["domain"], qmap[i.question_id]["cell"])
        for i in sample.selected
    )
    assert dict(got) == sample.allocation


def test_select_records_population_sizes():
    per = {("academic", "known_real"): 500, ("medical", "obscure_real"): 300}
    inputs, qmap = _population(per)
    sample = select_substudy_sample(inputs, qmap, n=80, floor=5, seed=1)
    assert sample.population == per


def test_select_is_deterministic_for_fixed_seed():
    per = {("academic", "known_real"): 500, ("medical", "obscure_real"): 300,
           ("legal", "known_real"): 120}
    inputs, qmap = _population(per)
    s1 = select_substudy_sample(inputs, qmap, n=100, floor=5, seed=42)
    s2 = select_substudy_sample(inputs, qmap, n=100, floor=5, seed=42)
    assert [i.custom_id for i in s1.selected] == [i.custom_id for i in s2.selected]


def test_select_draws_only_from_population():
    per = {("academic", "known_real"): 500, ("medical", "obscure_real"): 300}
    inputs, qmap = _population(per)
    sample = select_substudy_sample(inputs, qmap, n=80, floor=5, seed=1)
    pop_ids = {i.custom_id for i in inputs}
    assert all(i.custom_id in pop_ids for i in sample.selected)
    assert len({i.custom_id for i in sample.selected}) == len(sample.selected)


def test_select_different_seed_changes_draw():
    per = {("academic", "known_real"): 500, ("medical", "obscure_real"): 300}
    inputs, qmap = _population(per)
    s1 = select_substudy_sample(inputs, qmap, n=80, floor=5, seed=1)
    s2 = select_substudy_sample(inputs, qmap, n=80, floor=5, seed=2)
    assert [i.custom_id for i in s1.selected] != [i.custom_id for i in s2.selected]


def test_module_constants_are_frozen():
    assert SUBSTUDY_N == 200
    assert SUBSTUDY_FLOOR == 5
    assert isinstance(SUBSTUDY_SEED, int)
