"""EXISTENT_WRONG_ID sub-study sampler (PRE_REGISTRATION_OSF.md amendment #6 #3).

J2a accepts verifier-REAL citations at face value (the judge never fires on
them). If some of those are actually EXISTENT_WRONG_ID fabrications, the trust
step silently leaks them into the "good" pile. This module draws a stratified
subsample of the verifier-REAL population so each one can be re-judged by the
full strict-hybrid judge; the disagreement rate (judge != REAL) is then scored
against the registered 8% sub-amendment threshold.

Stratification: (domain x cell), 9 strata. Allocation is proportional to
stratum size with a floor of SUBSTUDY_FLOOR per non-empty stratum so the thin,
higher-risk obscure/unanswerable/legal strata are always spot-checked. Because
the headline disagreement rate is estimated population-reweighted
(sum_h (N_h / N) * r_h), the floor improves thin-stratum precision without
biasing the pooled estimate the threshold targets.

The draw is deterministic: a fixed SUBSTUDY_SEED plus a stable sort by
custom_id makes the selected subsample reproducible from the frozen artifacts
alone, as a pre-registered sample must be.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from src.judge_via_subagents import JudgeInput

SUBSTUDY_N = 200
SUBSTUDY_FLOOR = 5
# Frozen seed: the originally-scheduled sub-study date (2026-06-14), kept stable
# so the draw is reproducible regardless of when the study actually runs.
SUBSTUDY_SEED = 20260614

_Stratum = tuple[str, str]  # (domain, cell)


def stratified_allocation(
    sizes: dict[Any, int], total: int, *, floor: int
) -> dict[Any, int]:
    """Allocate `total` draws across strata proportional to size, with a floor.

    Rules: every non-empty stratum gets at least min(floor, size); no stratum
    gets more than its size; the remainder is water-filled one unit at a time to
    whichever eligible stratum is most under its proportional share. If `total`
    exceeds the population, the whole population is taken (each stratum = size).
    Deterministic.
    """
    keys = sorted(k for k, n in sizes.items() if n > 0)
    capacity = sum(sizes[k] for k in keys)
    target = min(total, capacity)
    total_size = sum(sizes[k] for k in keys)

    alloc = {k: min(floor, sizes[k]) for k in keys}

    # Floors can overshoot a small target; trim from the most-allocated stratum.
    while sum(alloc.values()) > target:
        k = max(keys, key=lambda k: (alloc[k], -sizes[k]))
        alloc[k] -= 1

    # Water-fill the remainder by largest proportional deficit.
    while sum(alloc.values()) < target:
        nxt_total = sum(alloc.values()) + 1
        candidates = [k for k in keys if alloc[k] < sizes[k]]

        def deficit(k: Any) -> float:
            return nxt_total * sizes[k] / total_size - alloc[k]

        k = max(candidates, key=lambda k: (deficit(k), sizes[k], k))
        alloc[k] += 1

    return alloc


@dataclass(frozen=True)
class SubstudySample:
    """A frozen stratified draw plus the population sizes needed to reweight the
    headline disagreement rate back to the verifier-REAL population."""
    selected: list[JudgeInput]
    allocation: dict[_Stratum, int]
    population: dict[_Stratum, int]
    seed: int
    n_requested: int


def _stratum_of(inp: JudgeInput, qmap: dict[str, dict[str, Any]]) -> _Stratum:
    row = qmap[inp.question_id]
    return (row["domain"], row["cell"])


def select_substudy_sample(
    inputs: list[JudgeInput],
    qmap: dict[str, dict[str, Any]],
    *,
    n: int = SUBSTUDY_N,
    floor: int = SUBSTUDY_FLOOR,
    seed: int = SUBSTUDY_SEED,
) -> SubstudySample:
    """Draw a deterministic stratified subsample of verifier-REAL JudgeInputs.

    `inputs` is the verifier-REAL population (build_judge_inputs with
    fire_on_statuses=("REAL",)); `qmap` maps question_id -> {domain, cell}.
    """
    by_stratum: dict[_Stratum, list[JudgeInput]] = defaultdict(list)
    for inp in inputs:
        by_stratum[_stratum_of(inp, qmap)].append(inp)

    population = {s: len(items) for s, items in by_stratum.items()}
    allocation = stratified_allocation(population, n, floor=floor)

    rng = random.Random(seed)
    selected: list[JudgeInput] = []
    for stratum in sorted(allocation):
        k = allocation[stratum]
        if k <= 0:
            continue
        pool = sorted(by_stratum[stratum], key=lambda i: i.custom_id)
        selected.extend(rng.sample(pool, k))

    return SubstudySample(
        selected=selected,
        allocation=allocation,
        population=population,
        seed=seed,
        n_requested=n,
    )
