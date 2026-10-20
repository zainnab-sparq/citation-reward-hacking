"""E4 pair construction: base against post-trained, holding everything else fixed.

H1.4 asks whether a naive citation reviewer prefers post-trained output over base
output. A "preference" is only interpretable if the two sides differ in
post-training and nothing else, so a pair is formed within a single question, a
single prompt condition, and a single model family. Domain, question difficulty,
and prompt wording are therefore identical across every pair by construction, and
none of them need to be modelled.

TWO CHOICES FIXED HERE, both settled 2026-08-13 before any model was fit:

1. PRIMARY PAIRS ARE BASE AGAINST THE FINAL CHECKPOINT ONLY, not base against
   each of SFT/DPO/RLVR. §5 names "cross-family base/Instruct deltas", which
   points at the base and the Instruct/final checkpoint specifically. It also
   keeps pairs independent: reusing one base generation in three pairs would
   make a plain bootstrap CI too narrow, and E4 registers no clustered
   bootstrap. The per-stage version, if reported at all, is a labelled
   exploratory addendum.

2. PLACEBO PAIRS ARE SFT AGAINST DPO. The pre-registered placebo evaluates the
   judge on two post-trained outputs, where no direction is expected. Using the
   two intermediate stages keeps the placebo from sharing generations with the
   primary, so a quirk in the final checkpoint cannot move both at once.
   Orientation is by stage ordinal, which is arbitrary with respect to quality
   and is exactly what makes 50% the null.

Both details are recorded in `docs/e4_surface_feature_spec.md`.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.analysis.e4_surface_features import surface_features

BASE_ORDINAL = 0
FINAL_ORDINAL = 3
PLACEBO_ORDINALS = (1, 2)

TRAIN_FAMILIES = ("tulu3_8b", "tulu3_70b")
EVAL_FAMILIES = ("olmo2_13b",)

COUNT_MATCH_TOLERANCE = 1
PRIMARY = "primary"
PLACEBO = "placebo"


@dataclass(frozen=True)
class Pair:
    """One base/post-trained comparison. `low` and `high` are stage positions,
    not quality judgements: `high` is the later checkpoint."""

    question_id: str
    condition: str
    kind: str
    split: str
    low_model_id: str
    high_model_id: str
    low_family: str
    high_family: str
    low_features: dict[str, float]
    high_features: dict[str, float]
    expected_preference: str | None

    @property
    def count_delta(self) -> int:
        return int(self.high_features["citation_count"] - self.low_features["citation_count"])

    @property
    def is_count_matched(self) -> bool:
        return abs(self.count_delta) <= COUNT_MATCH_TOLERANCE


def load_registry(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def family_checkpoints(registry: Iterable[Mapping[str, Any]]) -> dict[str, dict[int, str]]:
    """`family -> {stage_ordinal: model_id}`."""
    checkpoints: dict[str, dict[int, str]] = {}
    for record in registry:
        family = checkpoints.setdefault(record["model_family"], {})
        family[int(record["stage_ordinal"])] = record["model_id"]
    return checkpoints


def _split_of(family: str) -> str | None:
    if family in TRAIN_FAMILIES:
        return "train"
    if family in EVAL_FAMILIES:
        return "eval"
    return None


def _index_extractions(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    """Errored records are dropped here rather than scored as zero citations: an
    extraction failure is not evidence that the model cited nothing."""
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for record in records:
        if record.get("error"):
            continue
        indexed[(record["question_id"], record["model_id"], record["condition"])] = record
    return indexed


def _ordinals_for(kind: str) -> tuple[int, int]:
    if kind == PRIMARY:
        return BASE_ORDINAL, FINAL_ORDINAL
    if kind == PLACEBO:
        return PLACEBO_ORDINALS
    raise ValueError(f"unknown pair kind {kind!r}; expected {PRIMARY!r} or {PLACEBO!r}")


def build_pairs(
    extractions: Iterable[Mapping[str, Any]],
    registry: Iterable[Mapping[str, Any]],
    *,
    kind: str = PRIMARY,
) -> list[Pair]:
    """Every pair holding question, condition, and family fixed, in stable order.

    A pair is emitted only when both sides are present and error-free, so a
    missing generation removes the comparison instead of biasing it.
    """
    low_ordinal, high_ordinal = _ordinals_for(kind)
    checkpoints = family_checkpoints(registry)
    indexed = _index_extractions(extractions)
    keys = sorted({(question_id, condition) for question_id, _, condition in indexed})

    pairs: list[Pair] = []
    for question_id, condition in keys:
        for family in sorted(checkpoints):
            split = _split_of(family)
            if split is None:
                continue
            pair = _pair_for(indexed, checkpoints[family], family, split, question_id, condition, kind, low_ordinal, high_ordinal)
            if pair is not None:
                pairs.append(pair)
    return pairs


def _pair_for(
    indexed: Mapping[tuple[str, str, str], Mapping[str, Any]],
    stages: Mapping[int, str],
    family: str,
    split: str,
    question_id: str,
    condition: str,
    kind: str,
    low_ordinal: int,
    high_ordinal: int,
) -> Pair | None:
    low_model_id = stages.get(low_ordinal)
    high_model_id = stages.get(high_ordinal)
    if low_model_id is None or high_model_id is None:
        return None
    low = indexed.get((question_id, low_model_id, condition))
    high = indexed.get((question_id, high_model_id, condition))
    if low is None or high is None:
        return None
    return Pair(
        question_id=question_id,
        condition=condition,
        kind=kind,
        split=split,
        low_model_id=low_model_id,
        high_model_id=high_model_id,
        low_family=family,
        high_family=family,
        low_features=surface_features(low.get("citations")),
        high_features=surface_features(high.get("citations")),
        expected_preference="high" if kind == PRIMARY else None,
    )
