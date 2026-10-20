"""Build the E3 probe training set from TruthfulQA and SciQ.

Pre-registration section 3 (E3): "Train probes on a separate TruthfulQA + SciQ
true/false set (~5k examples, 80/10/10 split) ... Probe training set fully
disjoint from any citation-elicitation question."

Two design choices here carry the weight of that sentence.

Each source question contributes exactly one true and one false answer. That
balances the classes within each source without subsampling, and it stops the
probe from learning the question rather than truthfulness, which is what happens
when a question appears under only one label.

Splitting is by question, never by row. The true and false answers to one
question share almost all of their text, so putting one in train and the other
in test leaks the answer and inflates AUROC against the registered 0.70
threshold.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

SPLIT_FRACTIONS = (0.8, 0.1, 0.1)

TRUTHFULQA_DATASET = ("truthfulqa/truthful_qa", "multiple_choice")
SCIQ_DATASET = ("allenai/sciq", "default")


@dataclass(frozen=True)
class ProbeExample:
    question: str
    answer: str
    label: int  # 1 = true statement, 0 = false statement
    source: str


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _check_no_citation_overlap(
    questions: Iterable[str], citation_questions: Sequence[str] | None
) -> None:
    if not citation_questions:
        return
    banned = {_normalize(q) for q in citation_questions}
    hits = sorted({q for q in questions if _normalize(q) in banned})
    if hits:
        raise ValueError(
            "probe questions overlap the citation-elicitation set, which would "
            f"turn E3's cross-task transfer claim into a within-task one: {hits[:5]}"
        )


def _from_truthfulqa(rows: Sequence[dict]) -> list[ProbeExample]:
    out: list[ProbeExample] = []
    for row in rows:
        targets = row.get("mc1_targets") or {}
        choices = list(targets.get("choices") or [])
        labels = list(targets.get("labels") or [])
        if len(choices) != len(labels):
            continue
        true_answers = [c for c, lab in zip(choices, labels) if lab == 1]
        false_answers = [c for c, lab in zip(choices, labels) if lab == 0]
        if not true_answers or not false_answers:
            continue
        question = row["question"]
        out.append(ProbeExample(question, true_answers[0], 1, "truthfulqa"))
        out.append(ProbeExample(question, false_answers[0], 0, "truthfulqa"))
    return out


def _from_sciq(rows: Sequence[dict]) -> list[ProbeExample]:
    out: list[ProbeExample] = []
    for row in rows:
        correct = row.get("correct_answer")
        distractor = row.get("distractor1")
        question = row.get("question")
        if not (correct and distractor and question):
            continue
        out.append(ProbeExample(question, correct, 1, "sciq"))
        out.append(ProbeExample(question, distractor, 0, "sciq"))
    return out


def build_probe_dataset(
    truthfulqa_rows: Sequence[dict],
    sciq_rows: Sequence[dict],
    citation_questions: Sequence[str] | None = None,
) -> list[ProbeExample]:
    """One true and one false answer per source question, contamination-checked."""
    rows = _from_truthfulqa(truthfulqa_rows) + _from_sciq(sciq_rows)
    if not rows:
        raise ValueError(
            "probe dataset is empty; both TruthfulQA and SciQ yielded no usable "
            "true/false pairs"
        )
    _check_no_citation_overlap({r.question for r in rows}, citation_questions)
    return rows


def split_probe_dataset(
    rows: Sequence[ProbeExample], seed: int
) -> tuple[list[ProbeExample], list[ProbeExample], list[ProbeExample]]:
    """Split 80/10/10 by question, so no question straddles two splits."""
    by_question: dict[str, list[ProbeExample]] = {}
    for row in rows:
        by_question.setdefault(row.question, []).append(row)

    questions = sorted(by_question)
    random.Random(seed).shuffle(questions)

    n = len(questions)
    n_train = int(n * SPLIT_FRACTIONS[0])
    n_val = int(n * SPLIT_FRACTIONS[1])
    bounds = (questions[:n_train], questions[n_train:n_train + n_val],
              questions[n_train + n_val:])
    return tuple([row for q in part for row in by_question[q]] for part in bounds)


def load_source_datasets(limit_per_source: int | None = None):
    """Pull the two public sources from HuggingFace.

    Kept separate from `build_probe_dataset` so the shaping logic stays testable
    without a network round-trip.
    """
    from datasets import load_dataset

    tqa_name, tqa_config = TRUTHFULQA_DATASET
    sciq_name, sciq_config = SCIQ_DATASET
    tqa = load_dataset(tqa_name, tqa_config, split="validation")
    sciq = load_dataset(sciq_name, sciq_config, split="train")
    if limit_per_source is not None:
        tqa = tqa.select(range(min(limit_per_source, len(tqa))))
        sciq = sciq.select(range(min(limit_per_source, len(sciq))))
    return list(tqa), list(sciq)
