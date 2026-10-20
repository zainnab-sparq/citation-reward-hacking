"""Build a fabrication-enriched blind hand-check sample from the Phase C corpus.

WHY THIS EXISTS. The J2a judge's only human validation is the Step 0 check: 50
citations, of which just 6 were fabricated. That set was drawn to be
representative, and for a rare class "representative" means starved of
positives. The consequences, recomputed 2026-08-11:

    production J2a           48/50 = 96.0%  [86.5, 98.9]
    trivial "always REAL"    44/50 = 88.0%  [76.2, 94.4]   McNemar p = 0.125
    FABRICATED recall         4/6  = 66.7%  [30.0, 90.3]

So the existing check cannot separate the pipeline from a judge that says REAL
to everything, and the recall estimate on the class that defines the study's
outcome variable is nearly uninformative. (It is not that the judge does
nothing: the structured verifier ALONE scores 23/50 = 46.0%, so the judge layer
carries the pipeline from 46% to 96%. It is that this sample cannot measure the
part that matters.)

Oversampling the positive class fixes the precision problem, and design weights
keep the population estimate unbiased. Around 60 fabricated events takes the
recall interval from roughly +/-30pp to +/-13pp.

TWO DELIBERATE DEVIATIONS FROM THE STEP 0 DESIGN, both tightening it:

  1. No pipeline evidence is shown. Step 0's blind CSV pre-filled an
     `evidence_summary` drawn from the verifier's lookups. The verifier is part
     of the pipeline under validation, so pre-filling its findings risks
     measuring anchoring rather than accuracy. Here the annotator does their own
     search and records it in `your_evidence_url`.
  2. The sample is enriched rather than representative, with weights.

Both make the new check a stricter test than Step 0, so the two numbers are not
directly comparable and the paper should not present them as a trend.

Usage:
  python -m src.analysis.handcheck_sample \
      --extractions data/extractions/phase_c/extractions.jsonl \
      --verifications data/verifications/phase_c/verifications.jsonl \
      --judgments data/judgments/phase_c/claude-subagent-sonnet.jsonl \
      --generations data/generations/phase_c/*.jsonl \
      --questions data/questions/frozen_900q_legal_amended.jsonl \
      --output-dir data/handcheck/phase_c
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from src.analysis.build_handcheck_csv import _excerpt
from src.analysis.judge_era import era_of_row_index, parse_custom_id

SAMPLE_SEED = 20260811

# Enriched on purpose. judge_FABRICATED and judge_REAL_rescue are where the
# judge's decisions actually live, so they carry most of the sample;
# verifier_REAL_trusted is included as a small check that the Phase B trust-step
# result still holds on the Phase C corpus.
DEFAULT_ALLOCATION = {
    "judge_FABRICATED": 60,
    "judge_REAL_rescue": 45,
    "judge_UNVERIFIABLE": 25,
    "verifier_REAL_trusted": 20,
}

# Anything that would tell the annotator what the pipeline concluded. None of
# these may reach the blind file.
LEAKY_FIELDS = (
    "verifier_status",
    "judge_status",
    "combined_status",
    "judge_reasoning",
    "judge_evidence_url",
    "judge_search_count",
    "stratum",
    "design_weight",
    "stratum_population",
    "era",
)

BLIND_COLUMNS = [
    "presentation_order",
    "row_id",
    "question_id",
    "domain",
    "cell",
    "condition",
    "model_id",
    "citation_index",
    "raw_span",
    "title",
    "case_name",
    "year",
    "venue",
    "doi",
    "output_excerpt",
    "your_status",        # REAL / FABRICATED / UNVERIFIABLE
    "your_evidence_url",
    "your_notes",
]

REVEAL_COLUMNS = [
    "row_id",
    "custom_id",
    "stratum",
    "stratum_population",
    "design_weight",
    "verifier_status",
    "judge_status",
    "combined_status",
    "era",
]

_JUDGE_WINS = ("REAL", "FABRICATED")


def _classify(verifier_status: str, judge_status: str | None) -> tuple[str, str]:
    """Return (stratum, combined_status) under the amendment #6(c) combine rule.

    `never_judged` is kept SEPARATE from `judge_UNVERIFIABLE`. Both fall back to
    the verifier's label, so it is tempting to merge them, but they are different
    objects: one is a judgment the judge made, the other is a judgment it has not
    made yet because judging is still running. Merging them contaminates the
    stratum and, worse, spends hand-labelling effort on citations whose combined
    status will change once the judge reaches them.
    """
    if verifier_status == "REAL":
        return "verifier_REAL_trusted", "REAL"
    if not judge_status:
        return "never_judged", verifier_status
    if judge_status in _JUDGE_WINS:
        stratum = "judge_FABRICATED" if judge_status == "FABRICATED" else "judge_REAL_rescue"
        return stratum, judge_status
    return "judge_UNVERIFIABLE", verifier_status


def build_citation_records(
    *,
    generations: Iterable[dict[str, Any]],
    extractions: Iterable[dict[str, Any]],
    verifications: Iterable[dict[str, Any]],
    judgments: Iterable[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """One record per citation, carrying everything both files will need."""
    def key_of(record):
        return (record["question_id"], record["condition"], record["model_id"],
                int(record.get("trial_id", 0)))

    gen_by_key = {key_of(g): g for g in generations}
    ext_by_key = {key_of(e): e for e in extractions}

    judge_by_cid: dict[str, dict[str, Any]] = {}
    for index, judgment in enumerate(judgments):
        if judgment.get("error") is not None:
            continue
        judgment["_era"] = era_of_row_index(index)
        judge_by_cid[judgment["custom_id"]] = judgment

    records: list[dict[str, Any]] = []
    row_id = 0
    for verification in verifications:
        key = key_of(verification)
        extraction, generation = ext_by_key.get(key), gen_by_key.get(key)
        if not extraction or not generation:
            continue
        question = questions.get(verification["question_id"], {})
        citations = extraction.get("citations", [])
        question_id, condition, model_id, trial_id = key
        for index, verifier_status in enumerate(verification.get("citation_statuses", [])):
            row_id += 1
            citation = citations[index] if index < len(citations) else {}
            custom_id = (f"{question_id}__{model_id}__{condition}"
                         f"__t{trial_id}__c{index}")
            judgment = judge_by_cid.get(custom_id)
            judge_status = judgment.get("judge_status") if judgment else None
            stratum, combined = _classify(verifier_status, judge_status)
            raw_span = citation.get("raw_span", "") or ""
            records.append({
                "row_id": row_id,
                "custom_id": custom_id,
                "stratum": stratum,
                "question_id": question_id,
                "domain": question.get("domain", ""),
                "cell": question.get("cell", ""),
                "condition": condition,
                "model_id": model_id,
                "citation_index": index,
                "raw_span": raw_span,
                "title": citation.get("title", "") or "",
                "case_name": citation.get("case_name", "") or "",
                "year": citation.get("year", "") or "",
                "venue": citation.get("venue", "") or "",
                "doi": citation.get("doi", "") or "",
                "output_excerpt": _excerpt(generation.get("output_text", "") or "", raw_span),
                "verifier_status": verifier_status,
                "judge_status": judge_status or "",
                "combined_status": combined,
                "era": judgment.get("_era", "") if judgment else "",
            })
    return records


def sample_stratified(
    records: Sequence[dict[str, Any]],
    allocation: dict[str, int] | None = None,
    seed: int = SAMPLE_SEED,
) -> list[dict[str, Any]]:
    """Draw the enriched sample and assign a shuffled presentation order.

    Presentation order is shuffled ACROSS strata, so position carries no
    information about what the pipeline concluded.
    """
    allocation = allocation or DEFAULT_ALLOCATION
    pool: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        pool.setdefault(record["stratum"], []).append(record)

    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for stratum in sorted(allocation):
        available = pool.get(stratum, [])
        n_drawn = min(allocation[stratum], len(available))
        if not n_drawn:
            continue
        weight = len(available) / n_drawn
        for record in rng.sample(available, n_drawn):
            sample.append({**record,
                           "stratum_population": len(available),
                           "design_weight": weight})

    order = list(range(1, len(sample) + 1))
    rng.shuffle(order)
    for record, position in zip(sample, order):
        record["presentation_order"] = position
    return sample


def _write_csv(rows: Sequence[dict[str, Any]], path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def write_blind_csv(sample: Sequence[dict[str, Any]], path: Path) -> None:
    """The file the annotator opens. Carries no pipeline opinion at all."""
    blanked = []
    for record in sorted(sample, key=lambda r: r["presentation_order"]):
        row = {c: record.get(c, "") for c in BLIND_COLUMNS}
        row["your_status"] = ""
        row["your_evidence_url"] = ""
        row["your_notes"] = ""
        blanked.append(row)
    _write_csv(blanked, path, BLIND_COLUMNS)


def write_reveal_csv(sample: Sequence[dict[str, Any]], path: Path) -> None:
    _write_csv(sorted(sample, key=lambda r: int(r["row_id"])), path, REVEAL_COLUMNS)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a fabrication-enriched blind hand-check sample.")
    parser.add_argument("--extractions", required=True, type=Path)
    parser.add_argument("--verifications", required=True, type=Path)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--generations", nargs="+", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args(argv)

    generations: list[dict[str, Any]] = []
    for path in args.generations:
        generations.extend(r for r in _read_jsonl(path) if not r.get("error"))
    questions = {q["id"]: q for q in _read_jsonl(args.questions)}

    records = build_citation_records(
        generations=generations,
        extractions=_read_jsonl(args.extractions),
        verifications=_read_jsonl(args.verifications),
        judgments=_read_jsonl(args.judgments),
        questions=questions,
    )
    print(f"citation records: {len(records)}")
    population: dict[str, int] = {}
    for record in records:
        population[record["stratum"]] = population.get(record["stratum"], 0) + 1
    for stratum in sorted(population):
        print(f"  {stratum:<24} {population[stratum]}")

    sample = sample_stratified(records, DEFAULT_ALLOCATION, args.seed)
    blind_path = args.output_dir / "phase_c_blind_handcheck.csv"
    reveal_path = args.output_dir / "phase_c_labels_reveal.csv"
    write_blind_csv(sample, blind_path)
    write_reveal_csv(sample, reveal_path)

    print(f"\nsampled {len(sample)}")
    drawn: dict[str, int] = {}
    for record in sample:
        drawn[record["stratum"]] = drawn.get(record["stratum"], 0) + 1
    for stratum in sorted(drawn):
        print(f"  {stratum:<24} {drawn[stratum]:>4}  weight "
              f"{population[stratum] / drawn[stratum]:.1f}")
    print(f"\nblind file  (label this one): {blind_path}")
    print(f"reveal file (do NOT open until labelling is finished): {reveal_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
