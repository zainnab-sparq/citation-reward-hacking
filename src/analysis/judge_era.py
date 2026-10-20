"""Derive judge-era and grounding covariates from the J2a judgments file.

The J2a judge ran over roughly two months and its behaviour changed twice
mid-collection (see `scripts/_judge_drift_three_era.py`). Nothing in the
judgment record stores a timestamp, so the only evidence of WHEN a citation was
judged is its position in the append-only output JSONL, which is written in
judging order.

This module turns that row order into two covariates the drift-interaction
models need, at the generation level where the registered `fabricated` outcome
lives:

  * `era`      3-level factor, boundaries fixed from grounding metrics
  * `mean_search_count` / `n_search_zero`  continuous grounding, which lets the
    same question be asked without drawing era boundaries at all

Specification: `docs/drift_interaction_prespec.md`. Neither registered
confirmatory script is touched.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

BLOCK_SIZE = 500
# Boundaries are the empirical regime changes in judge grounding (block 33 and
# block 48), established BEFORE the interaction test was specified and never
# re-tuned to the outcome. See prespec §3.
EARLY_END = 33 * BLOCK_SIZE
DEGRADED_END = 48 * BLOCK_SIZE

MIXED_ERA = "MIXED"

GEN_KEYS = ("question_id", "condition", "model_id", "trial_id")


def era_of_row_index(index: int) -> str:
    if index < EARLY_END:
        return "EARLY"
    if index < DEGRADED_END:
        return "DEGRADED"
    return "RECOVERED"


def parse_custom_id(custom_id: str) -> tuple[str, str, str, int, int]:
    """`<question_id>__<model_id>__<condition>__t<trial>__c<citation_index>`.

    The model_id may contain '/' (open-weights ids like `allenai/OLMo-2-...`),
    which is why the split is on '__' and never on '/'.
    """
    question_id, model_id, condition, trial, citation = custom_id.split("__")
    return question_id, model_id, condition, int(trial[1:]), int(citation[1:])


def load_judgment_meta(path: Path) -> list[dict[str, Any]]:
    """One record per judged citation, carrying its file position and era.

    `row_index` is the position among non-blank lines, INCLUDING any row skipped
    for carrying an error. Dropping an error row must not renumber the rows after
    it, or era assignment here would silently disagree with the standing drift
    audit scripts, which enumerate every line.
    """
    out: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        index = -1
        for line in handle:
            if not line.strip():
                continue
            index += 1
            record = json.loads(line)
            if record.get("error") is not None:
                continue
            question_id, model_id, condition, trial_id, citation_index = parse_custom_id(
                record["custom_id"]
            )
            out.append(
                {
                    "custom_id": record["custom_id"],
                    "row_index": index,
                    "block": index // BLOCK_SIZE,
                    "era": era_of_row_index(index),
                    "question_id": question_id,
                    "model_id": model_id,
                    "condition": condition,
                    "trial_id": trial_id,
                    "citation_index": citation_index,
                    "judge_status": record.get("judge_status"),
                    "judge_search_count": record.get("judge_search_count"),
                }
            )
    return out


def generation_era_rows(meta: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Roll citation-level meta up to one row per generation.

    A generation whose citations straddle an era boundary has no well-defined
    era and is labelled MIXED rather than being assigned one, so the drift script
    can drop it and report how many it dropped (prespec §4).
    """
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for record in meta:
        grouped.setdefault(tuple(record[key] for key in GEN_KEYS), []).append(record)

    rows: list[dict[str, Any]] = []
    for key, records in grouped.items():
        eras = {r["era"] for r in records}
        counts = [r["judge_search_count"] for r in records if r["judge_search_count"] is not None]
        rows.append(
            {
                **dict(zip(GEN_KEYS, key)),
                "era": next(iter(eras)) if len(eras) == 1 else MIXED_ERA,
                "n_eras": len(eras),
                "n_judged": len(records),
                "mean_search_count": sum(counts) / len(counts) if counts else None,
                "n_search_zero": sum(1 for c in counts if c == 0),
                "first_row_index": min(r["row_index"] for r in records),
            }
        )
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit generation-level judge era + grounding covariates."
    )
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    meta = load_judgment_meta(args.judgments)
    rows = generation_era_rows(meta)
    write_jsonl(rows, args.output)

    by_era: dict[str, int] = {}
    for row in rows:
        by_era[row["era"]] = by_era.get(row["era"], 0) + 1
    print(f"judged citations: {len(meta)}")
    print(f"generations:      {len(rows)}")
    for era in ("EARLY", "DEGRADED", "RECOVERED", MIXED_ERA):
        if era in by_era:
            print(f"  {era:<10} {by_era[era]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
