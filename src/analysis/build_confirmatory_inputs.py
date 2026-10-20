"""Build the JSONL inputs the registered E1/E2 scripts read.

`aggregate.py` targets the DESCRIPTIVE path: it writes CSVs that
`phase_b_descriptive.R` consumes. The registered confirmatory scripts
(`e1_checkpoint_cascade.R`, `e2_prompt_manipulation.R`) are a different
contract, and nothing produced it until now:

  * they read JSONL (`read_jsonl` -> `jsonlite::fromJSON`), not CSV;
  * they take a SINGLE `--generations` file, while Phase C has twelve;
  * they merge the registry on `family`, while the Phase C registry names
    that column `model_family` and additionally carries `stage_ordinal`,
    which E1 needs and `aggregate.to_models_frame` drops;
  * they expect one verification record per CITATION carrying `status`,
    while `verifications.jsonl` is nested one record per GENERATION;
  * they have no `--judgments` input at all, so without this step the J2a
    verdicts never reach the confirmatory model and E1/E2 would silently
    score raw verifier labels.

That last point is why this module exists rather than a shell one-liner: the
failure is silent, and E1/E2 run exactly once.

The J2a combine rule is NOT reimplemented here. `flatten_verifications`,
`build_judgment_index` and `apply_j2a` are imported from `aggregate` so the
restricted-firing rule (pre-reg amendment #6(c)) has one implementation and
one set of tests.

`questions` needs no adaptation: the frozen question file already carries
`id`, `cell` and `domain`, so point `--questions` at it directly.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from src.analysis.aggregate import (
    apply_j2a,
    build_judgment_index,
    flatten_verifications,
)

# Fields E1/E2 document for --generations. Projecting to a fixed key set keeps
# every emitted line the same shape, which matters because the R reader builds
# a one-row data.frame per line and rbinds them.
_GENERATION_FIELDS = ("question_id", "condition", "model_id", "trial_id", "output_text")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_confirmatory_models(path: Path) -> list[dict[str, Any]]:
    """Registry JSONL -> the columns E1/E2 merge on.

    Accepts either registry spelling: Phase B uses `family`, the Phase C
    checkpoint registry uses `model_family`. `stage_ordinal` is carried when
    present (E1 requires it) and is None for registries without a cascade.
    """
    out: list[dict[str, Any]] = []
    for record in _read_jsonl(path):
        if "family" in record:
            family = record["family"]
        elif "model_family" in record:
            family = record["model_family"]
        else:
            raise KeyError(
                f"registry record {record.get('model_id')!r} has neither "
                "'family' nor 'model_family'"
            )
        stage_ordinal = record.get("stage_ordinal")
        out.append(
            {
                "model_id": record["model_id"],
                "family": family,
                "stage_ordinal": None if stage_ordinal is None else int(stage_ordinal),
                "is_post_trained": int(record["is_post_trained"]),
                "is_open_weights": int(record["is_open_weights"]),
            }
        )
    return out


def to_confirmatory_questions(path: Path) -> list[dict[str, Any]]:
    """Frozen question JSONL -> the three columns E1/E2 merge on.

    The frozen file carries nested `expected_citation` and `metadata` objects.
    The R reader builds a one-row data.frame per line, and a nested object makes
    `as.data.frame` fail with "arguments imply differing number of rows", so the
    raw file is unreadable by E1/E2 as-is. Projection is what fixes it.
    """
    return [
        {"id": record["id"], "cell": record["cell"], "domain": record["domain"]}
        for record in _read_jsonl(path)
    ]


def apply_verifier_real_rejudgments(
    rows: Iterable[dict[str, Any]],
    rejudgment_index: dict[Any, str | None],
) -> list[dict[str, Any]]:
    """Overlay verdicts on citations the verifier called REAL.

    Amendment #6(c) accepts verifier-`REAL` at face value without judging. The
    2026-08-11 blind hand-check measured that trust step at 8 of 20 wrong on the
    Phase C corpus (40.0% [21.9, 61.3]) against 4.3% [1.7, 7.0] on Phase B under
    a registered 8% gate, biased toward false-REAL and differentially exposed by
    condition. This applies the corrective re-judge.

    Only verifier-`REAL_CONFIRMED` rows are eligible. Verifier-FAB/UNV citations
    belong to the main judge, and a stray re-judgment on one must not override
    its verdict a second time. Judge abstention or error leaves the verifier
    label standing, matching the main combine rule.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        eligible = row["verifier_status"] == "REAL_CONFIRMED"
        verdict = rejudgment_index.get(
            (row["question_id"], row["model_id"], row["condition"],
             row["trial_id"], row["citation_index"])
        ) if eligible else None
        applied = verdict in ("REAL", "FABRICATED")
        status = row["status"]
        if applied:
            status = "REAL_CONFIRMED" if verdict == "REAL" else "FABRICATED"
        out.append({**row, "status": status, "verifier_real_rejudged": applied})
    return out


def build_confirmatory_verifications(
    verification_paths: Sequence[Path],
    judgment_paths: Sequence[Path],
    verifier_real_rejudgment_paths: Sequence[Path] = (),
) -> list[dict[str, Any]]:
    """Nested verifications + J2a verdicts -> one row per citation.

    Each row carries `status` (the J2a-combined label E1/E2 model) and
    `verifier_status` (the pre-judge label, retained as an audit column).
    Citations the judge never fired on keep their verifier label.

    `verifier_real_rejudgment_paths` is OPT-IN. Passing nothing reproduces the
    registered amendment #6(c) pipeline exactly, so the correction can never be
    applied by accident; passing it is the explicit, reversible deviation the
    §11 sub-amendment records.
    """
    records: list[dict[str, Any]] = []
    for path in verification_paths:
        records.extend(_read_jsonl(path))
    judgments: list[dict[str, Any]] = []
    for path in judgment_paths:
        judgments.extend(_read_jsonl(path))
    rejudgments: list[dict[str, Any]] = []
    for path in verifier_real_rejudgment_paths:
        rejudgments.extend(_read_jsonl(path))

    index = build_judgment_index(judgments)
    rows = apply_j2a(flatten_verifications(records), index)
    return apply_verifier_real_rejudgments(rows, build_judgment_index(rejudgments))


def concat_generations(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Concatenate per-checkpoint generation files, dropping API-error rows.

    Mirrors `aggregate.to_generations_frame`: rows with a non-null `error` are
    excluded per pre-reg §6 exclusion #2.
    """
    out: list[dict[str, Any]] = []
    for path in paths:
        for record in _read_jsonl(path):
            if record.get("error"):
                continue
            row = {key: record.get(key) for key in _GENERATION_FIELDS}
            if row["output_text"] is None:
                row["output_text"] = ""
            out.append(row)
    return out


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build E1/E2 confirmatory JSONL inputs (J2a-merged)."
    )
    parser.add_argument("--verifications", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--judgments",
        nargs="*",
        default=[],
        type=Path,
        help="J2a judgment JSONL files. Omitting them scores raw verifier "
        "labels, which is NOT the registered outcome.",
    )
    parser.add_argument(
        "--verifier-real-rejudgments",
        nargs="*",
        default=[],
        type=Path,
        help="OPT-IN correction for the verifier-REAL trust step (see the §11 "
        "sub-amendment). Omitting it reproduces the registered amendment #6(c) "
        "pipeline exactly.",
    )
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--generations", nargs="+", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    verifications = build_confirmatory_verifications(
        args.verifications, args.judgments, args.verifier_real_rejudgments)
    models = to_confirmatory_models(args.registry)
    generations = concat_generations(args.generations)
    questions = to_confirmatory_questions(args.questions)

    write_jsonl(verifications, args.output_dir / "verifications.jsonl")
    write_jsonl(models, args.output_dir / "models.jsonl")
    write_jsonl(generations, args.output_dir / "generations.jsonl")
    write_jsonl(questions, args.output_dir / "questions.jsonl")

    judged = sum(1 for row in verifications if row["status"] != row["verifier_status"])
    print(f"verifications: {len(verifications)} citations ({judged} relabelled by J2a)")
    if args.verifier_real_rejudgments:
        corrected = sum(1 for row in verifications if row["verifier_real_rejudged"])
        flipped = sum(1 for row in verifications
                      if row["verifier_real_rejudged"] and row["status"] == "FABRICATED")
        print(f"  verifier-REAL correction APPLIED: {corrected} trusted citations "
              f"re-judged, {flipped} flipped to FABRICATED")
    print(f"models:        {len(models)}")
    print(f"generations:   {len(generations)}")
    print(f"questions:     {len(questions)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
