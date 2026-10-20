"""Phase B verification runner.

Generalizes pilot_reverify.reverify() so the key includes trial_id +
model_id (Phase B runs five trials per (qid, condition, model); the
pilot ran one). Emits one VerificationRecord per extraction record,
preserving the per-citation status list in extracted order so the
aggregation layer can derive any_fabricated / any_misgrounded /
n_citations without re-running verification.

Domain routing (legal -> CourtListener first; academic/medical ->
Crossref / S2 / OpenAlex / PubMed) is handled inside verify_citation().
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.elicit import Question
from src.extract_citations import ExtractedCitation
from src.run_phase_b import load_phase_b_questions
from src.verify import VerificationCache, verify_citation


@dataclass
class VerificationRecord:
    """One row per generation; statuses are in extracted-citation order."""

    question_id: str
    condition: str
    model_id: str
    trial_id: int
    n_citations: int
    citation_statuses: list[str] = field(default_factory=list)
    verified_at: str = ""

    def to_jsonl_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_extractions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        rows.append(json.loads(raw))
    return rows


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def verify_extractions(
    *,
    extractions: Iterable[dict[str, Any]],
    qmap: dict[str, Question],
    cache_path: Path,
    verified_at: str | None = None,
) -> AsyncIterator[VerificationRecord]:
    """Verify every citation in every (non-errored) extraction record.

    Yields one VerificationRecord per extraction. Errored extractions
    (those carrying a truthy `error` field) are skipped so the output
    represents only actionable verification work.

    Raises KeyError if an extraction references a question_id absent
    from `qmap` — that would indicate the wrong --questions argument.
    """
    cache = VerificationCache(cache_path)
    stamp = verified_at or _now_iso()
    for record in extractions:
        if record.get("error"):
            continue
        qid = record["question_id"]
        if qid not in qmap:
            raise KeyError(
                f"extraction references unknown question_id {qid!r}; "
                f"qmap has {len(qmap)} entries"
            )
        question = qmap[qid]
        statuses: list[str] = []
        for cite_dict in record.get("citations", []):
            extracted = ExtractedCitation.from_dict(cite_dict)
            parsed = extracted.to_verify_citation(domain=question.domain)
            res = await verify_citation(
                extracted.raw_span,
                domain=question.domain,
                cache=cache,
                parsed=parsed,
            )
            statuses.append(res.status.value)
        yield VerificationRecord(
            question_id=qid,
            condition=record["condition"],
            model_id=record["model_id"],
            trial_id=int(record.get("trial_id", 0)),
            n_citations=len(statuses),
            citation_statuses=statuses,
            verified_at=stamp,
        )


def _load_existing_verification_keys(
    path: Path,
) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        keys.add(
            (
                record["question_id"],
                record["condition"],
                record["model_id"],
                int(record.get("trial_id", 0)),
            )
        )
    return keys


async def run(
    *,
    extractions_path: Path,
    qmap: dict[str, Question],
    output_path: Path,
    cache_path: Path,
) -> None:
    """End-to-end runner: load extractions, verify, append to output JSONL.

    Resume-on-restart: any (qid, condition, model_id, trial_id) already in
    the output is skipped so a re-run after a crash does not duplicate
    rows or re-hit external verification APIs for already-finished work.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_verification_keys(output_path)
    extractions = load_extractions(extractions_path)
    todo = [
        e
        for e in extractions
        if not e.get("error")
        and (
            e["question_id"],
            e["condition"],
            e["model_id"],
            int(e.get("trial_id", 0)),
        )
        not in existing
    ]
    with output_path.open("a", encoding="utf-8") as fh:
        async for record in verify_extractions(
            extractions=todo,
            qmap=qmap,
            cache_path=cache_path,
        ):
            fh.write(json.dumps(record.to_jsonl_dict()) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src.verify_batch",
        description="Phase B verification runner (one row per generation).",
    )
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=Path("data/verifications/phase_b_cache.jsonl")
    )
    return parser.parse_args(argv)


async def _amain(args: argparse.Namespace) -> None:
    questions = load_phase_b_questions(args.questions)
    qmap = {q.id: q for q in questions}
    await run(
        extractions_path=args.extractions,
        qmap=qmap,
        output_path=args.output,
        cache_path=args.cache,
    )


def main() -> None:
    asyncio.run(_amain(parse_args()))


if __name__ == "__main__":
    main()
