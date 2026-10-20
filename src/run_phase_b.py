"""Phase B CLI orchestrator: full Claude E1+E2 across the snapshot registry.

Glue layer over `generate_via_batch_api` from src/elicit.py:
  1. Load the frozen 900-question JSONL (mapping `question` -> `text`,
     preserving `cell` + `expected_citation` in Question.metadata for the
     aggregation join).
  2. Load the Phase B model registry from data/models/phase_b_models.jsonl.
  3. For each model (optionally filtered with --only-models), submit one
     batch with all 4 conditions x N trials of the question set.

All Phase B paid-API spend flows through this entrypoint. Snapshot
identity is rooted in the registry's `model_id` field (per pre-reg §11
amendment #2: dated build for Haiku 4.5; aliases for Opus 4.7 /
Sonnet 4.6, with response.model echoed into GenerationRecord.model_id
at fetch time).

Usage:
    python -m src.run_phase_b \
      --questions data/questions/frozen_900q_legal_amended.jsonl \
      --registry data/models/phase_b_models.jsonl \
      --output-dir data/generations/phase_b \
      --manifest data/generations/phase_b/batches.manifest.jsonl \
      --trials 5

To restrict to one model (e.g. Step 1 Haiku-only checkpoint):
    ... --only-models claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from src.elicit import (
    GenerationSpec,
    Question,
    generate_via_batch_api,
)
from src.pilot import CONDITIONS


_DEFAULT_TRIALS = 5  # §16: closed-API default at default temperature.
_NON_METADATA_FIELDS = frozenset({"id", "domain", "question", "expected_existence"})


def load_phase_b_questions(path: Path) -> list[Question]:
    """Read the frozen 900-question JSONL into Question objects.

    Phase B's frozen schema uses `question` for the prompt text and carries
    a `cell` field (one of `known_real` / `obscure_real` / `unanswerable`)
    plus `expected_citation` metadata that the aggregation layer needs.
    Both are folded into Question.metadata alongside any pre-existing
    metadata block so the downstream join can recover them by question_id.
    """
    questions: list[Question] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        metadata: dict[str, Any] = dict(record.get("metadata", {}))
        for key, value in record.items():
            if key in _NON_METADATA_FIELDS or key == "metadata":
                continue
            metadata[key] = value
        questions.append(
            Question(
                id=record["id"],
                domain=record["domain"],
                text=record["question"],
                expected_existence=record["expected_existence"],
                metadata=metadata,
            )
        )
    return questions


def load_model_registry(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL row per Phase B model. Order is preserved."""
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        rows.append(json.loads(raw))
    return rows


def filter_registry(
    registry: list[dict[str, Any]], only: list[str] | None
) -> list[dict[str, Any]]:
    """Optionally subset the registry by `model_id`, preserving registry order.

    Registry order matters: §16 commits to Haiku-first staging, and the
    registry file is the source of truth for that ordering.
    """
    if only is None:
        return registry
    requested = set(only)
    known = {row["model_id"] for row in registry}
    unknown = requested - known
    if unknown:
        raise ValueError(
            f"--only-models values not in registry: {sorted(unknown)}. "
            f"Registry has: {sorted(known)}"
        )
    return [row for row in registry if row["model_id"] in requested]


def _spec_from_registry_row(row: dict[str, Any]) -> GenerationSpec:
    return GenerationSpec(
        model_id=row["model_id"],
        provider="anthropic",
        temperature=1.0,
        max_output_tokens=1024,
    )


async def run(
    *,
    questions_path: Path,
    registry_path: Path,
    output_dir: Path,
    manifest_path: Path,
    trials: int,
    only_models: list[str] | None,
    client: Any,
) -> None:
    """Iterate the (filtered) registry, submit one batch per model."""
    questions = load_phase_b_questions(questions_path)
    registry = filter_registry(load_model_registry(registry_path), only_models)
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in registry:
        spec = _spec_from_registry_row(row)
        await generate_via_batch_api(
            questions=questions,
            conditions=list(CONDITIONS),
            spec=spec,
            client=client,
            output_path=output_dir / f"{spec.model_id}.jsonl",
            manifest_path=manifest_path,
            num_trials=trials,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src.run_phase_b",
        description="Phase B Claude E1+E2 orchestrator (Batch API).",
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=_DEFAULT_TRIALS)
    parser.add_argument(
        "--only-models",
        nargs="+",
        default=None,
        help="Subset registry to these model_ids (registry order preserved).",
    )
    return parser.parse_args(argv)


async def _amain(args: argparse.Namespace) -> None:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    await run(
        questions_path=args.questions,
        registry_path=args.registry,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        trials=args.trials,
        only_models=args.only_models,
        client=client,
    )


def main() -> None:
    args = parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
