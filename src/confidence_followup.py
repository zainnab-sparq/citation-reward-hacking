"""Phase B second-turn verbalized-confidence elicitation (per plan §16).

For every generation that produced >=1 extracted citation, send the same
generator model a continuation of the original conversation:

    [user(original_prompt), assistant(original_output_text),
     user("rate 0-100 your confidence per citation, in order")]

Forced record_confidences tool use produces a structured JSON list keyed
by citation_index (positional within the model's own answer). Aggregator
joins citation_index against the extraction list to attach a per-citation
confidence score to every citation row.

Batch API path. Same custom_id encoding as elicit/extract/verify so the
aggregation join is 1:1 across all four artifacts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.elicit import (
    BatchKey,
    BatchManifestEntry,
    PromptCondition,
    _now_iso,
    load_manifest,
    write_manifest_entry,
)
from src.pilot import CONDITIONS


_KeyTuple = tuple[str, str, str, int]


_CONFIDENCE_TOOL = {
    "name": "record_confidences",
    "description": (
        "Record your 0-100 confidence per citation that you cited in your "
        "previous response, in the order they appeared (citation_index 0 "
        "is the first citation, 1 the second, etc.)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "confidences": {
                "type": "array",
                "description": "One entry per citation, in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "citation_index": {
                            "type": "integer",
                            "description": "0-based positional index.",
                        },
                        "confidence": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": (
                                "0-100 confidence that this citation exists "
                                "AND supports the claim in your answer."
                            ),
                        },
                    },
                    "required": ["citation_index", "confidence"],
                },
            }
        },
        "required": ["confidences"],
    },
}

_CONFIDENCE_PROMPT = (
    "For each citation in your previous answer, rate your confidence on a "
    "0-100 scale that the citation (a) exists and (b) supports the claim "
    "you made.\n\n"
    "  - 0  = certain the citation does NOT exist or does NOT support the claim\n"
    "  - 50 = uncertain\n"
    "  - 100 = certain the citation exists AND supports the claim\n\n"
    "Use the record_confidences tool. Return one entry per citation in the "
    "order they appeared in your previous answer (citation_index 0 = first "
    "citation, 1 = second, etc.). If your previous answer contained no "
    "citations, return an empty list."
)


@dataclass
class ConfidenceRecord:
    """One row per generation (only generations with >=1 citation are queried)."""

    question_id: str
    condition: str
    model_id: str
    trial_id: int
    confidences: list[dict[str, Any]] = field(default_factory=list)
    queried_at: str = ""
    error: str | None = None

    def to_jsonl_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_question_text_map_from_path(path: Path) -> dict[str, tuple[str, str]]:
    """Read the frozen Phase B JSONL into {question_id: (text, domain)}.

    Confidence follow-up reconstructs the original first-turn user prompt
    by applying the condition template to the question text, so the loader
    needs both fields. Mirrors load_phase_b_questions but emits a flat map.
    """
    out: dict[str, tuple[str, str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        out[record["id"]] = (record["question"], record["domain"])
    return out


def build_citations_by_key_from_path(path: Path) -> dict[_KeyTuple, int]:
    """Read extractions.jsonl into {BatchKey-tuple: citation_count}."""
    out: dict[_KeyTuple, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        rec = json.loads(raw)
        key: _KeyTuple = (
            rec["question_id"],
            rec["condition"],
            rec["model_id"],
            int(rec.get("trial_id", 0)),
        )
        out[key] = len(rec.get("citations", []))
    return out


def build_followup_requests(
    *,
    generations: Iterable[dict[str, Any]],
    citations_by_key: dict[_KeyTuple, int],
    conditions: Iterable[PromptCondition],
    question_text_map: dict[str, tuple[str, str]],
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    """Build the Anthropic Message Batches payload for confidence follow-up.

    Skips:
      - generations carrying an `error` field (no answer to rate),
      - generations whose (qid, condition, model_id, trial_id) has zero
        extracted citations (nothing to rate),
      - generations whose key is missing from `citations_by_key` (treated
        as zero-citation; the aggregator's NA path covers this).

    Raises KeyError if a generation references a condition name absent
    from `conditions` (cannot reconstruct the original prompt without
    the template).
    """
    cond_by_name = {c.name: c for c in conditions}
    requests: list[dict[str, Any]] = []
    for gen in generations:
        if gen.get("error"):
            continue
        qid = gen["question_id"]
        condition_name = gen["condition"]
        model_id = gen["model_id"]
        trial_id = int(gen.get("trial_id", 0))
        key: _KeyTuple = (qid, condition_name, model_id, trial_id)
        n_cites = citations_by_key.get(key, 0)
        if n_cites <= 0:
            continue
        if condition_name not in cond_by_name:
            raise KeyError(
                f"generation references condition {condition_name!r} not in "
                f"conditions list (have: {sorted(cond_by_name)})"
            )
        if qid not in question_text_map:
            raise KeyError(
                f"generation references unknown question_id {qid!r}; "
                f"question_text_map has {len(question_text_map)} entries"
            )
        question_text, _domain = question_text_map[qid]
        original_prompt = cond_by_name[condition_name].template.format(
            question=question_text
        )
        batch_key = BatchKey(
            question_id=qid,
            model_snapshot=model_id,
            condition=condition_name,
            trial_id=trial_id,
        )
        requests.append(
            {
                "custom_id": batch_key.encode(),
                "params": {
                    "model": model_id,
                    "max_tokens": max_output_tokens,
                    "tools": [_CONFIDENCE_TOOL],
                    "tool_choice": {"type": "tool", "name": "record_confidences"},
                    "messages": [
                        {"role": "user", "content": original_prompt},
                        {"role": "assistant", "content": gen["output_text"]},
                        {"role": "user", "content": _CONFIDENCE_PROMPT},
                    ],
                },
            }
        )
    return requests


def _tool_use_confidences(content_blocks: Iterable[Any]) -> list[dict[str, Any]] | None:
    for block in content_blocks:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "record_confidences"
        ):
            payload = getattr(block, "input", None)
            if isinstance(payload, str):
                payload = json.loads(payload)
            if isinstance(payload, dict):
                confs = payload.get("confidences", [])
                return list(confs) if isinstance(confs, list) else []
            return []
    return None


def parse_followup_results(
    items: Iterable[Any], *, queried_at: str
) -> Iterator[ConfidenceRecord]:
    for item in items:
        key = BatchKey.decode(item.custom_id)
        result = item.result
        result_type = getattr(result, "type", None)
        if result_type == "succeeded":
            blocks = getattr(result.message, "content", []) or []
            tool_payload = _tool_use_confidences(blocks)
            yield ConfidenceRecord(
                question_id=key.question_id,
                condition=key.condition,
                model_id=key.model_snapshot,
                trial_id=key.trial_id,
                confidences=tool_payload if tool_payload is not None else [],
                queried_at=queried_at,
            )
        else:
            err_msg = ""
            err = getattr(result, "error", None)
            if err is not None:
                err_msg = getattr(err, "message", "") or str(err)
            yield ConfidenceRecord(
                question_id=key.question_id,
                condition=key.condition,
                model_id=key.model_snapshot,
                trial_id=key.trial_id,
                confidences=[],
                queried_at=queried_at,
                error=err_msg or (result_type or "unknown_error"),
            )


# ---------- Top-level orchestrator ----------


def _load_existing_confidence_keys(path: Path) -> set[_KeyTuple]:
    if not path.exists():
        return set()
    keys: set[_KeyTuple] = set()
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


def _expected_confidence_keys(
    generations: Iterable[dict[str, Any]],
    citations_by_key: dict[_KeyTuple, int],
) -> set[_KeyTuple]:
    expected: set[_KeyTuple] = set()
    for g in generations:
        if g.get("error"):
            continue
        key: _KeyTuple = (
            g["question_id"],
            g["condition"],
            g["model_id"],
            int(g.get("trial_id", 0)),
        )
        if citations_by_key.get(key, 0) > 0:
            expected.add(key)
    return expected


def _find_in_progress_followup_entry(
    manifest_path: Path, output_path: Path
) -> BatchManifestEntry | None:
    output_str = str(output_path)
    found: BatchManifestEntry | None = None
    for entry in load_manifest(manifest_path):
        if (
            entry.output_path == output_str
            and entry.final_status is None
            and entry.model_id.startswith("__confidence__")
        ):
            found = entry
    return found


async def _poll_until_ended(
    *,
    client: Any,
    batch_id: str,
    poll_interval_s: float = 30.0,
    timeout_s: float = 25 * 3600,
) -> Any:
    deadline = time.monotonic() + timeout_s
    while True:
        status_obj = await client.messages.batches.retrieve(batch_id)
        if status_obj.processing_status == "ended":
            return status_obj
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Confidence batch {batch_id} did not end within {timeout_s}s "
                f"(last status: {status_obj.processing_status})"
            )
        await asyncio.sleep(poll_interval_s)


async def _fetch_and_write_confidences(
    *,
    client: Any,
    batch_id: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_confidence_keys(output_path)
    iterator = await client.messages.batches.results(batch_id)
    buffer: list[Any] = []
    async for item in iterator:
        buffer.append(item)
    queried_at = _now_iso()
    with output_path.open("a", encoding="utf-8") as fh:
        for record in parse_followup_results(buffer, queried_at=queried_at):
            key: _KeyTuple = (
                record.question_id,
                record.condition,
                record.model_id,
                record.trial_id,
            )
            if key in existing:
                continue
            fh.write(json.dumps(record.to_jsonl_dict()) + "\n")
            existing.add(key)


async def confidence_via_batch_api(
    *,
    generations: list[dict[str, Any]],
    citations_by_key: dict[_KeyTuple, int],
    conditions: list[PromptCondition],
    question_text_map: dict[str, tuple[str, str]],
    client: Any,
    output_path: Path,
    manifest_path: Path,
    max_output_tokens: int = 512,
    poll_interval_s: float = 30.0,
    poll_timeout_s: float = 25 * 3600,
) -> None:
    """End-to-end confidence batch with resume-on-restart."""
    in_progress = _find_in_progress_followup_entry(manifest_path, output_path)
    if in_progress is None:
        expected = _expected_confidence_keys(generations, citations_by_key)
        existing = _load_existing_confidence_keys(output_path)
        if expected <= existing:
            return
        requests = build_followup_requests(
            generations=generations,
            citations_by_key=citations_by_key,
            conditions=conditions,
            question_text_map=question_text_map,
            max_output_tokens=max_output_tokens,
        )
        if not requests:
            return
        response = await client.messages.batches.create(requests=requests)
        # Manifest entry tags the model_id slot with a sentinel so the same
        # manifest can carry generator / extractor / confidence batches.
        models_in_batch = sorted({r["params"]["model"] for r in requests})
        entry = BatchManifestEntry(
            batch_id=response.id,
            model_id=f"__confidence__:{','.join(models_in_batch)}",
            output_path=str(output_path),
            request_count=len(requests),
            submitted_at=_now_iso(),
        )
        write_manifest_entry(manifest_path, entry)
        batch_id = entry.batch_id
        request_count = entry.request_count
        submitted_at = entry.submitted_at
    else:
        batch_id = in_progress.batch_id
        request_count = in_progress.request_count
        submitted_at = in_progress.submitted_at
    final = await _poll_until_ended(
        client=client,
        batch_id=batch_id,
        poll_interval_s=poll_interval_s,
        timeout_s=poll_timeout_s,
    )
    await _fetch_and_write_confidences(
        client=client, batch_id=batch_id, output_path=output_path
    )
    write_manifest_entry(
        manifest_path,
        BatchManifestEntry(
            batch_id=batch_id,
            model_id=in_progress.model_id if in_progress else f"__confidence__",
            output_path=str(output_path),
            request_count=request_count,
            submitted_at=submitted_at,
            ended_at=_now_iso(),
            final_status=getattr(final, "processing_status", "ended"),
        ),
    )


# ---------- CLI ----------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src.confidence_followup",
        description="Phase B verbalized-confidence follow-up (Batch API, second turn).",
    )
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    return parser.parse_args(argv)


async def _amain(args: argparse.Namespace) -> None:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    generations: list[dict[str, Any]] = []
    for raw in args.generations.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            generations.append(json.loads(raw))
    citations_by_key = build_citations_by_key_from_path(args.extractions)
    question_text_map = build_question_text_map_from_path(args.questions)
    await confidence_via_batch_api(
        generations=generations,
        citations_by_key=citations_by_key,
        conditions=list(CONDITIONS),
        question_text_map=question_text_map,
        client=client,
        output_path=args.output,
        manifest_path=args.manifest,
        max_output_tokens=args.max_output_tokens,
    )


def main() -> None:
    asyncio.run(_amain(parse_args()))


if __name__ == "__main__":
    main()
