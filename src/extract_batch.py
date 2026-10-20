"""Batch citation extraction (Phase B Step 1 pipeline element).

Reads a generations.jsonl produced by `generate_via_batch_api`, asks
Sonnet 4.6 (forced tool-use) to recover structured citations from each
generation, and writes an extractions.jsonl with one record per
generation. Same custom_id shape as elicit.BatchKey so the aggregation
join is 1:1.

This module is the batch analogue of `AnthropicExtractorClient`
(src/extract_citations.py:173), which is single-request async only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.elicit import (
    BatchKey,
    BatchManifestEntry,
    _now_iso,
    load_manifest,
    write_manifest_entry,
)
from src.extract_citations import _EXTRACTION_PROMPT, _EXTRACTION_TOOL


@dataclass
class ExtractionRecord:
    """One row per generation in the output extractions.jsonl."""

    question_id: str
    condition: str
    model_id: str  # the GENERATOR model (not the extractor)
    trial_id: int
    extractor_model_id: str
    extracted_at: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_jsonl_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_generations(path: Path) -> list[dict[str, Any]]:
    """Read a generations.jsonl into a list of dicts."""
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        out.append(json.loads(raw))
    return out


def build_question_domain_map_from_path(path: Path) -> dict[str, str]:
    """Build a {question_id: domain} map from the frozen Phase B JSONL.

    The frozen schema uses `question` for prompt text, but `domain` is
    consistent with the pilot loader, so a plain field read suffices.
    """
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        out[record["id"]] = record["domain"]
    return out


def build_extraction_requests(
    *,
    generations: Iterable[dict[str, Any]],
    question_domain: dict[str, str],
    extractor_model_id: str,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    """Build the Anthropic Message Batches payload for citation extraction.

    One request per generation, keyed by BatchKey-encoded custom_id whose
    `model_snapshot` field carries the GENERATOR model (so the aggregation
    layer can join back to the source generation; the EXTRACTOR model is
    carried separately in params and in the output ExtractionRecord).

    Rows with a truthy `error` field (failed-batch generations carrying
    no `output_text`) are skipped — there is nothing to extract from.

    Raises KeyError if any generation references a question_id absent
    from `question_domain`.
    """
    requests: list[dict[str, Any]] = []
    for gen in generations:
        if gen.get("error"):
            continue
        qid = gen["question_id"]
        if qid not in question_domain:
            raise KeyError(
                f"generation references unknown question_id {qid!r}; "
                f"map has {len(question_domain)} entries"
            )
        key = BatchKey(
            question_id=qid,
            model_snapshot=gen["model_id"],
            condition=gen["condition"],
            trial_id=int(gen.get("trial_id", 0)),
        )
        prompt = _EXTRACTION_PROMPT.format(
            domain=question_domain[qid], text=gen["output_text"]
        )
        requests.append(
            {
                "custom_id": key.encode(),
                "params": {
                    "model": extractor_model_id,
                    "max_tokens": max_output_tokens,
                    "tools": [_EXTRACTION_TOOL],
                    "tool_choice": {"type": "tool", "name": "record_citations"},
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )
    return requests


# --- Dedup post-process (PRE_REGISTRATION_OSF.md §11 amendment #6 (b)) ---
#
# Haiku 4.5 over-counts when the same canonical work is mentioned in multiple
# paragraphs of one response. We collapse citations sharing normalized title +
# normalized first-author surname + year (±1) within the same record,
# preserving the longest raw_span.

_DEDUP_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_DEDUP_WS_RE = re.compile(r"\s+")


def _dedup_norm_title(title: Any) -> str:
    if not isinstance(title, str):
        return ""
    s = title.lower()
    s = _DEDUP_PUNCT_RE.sub(" ", s)
    s = _DEDUP_WS_RE.sub(" ", s).strip()
    return s


def _dedup_first_surname(authors: Any) -> str | None:
    if not authors:
        return None
    if not isinstance(authors, (list, tuple)):
        return None
    first = authors[0]
    if not isinstance(first, str) or not first.strip():
        return None
    return _DEDUP_WS_RE.sub(" ", first.lower().strip())


def _dedup_year(year: Any) -> int | None:
    if isinstance(year, int):
        return year
    if isinstance(year, str) and year.strip().isdigit():
        return int(year.strip())
    return None


def _dedup_keys_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if _dedup_norm_title(a.get("title")) != _dedup_norm_title(b.get("title")):
        return False
    if _dedup_norm_title(a.get("title")) == "":
        return False  # don't collapse two empty-titled rows together
    sa = _dedup_first_surname(a.get("authors"))
    sb = _dedup_first_surname(b.get("authors"))
    if sa != sb:
        return False
    ya = _dedup_year(a.get("year"))
    yb = _dedup_year(b.get("year"))
    if ya is None and yb is None:
        return True
    if ya is None or yb is None:
        return False
    return abs(ya - yb) <= 1


def dedup_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse duplicate citations within a single extraction record.

    Two citations are duplicates iff:
      - normalized titles match (lowercase, punctuation-stripped, ws-collapsed),
        AND title is non-empty
      - first-author surnames match (both None / both present-and-equal)
      - years match (both None OR both present and within ±1)

    On collapse, the duplicate's data is dropped EXCEPT raw_span: the longest
    raw_span across the merged group is preserved on the kept record.
    """
    kept: list[dict[str, Any]] = []
    for cite in citations:
        merged = False
        for k in kept:
            if _dedup_keys_match(k, cite):
                cur_span = k.get("raw_span") or ""
                new_span = cite.get("raw_span") or ""
                if len(new_span) > len(cur_span):
                    k["raw_span"] = new_span
                merged = True
                break
        if not merged:
            kept.append(dict(cite))
    return kept


def _tool_use_citations(content_blocks: Iterable[Any]) -> list[dict[str, Any]] | None:
    """Return the record_citations tool input, or None if no tool_use block found."""
    for block in content_blocks:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "record_citations"
        ):
            payload = getattr(block, "input", None)
            if isinstance(payload, str):
                payload = json.loads(payload)
            if isinstance(payload, dict):
                citations = payload.get("citations", [])
                return list(citations) if isinstance(citations, list) else []
            return []
    return None


def parse_extraction_results(
    items: Iterable[Any],
    *,
    extractor_model_id: str,
    extracted_at: str,
) -> Iterator[ExtractionRecord]:
    """Convert per-batch results into ExtractionRecord objects.

    - succeeded with tool_use → record with parsed citations
    - succeeded with text-only → record with citations=[]
    - errored / expired / canceled → record with citations=[] and error message
    """
    for item in items:
        key = BatchKey.decode(item.custom_id)
        result = item.result
        result_type = getattr(result, "type", None)
        if result_type == "succeeded":
            blocks = getattr(result.message, "content", []) or []
            tool_payload = _tool_use_citations(blocks)
            citations = tool_payload if tool_payload is not None else []
            citations = dedup_citations(citations)
            yield ExtractionRecord(
                question_id=key.question_id,
                condition=key.condition,
                model_id=key.model_snapshot,
                trial_id=key.trial_id,
                extractor_model_id=extractor_model_id,
                extracted_at=extracted_at,
                citations=citations,
            )
        else:
            error_message = ""
            err = getattr(result, "error", None)
            if err is not None:
                error_message = getattr(err, "message", "") or str(err)
            yield ExtractionRecord(
                question_id=key.question_id,
                condition=key.condition,
                model_id=key.model_snapshot,
                trial_id=key.trial_id,
                extractor_model_id=extractor_model_id,
                extracted_at=extracted_at,
                citations=[],
                error=error_message or (result_type or "unknown_error"),
            )


# ---------- Top-level orchestrator (mirrors elicit.generate_via_batch_api) ----------


def _load_existing_extraction_keys(
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


def _expected_extraction_keys(
    generations: Iterable[dict[str, Any]],
) -> set[tuple[str, str, str, int]]:
    return {
        (
            g["question_id"],
            g["condition"],
            g["model_id"],
            int(g.get("trial_id", 0)),
        )
        for g in generations
        if not g.get("error")
    }


def _find_in_progress_extraction_entry(
    manifest_path: Path, output_path: Path
) -> BatchManifestEntry | None:
    output_str = str(output_path)
    found: BatchManifestEntry | None = None
    for entry in load_manifest(manifest_path):
        if (
            entry.output_path == output_str
            and entry.final_status is None
            and entry.model_id.startswith("__extractor__")
        ):
            found = entry
    return found


async def submit_extraction_batch(
    *,
    client: Any,
    generations: list[dict[str, Any]],
    question_domain: dict[str, str],
    extractor_model_id: str,
    max_output_tokens: int,
    manifest_path: Path,
    output_path: Path,
) -> BatchManifestEntry:
    requests = build_extraction_requests(
        generations=generations,
        question_domain=question_domain,
        extractor_model_id=extractor_model_id,
        max_output_tokens=max_output_tokens,
    )
    if not requests:
        raise ValueError(
            "submit_extraction_batch produced zero requests — "
            "no successful generations to extract from"
        )
    response = await client.messages.batches.create(requests=requests)
    entry = BatchManifestEntry(
        batch_id=response.id,
        # Tag with a sentinel so manifest scans can distinguish extractor
        # batches from generator batches over the same output_path.
        model_id=f"__extractor__:{extractor_model_id}",
        output_path=str(output_path),
        request_count=len(requests),
        submitted_at=_now_iso(),
    )
    write_manifest_entry(manifest_path, entry)
    return entry


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
                f"Extractor batch {batch_id} did not end within {timeout_s}s "
                f"(last status: {status_obj.processing_status})"
            )
        await asyncio.sleep(poll_interval_s)


async def _fetch_and_write_extractions(
    *,
    client: Any,
    batch_id: str,
    extractor_model_id: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_extraction_keys(output_path)
    iterator = await client.messages.batches.results(batch_id)
    buffer: list[Any] = []
    async for item in iterator:
        buffer.append(item)
    extracted_at = _now_iso()
    with output_path.open("a", encoding="utf-8") as fh:
        for record in parse_extraction_results(
            buffer, extractor_model_id=extractor_model_id, extracted_at=extracted_at
        ):
            key = (record.question_id, record.condition, record.model_id, record.trial_id)
            if key in existing:
                continue
            fh.write(json.dumps(record.to_jsonl_dict()) + "\n")
            existing.add(key)


async def extract_via_batch_api(
    *,
    generations: list[dict[str, Any]],
    question_domain: dict[str, str],
    client: Any,
    output_path: Path,
    manifest_path: Path,
    extractor_model_id: str = "claude-sonnet-4-6",
    max_output_tokens: int = 2048,
    poll_interval_s: float = 30.0,
    poll_timeout_s: float = 25 * 3600,
) -> None:
    """End-to-end extractor batch with resume-on-restart.

    Same control flow as elicit.generate_via_batch_api but parameterized
    by generations (instead of (questions, conditions, spec)).
    """
    in_progress = _find_in_progress_extraction_entry(manifest_path, output_path)
    if in_progress is None:
        expected = _expected_extraction_keys(generations)
        existing = _load_existing_extraction_keys(output_path)
        if expected <= existing:
            return
        entry = await submit_extraction_batch(
            client=client,
            generations=generations,
            question_domain=question_domain,
            extractor_model_id=extractor_model_id,
            max_output_tokens=max_output_tokens,
            manifest_path=manifest_path,
            output_path=output_path,
        )
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
    await _fetch_and_write_extractions(
        client=client,
        batch_id=batch_id,
        extractor_model_id=extractor_model_id,
        output_path=output_path,
    )
    write_manifest_entry(
        manifest_path,
        BatchManifestEntry(
            batch_id=batch_id,
            model_id=f"__extractor__:{extractor_model_id}",
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
        prog="src.extract_batch",
        description="Phase B citation extractor (Batch API; one row per generation).",
    )
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extractor-model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    return parser.parse_args(argv)


async def _amain(args: argparse.Namespace) -> None:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    generations = load_generations(args.generations)
    question_domain = build_question_domain_map_from_path(args.questions)
    await extract_via_batch_api(
        generations=generations,
        question_domain=question_domain,
        client=client,
        output_path=args.output,
        manifest_path=args.manifest,
        extractor_model_id=args.extractor_model,
        max_output_tokens=args.max_output_tokens,
    )


def main() -> None:
    asyncio.run(_amain(parse_args()))


if __name__ == "__main__":
    main()
