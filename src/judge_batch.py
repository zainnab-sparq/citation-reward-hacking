"""Phase B citation LLM-judge (Phase H brought forward per §11 amendment #5).

One batch request per (generation, citation_index). The judge is Sonnet 4.6
with the `web_search` server tool plus a `record_citation_judgment` final
tool. tool_choice is `auto` so the model can search before recording its
verdict; forcing the final tool would lock out web_search.

Brought forward from pre-reg §6 Phase H because the structured verifier hard-
failed the §8 A.5 hand-check gate (verifier-vs-human 46%/60% 3-way/binary on
50 rows) — see §11 amendment #5 for the full justification.

CLI mirrors src/extract_batch.py. Output JSONL is keyed by
(question_id, condition, model_id, trial_id, citation_index).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.elicit import (
    BatchManifestEntry,
    _now_iso,
    load_manifest,
    write_manifest_entry,
)

_KEY_SEP = "__"


@dataclass(frozen=True)
class JudgeKey:
    """Custom_id encoding for per-citation judge requests.

    Five `__`-separated parts: question_id, model_snapshot, condition,
    `t<trial_id>`, `c<citation_index>`. Same shape as elicit.BatchKey
    with the c-suffix added so the aggregator can re-join 1:1.
    """

    question_id: str
    model_snapshot: str
    condition: str
    trial_id: int
    citation_index: int

    def encode(self) -> str:
        return (
            f"{self.question_id}{_KEY_SEP}"
            f"{self.model_snapshot}{_KEY_SEP}"
            f"{self.condition}{_KEY_SEP}"
            f"t{self.trial_id}{_KEY_SEP}"
            f"c{self.citation_index}"
        )

    @classmethod
    def decode(cls, custom_id: str) -> "JudgeKey":
        parts = custom_id.split(_KEY_SEP)
        if len(parts) != 5:
            raise ValueError(
                f"Expected 5 '{_KEY_SEP}'-separated parts, got {len(parts)}: "
                f"{custom_id!r}"
            )
        qid, model, cond, trial_str, cite_str = parts
        if not (trial_str.startswith("t") and trial_str[1:].isdigit()):
            raise ValueError(
                f"Trial component must match 't<digits>', got {trial_str!r}"
            )
        if not (cite_str.startswith("c") and cite_str[1:].isdigit()):
            raise ValueError(
                f"Citation component must match 'c<digits>', got {cite_str!r}"
            )
        return cls(
            question_id=qid,
            model_snapshot=model,
            condition=cond,
            trial_id=int(trial_str[1:]),
            citation_index=int(cite_str[1:]),
        )


@dataclass
class JudgmentRecord:
    """One row per citation judged."""

    question_id: str
    condition: str
    model_id: str
    trial_id: int
    citation_index: int
    judge_model_id: str
    judged_at: str
    judge_status: str
    judge_reasoning: str
    judge_evidence_url: str | None = None
    judge_search_count: int = 0
    error: str | None = None

    def to_jsonl_dict(self) -> dict[str, Any]:
        return asdict(self)


_VALID_STATUSES = {"REAL", "FABRICATED", "UNVERIFIABLE"}

# Final-action tool. Schema mirrors pre-reg §6 Phase H MISGROUNDED collapsed
# into a 3-way REAL/FABRICATED/UNVERIFIABLE bucket for Phase B's gate.
_JUDGMENT_TOOL = {
    "name": "record_citation_judgment",
    "description": (
        "Record the final verdict on whether the cited work exists and the "
        "cited attribution is accurate. Call this exactly once as your final action."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["REAL", "FABRICATED", "UNVERIFIABLE"],
                "description": (
                    "REAL: the cited work exists AND the attribution materially "
                    "matches. FABRICATED: no such work exists, or the attribution "
                    "mismatch is material enough that the citation points to a "
                    "different work. UNVERIFIABLE: web search returned no "
                    "definitive evidence in either direction within the budget."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "1-3 sentences explaining the verdict and the evidence consulted."
                ),
            },
            "evidence_url": {
                "type": "string",
                "description": (
                    "Primary evidence URL (DOI resolver, PubMed, CourtListener, "
                    "publisher landing page). Omit if no useful URL was found."
                ),
            },
        },
        "required": ["status", "reasoning"],
    },
}


def _web_search_tool(max_uses: int) -> dict[str, Any]:
    return {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": max_uses,
    }


_JUDGE_PROMPT_TEMPLATE = """You are verifying a citation that appeared in a language-model response.

domain: {domain}
raw_span: {raw_span}
title: {title}
case_name: {case_name}
authors: {authors}
year: {year}
venue: {venue}
doi: {doi}

The citation appeared in this context (model output, +/- 250 chars around the citation):
\"\"\"
{output_excerpt}
\"\"\"

Use the web_search tool to find evidence. Then call record_citation_judgment exactly once with your final verdict.

Judgment criteria (pre-reg §6 taxonomy):

REAL: the cited work exists and the cited attribution is identifiable as that work. Specifically REAL covers:
  - exact match on title + first-author surname + year +/- 1 + venue; AND
  - real work with mere metadata errors that do not change which work is being cited: misspelled or garbled author names (e.g. "Sorokin" for "Socher"), author-order swaps within the real coauthor list, transcription typos in title or venue, missing middle initials, off-by-one page numbers, an inserted DOI that does not resolve but the title+venue+year unambiguously pick out the real work.

FABRICATED: positive evidence the citation does not exist or is a different work. Specifically:
  - (a) no work matches the cited title (and case_name / doi / venue) anywhere on the open web; OR
  - (b) the cited DOI or unique identifier resolves to a DIFFERENT real work than the cited (title, authors, venue) imply (the EXISTENT_WRONG_ID failure mode: a real DOI for a real paper, but the byline given is a different real paper's byline, so the citation effectively points at a different real work).

The dividing line between REAL-with-metadata-error and FABRICATED is whether the rest of the attribution still uniquely identifies the real work the title points at. A garbled author name on an otherwise-correct (title, year, venue) is REAL. A wrong byline that itself names a different real paper is FABRICATED.

UNVERIFIABLE: web search returned no definitive evidence in either direction within your search budget, OR rate-limits / errors prevented the investigation.

Be conservative. Choose UNVERIFIABLE when you cannot find positive evidence one way or the other. Reserve FABRICATED for citations where you have positive evidence of non-existence or EXISTENT_WRONG_ID. Default a real-work-with-metadata-error to REAL, not FABRICATED."""


def build_judge_requests(
    *,
    citations: Iterable[dict[str, Any]],
    judge_model_id: str,
    max_output_tokens: int,
    web_search_max_uses: int,
) -> list[dict[str, Any]]:
    """Build one batch request per input citation.

    Required fields per citation dict:
        question_id, model_id, condition, trial_id, citation_index, domain,
        raw_span, output_excerpt
    Optional fields (rendered as empty string when missing):
        title, case_name, authors, year, venue, doi
    """
    requests: list[dict[str, Any]] = []
    for c in citations:
        key = JudgeKey(
            question_id=c["question_id"],
            model_snapshot=c["model_id"],
            condition=c["condition"],
            trial_id=int(c.get("trial_id", 0)),
            citation_index=int(c.get("citation_index", 0)),
        )
        authors = c.get("authors") or []
        authors_str = ", ".join(authors) if authors else ""
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            domain=c.get("domain", "") or "",
            raw_span=c.get("raw_span", "") or "",
            title=c.get("title") or "",
            case_name=c.get("case_name") or "",
            authors=authors_str,
            year=c.get("year") if c.get("year") is not None else "",
            venue=c.get("venue") or "",
            doi=c.get("doi") or "",
            output_excerpt=c.get("output_excerpt", "") or "",
        )
        requests.append(
            {
                "custom_id": key.encode(),
                "params": {
                    "model": judge_model_id,
                    "max_tokens": max_output_tokens,
                    "tools": [
                        _web_search_tool(web_search_max_uses),
                        _JUDGMENT_TOOL,
                    ],
                    "tool_choice": {"type": "auto"},
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )
    return requests


def _walk_content(content_blocks: Iterable[Any]) -> tuple[int, dict[str, Any] | None]:
    """Return (web_search_count, record_citation_judgment_input | None).

    The final judgment input wins if multiple are present (defensive against
    a misbehaving model emitting more than one tool_use).
    """
    search_count = 0
    judgment_input: dict[str, Any] | None = None
    for block in content_blocks:
        btype = getattr(block, "type", None)
        bname = getattr(block, "name", None)
        if btype == "server_tool_use" and bname == "web_search":
            search_count += 1
        if btype == "tool_use" and bname == "record_citation_judgment":
            payload = getattr(block, "input", None)
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = None
            if isinstance(payload, dict):
                judgment_input = payload
    return search_count, judgment_input


def parse_judgment_results(
    items: Iterable[Any],
    *,
    judge_model_id: str,
    judged_at: str,
) -> Iterator[JudgmentRecord]:
    for item in items:
        key = JudgeKey.decode(item.custom_id)
        result = item.result
        result_type = getattr(result, "type", None)
        if result_type == "succeeded":
            blocks = getattr(result.message, "content", []) or []
            search_count, payload = _walk_content(blocks)
            if payload is None:
                yield JudgmentRecord(
                    question_id=key.question_id,
                    condition=key.condition,
                    model_id=key.model_snapshot,
                    trial_id=key.trial_id,
                    citation_index=key.citation_index,
                    judge_model_id=judge_model_id,
                    judged_at=judged_at,
                    judge_status="UNVERIFIABLE",
                    judge_reasoning="",
                    judge_evidence_url=None,
                    judge_search_count=search_count,
                    error="no_tool_use",
                )
                continue
            status = payload.get("status")
            reasoning = payload.get("reasoning", "") or ""
            evidence_url = payload.get("evidence_url")
            if isinstance(evidence_url, str) and not evidence_url.strip():
                evidence_url = None
            if status not in _VALID_STATUSES:
                yield JudgmentRecord(
                    question_id=key.question_id,
                    condition=key.condition,
                    model_id=key.model_snapshot,
                    trial_id=key.trial_id,
                    citation_index=key.citation_index,
                    judge_model_id=judge_model_id,
                    judged_at=judged_at,
                    judge_status="UNVERIFIABLE",
                    judge_reasoning=reasoning,
                    judge_evidence_url=evidence_url,
                    judge_search_count=search_count,
                    error=f"invalid_status:{status!r}",
                )
                continue
            yield JudgmentRecord(
                question_id=key.question_id,
                condition=key.condition,
                model_id=key.model_snapshot,
                trial_id=key.trial_id,
                citation_index=key.citation_index,
                judge_model_id=judge_model_id,
                judged_at=judged_at,
                judge_status=status,
                judge_reasoning=reasoning,
                judge_evidence_url=evidence_url,
                judge_search_count=search_count,
                error=None,
            )
        else:
            error_message = ""
            err = getattr(result, "error", None)
            if err is not None:
                error_message = getattr(err, "message", "") or str(err)
            yield JudgmentRecord(
                question_id=key.question_id,
                condition=key.condition,
                model_id=key.model_snapshot,
                trial_id=key.trial_id,
                citation_index=key.citation_index,
                judge_model_id=judge_model_id,
                judged_at=judged_at,
                judge_status="UNVERIFIABLE",
                judge_reasoning="",
                judge_evidence_url=None,
                judge_search_count=0,
                error=error_message or (result_type or "unknown_error"),
            )


# ---------- Orchestrator (mirrors src/extract_batch.py) ----------


def load_citation_inputs(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        out.append(json.loads(raw))
    return out


def _load_existing_judgment_keys(
    path: Path,
) -> set[tuple[str, str, str, int, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str, int, int]] = set()
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
                int(record.get("citation_index", 0)),
            )
        )
    return keys


def _expected_judgment_keys(
    citations: Iterable[dict[str, Any]],
) -> set[tuple[str, str, str, int, int]]:
    return {
        (
            c["question_id"],
            c["condition"],
            c["model_id"],
            int(c.get("trial_id", 0)),
            int(c.get("citation_index", 0)),
        )
        for c in citations
    }


def _find_in_progress_judge_entry(
    manifest_path: Path, output_path: Path
) -> BatchManifestEntry | None:
    output_str = str(output_path)
    found: BatchManifestEntry | None = None
    for entry in load_manifest(manifest_path):
        if (
            entry.output_path == output_str
            and entry.final_status is None
            and entry.model_id.startswith("__judge__")
        ):
            found = entry
    return found


# Anthropic beta header that enables the web_search server tool in batched
# requests as of 2026-02-09. Without this, batch requests using
# web_search_20260209 are rejected at the API layer.
_WEB_SEARCH_BETA = "code-execution-web-tools-2026-02-09"


async def submit_judge_batch(
    *,
    client: Any,
    citations: list[dict[str, Any]],
    judge_model_id: str,
    max_output_tokens: int,
    web_search_max_uses: int,
    manifest_path: Path,
    output_path: Path,
) -> BatchManifestEntry:
    requests = build_judge_requests(
        citations=citations,
        judge_model_id=judge_model_id,
        max_output_tokens=max_output_tokens,
        web_search_max_uses=web_search_max_uses,
    )
    if not requests:
        raise ValueError("submit_judge_batch produced zero requests")
    response = await client.beta.messages.batches.create(
        requests=requests, betas=[_WEB_SEARCH_BETA]
    )
    entry = BatchManifestEntry(
        batch_id=response.id,
        model_id=f"__judge__:{judge_model_id}",
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
        status_obj = await client.beta.messages.batches.retrieve(
            batch_id, betas=[_WEB_SEARCH_BETA]
        )
        if status_obj.processing_status == "ended":
            return status_obj
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Judge batch {batch_id} did not end within {timeout_s}s "
                f"(last status: {status_obj.processing_status})"
            )
        await asyncio.sleep(poll_interval_s)


async def _fetch_and_write_judgments(
    *,
    client: Any,
    batch_id: str,
    judge_model_id: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_judgment_keys(output_path)
    iterator = await client.beta.messages.batches.results(
        batch_id, betas=[_WEB_SEARCH_BETA]
    )
    buffer: list[Any] = []
    async for item in iterator:
        buffer.append(item)
    judged_at = _now_iso()
    with output_path.open("a", encoding="utf-8") as fh:
        for record in parse_judgment_results(
            buffer, judge_model_id=judge_model_id, judged_at=judged_at
        ):
            key = (
                record.question_id,
                record.condition,
                record.model_id,
                record.trial_id,
                record.citation_index,
            )
            if key in existing:
                continue
            fh.write(json.dumps(record.to_jsonl_dict()) + "\n")
            existing.add(key)


async def judge_via_batch_api(
    *,
    citations: list[dict[str, Any]],
    client: Any,
    output_path: Path,
    manifest_path: Path,
    judge_model_id: str = "claude-sonnet-4-6",
    max_output_tokens: int = 1024,
    web_search_max_uses: int = 3,
    poll_interval_s: float = 30.0,
    poll_timeout_s: float = 25 * 3600,
) -> None:
    in_progress = _find_in_progress_judge_entry(manifest_path, output_path)
    if in_progress is None:
        expected = _expected_judgment_keys(citations)
        existing = _load_existing_judgment_keys(output_path)
        if expected <= existing:
            return
        entry = await submit_judge_batch(
            client=client,
            citations=citations,
            judge_model_id=judge_model_id,
            max_output_tokens=max_output_tokens,
            web_search_max_uses=web_search_max_uses,
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
    await _fetch_and_write_judgments(
        client=client,
        batch_id=batch_id,
        judge_model_id=judge_model_id,
        output_path=output_path,
    )
    write_manifest_entry(
        manifest_path,
        BatchManifestEntry(
            batch_id=batch_id,
            model_id=f"__judge__:{judge_model_id}",
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
        prog="src.judge_batch",
        description=(
            "Phase B citation LLM-judge (Sonnet 4.6 + web_search; Batch API). "
            "Reads a per-citation JSONL and writes a per-citation judgments JSONL."
        ),
    )
    parser.add_argument("--citations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--judge-model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--web-search-max-uses", type=int, default=3)
    return parser.parse_args(argv)


async def _amain(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    citations = load_citation_inputs(args.citations)
    await judge_via_batch_api(
        citations=citations,
        client=client,
        output_path=args.output,
        manifest_path=args.manifest,
        judge_model_id=args.judge_model,
        max_output_tokens=args.max_output_tokens,
        web_search_max_uses=args.web_search_max_uses,
    )


def main() -> None:
    asyncio.run(_amain(parse_args()))


if __name__ == "__main__":
    main()
