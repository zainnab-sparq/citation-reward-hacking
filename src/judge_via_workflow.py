"""Path 7 J2a dispatcher: Workflow-tool (parent-session Max-OAuth) judge layer.

WHY THIS EXISTS
---------------
The §11 #6 (d) sub-amendment first dispatched the J2a judge via `claude -p`
subprocess (src/judge_via_subagents.py), expecting it to bill the Max plan's
interactive pool. On 2026-06-07/08, after 3,821 of 10,693 calls, the paid
Anthropic API balance went negative. Root cause: `claude -p` bills the paid
API even with OAuth credentials and no ANTHROPIC_API_KEY set — a documented
Claude Code behavior (anthropics/claude-code issues #43333, #37686). The
3,821 completed judgments are methodologically valid (Sonnet 4.6 + WebSearch
+ WebFetch + v2 §6 prompt); only the billing path was wrong.

This module dispatches the REMAINING 6,872 via the Claude Code `Workflow`
tool, whose `agent()` calls run in the parent session and bill against the
Max plan's interactive pool ($0). Methodology is held identical:
  - same judge model (Sonnet 4.6, pinned model="sonnet")
  - same tool stack (WebSearch + WebFetch)
  - same prompt: the per-citation prompt is rendered HERE by the canonical
    src.judge_via_subagents._render_prompt (via build_judge_inputs) and written
    verbatim to a file the workflow agent reads, so the judged content is
    byte-identical to the subprocess batch.

WORKFLOW CONSTRAINT
-------------------
Workflow scripts have no filesystem access; only their spawned agents do (via
the Read tool). So the orchestration is:
  1. (here, Python) build remaining JudgeInputs, write each prompt to a file in
     a non-Dropbox cache dir, emit compact {custom_id, prompt_path} items.
  2. (Workflow JS) each agent Reads its file, judges, returns the verdict; the
     workflow closure re-attaches custom_id.
  3. (here, Python) normalize returned verdicts to the canonical 6-key schema
     and append to the same JSONL the subprocess batch wrote.

Resume safety: a custom_id present as a clean (error=None, status set) row in
the output JSONL is skipped on the next build_remaining_items() call, so a
killed run is restarted by simply rebuilding the remaining set.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.judge_via_subagents import JudgeRecord, build_judge_inputs

_VALID_STATUSES = frozenset({"REAL", "FABRICATED", "UNVERIFIABLE"})
_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _prompt_filename(custom_id: str) -> str:
    """Map a custom_id to a filesystem-safe filename for its prompt file.

    Phase B Claude custom_ids contain only [A-Za-z0-9._-] and are used
    verbatim. Phase C open-weights custom_ids embed the model_id's '/'
    (e.g. 'allenai/OLMo-2-1124-13B'), an illegal path separator that would
    otherwise be written as a nested subdirectory (raising FileNotFoundError).
    Those are sanitized to '-' and given a short hash suffix of the ORIGINAL
    custom_id, so the mapping stays injective even when two distinct ids would
    sanitize to the same stem (e.g. 'org/model' vs 'org-model')."""
    if not _UNSAFE_FILENAME_RE.search(custom_id):
        return f"{custom_id}.txt"
    safe = _UNSAFE_FILENAME_RE.sub("-", custom_id)
    digest = hashlib.sha1(custom_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe}.{digest}.txt"


@dataclass(frozen=True)
class JudgeWorkflowItem:
    """One compact unit passed to the Workflow via args: the stable custom_id
    plus the absolute path of the file holding the rendered judge prompt."""
    custom_id: str
    prompt_path: str


def successful_custom_ids(output_path: Path) -> set[str]:
    """custom_ids that already have a clean judgment (error is None AND a
    judge_status is present). Error/null rows are NOT counted, so they get
    re-judged on resume."""
    done: set[str] = set()
    if not output_path.exists():
        return done
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("error") is None and row.get("judge_status"):
            cid = row.get("custom_id")
            if isinstance(cid, str):
                done.add(cid)
    return done


def build_remaining_items(
    extractions_path: Path,
    verifications_path: Path,
    qmap_path: Path,
    output_path: Path,
    prompts_dir: Path,
) -> list[JudgeWorkflowItem]:
    """Build the list of not-yet-judged J2a items, writing each one's rendered
    prompt to `prompts_dir/<custom_id>.txt` (idempotent — skips files that
    already exist). The prompt text is the canonical v2 §6 prompt from
    build_judge_inputs, written verbatim."""
    prompts_dir.mkdir(parents=True, exist_ok=True)
    done = successful_custom_ids(output_path)

    items: list[JudgeWorkflowItem] = []
    for inp in build_judge_inputs(extractions_path, verifications_path, qmap_path):
        if inp.custom_id in done:
            continue
        path = prompts_dir / _prompt_filename(inp.custom_id)
        if not path.exists():
            path.write_text(inp.prompt, encoding="utf-8")
        items.append(JudgeWorkflowItem(custom_id=inp.custom_id, prompt_path=str(path)))
    return items


def chunk(seq: list[Any], size: int) -> list[list[Any]]:
    """Split seq into consecutive sublists of length `size` (last may be
    shorter)."""
    if size <= 0:
        raise ValueError("size must be positive")
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def make_workflow_args(items: list[JudgeWorkflowItem]) -> dict[str, Any]:
    """Build the JSON-serializable args blob for a Workflow batch."""
    return {"items": [{"custom_id": it.custom_id, "prompt_path": it.prompt_path} for it in items]}


def _normalize_one(result_item: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one workflow result entry into a canonical JudgeRecord dict, or
    return None if it is a failure (no valid §6 status) that must be re-judged."""
    cid = result_item.get("custom_id")
    if not isinstance(cid, str):
        return None
    status = result_item.get("status")
    if status not in _VALID_STATUSES:
        return None
    rec = JudgeRecord(
        custom_id=cid,
        judge_status=status,
        judge_reasoning=str(result_item.get("reasoning", "") or ""),
        judge_evidence_url=str(result_item.get("evidence_url", "") or ""),
        judge_search_count=int(result_item.get("search_count", 0) or 0),
        error=None,
    )
    return asdict(rec)


def records_from_workflow_results(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split a workflow batch's returned `results` array into
    (canonical_success_records, failed_custom_ids). Failures (null verdict,
    agent error, or invalid status) are returned as custom_ids so the caller
    can leave them un-persisted for the resume pass."""
    records: list[dict[str, Any]] = []
    failed: list[str] = []
    for item in results:
        rec = _normalize_one(item)
        if rec is None:
            cid = item.get("custom_id")
            if isinstance(cid, str):
                failed.append(cid)
        else:
            records.append(rec)
    return records, failed


def persist_records(records: list[dict[str, Any]], output_path: Path) -> None:
    """Append canonical records to the output JSONL (creating parent dirs)."""
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
